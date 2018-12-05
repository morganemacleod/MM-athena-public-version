//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file pm_envelope.cpp: tidal perturbation of polytropic stellar envelope by one point mass
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#define NARRAY 10000


// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../utils/utils.hpp"
#include "../outputs/outputs.hpp"




Real Interpolate1DArrayEven(Real *x,Real *y,Real x0);

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);


void HSEInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		  FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		  FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
		   FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);


void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);


void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void ParticleAccelsPreInt(Real GMenv, Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void SumGasOnParticleAccels(Mesh *pm, Real (&xia)[3],Real (&ag1i)[3], Real (&ag2i)[3]);

void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

int RefinementCondition(MeshBlock *pmb);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real GetGM2factor(Real time);

void SumComPosVel(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
		     Real (&xgcom)[3],Real (&vgcom)[3],
		     Real (&xcom)[3],Real (&vcom)[3],
		     Real &mg);

void SumTrackfileDiagnostics(Mesh *pm, Real (&xi)[3],Real (&vi)[3],
				     Real (&xgcom)[3],Real (&vgcom)[3],
				     Real (&xcom)[3],Real (&vcom)[3],
			     Real (&lp)[3],Real (&lg)[3],Real (&ldo)[3], Real &Eorb,
			     Real &mb, Real &mu);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);

Real mxOmegaEnv(MeshBlock *pmb, int iout);
Real mEnv(MeshBlock *pmb, int iout);
Real mr1(MeshBlock *pmb, int iout);
Real mr12(MeshBlock *pmb, int iout);


//Real UpdatePMTimestep(MeshBlock *pmb);

// global (to this file) problem parameters
Real gamma_gas; 
Real da,pa; // ambient density, pressure
Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc[NARRAY];  // initial profile

Real GM2, GM1; // point masses
Real rsoft2; // softening length of PM 2
Real t_relax,t_mass_on; // time to damp fluid motion, time to turn on M2 over
int  include_gas_backreaction, corotating_frame; // flags for output, gas backreaction on EOM, frame choice
int n_particle_substeps; // substepping of particle integration
Real agas1_source_rel_thresh; // relative threshold for inclusion of gas accel on particle 1 in EOM 

Real xi[3], vi[3], agas1i[3], agas2i[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration
Real xcom[3], vcom[3]; // cartesian pos/vel of the COM of the particle/gas system
Real xgcom[3], vgcom[3]; // cartesian pos/vel of the COM of the gas
Real lp[3], lg[3], ldo[3];  // particle, gas, and rate of angular momentum loss
Real Eorb; // diagnostic output
Real mb,mu; // masses of bound/unbound gas outside r=1 for diagnostic output

Real Omega[3],  Omega_envelope;  // vector rotation of the frame, initial envelope

Real trackfile_next_time, trackfile_dt;
int  trackfile_number;
int  mode;  // mode=1 (polytrope), mode=2 (wind BC) 

Real Ggrav;

Real separation_start,separation_stop_min, separation_stop_max; // particle separation to abort the integration.

int is_restart;

// Static Refinement with AMR Params
//Real x1_min_level1, x1_max_level1,x2_min_level1, x2_max_level1;
Real x1_min_derefine;

bool do_pre_integrate;
bool fixed_orbit;
Real Omega_orb_fixed,sma_fixed;

Real output_next_sep,dsep_output; // controling user forced output (set with dt=999.)



//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  // read in some global params (to this file)
 
  // first non-mode-dependent settings
  pa   = pin->GetOrAddReal("problem","pamb",1.0);
  da   = pin->GetOrAddReal("problem","damb",1.0);
  gamma_gas = pin->GetReal("hydro","gamma");

  Ggrav = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
  GM2 = pin->GetOrAddReal("problem","GM2",0.0);
  //GM1 = pin->GetOrAddReal("problem","GM1",1.0);

  rsoft2 = pin->GetOrAddReal("problem","rsoft2",0.1);
  t_relax = pin->GetOrAddReal("problem","trelax",0.0);
  t_mass_on = pin->GetOrAddReal("problem","t_mass_on",0.0);
  corotating_frame = pin->GetInteger("problem","corotating_frame");

  trackfile_dt = pin->GetOrAddReal("problem","trackfile_dt",0.01);

  include_gas_backreaction = pin->GetInteger("problem","gas_backreaction");
  agas1_source_rel_thresh =  pin->GetOrAddReal("problem","agas1_source_rel_thresh",0.0);
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");

  separation_stop_min = pin->GetOrAddReal("problem","separation_stop_min",0.0);
  separation_stop_max = pin->GetOrAddReal("problem","separation_stop_max",1.e99);
  separation_start = pin->GetOrAddReal("problem","separation_start",1.e99);

  // These are used for static refinement when using AMR
  x1_min_derefine = pin->GetOrAddReal("problem","x1_min_derefine",0.0);

  // fixed orbit parameters
  fixed_orbit = pin->GetOrAddBoolean("problem","fixed_orbit",false);
  Omega_orb_fixed = pin->GetOrAddReal("problem","omega_orb_fixed",0.5);

  // separation based ouput, triggered when dt=999. for an output type
  dsep_output = pin->GetOrAddReal("problem","dsep_output",999.);

  // local vars
  Real rmin = pin->GetOrAddReal("mesh","x1min",0.0);
  Real rmax = pin->GetOrAddReal("mesh","x1max",0.0);
  Real thmin = pin->GetOrAddReal("mesh","x2min",0.0);
  Real thmax = pin->GetOrAddReal("mesh","x2max",0.0);

  Real sma = pin->GetOrAddReal("problem","sma",2.0);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Real fcorot = pin->GetOrAddReal("problem","fcorotation",0.0);
  Real Omega_orb, vcirc;
  

   // allocate MESH data for the particle pos/vel, Omega frame
  AllocateRealUserMeshDataField(3);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
  
  
  // enroll the BCs
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, DiodeOuterX1);
  }
  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X1, HSEInnerX1);
  }

  if(mesh_bcs[INNER_X2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X2, DiodeInnerX2);
  }
  if(mesh_bcs[OUTER_X2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X2, DiodeOuterX2);
  }


  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(TwoPointMass);

  // Enroll AMR
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  // Enroll extra history output
  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, mxOmegaEnv, "mxOmegaEnv");
  EnrollUserHistoryOutput(1, mEnv, "mEnv");
  EnrollUserHistoryOutput(2, mr1, "mr1");
  EnrollUserHistoryOutput(3, mr12, "mr12");
  //EnrollUserHistoryOutput(4, mb, "mb");
  //EnrollUserHistoryOutput(5, mu, "mu");
  

  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;


    
  // read in profile arrays from file
  std::ifstream infile("hse_profile.dat"); 
  for(int i=0;i<NARRAY;i++){
    infile >> rad[i] >> rho[i] >> p[i] >> menc[i];
    //std:: cout << rad[i] << "    " << rho[i] << std::endl;
  }
  infile.close();


  // set the inner point mass based on excised mass
  Real menc_rin = Interpolate1DArrayEven(rad,menc, rmin );
  GM1 = Ggrav*menc_rin;
  
  
  
  // need to do a 3D integral to get the gravitational acceleration
  int ntp=300;
  Real pi = 3.14159265359;
  Real dr = (1.0 - rmin)/ntp;
  Real dth = (thmax-thmin)/ntp;
  Real dph = 2*pi/ntp;
  
  // current position of the secondary
  Real x_2 = sma;
  Real y_2 = 0;
  Real z_2 = 0;
  
  // loop over the artifical domain and do the integral of the initial condition
  Real accel = 0;
  Real GMenv = 0;
  for(int i=0; i<ntp; i++){
    for(int j=0; j<ntp; j++){
      for(int k=0; k<ntp; k++){
	Real r  = rmin + dr/2 + i*dr;
	Real th = thmin + dth/2. + j*dth;
	Real ph = -pi + dph/2. + k*dph;
	
	// get local cartesian           
	Real x = r*sin(th)*cos(ph);
	Real y = r*sin(th)*sin(ph);
	Real z = r*cos(th);
	
	Real d2 = sqrt(pow(x-x_2, 2) +
		       pow(y-y_2, 2) +
		       pow(z-z_2, 2) );
	
	// mass element
	Real den = Interpolate1DArrayEven(rad,rho, r );
	Real dm = r*r*sin(th)*dr*dth*dph * den;
	
	//mass
	GMenv += Ggrav*dm;
	
	// gravitational accels in cartesian coordinates  
	accel += -Ggrav*dm*fspline(d2,rsoft2) * (x-x_2);
	
      }
    }
  }
    

  //ONLY enter ICs loop if this isn't a restart
  if(time==0){
    if(fixed_orbit){
      sma_fixed = pow((GM1+GM2+GMenv)/(Omega_orb_fixed*Omega_orb_fixed),1./3.);
      xi[0] = sma_fixed;
      xi[1] = 0.0;
      xi[2] = 0.0;  
      vi[0] = 0.0;
      vi[1]= Omega_orb_fixed*sma_fixed; 
      vi[2] = 0.0;

      Omega_envelope = fcorot*Omega_orb_fixed;
    }else{
      //Real vcirc = sqrt((GM1+GM2)/sma + accel*sma);    
      vcirc = sqrt((GM1+GM2+GMenv)/sma);
      Omega_orb = vcirc/sma;

      // set the initial conditions for the pos/vel of the secondary
      xi[0] = sma*(1.0 + ecc);  // apocenter
      xi[1] = 0.0;
      xi[2] = 0.0;
            
      vi[0] = 0.0;
      vi[1]= sqrt( vcirc*vcirc*(1.0 - ecc)/(1.0 + ecc) ); //v_apocenter
      vi[2] = 0.0;
    
      // now set the initial condition for Omega
      Omega[0] = 0.0;
      Omega[1] = 0.0;
      Omega[2] = 0.0;
    
      // In the case of a corotating frame,
      // subtract off the frame velocity and set Omega
      if(corotating_frame == 1){
	Omega[2] = Omega_orb;
	vi[1] -=  Omega[2]*xi[0]; 
      }
      
    
      // Angular velocity of the envelope (minus the frame?)
      Real f_pseudo = 1.0;
      if(ecc>0){
	f_pseudo=(1.0+7.5*pow(ecc,2) + 45./8.*pow(ecc,4) + 5./16.*pow(ecc,6));
	f_pseudo /= (1.0 + 3.0*pow(ecc,2) + 3./8.*pow(ecc,4))*pow(1-pow(ecc,2),1.5);
      }
      Omega_envelope = fcorot*Omega_orb*f_pseudo;

      // Decide whether to do pre-integration
      do_pre_integrate = (separation_start>sma*(1.0-ecc)) && (separation_start<sma*(1+ecc));
      
    } // end of if_not_fixed_orbit

    // save the ruser_mesh_data variables
    for(int i=0; i<3; i++){
      ruser_mesh_data[0](i)  = xi[i];
      ruser_mesh_data[1](i)  = vi[i];
      ruser_mesh_data[2](i)  = Omega[i];
    }
          
  }else{
    is_restart=1;
  }
  
    
  // Print out some info
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "==========   SIMULATION INFO =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "mode =" << mode << "\n";
    std::cout << "time =" << time << "\n";
    std::cout << "Ggrav = "<< Ggrav <<"\n";
    std::cout << "gamma = "<< gamma_gas <<"\n";
    std::cout << "GM1 = "<< GM1 <<"\n";
    std::cout << "GM2 = "<< GM2 <<"\n";
    std::cout << "GMenv="<< GMenv << "\n";
    std::cout << "GMenv/r^2, g_accel = " << GMenv/(sma*sma) << "  "<< accel << "\n";
    std::cout << "Omega_orb="<< Omega_orb << "\n";
    std::cout << "Omega_env="<< Omega_envelope << "\n";
    std::cout << "a = "<< sma <<"\n";
    std::cout << "e = "<< ecc <<"\n";
    std::cout << "P = "<< 6.2832*sqrt(sma*sma*sma/(GM1+GM2+GMenv)) << "\n";
    std::cout << "rsoft2 ="<<rsoft2<<"\n";
    std::cout << "corotating frame? = "<< corotating_frame<<"\n";
    std::cout << "gas backreaction? = "<< include_gas_backreaction<<"\n";
    std::cout << "particle substeping n="<<n_particle_substeps<<"\n";
    std::cout << "agas1_source_rel_thresh ="<<agas1_source_rel_thresh<<"\n";
    std::cout << "t_relax ="<<t_relax<<"\n";
    std::cout << "t_mass_on ="<<t_mass_on<<"\n";
    std::cout << "do_pre_integrate ="<<do_pre_integrate<<"\n";
    std::cout << "fixed_orbit ="<<fixed_orbit<<"\n";
    std::cout << "Omega_orb_fixed="<< Omega_orb_fixed << "\n";
    std::cout << "a_fixed = "<< sma_fixed <<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   Particle        =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "x ="<<xi[0]<<"\n";
    std::cout << "y ="<<xi[1]<<"\n";
    std::cout << "z ="<<xi[2]<<"\n";
    std::cout << "vx ="<<vi[0]<<"\n";
    std::cout << "vy ="<<vi[1]<<"\n";
    std::cout << "vz ="<<vi[2]<<"\n";
    std::cout << "==========================================================\n";
  }
  
    


} // end


Real mxOmegaEnv(MeshBlock *pmb, int iout){
  Real mxOmega = 0.0;
  
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	Real dens = pmb->phydro->u(IDN,k,j,i);
	Real dm = vol(i) * dens;

	if(dens > 1.e-3){
	  mxOmega += dm*(pmb->phydro->u(IM3,k,j,i)/dens) / (r*sin_th);
	}
	
      }
    }
  }
  
  return mxOmega;
}

Real mEnv(MeshBlock *pmb, int iout){
  Real menv = 0.0;
  
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
	Real dens = pmb->phydro->u(IDN,k,j,i);
	Real dm = vol(i) * dens;
	if(dens > 1.e-3){
	  menv += dm;
	}
      }
    }
  }
  
  return menv;
}

Real mr1(MeshBlock *pmb, int iout){
  Real mr1 = 0.0;
  
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
	if(pmb->pcoord->x1v(i) <= 1.0){
	  Real dens = pmb->phydro->u(IDN,k,j,i);
	  Real dm = vol(i) * dens;
	  mr1 += dm;
	}
      }
    }
  }
  
  return mr1;
}

Real mr12(MeshBlock *pmb, int iout){
  Real mr12 = 0.0;
  
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> vol;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
	if(pmb->pcoord->x1v(i) <= 1.2){
	  Real dens = pmb->phydro->u(IDN,k,j,i);
	  Real dm = vol(i) * dens;
	  mr12 += dm;
	}
      }
    }
  }
  
  return mr12;
}









int RefinementCondition(MeshBlock *pmb)
{
  Real mindist=1.e99;
  Real rmin = 1.e99;
  int inregion = 0;
  for(int k=pmb->ks; k<=pmb->ke; k++){
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for(int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for(int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
	Real dist = std::sqrt(SQR(x-xi[0]) +
			      SQR(y-xi[1]) +
			      SQR(z-xi[2]) );
	mindist = std::min(mindist,dist);
	rmin    = std::min(rmin,r);
      }
    }
  }
  // derefine when away from pm & static region
  if( (mindist > 4.0*rsoft2) && rmin>x1_min_derefine  ) return -1;
  // refine near point mass 
  if(mindist <= 3.0*rsoft2) return 1;
   // otherwise do nothing
  return 0;
}



Real GetGM2factor(Real time){
  Real GM2_factor;

  // turn the gravity of the secondary on over time...
  if(time<t_relax+t_mass_on){
    if(time<t_relax){
      // if t<t_relax, do not apply the acceleration of the secondary to the gas
      GM2_factor = 0.0;
    }else{
      // turn on the gravity of the secondary over the course of t_mass_on after t_relax
      GM2_factor = (time-t_relax)/t_mass_on;
    }
  }else{
    // if we're outside of the relaxation times, turn the gravity of the secondary fully on
    GM2_factor = 1.0;
  }
  
  return GM2_factor;
}



// Source Function for two point masses
void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{ 

  if(is_restart>0){
    // else this is a restart, read the current particle state
    for(int i=0; i<3; i++){
      xi[i]    = pmb->pmy_mesh->ruser_mesh_data[0](i);
      vi[i]    = pmb->pmy_mesh->ruser_mesh_data[1](i);
      Omega[i] = pmb->pmy_mesh->ruser_mesh_data[2](i);
    }
    // print some info
    if (Globals::my_rank==0){
      std::cout << "*** Setting initial conditions for t>0 ***\n";
      std::cout <<"xi="<<xi[0]<<" "<<xi[1]<<" "<<xi[2]<<"\n";
      std::cout <<"vi="<<vi[0]<<" "<<vi[1]<<" "<<vi[2]<<"\n";
      std::cout <<"Omega="<<Omega[0]<<" "<<Omega[1]<<" "<<Omega[2]<<"\n";
    }
    is_restart=0;
  }
  
  Real GM2_factor = GetGM2factor(time);

  // decide whether or not to use the -a_gas_1 source term
  // we don't apply this if unless it's important because it can sometimes strongly destabilize the HSE envelope
  int agas1_source = 0;
  if(include_gas_backreaction==1){
    Real a2m = sqrt(agas2i[0]*agas2i[0] + agas2i[1]*agas2i[1] + agas2i[2]*agas2i[2]);
    Real a1m = sqrt(agas1i[0]*agas1i[0] + agas1i[1]*agas1i[1] + agas1i[2]*agas1i[2]);
    
    if (a2m>0 && a1m/a2m > agas1_source_rel_thresh){
      //std::cout<< "applying agas1_source!\n"<< Globals::my_rank <<" "<<a1m<<"  "<<a2m<<"\n";
      agas1_source=1;
    }
  }
  

  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	
	// current position of the secondary
	Real x_2 = xi[0];
	Real y_2 = xi[1];
	Real z_2 = xi[2];
	Real d12c = pow(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2], 1.5);
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
  
	Real d2  = sqrt(pow(x-x_2, 2) +
			pow(y-y_2, 2) +
			pow(z-z_2, 2) );
  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1
	//Real a_r1 = -GM1/pow(r,2);
	// cell volume avg'd version, see pointmass.cpp sourceterm code. 
	//Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	Real GMenc1 = Ggrav*Interpolate1DArrayEven(rad,menc, r);
	Real a_r1 = -GMenc1*pmb->pcoord->coord_src1_i_(i)/r;
	//Real a_r1 = -GMenc1/pow(r,2);
	
	// PM2 gravitational accels in cartesian coordinates
	Real a_x = - GM2*GM2_factor * fspline(d2,rsoft2) * (x-x_2);   
	Real a_y = - GM2*GM2_factor * fspline(d2,rsoft2) * (y-y_2);  
	Real a_z = - GM2*GM2_factor * fspline(d2,rsoft2) * (z-z_2);
	
	// add the correction for the orbiting frame (relative to the COM)
	a_x += -  GM2*GM2_factor / d12c * x_2;
	a_y += -  GM2*GM2_factor / d12c * y_2;
	a_z += -  GM2*GM2_factor / d12c * z_2;
	
	if(corotating_frame == 1){
	  
	  Real vr  = prim(IVX,k,j,i);
	  Real vth = prim(IVY,k,j,i);
	  Real vph = prim(IVZ,k,j,i);
	  
	  // distance from the origin in cartesian (vector)
	  Real rxyz[3];
	  rxyz[0] = x;
	  rxyz[1] = y;
	  rxyz[2] = z;
	  
	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;
	  
	  // add the centrifugal and coriolis terms
	  
	  // centrifugal
	  Real Omega_x_r[3], Omega_x_Omega_x_r[3];
	  cross(Omega,rxyz,Omega_x_r);
	  cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
	  
	  a_x += - Omega_x_Omega_x_r[0];
	  a_y += - Omega_x_Omega_x_r[1];
	  a_z += - Omega_x_Omega_x_r[2];
	  
	  // coriolis
	  Real Omega_x_v[3];
	  cross(Omega,vgas,Omega_x_v);
	  
	  a_x += -2.0*Omega_x_v[0];
	  a_y += -2.0*Omega_x_v[1];
	  a_z += -2.0*Omega_x_v[2];
	}
	
	// add the gas acceleration of the frame of ref
	if(agas1_source==1){
	  a_x += -agas1i[0];
	  a_y += -agas1i[1];
	  a_z += -agas1i[2];    
	}
	
	// convert back to spherical
	Real a_r  = sin_th*cos_ph*a_x + sin_th*sin_ph*a_y + cos_th*a_z;
	Real a_th = cos_th*cos_ph*a_x + cos_th*sin_ph*a_y - sin_th*a_z;
	Real a_ph = -sin_ph*a_x + cos_ph*a_y;
	
	// add the PM1 accel
	a_r += a_r1;
	
	//
	// ADD SOURCE TERMS TO THE GAS MOMENTA/ENERGY
	//
	Real den = prim(IDN,k,j,i);
	
	Real src_1 = dt*den*a_r; 
	Real src_2 = dt*den*a_th;
	Real src_3 = dt*den*a_ph;
	
	// add the source term to the momenta  (source = - rho * a)
	cons(IM1,k,j,i) += src_1;
	cons(IM2,k,j,i) += src_2;
	cons(IM3,k,j,i) += src_3;
	
	// update the energy (source = - rho v dot a)
	//cons(IEN,k,j,i) += src_1*prim(IVX,k,j,i) + src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);
	cons(IEN,k,j,i) += src_1/den * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	//cons(IEN,k,j,i) += src_2/den * 0.5*(flux[X2DIR](IDN,k,j,i) + flux[X2DIR](IDN,k,j+1,i)); //not sure why this seg-faults
	//cons(IEN,k,j,i) += src_3/den * 0.5*(flux[X3DIR](IDN,k,j,i) + flux[X3DIR](IDN,k+1,j,i));
	cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);

      }
    }
  } // end loop over cells
  

}




//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres;

   // Prepare index bounds including ghost cells
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

	Real r  = pcoord->x1v(i);
	Real th = pcoord->x2v(j);
	Real ph = pcoord->x3v(k);

	
	Real sin_th = sin(th);
	Real Rcyl = r*sin_th;
	
	// get the density
	den = Interpolate1DArrayEven(rad,rho, r );
	den = std::max(den,da);
	
	// get the pressure 
	pres = Interpolate1DArrayEven(rad,p, r );
	pres = std::max(pres,pa);

	// set the density
	phydro->u(IDN,k,j,i) = den;
	
   	// set the momenta components
	phydro->u(IM1,k,j,i) = 0.0;
	phydro->u(IM2,k,j,i) = 0.0;
	if(r <= 1.0){
	  phydro->u(IM3,k,j,i) = den*(Omega_envelope*Rcyl - Omega[2]*Rcyl);
	}else{
	  phydro->u(IM3,k,j,i) = den*(Omega_envelope*sin_th*sin_th/Rcyl - Omega[2]*Rcyl);
	}
	  
	//set the energy 
	phydro->u(IEN,k,j,i) = pres/(gamma_gas-1);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//======================================================================================
void MeshBlock::UserWorkInLoop(void)
{
  Real time = pmy_mesh->time;
  Real dt = pmy_mesh->dt;
  //Real v_factor;
  
  // if less than the relaxation time, apply 
  // a damping to the fluid velocities
  if(time < t_relax){
    Real tau = 1.0;
    if(time > 0.2*t_relax){
      tau *= pow(10, 2.0*(time-0.2*t_relax)/(0.8*t_relax) );
    }
    if (Globals::my_rank==0){
      std::cout << "Relaxing: tau_damp ="<<tau<<std::endl;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie; i++) {

	  Real den = phydro->u(IDN,k,j,i);
	  Real vr  = phydro->u(IM1,k,j,i) / den;
	  Real vth = phydro->u(IM2,k,j,i) / den;
	  Real vph = phydro->u(IM3,k,j,i) / den;
	  
	  Real a_damp_r =  - vr/tau;
	  Real a_damp_th = - vth/tau;
	  Real a_damp_ph = 0.0;
	  
	  phydro->u(IM1,k,j,i) += dt*den*a_damp_r;
	  phydro->u(IM2,k,j,i) += dt*den*a_damp_th;
	  phydro->u(IM3,k,j,i) += dt*den*a_damp_ph;
	  
	  phydro->u(IEN,k,j,i) += dt*den*a_damp_r*vr + dt*den*a_damp_th*vth + dt*den*a_damp_ph*vph; 
	  
	}
      }
    } // end loop over cells                   
  } // end relax

  return;
} // end of UserWorkInLoop


//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){

  Real ai[3];
  Real mg;
  
  // ONLY ON THE FIRST CALL TO THIS FUNCTION
  // (NOTE: DOESN'T WORK WITH RESTARTS)
  if((ncycle==0) && (fixed_orbit==false)){
    
    // kick the initial conditions back a half step (v^n-1/2)

    // first sum the gas accel if needed
    if(include_gas_backreaction == 1){
      SumGasOnParticleAccels(pblock->pmy_mesh, xi,agas1i,agas2i);
    }

    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai);

    // Integrate from apocenter to separation_start
    if( do_pre_integrate ) {
      Real sep,vel,dt_pre_integrator;
      int n_steps_pre_integrator;

      SumComPosVel(pblock->pmy_mesh, xi, vi, xgcom, vgcom, xcom, vcom, mg);
      Real GMenv = Ggrav*mg;
      
      
      for (int ii=1; ii<=1e8; ii++) {
	sep = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);
	vel = sqrt(vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
	dt_pre_integrator = 1.e-4 * sep/vel;
	// check stopping condition
	n_steps_pre_integrator = ii;
	if (sep<separation_start) break;
	
	// add the particle acceleration to ai
	ParticleAccelsPreInt(GMenv,xi,vi,ai);
	// advance the particle
	particle_step(dt_pre_integrator,xi,vi,ai);
      }

      if (Globals::my_rank==0){
	std::cout << "Integrated to starting separation:"<<sep<<"\n";
	std::cout << " in "<<n_steps_pre_integrator<<" steps\n";
	if( std::abs(sep-separation_start) > 1.e-2*separation_start){
	  std::cout << "Pre-integrator failed!!! Exiting. \n";
	  SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
	}	
      }

    }
 
  } // ncycle=0, fixed_orbit = false

    
  // EVOLVE THE ORBITAL POSITION OF THE SECONDARY
  // do this on rank zero, then broadcast
  if (Globals::my_rank == 0 && time>t_relax){
    if(fixed_orbit){
      Real theta_orb = Omega_orb_fixed*time;
      xi[0] = sma_fixed*std::cos(theta_orb);
      xi[1] = sma_fixed*std::sin(theta_orb);
      xi[2] = 0.0;
      vi[0] = sma_fixed*Omega_orb_fixed*std::sin(theta_orb);
      vi[1] = sma_fixed*Omega_orb_fixed*std::cos(theta_orb);
      vi[2] = 0.0;
    }else{
      for (int ii=1; ii<=n_particle_substeps; ii++) {
	// add the particle acceleration to ai
	ParticleAccels(xi,vi,ai);
	// advance the particle
	particle_step(dt/n_particle_substeps,xi,vi,ai);
      }
    }
  }
  
#ifdef MPI_PARALLEL
  // broadcast the position update from proc zero
  MPI_Bcast(xi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // update the ruser_mesh_data variables
  for(int i=0; i<3; i++){
    ruser_mesh_data[0](i)  = xi[i];
    ruser_mesh_data[1](i)  = vi[i];
    ruser_mesh_data[2](i)  = Omega[i];
  }

  // check the separation stopping conditions
  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2] );
  if (d<separation_stop_min){ 
    if (Globals::my_rank == 0) {
      std::cout << "### Stopping because binary separation d<separation_stop_min" << std::endl
		<< "d= " << d << " separation_stop_min="<<separation_stop_min<<std::endl;
    }
    SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
  }

  if (d>separation_stop_max){ 
    if (Globals::my_rank == 0) {
      std::cout << "### Stopping because binary separation d>separation_stop_max" << std::endl
		<< "d= " << d << " separation_stop_max="<<separation_stop_max<<std::endl;
    }
    SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
  }

  // check whether to trigger forced output
  if ((d<separation_stop_min) ||
      (d>separation_stop_max) ||
      (d<=output_next_sep) ){
    user_force_output = true;
    //output_next_sep -= dsep_output;
    if (Globals::my_rank == 0) {
      std::cout << "triggering user separation based output, d="<<d<<"\n";
    } 
  }
  output_next_sep = floor(d/dsep_output)*dsep_output; // rounds to the nearest lower sep
  
  
  // sum the gas->part accel for the next step
  if(include_gas_backreaction == 1 && time>t_relax){
    SumGasOnParticleAccels(pblock->pmy_mesh, xi,agas1i,agas2i);
  }
  
  
  // write the output to the trackfile
  if(time >= trackfile_next_time || user_force_output ){
    SumComPosVel(pblock->pmy_mesh, xi, vi, xgcom, vgcom, xcom, vcom, mg);
    SumTrackfileDiagnostics(pblock->pmy_mesh, xi, vi, xgcom, vgcom, xcom, vcom, lp, lg, ldo, Eorb,mb,mu);
    WritePMTrackfile(pblock->pmy_mesh,pin);
  }

  // std output
  if (Globals::my_rank == 0  & ncycle%100==0) {
    std::cout << "sep:  d="<<d<<"\n";
  } 
  
  
}


void WritePMTrackfile(Mesh *pm, ParameterInput *pin){
  
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("pm_trackfile.dat");
    
    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if((pfile = fopen(fname.c_str(),"a")) == NULL){
      msg << "### FATAL ERROR in function [WritePMTrackfile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      throw std::runtime_error(msg.str().c_str());
    }
  
    if(trackfile_number==0){
      fprintf(pfile,"#    ncycle     ");
      fprintf(pfile,"time           ");
      fprintf(pfile,"dt             ");
      fprintf(pfile,"m1             ");
      fprintf(pfile,"m2             ");
      fprintf(pfile,"x              ");
      fprintf(pfile,"y              ");
      fprintf(pfile,"z              ");
      fprintf(pfile,"vx             ");
      fprintf(pfile,"vy             ");
      fprintf(pfile,"vz             ");
      fprintf(pfile,"agas1x          ");
      fprintf(pfile,"agas1y          ");
      fprintf(pfile,"agas1z          ");
      fprintf(pfile,"agas2x          ");
      fprintf(pfile,"agas2y          ");
      fprintf(pfile,"agas2z          ");
      fprintf(pfile,"xcom            ");
      fprintf(pfile,"ycom            ");
      fprintf(pfile,"zcom            ");
      fprintf(pfile,"vxcom            ");
      fprintf(pfile,"vycom            ");
      fprintf(pfile,"vzcom            ");
      fprintf(pfile,"lpz             ");
      fprintf(pfile,"lgz             ");
      fprintf(pfile,"ldoz            ");
      fprintf(pfile,"Eorb            ");
      fprintf(pfile,"mb              ");
      fprintf(pfile,"mu              ");
      fprintf(pfile,"\n");
    }


    // write the data line
    fprintf(pfile,"%20i",pm->ncycle);
    fprintf(pfile,"%20.6e",pm->time);
    fprintf(pfile,"%20.6e",pm->dt);
    fprintf(pfile,"%20.6e",GM1/Ggrav);
    fprintf(pfile,"%20.6e",GM2/Ggrav);
    fprintf(pfile,"%20.6e",xi[0]);
    fprintf(pfile,"%20.6e",xi[1]);
    fprintf(pfile,"%20.6e",xi[2]);
    fprintf(pfile,"%20.6e",vi[0]);
    fprintf(pfile,"%20.6e",vi[1]);
    fprintf(pfile,"%20.6e",vi[2]);
    fprintf(pfile,"%20.6e",agas1i[0]);
    fprintf(pfile,"%20.6e",agas1i[1]);
    fprintf(pfile,"%20.6e",agas1i[2]);
    fprintf(pfile,"%20.6e",agas2i[0]);
    fprintf(pfile,"%20.6e",agas2i[1]);
    fprintf(pfile,"%20.6e",agas2i[2]);
    fprintf(pfile,"%20.6e",xcom[0]);
    fprintf(pfile,"%20.6e",xcom[1]);
    fprintf(pfile,"%20.6e",xcom[2]);
    fprintf(pfile,"%20.6e",vcom[0]);
    fprintf(pfile,"%20.6e",vcom[1]);
    fprintf(pfile,"%20.6e",vcom[2]);
    fprintf(pfile,"%20.6e",lp[2]);
    fprintf(pfile,"%20.6e",lg[2]);
    fprintf(pfile,"%20.6e",ldo[2]);
    fprintf(pfile,"%20.6e",Eorb);
    fprintf(pfile,"%20.6e",mb);
    fprintf(pfile,"%20.6e",mu);
    fprintf(pfile,"\n");

    // close the file
    fclose(pfile);  

  } // end rank==0

  // increment counters
  trackfile_number++;
  trackfile_next_time += trackfile_dt;


  
  return;
}



void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  // Leapfrog algorithm (KDK)

  // kick a full step
  kick(dt,xi,vi,ai);

  // drift a full step
  drift(dt,xi,vi,ai);
  
}

// kick the velocities dt using the accelerations given in ai
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    vi[i] += dt*ai[i];
  }
}

// drift the velocities dt using the velocities given in vi
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    xi[i] += dt*vi[i];
  }
}

void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

  // fill in the accelerations for the orbiting frame
  for (int i = 0; i < 3; i++){
    ai[i] = - GM1/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];
  } 
  
  // IF WE'RE IN A ROTATING FRAME
  if(corotating_frame == 1){
    Real Omega_x_r[3],Omega_x_Omega_x_r[3], Omega_x_v[3];
 
    // compute cross products 
    cross(Omega,xi,Omega_x_r);
    cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
    
    cross(Omega,vi,Omega_x_v);
  
    // fill in the accelerations for the rotating frame
    for (int i = 0; i < 3; i++){
      ai[i] += -Omega_x_Omega_x_r[i];
      ai[i] += -2.0*Omega_x_v[i];
    }
  }

  // add the gas acceleration to ai
  if(include_gas_backreaction == 1){
    for (int i = 0; i < 3; i++){
      ai[i] += -agas1i[i]+agas2i[i];
    }
  }

}


void ParticleAccelsPreInt(Real GMenv, Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

  // fill in the accelerations for the orbiting frame
  for (int i = 0; i < 3; i++){
    ai[i] = - (GM1+GMenv)/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];
  } 
  
  // IF WE'RE IN A ROTATING FRAME
  if(corotating_frame == 1){
    Real Omega_x_r[3],Omega_x_Omega_x_r[3], Omega_x_v[3];
 
    // compute cross products 
    cross(Omega,xi,Omega_x_r);
    cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
    
    cross(Omega,vi,Omega_x_v);
  
    // fill in the accelerations for the rotating frame
    for (int i = 0; i < 3; i++){
      ai[i] += -Omega_x_Omega_x_r[i];
      ai[i] += -2.0*Omega_x_v[i];
    }
  }

}









void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3],Real (&ag1i)[3],Real (&ag2i)[3]){
  
  // start by setting accelerations / positions to zero
  for (int ii = 0; ii < 3; ii++){
    ag1i[ii] = 0.0;
    ag2i[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->pblock;
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  while (pmb != NULL) {
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real ph= pmb->pcoord->x3v(k);
      Real sin_ph = sin(ph);
      Real cos_ph = cos(ph);
      for (int j=pmb->js; j<=pmb->je; ++j) {
	Real th= pmb->pcoord->x2v(j);
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  	  
	  Real r = pmb->pcoord->x1v(i);

	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  Real d2 = sqrt(pow(x-xi[0], 2) +
			 pow(y-xi[1], 2) +
			 pow(z-xi[2], 2) );
  
	  Real d1c = pow(r,3);
	  
	   // gravitational accels in cartesian coordinates
	  
	  ag1i[0] += Ggrav*dm/d1c * x;
	  ag1i[1] += Ggrav*dm/d1c * y;
	  ag1i[2] += Ggrav*dm/d1c * z;
	  
	  ag2i[0] += Ggrav*dm * fspline(d2,rsoft2) * (x-xi[0]);
	  ag2i[1] += Ggrav*dm * fspline(d2,rsoft2) * (y-xi[1]);
	  ag2i[2] += Ggrav*dm * fspline(d2,rsoft2) * (z-xi[2]);
	  
	}
      }
    }//end loop over cells
    pmb=pmb->next;
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, ag1i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, ag2i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(ag1i,ag1i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(ag2i,ag2i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  // and broadcast the result
  MPI_Bcast(ag1i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(ag2i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

    
}


void SumComPosVel(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
				     Real (&xgcom)[3],Real (&vgcom)[3],
				     Real (&xcom)[3],Real (&vcom)[3],
				     Real &mg){

   mg = 0.0;
   Real m1 = GM1/Ggrav;
   Real m2 = GM2/Ggrav;
  
  // start by setting accelerations / positions to zero
  for (int ii = 0; ii < 3; ii++){
    xgcom[ii] = 0.0;
    vgcom[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->pblock;
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  while (pmb != NULL) {
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  	  
	  //coordinates
	  Real r = pmb->pcoord->x1v(i);
	  Real th= pmb->pcoord->x2v(j);
	  Real ph= pmb->pcoord->x3v(k);

	    //get some angles
	  Real sin_th = sin(th);
	  Real cos_th = cos(th);
	  Real sin_ph = sin(ph);
	  Real cos_ph = cos(ph);
	  
	 
	  // spherical velocities
	  Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	  // Correct for rotation of the frame? [TBDW]

	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;

	  // do the summation
	  mg += dm;
	  
	  xgcom[0] += dm*x;
	  xgcom[1] += dm*y;
	  xgcom[2] += dm*z;

	  vgcom[0] += dm*vgas[0];
	  vgcom[1] += dm*vgas[1];
	  vgcom[2] += dm*vgas[2];

	  
	}
      }
    }//end loop over cells
    pmb=pmb->next;
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &mg, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, xgcom, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, vgcom, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&mg,&mg,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(xgcom,xgcom,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(vgcom,vgcom,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  MPI_Bcast(&mg,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(xgcom,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vgcom,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // normalize to the total mass
  for (int ii = 0; ii < 3; ii++){
    xgcom[ii] /= mg;
    vgcom[ii] /= mg;
  }
  
  // FINISH CALC OF COM
  for (int ii = 0; ii < 3; ii++){
    xcom[ii] = (xi[ii]*m2 + xgcom[ii]*mg)/(m1+m2+mg);
    vcom[ii] = (vi[ii]*m2 + vgcom[ii]*mg)/(m1+m2+mg); 
  }

}



void SumTrackfileDiagnostics(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
				     Real (&xgcom)[3],Real (&vgcom)[3],
				     Real (&xcom)[3],Real (&vcom)[3],
			     Real (&lp)[3],Real (&lg)[3],Real (&ldo)[3], Real &Eorb,
			     Real &mb, Real &mu){

  // NOW COMPUTE THE ANGULAR MOMENTA
  Real m1 = GM1/Ggrav;
  Real m2 = GM2/Ggrav;
  Real d12 = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

 
  // start by setting accelerations / positions to zero
  Eorb = 0.0;
  Real Mg = 0.0;
  mb = 0.0;
  mu = 0.0;
  for (int ii = 0; ii < 3; ii++){
    lg[ii]  = 0.0;
    lp[ii]  = 0.0;
    ldo[ii] = 0.0;
  }
  
  // loop over cells here
  MeshBlock *pmb=pm->pblock;
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  while (pmb != NULL) {
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real ph= pmb->pcoord->x3v(k);
      Real sin_ph = sin(ph);
      Real cos_ph = cos(ph);
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	Real th= pmb->pcoord->x2v(j);
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real r = pmb->pcoord->x1v(i);

	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  
	  // outer-r face area of cell i
	  Real dA = pmb->pcoord->GetFace1Area(k,j,i+1);
	  	 
	  // spherical velocities
	  Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	  // Correct for rotation of the frame? [TBDW]
	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;

	  // do the summation
	  // position rel to COM
	  Real rg[3];
	  rg[0] = x - xcom[0];
	  rg[1] = y - xcom[1];
	  rg[2] = z - xcom[2];

	  // momentum rel to COM
	  Real pg[3];
	  pg[0] = dm*(vgas[0] - vcom[0]);
	  pg[1] = dm*(vgas[1] - vcom[1]);
	  pg[2] = dm*(vgas[2] - vcom[2]);

	  // rxp
	  Real rxp[3];
	  cross(rg,pg,rxp);
	  for (int ii = 0; ii < 3; ii++){
	    lg[ii] += rxp[ii];
	  }

	  // now the flux of angular momentum off of the outer boundary of the grid
	  if(pmb->pcoord->x1f(i+1)==pm->mesh_size.x1max){
	    Real md = phyd->u(IDN,k,j,i)*vr*dA;
	    Real pd[3];
	    pd[0] = md*(vgas[0] - vcom[0]);
	    pd[1] = md*(vgas[1] - vcom[1]);
	    pd[2] = md*(vgas[2] - vcom[2]);

	    Real rxpd[3];
	    cross(rg,pd,rxpd);
	    for (int ii = 0; ii < 3; ii++){
	      ldo[ii] += rxpd[ii];
	    }
	  } //endif


	  // enclosed mass (within current orbital separation)
	  if(r<d12){
	    Mg += dm;
	  }


	  // compute bound and unbound masses based on bernoulli param
	  Real d2 = std::sqrt(SQR(x-xi[0]) +
			      SQR(y-xi[1]) +
			      SQR(z-xi[2]) );
	  Real GMenc1 = Ggrav*Interpolate1DArrayEven(rad,menc, r);
	  Real h = gamma_gas * pmb->phydro->u(IPR,k,j,i)/((gamma_gas-1.0)*pmb->phydro->u(IDN,k,j,i));
	  Real epot = -GMenc1/r - GM2*pspline(d2,rsoft2);
	  Real ek = 0.5*(SQR(vgas[0]-vcom[0]) +SQR(vgas[1]-vcom[1]) +SQR(vgas[2]-vcom[2]));
	  Real bern = h+ek+epot;
	  if( r>1.0) {
	    if (bern < 0.0){
	      mb += dm;
	    }else{
	      mu += dm;
	    }
	  }
	  
			 
	    
	    
	}
      }
    }//end loop over cells
    pmb=pmb->next;
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, lg, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, ldo, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Mg, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mb, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mu, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(lg,lg,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(ldo,ldo,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&Mg,&Mg,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mb,&mb,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mu,&mu,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  MPI_Bcast(lg,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(ldo,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&Mg,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mb,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mu,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif


  // calculate the particle angular momenta
  Real r1[3], r2[3], p1[3], p2[3], r1xp1[3], r2xp2[3];
  
  for (int ii = 0; ii < 3; ii++){
    r1[ii] = -xcom[ii];
    p1[ii] = -m1*vcom[ii];

    r2[ii] = xi[ii] - xcom[ii];
    p2[ii] = m2*(vi[ii] - vcom[ii]);
  }

  cross(r1,p1,r1xp1);
  cross(r2,p2,r2xp2);

  for (int ii = 0; ii < 3; ii++){
    lp[ii] = r1xp1[ii] + r2xp2[ii];
  }


  // calculate the orbital energy (approximate, I think)
  Real v1_sq = vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2];
  Real v2_sq = SQR(vi[0]-vcom[0]) + SQR(vi[1]-vcom[1]) + SQR(vi[2]-vcom[2]);
  Eorb = 0.5*(m1+Mg)*v1_sq + 0.5*m2*v2_sq - Ggrav*(m1+Mg)*m2/d12; 

  
}






//--------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  // copy hydro variables into ghost zones, don't allow inflow
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(IVX,k,j,ie+i) =  std::max( 0.0, prim(IVX,k,j,(ie-i+1)) );  // positive velocities only
	  }
	}}
    } else {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(n,k,j,ie+i) = prim(n,k,j,(ie-i+1));
	  }
	}}
    }
  }


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
	}
      }}

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
	}
      }}

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
	}
      }}
  }

  return;
}


//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  return;
}




void HSEInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{

 
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
	prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));
	prim(IVY,k,j,is-i) = prim(IVY,k,j,(is+i-1));
	prim(IVZ,k,j,is-i) = prim(IVZ,k,j,(is+i-1));
	prim(IDN,k,j,is-i) = prim(IDN,k,j,(is+i-1));

	Real r = pco->x1v(is-i+1);

	// HSE radial gravity
	Real geff = pco->coord_src1_i_(is-i+1)*GM1/r;

	// Rotational support
	Real Rcyl = r*sin(pco->x2v(j));
	geff +=  - prim(IVZ,k,j,is-i)*prim(IVZ,k,j,is-i)/Rcyl;

	// Rotating frame
	geff += - Omega[2]*Omega[2]*Rcyl - 2.0*Omega[2]*prim(IVZ,k,j,is-i);

	Real dr = pco->dx1v(is-i);
	Real dp = geff * prim(IDN,k,j,is-i+1) * dr;
	
	prim(IPR,k,j,is-i) = prim(IPR,k,j,(is-i+1)) + dp;
      }
    }
  }
  
  // copy face-centered magnetic fields into ghost zones, reflect x1
  if (MAGNETIC_FIELDS_ENABLED) {
    for
      (int k=ks; k<=ke; ++k) { 
      for (int j=js; j<=je; ++j) { 
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) { 
	  b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  
	} 
      }}
    
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
	}
      }}  
    
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
	}
      }}
  }
}
  


//----------------------------------------------------------------------------------------
//! \fn void ReflecInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, const Real time, const Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief REFLECTING boundary conditions, inner x2 boundary

void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  // copy hydro variables into ghost zones, reflecting v2
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVY)) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i) {
          prim(IVY,k,js-j,i) = 0.0; //std::min(0.0,prim(IVY,k,js+j-1,i));  // allow outflow only
        }
      }}
    } else {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i) {
          prim(n,k,js-j,i) = prim(n,k,js+j-1,i);
        }
      }}
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) =  b.x1f(k,(js+j-1),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = b.x2f(k,(js+j  ),i);  
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) =  b.x3f(k,(js+j-1),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflectOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke)
//  \brief REFLECTING boundary conditions, outer x2 boundary

void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  // copy hydro variables into ghost zones, reflecting v2
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVY)) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i) {
          prim(IVY,k,je+j,i) = 0.0; //std::max(0.0,prim(IVY,k,je-j+1,i) ); //allow outflow only 
        }
      }}
    } else {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i) {
          prim(n,k,je+j,i) = prim(n,k,je-j+1,i);
        }
      }}
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) =  b.x1f(k,(je-j+1),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = b.x2f(k,(je-j+1),i);  
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) =  b.x3f(k,(je-j+1),i);
      }
    }}
  }

  return;
}



// Real Interpolate1DArray(Real *x,Real *y,Real x0){
//   int i;
//   Real d,dx,s,y0;
//   //std::cout.precision(17);
  

//   // check the lower bound
//   if(x[0] >= x0){
//     //std::cout << "hit lower bound!\n";
//     return y[0];
//   }
//   // check the upper bound
//   if(x[NARRAY-1] <= x0){
//     //std::cout << "hit upper bound!\n";
//     return y[NARRAY-1];
//   }

//   // if in the interior, do a linear interpolation
//   for(i=0;i<NARRAY-1;i++){
//     if (x[i+1] >= x0){
//       dx =  (x[i+1]-x[i]);
//       d = (x0 - x[i]);
//       s = (y[i+1]-y[i]) /dx;
//       y0 = s*d + y[i];
//       return y0;
//     }
//   }
//   // should never get here, -9999.9 represents an error
//   return -9999.9;
// }


// 1D Interpolation that assumes EVEN spacing in x array
Real Interpolate1DArrayEven(Real *x,Real *y,Real x0){ 
  // check the lower bound
  if(x[0] >= x0){
    //std::cout << "hit lower bound!\n";
    return y[0];
  }
  // check the upper bound
  if(x[NARRAY-1] <= x0){
    //std::cout << "hit upper bound!\n";
    return y[NARRAY-1];
  }

  int i = floor( (x0-x[0])/(x[1]-x[0]) );
  
  // if in the interior, do a linear interpolation
  if (x[i+1] >= x0){ 
    Real dx =  (x[i+1]-x[i]);
    Real d = (x0 - x[i]);
    Real s = (y[i+1]-y[i]) /dx;
    return s*d + y[i];
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}



Real fspline(Real r, Real eps){
  // Hernquist & Katz 1989 spline kernel F=-GM r f(r,e) EQ A2
  Real u = r/eps;
  Real u2 = u*u;

  if (u<1.0){
    return pow(eps,-3) * (4./3. - 1.2*pow(u,2) + 0.5*pow(u,3) );
  } else if(u<2.0){
    return pow(r,-3) * (-1./15. + 8./3.*pow(u,3) - 3.*pow(u,4) + 1.2*pow(u,5) - 1./6.*pow(u,6));
  } else{
    return pow(r,-3);
  }

}


Real pspline(Real r, Real eps){
  Real u = r/eps;
  if (u<1.0){
    return -2/eps *(1./3.*pow(u,2) -0.15*pow(u,4) + 0.05*pow(u,5)) +7./(5.*eps);
  } else if(u<2.0){
    return -1./(15.*r) - 1/eps*( 4./3.*pow(u,2) - pow(u,3) + 0.3*pow(u,4) -1./30.*pow(u,5)) + 8./(5.*eps);
  } else{
    return 1/r;
  }

}







void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]){
  // set the vector AxB = A x B
  AxB[0] = A[1]*B[2] - A[2]*B[1];
  AxB[1] = A[2]*B[0] - A[0]*B[2];
  AxB[2] = A[0]*B[1] - A[1]*B[0];
}

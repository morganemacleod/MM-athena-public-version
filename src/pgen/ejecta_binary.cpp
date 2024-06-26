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
//! \file planet_wind_lambda.cpp: tidal perturbation of planet wind defined by hydrodynamic escape parameter lambda
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

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

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

int RefinementCondition(MeshBlock *pmb);


void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);

Real vr_ejecta(Real r);
Real dmdv_ejecta(Real v);


// global (to this file) problem parameters
Real gamma_gas; 
//Real da,pa; // ambient density, pressure

Real Ggrav;   // G 
Real GM2, GM1; // point masses
Real rsoft2; // softening length of PM 2
int corotating_frame; // flags for output, gas backreaction on EOM, frame choice
int n_particle_substeps; // substepping of particle integration

Real xi[3], vi[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration
Real Omega[3];  // vector rotation of the frame, initial wind

Real trackfile_next_time, trackfile_dt;
int  trackfile_number;

int is_restart;

Real m_ejecta, vinf_min_ejecta, vinf_max_ejecta, dlogmdlogv_ejecta; // ejecta variables
Real r_max_ejecta;
Real r_inner;
Real x1_min_derefine;

Real da,pa;

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
  GM2 = pin->GetOrAddReal("problem","M2",0.0)*Ggrav;
  GM1 = pin->GetOrAddReal("problem","M1",1.9e30)*Ggrav;

  rsoft2 = pin->GetOrAddReal("problem","rsoft2",0.1);
  corotating_frame = pin->GetInteger("problem","corotating_frame");

  trackfile_dt = pin->GetOrAddReal("problem","trackfile_dt",0.01);
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");

  m_ejecta = pin->GetReal("problem","m_ejecta");
  vinf_min_ejecta =pin->GetReal("problem","vinf_min_ejecta");
  vinf_max_ejecta =pin->GetReal("problem","vinf_max_ejecta");
  dlogmdlogv_ejecta =pin->GetReal("problem","dlogmdlogv_ejecta");
   
  r_max_ejecta = pin->GetReal("problem","r_max_ejecta");
  r_inner = pin->GetReal("mesh","x1min");
  x1_min_derefine = pin->GetReal("problem","r_max_ejecta");



  // local vars
  Real sma = pin->GetOrAddReal("problem","sma",1.5e12);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Real Omega_orb, vcirc;
 
  // allocate MESH data for the particle pos/vel, Omega frame, omega_planet & omega_star
  AllocateRealUserMeshDataField(3);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
  

  // enroll the BCs
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, DiodeOuterX1);
  }

   // Enroll AMR
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(TwoPointMass);

  // Enroll extra history output
  //AllocateUserHistoryOutput(1);
  //EnrollUserHistoryOutput(0, mdotp, "mdotp");

  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;
    
  // PARTICLE ICs
  //ONLY enter ICs loop if this isn't a restart
  if(time==0){
    //Real vcirc = sqrt((GM1+GM2)/sma + accel*sma);    
    vcirc = sqrt((GM1+GM2)/sma);
    Omega_orb = vcirc/sma;
    // rotation of star and planet
    //omega_star = f_corot_star * Omega_orb;
    //omega_planet = f_corot_planet * Omega_orb;
    
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

    // save the ruser_mesh_data variables
    for(int i=0; i<3; i++){
      ruser_mesh_data[0](i)  = xi[i];
      ruser_mesh_data[1](i)  = vi[i];
      ruser_mesh_data[2](i)  = Omega[i];
    }

    //ruser_mesh_data[3](0) = omega_planet;
    //ruser_mesh_data[3](1) = omega_star;
    
    
  }else{
    is_restart=1;
  }
  
    
  // Print out some info
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "==========   SIMULATION INFO =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "time =" << time << "\n";
    std::cout << "Ggrav = "<< Ggrav <<"\n";
    std::cout << "gamma = "<< gamma_gas <<"\n";
    std::cout << "GM1 = "<< GM1 <<"\n";
    std::cout << "GM2 = "<< GM2 <<"\n";
    std::cout << "Omega_orb="<< Omega_orb << "\n";
    std::cout << "a = "<< sma <<"\n";
    std::cout << "e = "<< ecc <<"\n";
    std::cout << "P = "<< 6.2832*sqrt(sma*sma*sma/(GM1+GM2)) << "\n";
    std::cout << "rsoft2 ="<<rsoft2<<"\n";
    std::cout << "corotating frame? = "<< corotating_frame<<"\n";
    std::cout << "particle substeping n="<<n_particle_substeps<<"\n";
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



Real vr_ejecta(Real r)
{
  Real vesc_r = std::sqrt(2*GM1/r);
  return (r - r_inner)* (vinf_max_ejecta - vinf_min_ejecta)/(r_max_ejecta-r_inner) + vinf_min_ejecta + vesc_r;
}

Real dmdv_ejecta(Real v)
{
  // normalization of the ejecta distribution
  // dmdv = dmdv0 * (v/v0)^dlogmdlogv_ejecta
  Real v0 = std::sqrt(GM1/r_inner);
  Real v_integral_factor =  pow((1.0/v0),dlogmdlogv_ejecta) * ( pow(vinf_max_ejecta,dlogmdlogv_ejecta+1.0) - pow(vinf_min_ejecta,dlogmdlogv_ejecta+1.0) ) / (dlogmdlogv_ejecta +1.0);
  Real dmdv0 = m_ejecta / v_integral_factor;
  return  dmdv0 * pow(v/v0, dlogmdlogv_ejecta);
}




//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres, vr;
  Real sma = pin->GetReal("problem","sma");
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

	Real r  = pcoord->x1v(i);
	Real th = pcoord->x2v(j);
	Real ph = pcoord->x3v(k);

	Real sin_th = sin(th);
	Real Rcyl = r*sin_th;

	// INITIALIZE STAR BOUNDARY AND STELLAR WIND
	// current position of the secondary
	Real x_2 = xi[0];
	Real y_2 = xi[1];
	Real z_2 = xi[2];
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin(th)*cos(ph);
	Real y = r*sin(th)*sin(ph);
	Real z = r*cos(th);

	Real vr,vth,vph;


	// Initialize ejecta 
	if(r<=r_max_ejecta){
	  // ejecta given the velocity distribution
	  vr = vr_ejecta(r);
	  vth  = 0.0;
	  vph  = - Omega[2]*Rcyl;   //omega_planet*sin_th*sin_th/Rcyl - Omega[2]*Rcyl;

	  Real dmdv = dmdv_ejecta(vr);
	  Real delta_v = vr_ejecta(pcoord->x1f(i+1)) -  vr_ejecta(pcoord->x1f(i));
	  Real mr = dmdv * delta_v;
	  den  = mr / (4.*3.14159* r*r*(pcoord->x1f(i+1) - pcoord->x1f(i)) ); 
	  pres = pa;
	  
	}else{
	  den = da;
	  pres= pa;
	  vr  = 0.0;
	  vth = 0.0;
	  vph = - Omega[2]*Rcyl;
	}
	
	phydro->u(IDN,k,j,i) = std::max(den,da);
	phydro->u(IM1,k,j,i) = den*vr;
	phydro->u(IM2,k,j,i) = den*vth;
	phydro->u(IM3,k,j,i) = den*vph;
	phydro->u(IEN,k,j,i) = std::max(pres,pa)/(gamma_gas-1.0);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator






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
	Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	
	// PM2 gravitational accels in cartesian coordinates
	Real a_x = - GM2 * fspline(d2,rsoft2) * (x-x_2);   
	Real a_y = - GM2 * fspline(d2,rsoft2) * (y-y_2);  
	Real a_z = - GM2 * fspline(d2,rsoft2) * (z-z_2);
	
	// add the correction for the orbiting frame (relative to the COM)
	a_x += -  GM2 / d12c * x_2;
	a_y += -  GM2 / d12c * y_2;
	a_z += -  GM2 / d12c * z_2;
	
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
	cons(IEN,k,j,i) += src_1/den * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);


      }
    }
  } // end loop over cells
  
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



//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){

  Real ai[3];
  Real mg;
  
  // kick the initial conditions back a half step (v^n-1/2)
  if(ncycle==0){
    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai); 
  } // ncycle=0 

    
  // EVOLVE THE ORBITAL POSITION OF THE SECONDARY
  // do this on rank zero, then broadcast
  if (Globals::my_rank == 0){
    for (int ii=1; ii<=n_particle_substeps; ii++) {
      // add the particle acceleration to ai
      ParticleAccels(xi,vi,ai);
      // advance the particle
      particle_step(dt/n_particle_substeps,xi,vi,ai);
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
  
  // write the output to the trackfile
  if(time >= trackfile_next_time || user_force_output ){
    WritePMTrackfile(pblock->pmy_mesh,pin);
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
    fprintf(pfile,"\n");

    // close the file
    fclose(pfile);  

  } // end rank==0

  // increment counters
  trackfile_number++;
  trackfile_next_time += trackfile_dt;
  
  return;
}


//--------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
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

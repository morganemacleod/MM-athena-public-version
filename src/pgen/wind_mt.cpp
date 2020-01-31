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
//! \file wind_mt.cpp: wind mass transfer interaction in a binary system
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#define NARRAY 1000


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

Real Interpolate1DArray(Real *x,Real *y,Real x0);

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void WindInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3],Real (&ag1i)[3], Real (&ag2i)[3]);

void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
int RefinementCondition(MeshBlock *pmb);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real fspline(Real r, Real eps);


void ParticleAccrete(Mesh *pm, Real(&xi)[3],Real(&vi)[3],Real(&mdot), Real(&pdot)[3]);



// global (to this file) problem parameters
Real gamma_gas; 
Real da,pa; // ambient density, pressure

Real GM2, GM1; // point masses
Real rsoft2; // softening length of PM 2
int  include_gas_backreaction, corotating_frame; // flags for output, gas backreaction on EOM, frame choice
int n_particle_substeps; // substepping of particle integration

Real xi[3], vi[3], agas1i[3], agas2i[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration

Real Omega[3],  Omega_wind;  // vector rotation of the frame, initial wind

Real trackfile_next_time, trackfile_dt;
int  trackfile_number;

Real Ggrav;

Real wind_mdot, wind_v_o_vesc, wind_mach_initial; // variables describing the wind BC

int is_restart;
int particle_accrete;
Real mdot, pdot[3]; // accretion parameters



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
  GM1 = pin->GetOrAddReal("problem","M1",1.0)*Ggrav;

  rsoft2 = pin->GetOrAddReal("problem","rsoft2",0.1);
  corotating_frame = pin->GetInteger("problem","corotating_frame");

  trackfile_dt = pin->GetOrAddReal("problem","trackfile_dt",0.01);

  include_gas_backreaction = pin->GetInteger("problem","gas_backreaction");
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");

  wind_mdot = pin->GetOrAddReal("problem","wind_mdot",1.e-10);
  wind_v_o_vesc = pin->GetOrAddReal("problem","wind_vel_factor",1.0);
  wind_mach_initial = pin->GetOrAddReal("problem","wind_mach_initial",1.0);
  
  particle_accrete = pin->GetInteger("problem","particle_accrete");

  // local vars
  Real rmin = pin->GetOrAddReal("mesh","x1min",0.0);
  Real rmax = pin->GetOrAddReal("mesh","x1max",0.0);
  Real thmin = pin->GetOrAddReal("mesh","x2min",0.0);
  Real thmax = pin->GetOrAddReal("mesh","x2max",0.0);

  Real sma = pin->GetOrAddReal("problem","sma",2.0);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);

  Real wind_f_corot = pin->GetOrAddReal("problem","wind_f_corotation",1.0);
 
  Real Omega_orb, vcirc;
  

  // enroll the BCs
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, DiodeOuterX1);
  }
  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X1, WindInnerX1);
  }


  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(TwoPointMass);


  // Enroll AMR
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);


  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;

  // allocate MESH data for the particle pos/vel, Omega frame
  AllocateRealUserMeshDataField(3);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
    
  //ONLY enter ICs loop if this isn't a restart
  if(time==0){
    // Print out some info                                                                                     
    if (Globals::my_rank==0){
      std::cout << "*** Setting initial conditions for t=0 ***\n";
    }
  
    // set the initial conditions for the pos/vel of the secondary
    xi[0] = sma*(1.0 + ecc);  // apocenter
    xi[1] = 0.0;
    xi[2] = 0.0;
    
    vcirc = sqrt((GM1+GM2)/sma);    
    Omega_orb = vcirc/sma;
    
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
    
  
  }else{
    is_restart=1;
  }

  // set the wind Omega
  Omega_wind = wind_f_corot * Omega_orb;
  
    
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
    std::cout << "gas backreaction? = "<< include_gas_backreaction<<"\n";
    std::cout << "particle substeping n="<<n_particle_substeps<<"\n";
    if(time==0){
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
  }
  
    


} // end





int RefinementCondition(MeshBlock *pmb)
{
  Real mindist=1.e10;
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
      }
    }
  }
  if(mindist >  3.0*rsoft2) return -1;
  if(mindist <= 3.0*rsoft2) return 1;
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
 

  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

	Real r = pmb->pcoord->x1v(i);
	Real th= pmb->pcoord->x2v(j);
	Real ph= pmb->pcoord->x3v(k);

	Real vr  = prim(IVX,k,j,i);
	Real vth = prim(IVY,k,j,i);
	Real vph = prim(IVZ,k,j,i);

	//get some angles
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	Real sin_ph = sin(ph);
	Real cos_ph = cos(ph);

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
	if(include_gas_backreaction == 1){
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
	
	// update the energy (source = - rho v dot a
	cons(IEN,k,j,i) += src_1/den * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);

      }
    }
  } // end loop over cells
  

}


//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//  used to initialize variables which are global to other functions in this file. 
//  Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  return;
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
	den = da;
	
	// get the pressure 
	pres = pa;

	// set the density
	phydro->u(IDN,k,j,i) = den;

	//set the pressure              
        phydro->u(IPR,k,j,i) = pres;
	
   	// set the momenta components
	phydro->u(IM1,k,j,i) = 0.0;
	phydro->u(IM2,k,j,i) = 0.0;
	phydro->u(IM3,k,j,i) = 0.0;

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
  return;
} // end of UserWorkInLoop


//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){

  // first let the particle accrete
  if(particle_accrete>0){
    ParticleAccrete(pblock->pmy_mesh,xi,vi,mdot,pdot);
  }


  Real ai[3];
  // ONLY ON THE FIRST CALL TO THIS FUNCTION
  // (NOTE: DOESN'T WORK WITH RESTARTS)
  if(ncycle==0){
    // kick the initial conditions back a half step (v^n-1/2)

    // first sum the gas accel if needed
    if(include_gas_backreaction == 1){
      SumGasOnParticleAccels(pblock->pmy_mesh, xi,agas1i,agas2i);
    }

    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai);
  }
    
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

  // sum the gas->part accel for the next step
  if(include_gas_backreaction == 1){
    SumGasOnParticleAccels(pblock->pmy_mesh, xi,agas1i,agas2i);

  }else if(time >= trackfile_next_time){
    SumGasOnParticleAccels(pblock->pmy_mesh, xi,agas1i,agas2i);
  }
  
  
  // write the   to the trackfile
  if(time >= trackfile_next_time){
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
      fprintf(pfile,"mdot            ");
      fprintf(pfile,"pdotx           ");
      fprintf(pfile,"pdoty           ");
      fprintf(pfile,"pdotz           ");
      fprintf(pfile,"\n");
    }


    // write the data line
    fprintf(pfile,"%20i",pm->ncycle);
    fprintf(pfile,"%20.6e",pm->time);
    fprintf(pfile,"%20.6e",pm->dt);
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
    fprintf(pfile,"%20.6e",mdot);
    fprintf(pfile,"%20.6e",pdot[0]);
    fprintf(pfile,"%20.6e",pdot[1]);
    fprintf(pfile,"%20.6e",pdot[2]);
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





void ParticleAccrete(Mesh *pm, Real(&xi)[3],Real(&vi)[3], Real(&mdot), Real(&pdot)[3] ){
  
  // start by setting accelerations / positions to zero                                                                 
  mdot = 0.0;
  for (int ii = 0; ii < 3; ii++){
    pdot[ii] = 0.0;
  }
  
  Real mshell = 0.0;
  Real Vshell = 0.0;

  MeshBlock *pmb=pm->pblock;
  Real dt = pm->dt;
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
	  //coordinates
	  Real r = pmb->pcoord->x1v(i);
	  Real th= pmb->pcoord->x2v(j);
	  Real ph= pmb->pcoord->x3v(k);

	  //get some angles
	  Real sin_th = sin(th);
	  Real cos_th = cos(th);
	  Real sin_ph = sin(ph);
	  Real cos_ph = cos(ph);
	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // current position of the secondary
	  Real x_2 = xi[0];
	  Real y_2 = xi[1];
	  Real z_2 = xi[2];

	  Real d2 = sqrt(pow(x-x_2, 2) +
			 pow(y-y_2, 2) +
			 pow(z-z_2, 2) );

	  // conditions just outside sink
	  if((d2>rsoft2) && (d2<2.0*rsoft2)){
	    // add to the local sum
	    mshell += vol(i) * phyd->u(IDN,k,j,i);
	    Vshell += vol(i);
	  }

	}
      }
    }//end loop over cells                                                                                            
  pmb=pmb->next;
  } // end loop over blocks

#ifdef MPI_PARALLEL
  // sum over all ranks                                                                                                  
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &mshell, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Vshell, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
   } else {
    MPI_Reduce(&mshell,&mshell,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&Vshell,&Vshell,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
   }

  // broadcast the result
  MPI_Bcast(&mshell,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&Vshell,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
 #endif



  // SECOND LOOP: apply sink
  Real rho_sink = 0.1 * mshell/Vshell; 
  Real pres_sink = 0.5*rho_sink;

  if (Globals::my_rank == 0) {
    std::cout<<"rho_sink="<<rho_sink<<"\n";
  }

  pmb=pm->pblock;
  while (pmb != NULL) {
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  //coordinates
	  Real r = pmb->pcoord->x1v(i);
	  Real th= pmb->pcoord->x2v(j);
	  Real ph= pmb->pcoord->x3v(k);

	  //get some angles
	  Real sin_th = sin(th);
	  Real cos_th = cos(th);
	  Real sin_ph = sin(ph);
	  Real cos_ph = cos(ph);
	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // current position of the secondary
	  Real x_2 = xi[0];
	  Real y_2 = xi[1];
	  Real z_2 = xi[2];

	  Real d2 = sqrt(pow(x-x_2, 2) +
			 pow(y-y_2, 2) +
			 pow(z-z_2, 2) );

	  // ADD a sink BC near the secondary mass
	  if(d2<rsoft2){
	    // get the cartesian velocities from the spherical (vector)                                                 
	    Real vr  = phyd->u(IM1,k,j,i)/phyd->u(IDN,k,j,i);
	    Real vth = phyd->u(IM2,k,j,i)/phyd->u(IDN,k,j,i);
	    Real vph = phyd->u(IM3,k,j,i)/phyd->u(IDN,k,j,i);

	    Real vgas[3];
	    vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	    vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	    vgas[2] = cos_th*vr - sin_th*vth;

	    // cell mass dm                                                                                             
	    Real dm = vol(i) * (phyd->u(IDN,k,j,i)-rho_sink);
	    // accreted momentum depends on velocity difference between particle/gas (cartesian)
	    Real dp1 = dm*(vgas[0]-vi[0]);
	    Real dp2 = dm*(vgas[1]-vi[1]);
	    Real dp3 = dm*(vgas[2]-vi[2]);

	    // reset values within the "sink"
	    phyd->u(IDN,k,j,i) = rho_sink;
	    phyd->u(IPR,k,j,i) = pres_sink;
	    phyd->u(IM1,k,j,i) = 0.0;
	    phyd->u(IM2,k,j,i) = 0.0;
	    phyd->u(IM3,k,j,i) = 0.0;
	    phyd->u(IEN,k,j,i) = pres_sink/(gamma_gas-1);

	    // add to the local sums
	    mdot += dm;
	    pdot[0] += dp1;
	    pdot[1] += dp2;
	    pdot[2] += dp3;
	  }

	}
      }
    }//end loop over cells                                                                                            
  pmb=pmb->next;
  } // end loop over blocks

#ifdef MPI_PARALLEL
  // sum over all ranks                                                                                                  
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &mdot, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, pdot, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&mdot,&mdot,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(pdot,pdot,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  // divide by dt, then broadcast the result
  mdot=mdot/dt;
  pdot[0]=pdot[0]/dt;
  pdot[1]=pdot[1]/dt;
  pdot[2]=pdot[2]/dt;

  MPI_Bcast(&mdot,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(pdot,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif


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
	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // current position of the secondary
	  Real x_2 = xi[0];
	  Real y_2 = xi[1];
	  Real z_2 = xi[2];

	  Real d2 = sqrt(pow(x-x_2, 2) +
			 pow(y-y_2, 2) +
			 pow(z-z_2, 2) );
  
	  Real d1c = pow(r,3);
	  
	   // gravitational accels in cartesian coordinates
	  
	  ag1i[0] += Ggrav*dm/d1c * x;
	  ag1i[1] += Ggrav*dm/d1c * y;
	  ag1i[2] += Ggrav*dm/d1c * z;
	  
	  ag2i[0] += Ggrav*dm * fspline(d2,rsoft2) * (x-x_2);
	  ag2i[1] += Ggrav*dm * fspline(d2,rsoft2) * (y-y_2);
	  ag2i[2] += Ggrav*dm * fspline(d2,rsoft2) * (z-z_2);
	  
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




void WindInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real r = pco->x1f(is);
  Real vesc = sqrt(2.0*GM1/r);
  Real v_wind = wind_v_o_vesc * vesc;
  Real rho_wind = wind_mdot /(4.0 * 3.14159 * r*r * v_wind);
  Real cs_wind = v_wind / wind_mach_initial;
  Real p_wind = rho_wind * cs_wind*cs_wind / gamma_gas;
 
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
	Real r = pco->x1v(is-i+1);
	Real Rcyl = r*sin(pco->x2v(j));

	
	prim(IVX,k,j,is-i) = v_wind;
	prim(IVY,k,j,is-i) = 0.0;
	prim(IVZ,k,j,is-i) = (Omega_wind-Omega[2])*Rcyl;
	prim(IDN,k,j,is-i) = rho_wind;
	prim(IPR,k,j,is-i) = p_wind;
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
  
  

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]){
  // set the vector AxB = A x B
  AxB[0] = A[1]*B[2] - A[2]*B[1];
  AxB[1] = A[2]*B[0] - A[0]*B[2];
  AxB[2] = A[0]*B[1] - A[1]*B[0];
}





Real Interpolate1DArray(Real *x,Real *y,Real x0){
  int i;
  Real d,dx,s,y0;
  //std::cout.precision(17);
  

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

  // if in the interior, do a linear interpolation
  for(i=0;i<NARRAY-1;i++){
    if (x[i+1] >= x0){
      dx =  (x[i+1]-x[i]);
      d = (x0 - x[i]);
      s = (y[i+1]-y[i]) /dx;
      y0 = s*d + y[i];
      return y0;
    }
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}

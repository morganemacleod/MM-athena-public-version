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

void StarPlanetWinds(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);


void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real mdotp(MeshBlock *pmb, int iout);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);

int RefinementCondition(MeshBlock *pmb);

void SphericaltoCartesian(Real &r, Real &th, Real &ph, Real &x, Real &y, Real &z);
void CartesiantoSpherical(Real &r, Real &th, Real &ph, Real &x, Real &y, Real &z);
void SphericaltoCartesian_VEC(Real &th, Real &ph, Real &vr, Real &vth, Real &vph, Real &vx, Real &vy, Real &vz);
void CartesiantoSpherical_VEC(Real &th, Real &ph, Real &vr, Real &vth, Real &vph, Real &vx, Real &vy, Real &vz);



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

Real rho_surface, lambda; // planet surface variables
Real r_inner;

Real rho_surface_star, lambda_star, radius_star; //stellar surface variables
Real omega_planet, omega_star; // rotation of planet and star boundaries

bool initialize_planet_wind; // true=planetary wind backgorund ic, false stellar wind ic background
Real da,pa;

Real x1_min_derefine; // for AMR

Real POLE_DIR = 0;  // DIRECTION OF POLE OF THE SPHERICAL POLAR COORDINATE SYSTEM 0=x, 2=z




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

  rho_surface = pin->GetOrAddReal("problem","rho_surface",1.e-15);
  lambda = pin->GetOrAddReal("problem","lambda",5.0);

  rho_surface_star = pin->GetOrAddReal("problem","rho_surface_star",1.e-15);
  lambda_star = pin->GetOrAddReal("problem","lambda_star",5.0);
  radius_star = pin->GetOrAddReal("problem","radius_star",6.955e10);
   
  r_inner = pin->GetReal("mesh","x1min");
  x1_min_derefine = pin->GetOrAddReal("problem","x1_min_derefine",0.0);

  initialize_planet_wind = pin->GetBoolean("problem","initialize_planet_wind"); 



  // local vars
  Real sma = pin->GetOrAddReal("problem","sma",1.5e12);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Real f_corot_planet = pin->GetOrAddReal("problem","f_corotation_planet",1.0);
  Real f_corot_star   = pin->GetOrAddReal("problem","f_corotation_star",1.0);
  Real Omega_orb, vcirc;
 
  // allocate MESH data for the particle pos/vel, Omega frame, omega_planet & omega_star
  AllocateRealUserMeshDataField(4);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
  ruser_mesh_data[3].NewAthenaArray(2);

  // enroll the BCs
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, DiodeOuterX1);
  }


  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(StarPlanetWinds);

  // Enroll extra history output
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, mdotp, "mdotp");

   // Enroll AMR
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

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
    omega_star = f_corot_star * Omega_orb;
    omega_planet = f_corot_planet * Omega_orb;

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
    
    ruser_mesh_data[3](0) = omega_planet;
    ruser_mesh_data[3](1) = omega_star;
    
    
  }else{
    is_restart=1;
  }
  
    
  // Print out some info
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "==========   SIMULATION INFO =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "POLE DIR =" << POLE_DIR << "\n";
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
    std::cout << "==========   BC INFO         =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "rho_surface (planet) = "<< rho_surface <<"\n";
    std::cout << "lambda (planet) = "<< lambda <<"\n";
    std::cout << "press_surface (planet) =" << rho_surface*GM1/(r_inner*gamma_gas*lambda) <<"\n";
    std::cout << "temperature_surface (planet) =" << (rho_surface*GM1/(r_inner*gamma_gas*lambda)) / ( rho_surface * 8.3145e7) <<"\n";
    std::cout << "rho_surface (star) = "<< rho_surface_star <<"\n";
    std::cout << "lambda (star) = "<< lambda_star <<"\n";
    std::cout << "press_surface (star) =" << rho_surface_star*GM2/(radius_star*gamma_gas*lambda_star) <<"\n";
    std::cout << "temperature_surface (star) =" << (rho_surface_star*GM2/(radius_star*gamma_gas*lambda_star)) / ( rho_surface_star * 8.3145e7) <<"\n";
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


void SphericaltoCartesian(Real &r, Real &th, Real &ph, Real &x, Real &y, Real &z){
  // spherical polar coordinates, get local cartesian           
  if(POLE_DIR==2){
    x = r*sin(th)*cos(ph);
    y = r*sin(th)*sin(ph);
    z = r*cos(th);
  }else if(POLE_DIR==0){
    y = r*sin(th)*cos(ph);
    z = r*sin(th)*sin(ph);
    x = r*cos(th);
  }
}

void SphericaltoCartesian_VEC(Real &th, Real &ph, Real &vr, Real &vth, Real &vph, Real &vx, Real &vy, Real &vz){
  Real sin_th = sin(th);
  Real cos_th = cos(th);
  Real sin_ph = sin(ph);
  Real cos_ph = cos(ph);
  if(POLE_DIR==2){
    vx = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
    vy = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
    vz = cos_th*vr - sin_th*vph;
  }else if(POLE_DIR==0){
    vy = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
    vz = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
    vx = cos_th*vr - sin_th*vph;
  }
  
}
void CartesiantoSpherical_VEC(Real &th, Real &ph, Real &vr, Real &vth, Real &vph, Real &vx, Real &vy, Real &vz){
  Real sin_th = sin(th);
  Real cos_th = cos(th);
  Real sin_ph = sin(ph);
  Real cos_ph = cos(ph);
  if(POLE_DIR==2){
    vr  = sin_th*cos_ph*vx + sin_th*sin_ph*vy + cos_th*vz;
    vth = cos_th*cos_ph*vx + cos_th*sin_ph*vy - sin_th*vz;
    vph = -sin_ph*vx + cos_ph*vy;
  }else if(POLE_DIR==0){
    vr  = sin_th*cos_ph*vy + sin_th*sin_ph*vz + cos_th*vx;
    vth = cos_th*cos_ph*vy + cos_th*sin_ph*vz - sin_th*vx;
    vph = -sin_ph*vy + cos_ph*vz;
  }
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

	Real x,y,z;
	SphericaltoCartesian(r,th,ph,x,y,z);
	Real Rcyl = sqrt(x*x + y*y);
	
	
	// INITIALIZE STAR BOUNDARY AND STELLAR WIND
	// location relative to point mass 2 (star)
	Real d2  = sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) +
			pow(z-xi[2], 2) );

	Real R2 =  sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) ); // in the orb plane
	Real phi2 = std::atan2(y-xi[1],x-xi[0]); // az angle in the orbital plane
	Real th2 = std::acos((z-xi[2])/d2); // polar angle in the orbital plane
	
	// surface parameters (star and planet)
	Real press_surface_star = rho_surface_star*GM2/(radius_star*gamma_gas*lambda_star);
	Real cs_star = std::sqrt(gamma_gas *press_surface_star/rho_surface_star);
	Real press_surface = rho_surface*GM1/(r*gamma_gas*lambda);
	Real cs_planet = std::sqrt(gamma_gas *press_surface/rho_surface);
	
	Real vx,vy,vz;
	Real vr,vth,vph;

	// Near Star
	if(d2 <= radius_star){
	  den = rho_surface_star;
	  pres = press_surface_star;
	  vx = vi[0] - sin(phi2)*(omega_star-Omega[2])*R2;
	  vy = vi[1] + cos(phi2)*(omega_star-Omega[2])*R2;
	  vz = vi[2];	  
	}else if(d2<= sma/2){
	  den = rho_surface_star * pow((d2/radius_star),-8);
	  pres = press_surface_star * pow(den / rho_surface_star, gamma_gas);
	  // wind directed outward at v=cs
	  // constant angular momentum of surface
	  vx = (x-xi[0])/d2 * cs_star + vi[0] - sin(phi2)*(omega_star*std::sin(th2)*std::sin(th2)/R2 - Omega[2]*Rcyl);  
	  vy = (y-xi[1])/d2 * cs_star + vi[1] + cos(phi2)*(omega_star*std::sin(th2)*std::sin(th2)/R2 - Omega[2]*Rcyl);  
	  vz = (z-xi[2])/d2 * cs_star + vi[2];
	}else{
	  den = da;
	  pres = pa;
	  vx = 0.0;
	  vy = 0.0;
	  vz = 0.0; 
	}

	if(initialize_planet_wind == true & r<=sma/2){
	  // wind directed outward at v=cs outside of sonic point, linear increase to sonic point
	  // constant angular momentum of surface
	  den = rho_surface * pow((r/r_inner),-8);
	  pres = press_surface * pow(den / rho_surface, gamma_gas);
	  vr = cs_planet * std::min(r/(lambda/2. * r_inner), 1.0);  
	  Real az = std::atan2(y,x);
	  Real polar = std::acos(z/r);
	  vx = x/r*vr -sin(az)*(omega_planet*SQR(sin(polar))/Rcyl  - Omega[2]*Rcyl);
	  vy = y/r*vr +cos(az)*(omega_planet*SQR(sin(polar))/Rcyl  - Omega[2]*Rcyl);
	  vz = z/r*vr;   
	}
	
        CartesiantoSpherical_VEC(th,ph,vr,vth,vph,vx,vy,vz); // get vr,vth,vpi
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
void StarPlanetWinds(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{ 

  if(is_restart>0){
    // else this is a restart, read the current particle state
    for(int i=0; i<3; i++){
      xi[i]    = pmb->pmy_mesh->ruser_mesh_data[0](i);
      vi[i]    = pmb->pmy_mesh->ruser_mesh_data[1](i);
      Omega[i] = pmb->pmy_mesh->ruser_mesh_data[2](i);
    }
    omega_planet = pmb->pmy_mesh->ruser_mesh_data[3](0);
    omega_star   = pmb->pmy_mesh->ruser_mesh_data[3](1);

    // print some info
    if (Globals::my_rank==0){
      std::cout << "*** Setting initial conditions for t>0 ***\n";
      std::cout <<"xi="<<xi[0]<<" "<<xi[1]<<" "<<xi[2]<<"\n";
      std::cout <<"vi="<<vi[0]<<" "<<vi[1]<<" "<<vi[2]<<"\n";
      std::cout <<"Omega="<<Omega[0]<<" "<<Omega[1]<<" "<<Omega[2]<<"\n";
      std::cout << "omega_planet ="<<omega_planet<<"  omega_star ="<<omega_star<<"\n";
    }
    is_restart=0;
  }
  
  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real ph= pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	
	// current position of the secondary
	Real d12c = pow(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2], 1.5);
	
	// spherical polar coordinates, get local cartesian           
	Real x,y,z;
	SphericaltoCartesian(r,th,ph,x,y,z);
  
	Real d2  = sqrt(pow(x-xi[0], 2) +
			pow(y-xi[1], 2) +
			pow(z-xi[2], 2) );
  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1
	Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	
	// PM2 gravitational accels in cartesian coordinates
	Real a_x = - GM2 * fspline(d2,rsoft2) * (x-xi[0]);   
	Real a_y = - GM2 * fspline(d2,rsoft2) * (y-xi[1]);  
	Real a_z = - GM2 * fspline(d2,rsoft2) * (z-xi[2]);
	
	// add the correction for the orbiting frame (relative to the COM)
	a_x += -  GM2 / d12c * xi[0];
	a_y += -  GM2 / d12c * xi[1];
	a_z += -  GM2 / d12c * xi[2];
	
	Real vr  = prim(IVX,k,j,i);
	Real vth = prim(IVY,k,j,i);
	Real vph = prim(IVZ,k,j,i);

	if(corotating_frame == 1){

	  // distance from the origin in cartesian (vector)
	  Real rxyz[3];
	  rxyz[0] = x;
	  rxyz[1] = y;
	  rxyz[2] = z;
	  
	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  SphericaltoCartesian_VEC(th,ph,vr,vth,vph,vgas[0],vgas[1],vgas[2]);
	  	  
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
	Real a_r,a_th,a_ph;
	CartesiantoSpherical_VEC(th,ph,a_r,a_th,a_ph,a_x,a_y,a_z);
	
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


	
	// STAR BOUNDARY (note, overwrites the grav accel, ie gravitational accel is not applied in this region)
	if(d2 <= radius_star){
	  Real R2 =  sqrt(pow(x-xi[0], 2) +
			  pow(y-xi[1], 2) ); // orb plane cyl radius
	  Real phi2 = std::atan2(y-xi[1],x-xi[0]); // orb plane az angle
	  
	  Real press_surface_star = rho_surface_star*GM2/(radius_star*gamma_gas*lambda_star);
	  Real cs = std::sqrt(gamma_gas *press_surface_star/rho_surface_star);
	  Real vx = vi[0] - sin(phi2)*(omega_star-Omega[2])*R2;
	  Real vy = vi[1] + cos(phi2)*(omega_star-Omega[2])*R2;
	  Real vz = vi[2];

	  // convert back to spherical polar
	  CartesiantoSpherical_VEC(th,ph,vr,vth,vph,vx,vy,vz);
	  
	  cons(IDN,k,j,i) = rho_surface_star;
	  cons(IM1,k,j,i) = rho_surface_star*vr;
	  cons(IM2,k,j,i) = rho_surface_star*vth;  
	  cons(IM3,k,j,i) = rho_surface_star*vph;  
	  cons(IEN,k,j,i) = press_surface_star/(gamma_gas-1.0);
	  cons(IEN,k,j,i) += 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
				       + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
	}


	// PLANET BOUNDARY(note, overwrites the grav accel, ie gravitational accel is not applied in this region)
	if((pmb->pbval->block_bcs[INNER_X1] == REFLECTING_BNDRY) && (i==pmb->is)) {
	  Real press_surface = rho_surface*GM1/(r*gamma_gas*lambda);
	  Real Rcyl = sqrt(x*x + y*y);
	  Real az = std::atan2(y,x);
	  Real polar = std::acos(z/r);
	  Real vx = -sin(az)*(omega_planet*SQR(sin(polar))/Rcyl  - Omega[2]*Rcyl);
	  Real vy = cos(az)*(omega_planet*SQR(sin(polar))/Rcyl  - Omega[2]*Rcyl);
	  Real vz = 0.0;
	  Real vr,vth,vph;
	  CartesiantoSpherical_VEC(th,ph,vr,vth,vph,vx,vy,vz); // get vr,vth,vpi
	  cons(IDN,k,j,pmb->is) = rho_surface;
	  cons(IM1,k,j,pmb->is) = rho_surface*vr;
	  cons(IM2,k,j,pmb->is) = rho_surface*vth;
	  cons(IM3,k,j,pmb->is) = rho_surface*vph;
	  cons(IEN,k,j,pmb->is) = press_surface/(gamma_gas-1.0);
	  cons(IEN,k,j,pmb->is) += 0.5*(SQR(cons(IM1,k,j,pmb->is))+SQR(cons(IM2,k,j,pmb->is))
					+ SQR(cons(IM3,k,j,pmb->is)))/cons(IDN,k,j,pmb->is);
	  
	}

	

      }
    }
  } // end loop over cells
}





Real mdotp(MeshBlock *pmb, int iout){
  Real mdot = 0.0;
  
  if(pmb->pbval->block_bcs[INNER_X1] == REFLECTING_BNDRY) {
     int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
     AthenaArray<Real> area;
     int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
     area.NewAthenaArray(ncells1);
  
     for(int k=ks; k<=ke; k++) {
       for(int j=js; j<=je; j++) {
          pmb->pcoord->VolCenterFace1Area(k,j,pmb->is,pmb->ie,area);
          mdot += area(pmb->is+10)*pmb->phydro->u(IM1,k,j,pmb->is+10); // dmdot = dA*rho*v
        }
      }
    } // end if
  
  return mdot;
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
  if( (mindist > 4.0*radius_star) && rmin>x1_min_derefine  ) return -1;
  // refine near point mass
  if(mindist <= 3.0*radius_star) return 1;
   // otherwise do nothing
  return 0;
}




void MeshBlock::UserWorkInLoop(void) {
  // Add timestep diagnostics
  if(pmy_mesh->ncycle % 10 == 0){
    if(new_block_dt == pmy_mesh->dt){
      // call NewBlockTimeStep with extra diagnostic output
      phydro->NewBlockTimeStep(1);
    }
  }
  
  return;
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

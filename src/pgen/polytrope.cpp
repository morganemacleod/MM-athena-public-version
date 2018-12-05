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
//! \file pm_envelope.cpp: tidal perturbation of polytropic stellar envelope
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




Real Interpolate1DArrayEven(Real *x,Real *y,Real x0);

// global (to this file) problem parameters
Real gamma_gas; 
Real da,pa; // ambient density, pressure
Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc[NARRAY];  // initial profile

Real t_relax; // time to damp fluid motion, time to turn on M2 over

Real Ggrav;


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
  t_relax = pin->GetOrAddReal("problem","trelax",0.0);

  Real four_pi_G = 12.566370614359172*Ggrav;
  Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
  SetFourPiG(four_pi_G);
  SetGravityThreshold(eps);


  // read in profile arrays from file
  std::ifstream infile("hse_profile.dat"); 
  for(int i=0;i<NARRAY;i++){
    infile >> rad[i] >> rho[i] >> p[i] >> menc[i];
    //std:: cout << rad[i] << "    " << rho[i] << std::endl;
  }
  infile.close();
    


} // end







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

	Real x  = pcoord->x1v(i);
	Real y = pcoord->x2v(j);
	Real z = pcoord->x3v(k);

	Real r = sqrt(x*x + y*y + z*z);
		
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
  Real time = pmy_mesh->time;
  Real dt = pmy_mesh->dt;
  //Real v_factor;
  
  // if less than the relaxation time, apply 
  // a damping to the fluid velocities
  if(time < t_relax){
    //v_factor = 0.99 + 0.01*time / t_relax; // less than 1
    Real tau = 1.0;
    if(time > 0.2*t_relax){
      tau *= pow(10, 1.0*(time-0.2*t_relax)/(0.8*t_relax) );
    }
    if (Globals::my_rank==0){
      std::cout << "Relaxing: tau_damp ="<<tau<<std::endl;
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie; i++) {

	  Real den = phydro->u(IDN,k,j,i);
	  Real vx  = phydro->u(IM1,k,j,i) / den;
	  Real vy = phydro->u(IM2,k,j,i) / den;
	  Real vz = phydro->u(IM3,k,j,i) / den;
	  
	  Real a_damp_x = - vx/tau;
	  Real a_damp_y = - vy/tau;
	  Real a_damp_z = - vz/tau;
	  
	  phydro->u(IM1,k,j,i) += dt*den*a_damp_x;
	  phydro->u(IM2,k,j,i) += dt*den*a_damp_y;
	  phydro->u(IM3,k,j,i) += dt*den*a_damp_z;
	  
	  phydro->u(IEN,k,j,i) += dt*den*a_damp_x*vx + dt*den*a_damp_y*vy + dt*den*a_damp_z*vz; 
	  
	}
      }
    } // end loop over cells                   
  } // end relax

  return;
} // end of UserWorkInLoop





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




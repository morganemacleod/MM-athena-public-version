//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//  \brief computes timestep using CFL condition on a MEshBlock

// C/C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena++ headers
#include "hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn Real Hydro::NewBlockTimeStep(void)
// \brief calculate the minimum timestep within a MeshBlock

Real Hydro::NewBlockTimeStep(int diagnostic_output) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> w,bcc,b_x1f,b_x2f,b_x3f;
  w.InitWithShallowCopy(pmb->phydro->w);
  if (MAGNETIC_FIELDS_ENABLED) {
    bcc.InitWithShallowCopy(pmb->pfield->bcc);
    b_x1f.InitWithShallowCopy(pmb->pfield->b.x1f);
    b_x2f.InitWithShallowCopy(pmb->pfield->b.x2f);
    b_x3f.InitWithShallowCopy(pmb->pfield->b.x3f);
  }

  AthenaArray<Real> dt1, dt2, dt3;
  dt1.InitWithShallowCopy(dt1_);
  dt2.InitWithShallowCopy(dt2_);
  dt3.InitWithShallowCopy(dt3_);
  Real wi[(NWAVE)];

  Real min_dt = (FLT_MAX);
  Real i_min, j_min, k_min; // MM
  Real dt1_min, dt2_min, dt3_min; // MM

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CenterWidth1(k,j,is,ie,dt1);
      pmb->pcoord->CenterWidth2(k,j,is,ie,dt2);
      pmb->pcoord->CenterWidth3(k,j,is,ie,dt3);
      if (!RELATIVISTIC_DYNAMICS) {
#pragma ivdep
        for (int i=is; i<=ie; ++i) {

	  //MM: zone averaging, changes the center widths in the 3-direction.
          if(do_average_==true & n_avg_(j)>0){
            dt3(i) *= n_avg_(j);
          }
	  
          wi[IDN]=w(IDN,k,j,i);
          wi[IVX]=w(IVX,k,j,i);
          wi[IVY]=w(IVY,k,j,i);
          wi[IVZ]=w(IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) wi[IPR]=w(IPR,k,j,i);

          if (MAGNETIC_FIELDS_ENABLED) {

            Real bx = bcc(IB1,k,j,i) + fabs(b_x1f(k,j,i)-bcc(IB1,k,j,i));
            wi[IBY] = bcc(IB2,k,j,i);
            wi[IBZ] = bcc(IB3,k,j,i);
            Real cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt1(i) /= (fabs(wi[IVX]) + cf);

            wi[IBY] = bcc(IB3,k,j,i);
            wi[IBZ] = bcc(IB1,k,j,i);
            bx = bcc(IB2,k,j,i) + fabs(b_x2f(k,j,i)-bcc(IB2,k,j,i));
            cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt2(i) /= (fabs(wi[IVY]) + cf);

            wi[IBY] = bcc(IB1,k,j,i);
            wi[IBZ] = bcc(IB2,k,j,i);
            bx = bcc(IB3,k,j,i) + fabs(b_x3f(k,j,i)-bcc(IB3,k,j,i));
            cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt3(i) /= (fabs(wi[IVZ]) + cf);

          } else {

            Real cs = pmb->peos->SoundSpeed(wi);
            dt1(i) /= (fabs(wi[IVX]) + cs);
            dt2(i) /= (fabs(wi[IVY]) + cs);
            dt3(i) /= (fabs(wi[IVZ]) + cs);

          }
        }
      }

      // compute minimum of (v1 +/- C)
      for (int i=is; i<=ie; ++i) {
        Real& dt_1 = dt1(i);
        min_dt = std::min(min_dt,dt_1);
	// MM:
	if(diagnostic_output==1 && dt_1==min_dt){
	  i_min = i;
	  j_min = j;
	  k_min = k;
	  dt1_min = dt1(i);
	  dt2_min = dt2(i);
	  dt3_min = dt3(i);
	}
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_2 = dt2(i);
          min_dt = std::min(min_dt,dt_2);
	  // MM:
	  if(diagnostic_output==1 && dt_2==min_dt){
	    i_min = i;
	    j_min = j;
	    k_min = k;
	    dt1_min = dt1(i);
	    dt2_min = dt2(i);
	    dt3_min = dt3(i);
	  }
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_3 = dt3(i);
          min_dt = std::min(min_dt,dt_3);
	  // MM:
	  if(diagnostic_output==1 && dt_3==min_dt){
	    i_min = i;
	    j_min = j;
	    k_min = k;
	    dt1_min = dt1(i);
	    dt2_min = dt2(i);
	    dt3_min = dt3(i);
	  }
        }
      }
    }
  }

// calculate the timestep limited by the diffusion process
  if (phdif->hydro_diffusion_defined) {
    Real mindt_vis, mindt_cnd;
    phdif->NewHydroDiffusionDt(mindt_vis, mindt_cnd);
    min_dt = std::min(min_dt,mindt_vis);
    min_dt = std::min(min_dt,mindt_cnd);
  } // hydro diffusion

  if(MAGNETIC_FIELDS_ENABLED &&
     pmb->pfield->pfdif->field_diffusion_defined) {
    Real mindt_oa, mindt_h;
    pmb->pfield->pfdif->NewFieldDiffusionDt(mindt_oa, mindt_h);
    min_dt = std::min(min_dt,mindt_oa);
    min_dt = std::min(min_dt,mindt_h);
  } // field diffusion

  min_dt *= pmb->pmy_mesh->cfl_number;

  if (UserTimeStep_!=NULL) {
    min_dt = std::min(min_dt, UserTimeStep_(pmb));
  }

  pmb->new_block_dt=min_dt;

  // MM: diagnostic output
  if(diagnostic_output==1){
    std::cout<<"min dt: x1="<<pmb->pcoord->x1v(i_min)
	     <<" x2="<<pmb->pcoord->x2v(j_min)
	     <<" x3="<<pmb->pcoord->x3v(k_min)
	     <<" dt1="<<dt1_min*pmb->pmy_mesh->cfl_number
	     <<" dt2="<<dt2_min*pmb->pmy_mesh->cfl_number
	     <<" dt3="<<dt3_min*pmb->pmy_mesh->cfl_number<<"\n";
  }




  return min_dt;
}

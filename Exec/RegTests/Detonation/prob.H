#ifndef _PROB_H_
#define _PROB_H_

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "mechanism.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "Constants.H"
#include "PelePhysics.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"

#include <complex.h>
#include <fftw3.h>

#include <cmath>
//#include <Dfft.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
pmf(
  amrex::Real xlo,
  amrex::Real xhi,
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4>& y_vector,
  const ProbParmDevice& prob_parm)
{
  if (prob_parm.pmf_do_average) {
    int lo_loside = 0;
    int lo_hiside = 0;
    int hi_loside = 0;
    int hi_hiside = 0;
    if (xlo < prob_parm.d_pmf_X[0]) {
      lo_loside = 0;
      lo_hiside = 0;
    }
    if (xhi < prob_parm.d_pmf_X[0]) {
      hi_loside = 0;
      hi_hiside = 0;
    }
    if (xlo > prob_parm.d_pmf_X[prob_parm.pmf_N - 1]) {
      lo_loside = prob_parm.pmf_N - 1;
      lo_hiside = prob_parm.pmf_N - 1;
    }
    if (xhi > prob_parm.d_pmf_X[prob_parm.pmf_N - 1]) {
      hi_loside = prob_parm.pmf_N - 1;
      hi_hiside = prob_parm.pmf_N - 1;
    }
    if (lo_loside == 0) {
      for (unsigned int i = 0; i < prob_parm.pmf_N - 1; i++) {
        if ((xlo > prob_parm.d_pmf_X[i]) && (xlo < prob_parm.d_pmf_X[i + 1])) {
          lo_loside = i;
          lo_hiside = i + 1;
        }
      }
    }
    if (hi_loside == 0) {
      for (unsigned int i = 0; i < prob_parm.pmf_N - 1; i++) {
        if ((xhi > prob_parm.d_pmf_X[i]) && (xhi < prob_parm.d_pmf_X[i + 1])) {
          hi_loside = i;
          hi_hiside = i + 1;
        }
      }
    }
    for (unsigned int j = 0; j < prob_parm.pmf_M; j++) {
      amrex::Real x1 = prob_parm.d_pmf_X[lo_loside];
      amrex::Real y1 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + lo_loside];
      amrex::Real x2 = prob_parm.d_pmf_X[lo_hiside];
      amrex::Real y2 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + lo_hiside];
      amrex::Real dydx = 0.0;
      if (lo_loside == lo_hiside)
        dydx = 0.0;
      else
        dydx = (y2 - y1) / (x2 - x1);
      amrex::Real ylo = y1 + dydx * (xlo - x1);
      amrex::Real yhi = 0.0;
      if (lo_loside == hi_loside) {
        yhi = y1 + dydx * (xhi - x1);
        y_vector[j] = 0.5 * (ylo + yhi);
      } else {
        amrex::Real sum = (x2 - xlo) * 0.5 * (ylo + y2);
        x1 = prob_parm.d_pmf_X[hi_loside];
        y1 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + hi_loside];
        x2 = prob_parm.d_pmf_X[hi_hiside];
        y2 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + hi_hiside];
        if (hi_loside == hi_hiside)
          dydx = 0.0;
        else
          dydx = (y2 - y1) / (x2 - x1);
        yhi = y1 + dydx * (xhi - x1);
        sum += (xhi - x1) * 0.5 * (yhi + y1);
        for (int k = lo_hiside; k < hi_loside - 1; k++) {
          sum += (prob_parm.d_pmf_X[k + 1] - prob_parm.d_pmf_X[k]) * 0.5 *
                 (prob_parm.d_pmf_Y[prob_parm.pmf_N * j + k] +
                  prob_parm.d_pmf_Y[prob_parm.pmf_N * j + k + 1]);
        }
        y_vector[j] = sum / (xhi - xlo);
      }
    }
  } else {
    amrex::Real xmid = 0.5 * (xlo + xhi);
    int loside = -1;
    int hiside = -1;
    if (xmid < prob_parm.d_pmf_X[0]) {
      loside = 0;
      hiside = 0;
    }
    if (xmid > prob_parm.d_pmf_X[prob_parm.pmf_N - 1]) {
      loside = prob_parm.pmf_N - 1;
      hiside = prob_parm.pmf_N - 1;
    }
    if (loside == -1) {
      for (unsigned int i = 0; i < prob_parm.pmf_N - 1; i++) {
        if (
          (xmid >= prob_parm.d_pmf_X[i]) &&
          (xmid <= prob_parm.d_pmf_X[i + 1])) {
          loside = i;
          hiside = i + 1;
        }
      }
    }
    for (unsigned int j = 0; j < prob_parm.pmf_M; j++) {
      const amrex::Real x1 = prob_parm.d_pmf_X[loside];
      const amrex::Real y1 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + loside];
      const amrex::Real x2 = prob_parm.d_pmf_X[hiside];
      const amrex::Real y2 = prob_parm.d_pmf_Y[prob_parm.pmf_N * j + hiside];
      amrex::Real dydx = 0.0;
      if (loside == hiside)
        dydx = 0.0;
      else
        dydx = (y2 - y1) / (x2 - x1);
      y_vector[j] = y1 + dydx * (xlo - x1);
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
  // Geometry
  
   const amrex::Real* prob_lo = geomdata.ProbLo();
   const amrex::Real* dx = geomdata.CellSize();

   const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
   const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
   const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

  amrex::Real x_non_dim = x/prob_parm.delta_half;
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; n++)
    massfrac[n] = prob_parm.rho_amb * prob_parm.massfrac[n];
    
 state(i, j, k, USOOT) = prob_parm.P_amb;
  //Quiescent and unburned gas
  if (x_non_dim>= 24)
  {
  state(i, j, k, URHO) = prob_parm.rho_amb;
  state(i, j, k, UMX) = 0.0;
  state(i, j, k, UMY) = 0.0;
  state(i, j, k, UMZ) = 0.0;
  state(i, j, k, UEINT) = prob_parm.rho_amb * prob_parm.eint_amb;
  state(i, j, k, UEDEN) =
    prob_parm.rho_amb * prob_parm.eint_amb;
  state(i, j, k, UTEMP) = prob_parm.T_amb;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = prob_parm.rho_amb * prob_parm.massfrac[n];
  }
  
  //Random density perturbations
  else if (x_non_dim>=20 && x_non_dim<24)
  {
     amrex::Real r = 0;
     amrex::Real LO = 0.0;
     amrex::Real HI = 1.0;
     r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
     amrex::Real rho_local, p_local, e_int_local;
     
     if (prob_parm.do_perturb==1)
     {
      rho_local = (1.25 - 0.5*r)*prob_parm.rho_amb;
     }
     else
     {
      rho_local = prob_parm.rho_amb;
     }
     //recalculate local pressure and internal energy
     auto eos = pele::physics::PhysicsType::eos();
     eos.RTY2P(rho_local, prob_parm.T_amb, massfrac, p_local);
     eos.RYP2E(rho_local, massfrac, p_local, e_int_local);
     //assign
     state(i, j, k, URHO) = rho_local;
     state(i, j, k, UMX) = 0.0;
     state(i, j, k, UMY) = 0.0;
     state(i, j, k, UMZ) = 0.0;
     state(i, j, k, UEINT) = rho_local * e_int_local;
     state(i, j, k, UEDEN) = rho_local * e_int_local;
     state(i, j, k, UTEMP) = prob_parm.T_amb;
     for (int n = 0; n < NUM_SPECIES; n++)
     	state(i, j, k, UFS + n) = rho_local * prob_parm.massfrac[n];
   }
  //ZND solution 
  else
  {
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real P, rho, U, T, eint, rho_comp, eint_comp;
  
  amrex::Real x_reversed = (20*prob_parm.delta_half - x)/100;
  pmf(x_reversed, x_reversed, pmf_vals, prob_parm);
  
  //MKS->CGS conversion
  P = pmf_vals[0]*10; 
  T = pmf_vals[1];
  rho = pmf_vals[2]/1000;
  U = abs(pmf_vals[3]-2352.88109392)*100;
  
  //assign massfracs
  amrex::Real sum = 0;
  for (int n = 0; n < NUM_SPECIES; n++)
    {sum = sum + pmf_vals[4+n];
    massfrac[n] = pmf_vals[4+n];} 
  //correct the last specie
  sum = sum - pmf_vals[4+NUM_SPECIES];
  massfrac[NUM_SPECIES] = 1.00 - sum;
  
  //compute other params
  auto eos = pele::physics::PhysicsType::eos();
  eos.TY2E(T, massfrac, eint);
  eos.PYT2RE(P, massfrac, T, rho_comp, eint_comp);
  
  //assign to state
  state(i, j, k, URHO) = rho;
  state(i, j, k, UMX) = rho*U;
  state(i, j, k, UMY) = 0.0;
  state(i, j, k, UMZ) = 0.0;
  state(i, j, k, UEINT) = rho*eint;
  state(i, j, k, UEDEN) = rho*(eint + 0.5*U*U);
  state(i, j, k, UTEMP) = T;

  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = rho*massfrac[n];
  }
}

//--------------------------------Boundary Conditions---------------------------------//
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real* /*x[AMREX_SPACEDIM]*/,
  const amrex::Real* /*s_int[NVAR]*/,
  amrex::Real s_ext[NVAR],
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParmDevice const& prob_parm)
{
  
}
//-------------------------------After every timestep----------------------------------//
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_poststep(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state_new,
  amrex::Array4<amrex::Real> const& state_old,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{ 
  //---------Code block that does soot foil computation-----------------------//:
  amrex::Real P_old, massfrac_old[NUM_SPECIES], T_old, rho_old, P_new, massfrac_new[NUM_SPECIES], T_new, rho_new;
  
  const amrex::Real* prob_lo = geomdata.ProbLo();
   const amrex::Real* dx = geomdata.CellSize();

   const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
   const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
   const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

  //densities and their inverses
  rho_new = state_new(i, j, k, URHO);
  rho_old = state_old(i, j, k, URHO);
  amrex::Real rhoInv_new = 1/rho_new;
  amrex::Real rhoInv_old = 1/rho_old;
    
  //massfracs
  for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac_old[n] = state_old(i, j, k, UFS + n) * rhoInv_old;
      massfrac_new[n] = state_new(i, j, k, UFS + n) * rhoInv_new;      
    }  
    
   //Temperatures 
   T_old = state_old(i, j, k, UTEMP);
   T_new = state_new(i, j, k, UTEMP);
   
   //EOS calls to do their job
   auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2P(rho_old, T_old, massfrac_old, P_old);
    eos.RTY2P(rho_new, T_new, massfrac_new, P_new);    
    
   //modify new state properly 
  if (state_old(i, j, k, USOOT) < P_new)
    {state_new(i, j, k, USOOT) = P_new;}
   else
    {state_new(i, j, k, USOOT) = state_old(i, j, k, USOOT);}
  //---------Code block that does FFT of obtained data------------------------//;
  //play around with garbage as of now to explore FFT to greater depth 
  //do it only for 1 cell otherwise it does for all and you don't know if the data is accurate
 /* if (x<=1e-1)
  {
   std::cout<<"TESTING FFT!!!";
   int N = 1000;
   amrex::Real L_1 = 6;
   amrex::Real fs = N/L_1;
   amrex::Real x_1[N], f_1[N];
   fftw_complex out_fft[N];
   
   for (int n=0; n<N; n++)
   {
	   x_1[n] = (n-1)*L_1/N;
	   f_1[n] = cos(M_PI*x_1[n]) + 2*exp(-(6*(x_1[n]-2))*(6*(x_1[n]-2)));
   }
   std::ofstream out;
   std::string file_out = "FFT.dat";
   out.open(file_out, std::ios_base::app | std::ios_base::out);
   for (int n=0; n<N; n++)
   {
	   out<<x_1[n]<<","<<f_1[n]<<"\n";
   }
   fftw_plan plan; 
   plan = fftw_plan_dft_r2c_1d(2*N, f_1, out_fft, FFTW_FORWARD);
   fftw_execute(plan);   
  out.close();
  }*/
  //---------End of FFT computation-------------------------------------------//;
}
//----------------------------------Delete instances------------------------------------//
void pc_prob_close();

using ProblemTags = EmptyProbTagStruct;
using ProblemDerives = EmptyProbDeriveStruct;

#endif

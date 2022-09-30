#ifndef _PROB_PARM_H_
#define _PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  int do_perturb = 1;
  amrex::Real P_amb = 1013250.0;
  amrex::Real T_amb = 0.0;
  amrex::Real rho_amb = 0.00116;
  amrex::Real eint_amb = 0.0;
  amrex::Real Y_CH4 = 0.028522;
  amrex::Real Y_O2 = 0.22635;
  amrex::Real Y_N2 = 0.74512;
  amrex::Real delta_half = 0.0013305;
  amrex::Real frac_rho_fluc = 1.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.0};
// soot foil
  int PostStep = 0;
  //pmf stuff
  unsigned int pmf_N = 0;
  unsigned int pmf_M = 0;
  int pmf_do_average = 0;

  amrex::Real* d_pmf_X = nullptr;
  amrex::Real* d_pmf_Y = nullptr;
};

struct ProbParmHost
{
  amrex::Vector<amrex::Real> h_pmf_X;
  amrex::Vector<amrex::Real> h_pmf_Y;
  amrex::Gpu::DeviceVector<amrex::Real> pmf_X;
  amrex::Gpu::DeviceVector<amrex::Real> pmf_Y;
  ProbParmHost() : pmf_X(0), pmf_Y(0) {}
};

#endif

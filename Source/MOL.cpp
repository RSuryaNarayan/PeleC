#include "MOL.H"
#include "Godunov.H"

void
pc_compute_hyp_mol_flux(
  const amrex::Box& cbox,
  const amrex::Array4<const amrex::Real>& q,
  const amrex::Array4<const amrex::Real>& qaux,
  const amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>
    area,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del
  /*unused*/,
  const int plm_iorder,
  const bool use_laxf_flux,
  const amrex::Array4<amrex::EBCellFlag const>& flags,
  const EBBndryGeom* ebg,
  const int /*Nebg*/,
  amrex::Real* ebflux,
  const int nebflux)
{
  const int R_RHO = 0;
  const int R_UN = 1;
  const int R_UT1 = 2;
  const int R_UT2 = 3;
  const int R_P = 4;
  const int R_ADV = 5;
  const int R_Y = R_ADV + NUM_ADV;
  const int R_AUX = R_Y + NUM_SPECIES;
  const int R_LIN = R_AUX + NUM_AUX;
  const int R_NUM = 5 + NUM_SPECIES + NUM_ADV + NUM_LIN + NUM_AUX;
  const int bc_test_val = 1;

  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    amrex::FArrayBox dq_fab(cbox, QVAR, amrex::The_Async_Arena());
    auto const& dq = dq_fab.array();
    setV(cbox, QVAR, dq, 0.0);

    // dimensional indexing
    const amrex::GpuArray<const int, 3> bdim{
      {static_cast<int>(dir == 0), static_cast<int>(dir == 1),
       static_cast<int>(dir == 2)}};
    const amrex::GpuArray<const int, 3> q_idx{
      {bdim[0] * QU + bdim[1] * QV + bdim[2] * QW,
       bdim[0] * QV + bdim[1] * QU + bdim[2] * QU,
       bdim[0] * QW + bdim[1] * QW + bdim[2] * QV}};
    const amrex::GpuArray<const int, 3> f_idx{
      {bdim[0] * UMX + bdim[1] * UMY + bdim[2] * UMZ,
       bdim[0] * UMY + bdim[1] * UMX + bdim[2] * UMX,
       bdim[0] * UMZ + bdim[1] * UMZ + bdim[2] * UMY}};

    if (plm_iorder != 1) {
      amrex::ParallelFor(
        cbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          mol_slope(i, j, k, dir, q_idx, q, qaux, dq, flags);
        });
    }
    const amrex::Box tbox = amrex::grow(cbox, dir, -1);
    const amrex::Box ebox = amrex::surroundingNodes(tbox, dir);
    amrex::ParallelFor(
      ebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
        const amrex::IntVect ivm(iv - amrex::IntVect::TheDimensionVector(dir));

        amrex::Real qtempl[R_NUM] = {0.0};
        qtempl[R_UN] =
          q(ivm, q_idx[0]) + 0.5 * ((dq(ivm, 1) - dq(ivm, 0)) / q(ivm, QRHO));
        qtempl[R_P] =
          q(ivm, QPRES) + 0.5 * (dq(ivm, 0) + dq(ivm, 1)) * qaux(ivm, QC);
        qtempl[R_UT1] = q(ivm, q_idx[1]) + 0.5 * dq(ivm, 2);
        qtempl[R_UT2] =
          AMREX_D_PICK(0.0, 0.0, q(ivm, q_idx[2]) + 0.5 * dq(ivm, 3));
        qtempl[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] =
            q(ivm, QFS + n) * q(ivm, QRHO) +
            0.5 * (dq(ivm, QFS + n) +
                   q(ivm, QFS + n) * (dq(ivm, 0) + dq(ivm, 1)) / qaux(ivm, QC));
          qtempl[R_RHO] += qtempl[R_Y + n];
        }

        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = qtempl[R_Y + n] / qtempl[R_RHO];
        }

        amrex::Real qtempr[R_NUM] = {0.0};
        qtempr[R_UN] =
          q(iv, q_idx[0]) - 0.5 * ((dq(iv, 1) - dq(iv, 0)) / q(iv, QRHO));
        qtempr[R_P] =
          q(iv, QPRES) - 0.5 * (dq(iv, 0) + dq(iv, 1)) * qaux(iv, QC);
        qtempr[R_UT1] = q(iv, q_idx[1]) - 0.5 * dq(iv, 2);
        qtempr[R_UT2] =
          AMREX_D_PICK(0.0, 0.0, q(iv, q_idx[2]) - 0.5 * dq(iv, 3));
        qtempr[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] =
            q(iv, QFS + n) * q(iv, QRHO) -
            0.5 * (dq(iv, QFS + n) +
                   q(iv, QFS + n) * (dq(iv, 0) + dq(iv, 1)) / qaux(iv, QC));
          qtempr[R_RHO] += qtempr[R_Y + n];
        }
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] = qtempr[R_Y + n] / qtempr[R_RHO];
        }

        for (int n = 0; n < NUM_ADV; n++) {
          qtempl[R_ADV + n] = q(ivm, QFA + n) + 0.5 * dq(ivm, QFA + n);
          qtempr[R_ADV + n] = q(iv, QFA + n) - 0.5 * dq(iv, QFA + n);
        }
        for (int n = 0; n < NUM_AUX; n++) {
          qtempl[R_AUX + n] = q(ivm, QFX + n) + 0.5 * dq(ivm, QFX + n);
          qtempr[R_AUX + n] = q(iv, QFX + n) - 0.5 * dq(iv, QFX + n);
        }
        for (int n = 0; n < NUM_LIN; n++) {
          qtempl[R_LIN + n] = q(ivm, QLIN + n) + 0.5 * dq(ivm, QLIN + n);
          qtempr[R_LIN + n] = q(iv, QLIN + n) - 0.5 * dq(iv, QLIN + n);
        }
        const amrex::Real cavg = 0.5 * (qaux(iv, QC) + qaux(ivm, QC));

        amrex::Real spl[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spl[n] = qtempl[R_Y + n];
        }

        amrex::Real spr[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spr[n] = qtempr[R_Y + n];
        }

        amrex::Real flux_tmp[NVAR] = {0.0};
        amrex::Real ustar = 0.0;

        if (!use_laxf_flux) {
          amrex::Real qint_iu = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0,
                      tmp4 = 0.0;
          riemann(
            qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
            qtempl[R_P], spl, qtempr[R_RHO], qtempr[R_UN], qtempr[R_UT1],
            qtempr[R_UT2], qtempr[R_P], spr, bc_test_val, cavg, ustar,
            flux_tmp[URHO], &flux_tmp[UFS], flux_tmp[f_idx[0]],
            flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
            flux_tmp[UEINT], qint_iu, tmp1, tmp2, tmp3, tmp4);
          const amrex::Real flxrho = flux_tmp[URHO];
          for (int n = 0; n < NUM_ADV; n++) {
            pc_cmpflx_passive(
              ustar, flxrho, qtempl[R_ADV + n], qtempr[R_ADV + n],
              flux_tmp[UFA + n]);
          }
          for (int n = 0; n < NUM_AUX; n++) {
            pc_cmpflx_passive(
              ustar, flxrho, qtempl[R_AUX + n], qtempr[R_AUX + n],
              flux_tmp[UFX + n]);
          }
          for (int n = 0; n < NUM_LIN; n++) {
            pc_cmpflx_passive(
              ustar, qint_iu, qtempl[R_LIN + n], qtempr[R_LIN + n],
              flux_tmp[ULIN + n]);
          }
        } else {
          amrex::Real maxeigval = 0.0;
          laxfriedrich_flux(
            qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
            qtempl[R_P], spl, qtempr[R_RHO], qtempr[R_UN], qtempr[R_UT1],
            qtempr[R_UT2], qtempr[R_P], spr, bc_test_val, cavg, ustar,
            maxeigval, flux_tmp[URHO], &flux_tmp[UFS], flux_tmp[f_idx[0]],
            flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
            flux_tmp[UEINT]);
          const amrex::Real ul = qtempl[R_UN];
          const amrex::Real ur = qtempr[R_UN];
          const amrex::Real rl = qtempl[R_RHO];
          const amrex::Real rr = qtempr[R_RHO];
          for (int n = 0; n < NUM_ADV; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, rl, rr, qtempl[R_ADV + n], qtempr[R_ADV + n], maxeigval,
              flux_tmp[UFA + n]);
          }
          for (int n = 0; n < NUM_AUX; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, rl, rr, qtempl[R_AUX + n], qtempr[R_AUX + n], maxeigval,
              flux_tmp[UFX + n]);
          }
          for (int n = 0; n < NUM_LIN; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, 1., 1., qtempl[R_LIN + n], qtempr[R_LIN + n], maxeigval,
              flux_tmp[ULIN + n]);
          }
        }
        flux_tmp[UTEMP] = 0.0;
        for (int ivar = 0; ivar < NVAR; ivar++) {
          flx[dir](iv, ivar) += flux_tmp[ivar] * area[dir](i, j, k);
        }
      });
  }

  // nextra was 3 for EB in PeleC but we are operating on a different
  // box here, so this should be zero.
  const int nextra = 0;

  const amrex::Real full_area = std::pow(del[0], AMREX_SPACEDIM - 1);
  const amrex::Box bxg = amrex::grow(cbox, nextra - 1);

  amrex::ParallelFor(nebflux, [=] AMREX_GPU_DEVICE(int L) {
    const amrex::IntVect& iv = ebg[L].iv;
    if (bxg.contains(iv)) {
      amrex::Real ebnorm[AMREX_SPACEDIM] = {AMREX_D_DECL(
        ebg[L].eb_normal[0], ebg[L].eb_normal[1], ebg[L].eb_normal[2])};
      const amrex::Real ebnorm_mag = std::sqrt(AMREX_D_TERM(
        ebnorm[0] * ebnorm[0], +ebnorm[1] * ebnorm[1], +ebnorm[2] * ebnorm[2]));
      for (amrex::Real& dir : ebnorm) {
        dir /= ebnorm_mag;
      }

      amrex::Real flux_tmp[NVAR] = {0.0};
      AMREX_D_TERM(flux_tmp[UMX] = -q(iv, QPRES) * ebnorm[0];
                   , flux_tmp[UMY] = -q(iv, QPRES) * ebnorm[1];
                   , flux_tmp[UMZ] = -q(iv, QPRES) * ebnorm[2];)

      // Copy result into ebflux vector. Being a bit chicken here and only
      // copy values where ebg % iv is within box
      for (int n = 0; n < NVAR; n++) {
        ebflux[n * nebflux + L] += flux_tmp[n] * ebg[L].eb_area * full_area;
      }
    }
  });
}

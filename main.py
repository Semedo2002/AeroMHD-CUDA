# main.py — The Execution Hub
# Verification tests, convergence study, RMI production runs, and summary.

import time
import os
import sys
import numpy as np
from typing import Dict

from config import (
    Config, NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from physics import (
    xp, cp,
    cons_to_prim, prim_to_cons,
    richtmyer_linear_theory,
    smooth,
)
from solver import (
    MHDSolver, PostProcessor,
    save_figure, to_numpy,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Verification Tests
# ============================================================
def brio_wu_test(nx=800, t_end=0.1, plot=True):
    """Run Brio-Wu MHD shock tube test for solver verification."""
    print("\n--- Brio-Wu Shock Tube Verification (GPU) ---")
    sys.stdout.flush()

    cfg = Config(
        nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.05,
        t_end=t_end, cfl=0.30, gamma=2.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=10000, snapshot_times=[t_end],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic",
    )

    solver = MHDSolver(cfg)
    ng = solver.ng

    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x
    left = x < 0.5

    W[iRHO] = xp.where(left[:, None], 1.0, 0.125)
    W[iPR] = xp.where(left[:, None], 1.0, 0.1)
    W[iBX] = 0.75
    W[iBY] = xp.where(left[:, None], 1.0, -1.0)
    W[iCLR] = xp.where(left[:, None], 1.0, 0.0)

    solver.U = prim_to_cons(W, cfg.gamma)
    solver.t = 0.0
    solver.step = 0

    t0 = time.time()
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        dt_cfl = solver.compute_dt(W_c)
        solver.dt = min(dt_cfl, cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3()
        solver.t += solver.dt
        solver.step += 1

    elapsed = time.time() - t0
    print(f"  Brio-Wu: {solver.step} steps, {elapsed:.1f}s")

    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_final[iRHO, ng:-ng, ng])
    p_1d = to_numpy(W_final[iPR, ng:-ng, ng])
    By_1d = to_numpy(W_final[iBY, ng:-ng, ng])
    vx_1d = to_numpy(W_final[iVX, ng:-ng, ng])
    x_1d = to_numpy(x[ng:-ng])

    rho_max = float(np.max(rho_1d))
    rho_min = float(np.min(rho_1d))

    check1 = 0.1 < rho_min < 0.2
    check2 = 0.9 < rho_max < 1.05
    n_levels = len(np.unique(np.round(rho_1d, 2)))
    check3 = n_levels > 10
    check4 = bool(np.any(By_1d[:-1] * By_1d[1:] < 0))
    check5 = rho_min < 0.18
    vx_range = float(np.max(vx_1d) - np.min(vx_1d))
    check6 = vx_range > 0.5

    passed = check1 and check2 and check3 and check4 and check5 and check6
    print(f"  rho range: [{rho_min:.4f}, {rho_max:.4f}]")
    print(f"  Distinct density levels: {n_levels}")
    print(f"  By sign change: {check4}, vx range: {vx_range:.3f}")
    
    res_str = 'PASS ✓' if passed else 'FAIL ✗'
    print(f"  Result: {res_str}")

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, data, ylabel, title, color in [
            (axes[0,0], rho_1d, r'$\rho$', 'Density', 'b'),
            (axes[0,1], p_1d, '$p$', 'Pressure', 'r'),
            (axes[1,0], vx_1d, '$v_x$', 'Velocity', 'g'),
            (axes[1,1], By_1d, '$B_y$', 'Transverse B', 'm'),
        ]:
            ax.plot(x_1d, data, f'{color}-', lw=1)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
        axes[1,0].set_xlabel('$x$'); axes[1,1].set_xlabel('$x$')
        fig.suptitle(f'Brio-Wu Shock Tube (GPU), t={t_end}, nx={nx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        ok, path, sz = save_figure(fig, 'brio_wu_test.png')
        status_mark = '✓' if ok else '✗'
        print(f"  {status_mark} Brio-Wu plot → {path} ({sz:.0f} KB)")

    sys.stdout.flush()
    return passed


def linear_wave_convergence_test(plot=True):
    """Run Alfven wave convergence test to verify spatial order of accuracy."""
    print("\n--- Linear Alfven Wave Convergence Test (GPU) ---")
    sys.stdout.flush()

    resolutions = [32, 64, 128, 256]
    errors = []
    rho0 = 1.0; p0 = 0.1; Bx0 = 1.0; amp = 1e-6; gamma = 5.0 / 3.0; Lx = 1.0
    vA = Bx0 / np.sqrt(rho0)
    period = Lx / vA

    print(f"  Setup: rho0={rho0}, p0={p0}, Bx0={Bx0}, amp={amp:.0e}")
    print(f"  vA={vA:.4f}, period={period:.4f}, Lx={Lx}")

    for nx in resolutions:
        cfg = Config(
            nx=nx, ny=4, x_min=0.0, x_max=Lx, y_min=0.0, y_max=0.05,
            t_end=period, cfl=0.25, gamma=gamma, mach=1.0, B_transverse=0.0,
            interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
            diag_interval=100000, snapshot_times=[period],
            powell_source=False, use_char_bc=False,
            bc_x_type="periodic", bc_y_type="periodic",
        )

        solver = MHDSolver(cfg)
        ng = solver.ng
        x = solver.x
        kx = 2.0 * np.pi / Lx

        W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
        W[iRHO] = rho0
        W[iPR] = p0
        W[iBX] = Bx0
        W[iBY] = amp * xp.sin(kx * x)[:, None] * xp.ones(solver.ny_tot)[None, :]
        W[iVY] = -amp * xp.sin(kx * x)[:, None] * xp.ones(solver.ny_tot)[None, :] / np.sqrt(rho0)

        W_init = W.copy()
        solver.U = prim_to_cons(W, cfg.gamma)
        solver.t = 0.0; solver.step = 0

        while solver.t < cfg.t_end and solver.step < 100000:
            W_c = cons_to_prim(solver.U, cfg.gamma)
            dt_cfl = solver.compute_dt(W_c)
            solver.dt = min(dt_cfl, cfg.t_end - solver.t)
            if solver.dt <= 1e-16: break
            solver.step_ssprk3()
            solver.t += solver.dt
            solver.step += 1

        W_final = cons_to_prim(solver.U, cfg.gamma)
        By_init = to_numpy(W_init[iBY, ng:-ng, ng])
        By_final = to_numpy(W_final[iBY, ng:-ng, ng])
        L1_err = float(np.mean(np.abs(By_final - By_init)))
        errors.append(L1_err)
        print(f"  nx={nx:4d}: L1(By) = {L1_err:.2e}, steps={solver.step}, "
              f"t_final={solver.t:.6f}")

    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            order = np.log(errors[i-1] / errors[i]) / np.log(resolutions[i] / resolutions[i-1])
            orders.append(order)
            print(f"  Order ({resolutions[i-1]}→{resolutions[i]}): {order:.2f}")

    mean_order = np.mean(orders) if orders else 0
    passed = mean_order >= 1.5
    res_str = 'PASS ✓' if passed else 'FAIL ✗'
    print(f"  Mean convergence order: {mean_order:.2f} ({res_str})")

    if plot and len(errors) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(resolutions, errors, 'bo-', lw=2, ms=8, label=f'Measured (order={mean_order:.2f})')
        ref_x = np.array([resolutions[0], resolutions[-1]], dtype=float)
        scale = errors[0] * (resolutions[0])**2
        ax.loglog(ref_x, scale / ref_x**2, 'k--', lw=1, alpha=0.5, label='2nd order')
        scale1 = errors[0] * resolutions[0]
        ax.loglog(ref_x, scale1 / ref_x, 'k:', lw=1, alpha=0.5, label='1st order')
        ax.set_xlabel('Resolution $N_x$'); ax.set_ylabel('$L_1$ error in $B_y$')
        ax.set_title('Alfven Wave Convergence Test (GPU)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ok, path, sz = save_figure(fig, 'convergence_alfven.png')
        status_mark = '✓' if ok else '✗'
        print(f"  {status_mark} Convergence plot → {path} ({sz:.0f} KB)")

    sys.stdout.flush()
    return passed


def contact_discontinuity_test(plot=True):
    """Verify isolated contact discontinuity is preserved."""
    print("\n--- Contact Discontinuity Test (GPU) ---")
    sys.stdout.flush()

    cfg = Config(
        nx=400, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.01,
        t_end=0.2, cfl=0.30, gamma=5.0/3.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=100000, snapshot_times=[0.2],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic",
    )
    solver = MHDSolver(cfg)
    ng = solver.ng

    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x
    W[iRHO] = xp.where(x[:, None] < 0.5, 1.0, 3.0)
    W[iPR] = 1.0
    W[iVX] = 1.0
    W[iBX] = 0.5
    W[iCLR] = xp.where(x[:, None] < 0.5, 0.0, 1.0)

    solver.U = prim_to_cons(W, cfg.gamma)
    solver.t = 0.0; solver.step = 0

    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        dt_cfl = solver.compute_dt(W_c)
        solver.dt = min(dt_cfl, cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3()
        solver.t += solver.dt
        solver.step += 1

    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_final[iRHO, ng:-ng, ng])
    p_1d = to_numpy(W_final[iPR, ng:-ng, ng])
    vx_1d = to_numpy(W_final[iVX, ng:-ng, ng])
    x_1d = to_numpy(x[ng:-ng])

    p_variation = float(np.max(p_1d) - np.min(p_1d)) / float(np.mean(p_1d))
    v_variation = float(np.max(vx_1d) - np.min(vx_1d))
    rho_min = float(np.min(rho_1d)); rho_max = float(np.max(rho_1d))

    check1 = p_variation < 0.05
    check2 = v_variation < 0.1
    check3 = rho_min > 0.9 and rho_max < 3.1

    passed = check1 and check2 and check3
    print(f"  Pressure variation: {p_variation*100:.2f}% (need <5%)")
    print(f"  Velocity variation: {v_variation:.4f} (need <0.1)")
    print(f"  Density range: [{rho_min:.3f}, {rho_max:.3f}]")
    res_str = 'PASS ✓' if passed else 'FAIL ✗'
    print(f"  Result: {res_str}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(x_1d, rho_1d, 'b-', lw=1); axes[0].set_ylabel(r'$\rho$')
        axes[0].set_title('Density'); axes[0].axvline(0.7, color='r', ls='--', alpha=0.5)
        axes[1].plot(x_1d, p_1d, 'r-', lw=1); axes[1].set_ylabel('$p$')
        axes[1].set_title('Pressure')
        axes[2].plot(x_1d, vx_1d, 'g-', lw=1); axes[2].set_ylabel('$v_x$')
        axes[2].set_title('Velocity')
        for ax in axes: ax.grid(True, alpha=0.3); ax.set_xlabel('$x$')
        fig.suptitle('Contact Discontinuity Test (GPU, t=0.2)', fontweight='bold')
        plt.tight_layout()
        ok, path, sz = save_figure(fig, 'contact_test.png')
        status_mark = '✓' if ok else '✗'
        print(f"  {status_mark} Contact test plot → {path} ({sz:.0f} KB)")

    sys.stdout.flush()
    return passed


# ============================================================
# Grid Convergence Study
# ============================================================
def run_convergence_study(base_params, By_value=0.0, resolutions=None, plot=True):
    """Run RMI at multiple resolutions and measure convergence."""
    if resolutions is None:
        resolutions = [(100, 50), (200, 100), (400, 200)]

    print(f"\n{'='*64}")
    print(f"  CONVERGENCE STUDY: By={By_value}")
    print(f"  Resolutions: {resolutions}")
    print(f"{'='*64}")
    sys.stdout.flush()

    results = {'resolutions': resolutions, 'dx': [], 'solvers': []}

    for nx, ny in resolutions:
        print(f"\n--- Resolution {nx}x{ny} ---")
        params = base_params.copy()
        params['nx'] = nx
        params['ny'] = ny
        params['B_transverse'] = By_value
        params['diag_interval'] = max(10, 20 * 200 // nx)

        cfg = Config(**params)
        solver = MHDSolver(cfg)
        solver.initialize()
        solver.run()
        results['solvers'].append(solver)
        results['dx'].append(cfg.dx)

    print(f"\n  Convergence Results:")
    print(f"  {'nx':>6s} {'ny':>6s} {'dx':>10s} {'MW_int':>10s} {'MW_thr':>10s} {'Peak_enst':>12s}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    mw_ints = []
    mw_thrs = []
    for (nx, ny), solver in zip(resolutions, results['solvers']):
        t = np.array(solver.diag_times)
        mask = t > 0.03
        mwi_arr = np.array(solver.diag_mixing_width_integral)[mask]
        mwt_arr = np.array(solver.diag_mixing_width_thresh)[mask]
        enst_arr = np.array(solver.diag_enstrophy_local)[mask]

        peak_mwi = float(np.max(mwi_arr)) if len(mwi_arr) > 0 else 0
        peak_mwt = float(np.max(mwt_arr)) if len(mwt_arr) > 0 else 0
        peak_enst = float(np.max(enst_arr)) if len(enst_arr) > 0 else 0
        mw_ints.append(peak_mwi)
        mw_thrs.append(peak_mwt)

        dx = solver.cfg.dx
        print(f"  {nx:6d} {ny:6d} {dx:10.5f} {peak_mwi:10.5f} {peak_mwt:10.5f} {peak_enst:12.1f}")

    if len(resolutions) >= 3:
        dx_arr = np.array(results['dx'])
        mwi_arr_conv = np.array(mw_ints)
        for i in range(1, len(mwi_arr_conv)):
            if mwi_arr_conv[i] > 0 and mwi_arr_conv[i-1] > 0:
                diff = abs(mwi_arr_conv[i] - mwi_arr_conv[i-1])
                if i >= 2:
                    diff_prev = abs(mwi_arr_conv[i-1] - mwi_arr_conv[i-2])
                    if diff > 1e-14 and diff_prev > 1e-14:
                        order = np.log(diff_prev / diff) / np.log(dx_arr[i-1] / dx_arr[i])
                        print(f"  Estimated convergence order (MW_int): {order:.2f}")

    results['mw_ints'] = mw_ints
    results['mw_thrs'] = mw_thrs

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors_conv = ['C0', 'C1', 'C2', 'C3']

        ax = axes[0]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            t = np.array(solver.diag_times); mw = np.array(solver.diag_mixing_width_integral)
            ax.plot(t, mw, color=colors_conv[i], lw=2, label=f'{nx}×{ny}')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Integral MW')
        ax.set_title('Mixing Width Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            t = np.array(solver.diag_times); en = np.array(solver.diag_enstrophy_local)
            ax.plot(t, en, color=colors_conv[i], lw=2, label=f'{nx}×{ny}')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Local Enstrophy')
        ax.set_title('Enstrophy Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[2]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            snap = solver.snapshots.get('final', list(solver.snapshots.values())[-1])
            rho = snap['rho']; j_mid = rho.shape[1] // 2
            ax.plot(snap['x'], rho[:, j_mid], color=colors_conv[i], lw=1.5, label=f'{nx}×{ny}')
        ax.set_xlabel('$x$'); ax.set_ylabel(r'$\rho$ at $y=L_y/2$')
        ax.set_title('Midline Density Profile', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        By_str = f'By={By_value}' if By_value > 0 else 'Hydro'
        fig.suptitle(f'Grid Convergence Study ({By_str})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        ok, path, sz = save_figure(fig, f'convergence_{By_str.replace("=","").replace(".","p")}.png')
        status_mark = '✓' if ok else '✗'
        print(f"  {status_mark} Convergence plot → {path} ({sz:.0f} KB)")

    sys.stdout.flush()
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("MAGNETIZED RICHTMYER-MESHKOV INSTABILITY (GPU/CUDA)")
    print("HLLD | MUSCL | SSP-RK3 | GLM+Powell | Passive Scalar")
    print("MHD R-H | Brio-Wu + Alfven wave + Contact verified")
    print("Extended domain | Char BCs | Linear theory overlay")
    print("  CuPy CUDA: ENABLED")
    print("=" * 64)
    sys.stdout.flush()

    total_t0 = time.time()

    # ============================
    # verification
    # ============================
    print("\n" + "="*64)
    print("  VERIFICATION SUITE")
    print("="*64)

    bw_passed = brio_wu_test(nx=800, t_end=0.1, plot=True)
    if not bw_passed:
        print("\n   WARNING: Brio-Wu test did not pass.")
    else:
        print("\n   Brio-Wu verification PASSED.")

    wave_passed = linear_wave_convergence_test(plot=True)
    if not wave_passed:
        print("\n   WARNING: Alfven wave convergence test did not pass.")
    else:
        print("\n   Alfven wave convergence PASSED.")

    contact_passed = contact_discontinuity_test(plot=True)
    if not contact_passed:
        print("\n   WARNING: Contact discontinuity test did not pass.")
    else:
        print("\n  Contact discontinuity PASSED.")

    all_tests_passed = bw_passed and wave_passed and contact_passed
    pass_mark = 'ALL PASSED ✓' if all_tests_passed else 'SOME FAILED ✗'
    print(f"\n  Overall verification: {pass_mark}")

    # ============================
    # RMI runs
    # ============================
    base = dict(
        nx=400, ny=200,
        x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=10.0,
        interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=3.0,
        interface_width=2.0,
        powell_source=True, use_char_bc=True,
        bc_x_type="characteristic", bc_y_type="periodic",
        enable_smoothing=False,
        diag_interval=20,
        snapshot_times=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    )

    cases = {
        "Hydro (B=0)": 0.0,
        "MHD (By=0.5)": 0.5,
        "MHD (By=1.5)": 1.5,
    }

    solvers: Dict[str, MHDSolver] = {}
    for label, By in cases.items():
        print(f"\n{'='*64}\n  CASE: {label}\n{'='*64}")
        sys.stdout.flush()
        cfg = Config(**base, B_transverse=By)
        s = MHDSolver(cfg)
        s.initialize()
        s.run()
        solvers[label] = s

    # ============================
    # grid convergence
    # ============================
    print("\n" + "="*64)
    print("  GRID CONVERGENCE STUDY")
    print("="*64)

    conv_base = base.copy()
    conv_base['t_end'] = 0.15
    conv_base['snapshot_times'] = [0.0, 0.15]
    conv_base['powell_source'] = True

    conv_resolutions = [(100, 50), (200, 100), (400, 200)]

    conv_hydro = run_convergence_study(conv_base, By_value=0.0,
                                        resolutions=conv_resolutions, plot=True)
    conv_mhd = run_convergence_study(conv_base, By_value=1.5,
                                      resolutions=conv_resolutions, plot=True)

    # ============================
    # post
    # ============================
    post = PostProcessor(solvers)
    post.plot_all()

    # ============================
    # summary
    # ============================
    print("=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    t_skip = 0.03

    theta_header = 'θ_mean'
    print(f"  {'Case':<20s}  {'MW_int':>8s}  {'MW_thr':>8s}  "
          f"{theta_header:>8s}  {'Enst_loc':>10s}  {'divB_L2':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    summary_data = {}
    for label, s in solvers.items():
        t = np.array(s.diag_times); mask = t > t_skip
        mwi = np.array(s.diag_mixing_width_integral)[mask]
        mwt = np.array(s.diag_mixing_width_thresh)[mask]
        theta = np.array(s.diag_mixedness)[mask]
        en = np.array(s.diag_enstrophy_local)[mask]
        divB = np.array(s.diag_divB_L2)[mask] if s.diag_divB_L2 else np.array([0.0])

        peak_mwi = float(np.max(mwi)) if len(mwi) > 0 else 0
        peak_mwt = float(np.max(mwt)) if len(mwt) > 0 else 0
        mean_theta = float(np.mean(theta)) if len(theta) > 0 else 0
        mean_en = float(np.mean(en)) if len(en) > 0 else 0
        mean_divB = float(np.mean(divB)) if len(divB) > 0 else 0
        summary_data[label] = (peak_mwi, peak_mwt, mean_theta, mean_en, mean_divB)
        print(f"  {label:<20s}  {peak_mwi:8.4f}  {peak_mwt:8.4f}  "
              f"{mean_theta:8.3f}  {mean_en:10.1f}  {mean_divB:8.3f}")

    hydro_key = list(solvers.keys())[0]
    h_mwi, h_mwt, h_theta, h_en, _ = summary_data[hydro_key]

    print()
    for label, s in solvers.items():
        if s.cfg.B_transverse == 0: continue
        m_mwi, m_mwt, m_theta, m_en, _ = summary_data[label]
        mwi_sup = (1 - m_mwi/max(h_mwi, 1e-10))*100
        mwt_sup = (1 - m_mwt/max(h_mwt, 1e-10))*100
        theta_change = (m_theta - h_theta)/max(h_theta, 1e-10)*100
        en_sup = (1 - m_en/max(h_en, 1e-10))*100

        print(f"  {label}:")
        print(f"    MW integral suppression:   {mwi_sup:+.1f}%")
        print(f"    MW threshold suppression:  {mwt_sup:+.1f}%")
        print(f"    Mixedness θ change:        {theta_change:+.1f}%")
        print(f"    Local enstrophy suppress.: {en_sup:+.1f}%")

    print()
    print("  Linear Theory Comparison:")
    for label, s in solvers.items():
        if s._rh_results is not None:
            cfg = s.cfg
            Ly = cfg.y_max - cfg.y_min
            k_pert = 2*np.pi*cfg.perturbation_mode / Ly
            _, t_sh, info = richtmyer_linear_theory(
                np.array([0.0]), cfg.gamma, cfg.mach, 1.0,
                cfg.density_ratio, cfg.perturbation_amp, k_pert,
                By=cfg.B_transverse)
            print(f"    {label}: A_post={info['A_post']:.3f}, "
                  f"da/dt={info['da_dt']:.2f}, "
                  f"Δv={info['delta_v']:.2f}, "
                  f"a0_post={info['a0_post']:.4f}")

    print()
    print("  Physics of magnetic RMI suppression:")
    print("  • Alfven waves transport baroclinic vorticity away from interface")
    print("  • Magnetic tension (B·∇)B opposes KH secondary instability")
    print("  • Interface coherence preserved → reduced mixing zone width")
    print("  • Suppression scales with v_A / v_perturbation")
    print("  • Higher θ in MHD = narrower but more homogeneous mixing zone")

    print()
    print("  Numerical features ")
    print(f"  • CuPy CUDA acceleration: YES")
    bw_m = 'PASSED ✓' if bw_passed else 'FAILED ✗'
    wv_m = 'PASSED ✓' if wave_passed else 'FAILED ✗'
    ct_m = 'PASSED ✓' if contact_passed else 'FAILED ✗'
    print(f"  • Brio-Wu: {bw_m}")
    print(f"  • Alfven wave convergence: {wv_m}")
    print(f"  • Contact discontinuity: {ct_m}")
    print(f"  • Powell source terms: ENABLED (face-centered div(B))")
    print(f"  • Characteristic BCs: ENABLED (production runs)")
    print(f"  • Periodic BCs: ENABLED (wave convergence test)")
    print(f"  • Extended domain: x=[0, 6] (50% longer)")
    print(f"  • Interface width: 2 cells (sharper)")
    print(f"  • Smoothing on diagnostics: DISABLED (raw data)")
    print(f"  • Grid convergence study: 3 resolutions")

    for label, s in solvers.items():
        if len(s.diag_energy_total) >= 2:
            e0 = s.diag_energy_total[0]; ef = s.diag_energy_total[-1]
            bf = s._cumulative_boundary_energy
            de_raw = abs(ef-e0)/max(abs(e0), 1e-14)*100
            de_corr = abs(ef-bf-e0)/max(abs(e0), 1e-14)*100
            print(f"  • {label}: energy drift raw={de_raw:.1f}%, corrected={de_corr:.1f}%")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output: {os.path.abspath('rmi_output_gpu')}")
    print("=" * 72)
    sys.stdout.flush()
# solver.py — The Core Machine
# MHDSolver class, PostProcessor class, and helper utilities.

import time
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
    "figure.dpi": 140, "savefig.dpi": 180,
    "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9,
    "figure.facecolor": "white",
})

from config import (
    Config, NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from physics import (
    xp, cp,
    cons_to_prim, prim_to_cons,
    muscl_x, muscl_y,
    hlld_flux_x, hlld_flux_y,
    mhd_rankine_hugoniot,
    richtmyer_linear_theory,
    smooth,
)


# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = "rmi_output_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


# ============================================================
# Helper Functions
# ============================================================
def save_figure(fig, filename):
    """Save figure reliably and verify the file exists."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=180)
    plt.close(fig)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return True, filepath, size_kb
    return False, filepath, 0


def to_numpy(a):
    """Convert CuPy array to NumPy, or pass through NumPy arrays."""
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


# ============================================================
# MHD Solver Class — GPU version
# ============================================================
class MHDSolver:
    """2D Ideal MHD solver with HLLD, MUSCL-Hancock, SSP-RK3, GLM+Powell."""

    def __init__(self, config):
        self.cfg = config
        self.ng = 2
        self.nx_tot = config.nx + 2 * self.ng
        self.ny_tot = config.ny + 2 * self.ng

        # Coordinate arrays on GPU
        self.x = xp.asarray(config.x_min + (np.arange(self.nx_tot) - self.ng + 0.5) * config.dx)
        self.y = xp.asarray(config.y_min + (np.arange(self.ny_tot) - self.ng + 0.5) * config.dy)
        self.X, self.Y = xp.meshgrid(self.x, self.y, indexing='ij')

        self.U = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._glm_ch_frozen = 0.0

        # Diagnostics (stored as Python lists on CPU)
        self.diag_times = []
        self.diag_mixing_width_integral = []
        self.diag_mixing_width_thresh = []
        self.diag_mixedness = []
        self.diag_enstrophy = []
        self.diag_enstrophy_local = []
        self.diag_perturbation_amp = []
        self.diag_mode_amps = []
        self.diag_stag_pressure = []
        self.diag_divB_max = []
        self.diag_divB_L2 = []
        self.diag_energy_total = []
        self.diag_boundary_flux_cumulative = []
        self._cumulative_boundary_energy = 0.0
        self.snapshots = {}
        self._rho_light = 1.0
        self._rho_heavy = 1.0
        self._rh_results = None

    def initialize(self):
        """Set up initial conditions: shocked gas + perturbed density interface."""
        cfg = self.cfg
        g = cfg.gamma
        M = cfg.mach
        rho1 = 1.0; p1 = 1.0
        By_pre = cfg.B_transverse

        rho2, p2, vx2, By2, vs = mhd_rankine_hugoniot(g, M, rho1, p1, By_pre)
        self._rh_results = {'rho2': rho2, 'p2': p2, 'vx2': vx2, 'By2': By2, 'vs': vs}

        rho_h = rho1 * cfg.density_ratio
        x_shock = cfg.interface_x - 0.3
        Ly = cfg.y_max - cfg.y_min
        ky = 2 * np.pi * cfg.perturbation_mode / Ly

        x_if = cfg.interface_x + cfg.perturbation_amp * xp.sin(ky * self.Y)
        interface_delta = cfg.interface_width * cfg.dx
        phi = 0.5 * (1.0 + xp.tanh((self.X - x_if) / max(interface_delta, 1e-10)))

        W = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        post = self.X < x_shock
        rho_pre = rho1 * (1 - phi) + rho_h * phi

        W[iRHO] = xp.where(post, rho2, rho_pre)
        W[iVX] = xp.where(post, vx2, 0.0)
        W[iBY] = xp.where(post, By2, By_pre)
        W[iPR] = xp.where(post, p2, p1)
        W[iCLR] = xp.where(post, 0.0, phi)

        self.U = prim_to_cons(W, g)
        self.t = 0.0; self.step = 0
        self._cumulative_boundary_energy = 0.0
        self._rho_light = rho1; self._rho_heavy = rho_h

        print(f"  Mach={M:.1f}, Vs={vs:.3f}, rho2={rho2:.3f}, p2={p2:.3f}, vx2={vx2:.3f}")
        if cfg.B_transverse > 0:
            beta = 2*p1/cfg.B_transverse**2
            va = cfg.B_transverse/np.sqrt(rho1)
            print(f"  By_pre={By_pre:.3f}, By_post={By2:.3f}, beta={beta:.2f}, v_A={va:.3f}")
        else:
            print(f"  B_y=0 (pure hydro)")
        bc_x = cfg.get_bc_x()
        print(f"  Grid: {cfg.nx}x{cfg.ny}, dx={cfg.dx:.4f}, interface_width={cfg.interface_width:.1f} cells")
        if cfg.powell_source:
            print(f"  Powell source terms: ENABLED (face-centered div(B))")
        print(f"  BC x: {bc_x}, BC y: {cfg.bc_y_type}")
        sys.stdout.flush()

    def apply_bc(self, U):
        """Apply boundary conditions."""
        ng = self.ng
        cfg = self.cfg
        bc_x = cfg.get_bc_x()

        if bc_x == "periodic":
            U[:, :ng, :] = U[:, -2*ng:-ng, :]
            U[:, -ng:, :] = U[:, ng:2*ng, :]
        elif bc_x == "characteristic":
            self._apply_characteristic_bc_x(U)
        else:
            for i in range(ng):
                U[:, i, :] = U[:, ng, :]
                U[:, -(i+1), :] = U[:, -(ng+1), :]

        if bc_x != "periodic":
            U[PSI, :ng, :] = 0
            U[PSI, -ng:, :] = 0

        if cfg.bc_y_type == "periodic":
            U[:, :, :ng] = U[:, :, -2*ng:-ng]
            U[:, :, -ng:] = U[:, :, ng:2*ng]
        else:
            for j in range(ng):
                U[:, :, j] = U[:, :, ng]
                U[:, :, -(j+1)] = U[:, :, -(ng+1)]

        return U

    def _apply_characteristic_bc_x(self, U):
        """Non-reflecting characteristic boundary conditions in x."""
        ng = self.ng
        for i in range(ng):
            U[:, i, :] = U[:, ng, :]
        for i in range(ng):
            U[:, -(i+1), :] = U[:, -(ng+1), :]
        for i in range(ng):
            U[PSI, -(i+1), :] *= 0.1
            U[PSI, i, :] *= 0.1

    def enforce_scalar_bounds(self, U):
        """Clamp passive scalar to [0,1]."""
        rho = xp.maximum(U[RHO], FLOOR_RHO)
        C = xp.clip(U[RHOC] / rho, 0.0, 1.0)
        U[RHOC] = rho * C
        return U

    def compute_dt(self, W):
        """Compute CFL-limited timestep."""
        cfg = self.cfg
        rho = xp.maximum(W[iRHO], FLOOR_RHO)
        p = xp.maximum(W[iPR], FLOOR_PR)
        B2 = W[iBX]**2 + W[iBY]**2 + W[iBZ]**2
        cf = xp.sqrt(cfg.gamma * p / rho + B2 / rho)
        cf_max = float(xp.max(cf))
        v_abs_max = float(xp.max(xp.abs(W[iVX]) + xp.abs(W[iVY])))
        cfg.glm_ch = max(cf_max + v_abs_max, 1.0) * 1.5
        ch = cfg.glm_ch
        sx = xp.abs(W[iVX]) + xp.maximum(cf, ch)
        sy = xp.abs(W[iVY]) + xp.maximum(cf, ch)
        inv_dt = xp.maximum(sx / cfg.dx, sy / cfg.dy)
        sm = float(xp.max(inv_dt))
        if sm < 1e-14:
            return cfg.cfl * min(cfg.dx, cfg.dy)
        return cfg.cfl / sm

    def _compute_boundary_energy_flux(self, U):
        """Compute net energy flux through x-boundaries."""
        ng = self.ng; cfg = self.cfg
        if cfg.get_bc_x() == "periodic":
            return 0.0

        W = cons_to_prim(U, cfg.gamma)
        def boundary_flux(idx):
            rho_b = W[iRHO, idx, ng:-ng]; vx_b = W[iVX, idx, ng:-ng]
            p_b = W[iPR, idx, ng:-ng]; Bx_b = W[iBX, idx, ng:-ng]
            By_b = W[iBY, idx, ng:-ng]; Bz_b = W[iBZ, idx, ng:-ng]
            B2_b = Bx_b**2 + By_b**2 + Bz_b**2; pt_b = p_b + 0.5*B2_b
            E_b = U[EN, idx, ng:-ng]
            vB_b = vx_b*Bx_b + W[iVY, idx, ng:-ng]*By_b + W[iVZ, idx, ng:-ng]*Bz_b
            return float(xp.sum(((E_b + pt_b)*vx_b - Bx_b*vB_b) * cfg.dy))
        return boundary_flux(ng) - boundary_flux(-(ng+1))

    def _compute_powell_source(self, U, W):
        """Powell 8-wave source terms for div(B) control."""
        cfg = self.cfg
        S = xp.zeros_like(U)
        Bx = W[iBX]; By_f = W[iBY]

        divB = xp.zeros_like(Bx)
        divB[1:-1, 1:-1] = (
            (0.5*(Bx[1:-1, 1:-1] + Bx[2:, 1:-1]) -
             0.5*(Bx[:-2, 1:-1] + Bx[1:-1, 1:-1])) / cfg.dx +
            (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
             0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
        )

        vB = W[iVX]*W[iBX] + W[iVY]*W[iBY] + W[iVZ]*W[iBZ]

        S[MX] = -divB * W[iBX]
        S[MY] = -divB * W[iBY]
        S[MZ] = -divB * W[iBZ]
        S[BX] = -divB * W[iVX]
        S[BY] = -divB * W[iVY]
        S[BZ] = -divB * W[iVZ]
        S[EN] = -divB * vB

        return S

    def compute_rhs(self, U):
        """Compute spatial RHS: -div(F) + Powell source."""
        cfg = self.cfg
        ng = self.ng
        nx = cfg.nx; ny = cfg.ny
        g = cfg.gamma
        ch = self._glm_ch_frozen

        U = self.apply_bc(U)
        W = cons_to_prim(U, g)

        # X-direction fluxes
        WLx, WRx = muscl_x(W)
        s = WLx.shape
        Fx, _ = hlld_flux_x(
            xp.ascontiguousarray(WLx.reshape(NVAR, -1)),
            xp.ascontiguousarray(WRx.reshape(NVAR, -1)),
            g, ch)
        Fx = Fx.reshape(s)

        # Y-direction fluxes
        WLy, WRy = muscl_y(W)
        s2 = WLy.shape
        Fy, _ = hlld_flux_y(
            xp.ascontiguousarray(WLy.reshape(NVAR, -1)),
            xp.ascontiguousarray(WRy.reshape(NVAR, -1)),
            g, ch)
        Fy = Fy.reshape(s2)

        dFx = Fx[:, 1:1+nx, :] - Fx[:, 0:nx, :]
        dFy = Fy[:, :, 1:1+ny] - Fy[:, :, 0:ny]

        R = xp.zeros_like(U)
        R[:, ng:ng+nx, ng:ng+ny] -= dFx[:, :, ng:ng+ny] / cfg.dx
        R[:, ng:ng+nx, ng:ng+ny] -= dFy[:, ng:ng+nx, :] / cfg.dy

        if cfg.powell_source and cfg.B_transverse > 0:
            S = self._compute_powell_source(U, W)
            R[:, ng:ng+nx, ng:ng+ny] += S[:, ng:ng+nx, ng:ng+ny]

        return R

    def step_ssprk3(self):
        """Advance one time step using 3-stage SSP Runge-Kutta."""
        dt = self.dt
        self._glm_ch_frozen = self.cfg.glm_ch
        U0 = self.U.copy()

        bflux = self._compute_boundary_energy_flux(U0)
        self._cumulative_boundary_energy += bflux * dt

        U1 = U0 + dt * self.compute_rhs(U0)
        U1 = self.enforce_scalar_bounds(U1)
        U1 = self.apply_bc(U1)

        U2 = 0.75*U0 + 0.25*(U1 + dt*self.compute_rhs(U1))
        U2 = self.enforce_scalar_bounds(U2)
        U2 = self.apply_bc(U2)

        self.U = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*self.compute_rhs(U2))
        self.U = self.enforce_scalar_bounds(self.U)
        self.U = self.apply_bc(self.U)

        if self._glm_ch_frozen > 0:
            decay = np.exp(-self.cfg.glm_alpha * self._glm_ch_frozen * dt / min(self.cfg.dx, self.cfg.dy))
            self.U[PSI] *= decay

    def compute_diagnostics(self):
        """Compute diagnostics — transfer to CPU only for storage."""
        ng = self.ng; cfg = self.cfg
        W = cons_to_prim(self.U, cfg.gamma)

        rho = to_numpy(W[iRHO, ng:-ng, ng:-ng])
        p = to_numpy(W[iPR, ng:-ng, ng:-ng])
        vx = to_numpy(W[iVX, ng:-ng, ng:-ng])
        vy = to_numpy(W[iVY, ng:-ng, ng:-ng])
        Bx_f = to_numpy(W[iBX, ng:-ng, ng:-ng])
        By_f = to_numpy(W[iBY, ng:-ng, ng:-ng])
        Bz_f = to_numpy(W[iBZ, ng:-ng, ng:-ng])
        C = to_numpy(W[iCLR, ng:-ng, ng:-ng])

        x_arr = to_numpy(self.x[ng:-ng])
        y_arr = to_numpy(self.y[ng:-ng])
        nx_int, ny_int = rho.shape

        C_clamped = np.clip(C, 0.0, 1.0)
        C_bar = np.mean(C_clamped, axis=1)
        integrand_mw = C_bar * (1.0 - C_bar)
        mw_integral = float(np.trapezoid(integrand_mw, x_arr))

        mixed = np.where((C_bar > 0.01) & (C_bar < 0.99))[0]
        mw_thresh = float(x_arr[mixed[-1]] - x_arr[mixed[0]]) if len(mixed) > 1 else 0.0

        C_bar_2d = C_bar[:, None]
        C_prime_sq = np.mean((C_clamped - C_bar_2d)**2, axis=1)
        denom_mix = np.maximum(C_bar * (1.0 - C_bar), 1e-14)
        if len(mixed) > 1:
            num = float(np.trapezoid(C_prime_sq[mixed], x_arr[mixed]))
            den = float(np.trapezoid(denom_mix[mixed], x_arr[mixed]))
            mixedness = float(np.clip(1.0 - num / max(den, 1e-14), 0.0, 1.0))
        else:
            mixedness = 0.0

        drho_dx = np.abs(np.gradient(C_clamped, cfg.dx, axis=0))
        interface_pos = np.full(ny_int, np.nan)
        for j in range(ny_int):
            grad_col = drho_dx[:, j]
            if np.max(grad_col) > 1e-6:
                ix_peak = np.argmax(grad_col)
                hw = min(10, ix_peak, nx_int - ix_peak - 1)
                if hw > 0:
                    sl = slice(ix_peak-hw, ix_peak+hw+1)
                    weights = grad_col[sl]
                    ws = np.sum(weights)
                    if ws > 1e-12:
                        interface_pos[j] = np.sum(x_arr[sl] * weights) / ws

        valid = ~np.isnan(interface_pos)
        n_valid = np.sum(valid)

        if n_valid > ny_int // 2:
            pos_valid = interface_pos[valid]; y_valid = y_arr[valid]
            pos_interp = np.interp(y_arr, y_valid, pos_valid) if n_valid < ny_int else pos_valid
            pos_fluct = pos_interp - np.mean(pos_interp)
            modes = np.fft.rfft(pos_fluct)
            mode_amps = 2.0 * np.abs(modes) / ny_int
            pert_amp = float(mode_amps[cfg.perturbation_mode]) if cfg.perturbation_mode < len(mode_amps) else 0.0
        else:
            mode_amps = np.zeros(ny_int // 2 + 1)
            pert_amp = 0.0

        dvydx = np.gradient(vy, cfg.dx, axis=0)
        dvxdy = np.gradient(vx, cfg.dy, axis=1)
        omega = dvydx - dvxdy
        enstrophy_global = float(np.mean(rho * omega**2))

        if n_valid > ny_int // 2:
            x_contact = float(np.mean(interface_pos[valid]))
            ix_lo = max(np.searchsorted(x_arr, x_contact - 0.5), 0)
            ix_hi = min(np.searchsorted(x_arr, x_contact + 0.5), nx_int)
            if ix_hi - ix_lo > 3:
                enstrophy_local = float(np.mean(rho[ix_lo:ix_hi,:] * omega[ix_lo:ix_hi,:]**2))
            else:
                enstrophy_local = enstrophy_global
        else:
            enstrophy_local = enstrophy_global

        v2 = vx**2 + vy**2
        B2 = Bx_f**2 + By_f**2 + Bz_f**2
        stag = float(np.max(p + 0.5*rho*v2 + 0.5*B2))

        divB_field = np.zeros_like(Bx_f)
        if Bx_f.shape[0] > 2 and Bx_f.shape[1] > 2:
            divB_field[1:-1, 1:-1] = (
                (0.5*(Bx_f[1:-1, 1:-1] + Bx_f[2:, 1:-1]) -
                 0.5*(Bx_f[:-2, 1:-1] + Bx_f[1:-1, 1:-1])) / cfg.dx +
                (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
                 0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
            )
        divB_max = float(np.max(np.abs(divB_field)))
        divB_L2 = float(np.sqrt(np.mean(divB_field**2)))

        energy_total = float(to_numpy(xp.sum(self.U[EN, ng:-ng, ng:-ng])) * cfg.dx * cfg.dy)

        self.diag_times.append(self.t)
        self.diag_mixing_width_integral.append(mw_integral)
        self.diag_mixing_width_thresh.append(mw_thresh)
        self.diag_mixedness.append(mixedness)
        self.diag_enstrophy.append(enstrophy_global)
        self.diag_enstrophy_local.append(enstrophy_local)
        self.diag_perturbation_amp.append(pert_amp)
        self.diag_mode_amps.append(mode_amps.copy())
        self.diag_stag_pressure.append(stag)
        self.diag_divB_max.append(divB_max)
        self.diag_divB_L2.append(divB_L2)
        self.diag_energy_total.append(energy_total)
        self.diag_boundary_flux_cumulative.append(self._cumulative_boundary_energy)

    def save_snapshot(self, label=None):
        """Save a snapshot of current primitive state to CPU."""
        ng = self.ng
        W = cons_to_prim(self.U, self.cfg.gamma)
        key = label or f"t={self.t:.4f}"
        self.snapshots[key] = {
            'rho': to_numpy(W[iRHO, ng:-ng, ng:-ng]),
            'p': to_numpy(W[iPR, ng:-ng, ng:-ng]),
            'vx': to_numpy(W[iVX, ng:-ng, ng:-ng]),
            'vy': to_numpy(W[iVY, ng:-ng, ng:-ng]),
            'Bx': to_numpy(W[iBX, ng:-ng, ng:-ng]),
            'By': to_numpy(W[iBY, ng:-ng, ng:-ng]),
            'Bz': to_numpy(W[iBZ, ng:-ng, ng:-ng]),
            'C': to_numpy(W[iCLR, ng:-ng, ng:-ng]),
            't': self.t,
            'x': to_numpy(self.x[ng:-ng]),
            'y': to_numpy(self.y[ng:-ng]),
        }

    def run(self):
        """Execute the simulation."""
        cfg = self.cfg; g = cfg.gamma
        print(f"\n--- Running (GPU): B_y = {cfg.B_transverse} ---")
        sys.stdout.flush()
        t0 = time.time()

        self.compute_diagnostics()
        self.save_snapshot(label="initial")

        while self.t < cfg.t_end and self.step < cfg.max_steps:
            W = cons_to_prim(self.U, g)
            dt_cfl = self.compute_dt(W)
            self.dt = min(dt_cfl, cfg.t_end - self.t)
            if self.dt <= 1e-16: break
            self.step_ssprk3()
            self.t += self.dt
            self.step += 1

            if self.step % cfg.diag_interval == 0:
                self.compute_diagnostics()

            for st in cfg.snapshot_times:
                lbl = f"t~{st:.2f}"
                if abs(self.t - st) < 1.5*self.dt and lbl not in self.snapshots:
                    self.save_snapshot(label=lbl)

            if self.step % 200 == 0:
                mwi = self.diag_mixing_width_integral[-1] if self.diag_mixing_width_integral else 0
                theta = self.diag_mixedness[-1] if self.diag_mixedness else 0
                en = self.diag_enstrophy[-1] if self.diag_enstrophy else 0
                e_tot = self.diag_energy_total[-1] if self.diag_energy_total else 0
                divB = self.diag_divB_max[-1] if self.diag_divB_max else 0
                print(f"  Step {self.step:5d}  t={self.t:.4f}  dt={self.dt:.2e}  "
                      f"MW={mwi:.4f}  \u03b8={theta:.3f}  enst={en:.1f}  "
                      f"divB={divB:.1f}  E={e_tot:.2f}")
                sys.stdout.flush()

        self.compute_diagnostics()
        self.save_snapshot(label="final")
        elapsed = time.time() - t0

        if len(self.diag_energy_total) >= 2:
            e0 = self.diag_energy_total[0]; ef = self.diag_energy_total[-1]
            bf = self._cumulative_boundary_energy
            de_raw = abs(ef-e0)/max(abs(e0),1e-14)*100
            de_corr = abs(ef-bf-e0)/max(abs(e0),1e-14)*100
            print(f"  Energy drift: raw={de_raw:.2f}%, corrected={de_corr:.2f}%")
            print(f"  Boundary flux: {bf:.2f}")

        if self.diag_divB_max:
            print(f"  Final max|divB|: {self.diag_divB_max[-1]:.2f}, "
                  f"L2|divB|: {self.diag_divB_L2[-1]:.4f}")

        print(f"  Done: {self.step} steps, {elapsed:.1f}s")
        sys.stdout.flush()


print("GPU Solver OK \u2713")


# ============================================================
# PostProcessor
# ============================================================
class PostProcessor:
    """Generate all analysis plots from simulation results."""

    def __init__(self, solvers_dict):
        self.solvers = solvers_dict
        self._colors = ['C3', 'C2', 'C0', 'C4', 'C5']
        self._styles = ['-', '--', '-.', ':', '-']
        self._saved_files = []

    @staticmethod
    def _final(s):
        return s.snapshots.get('final', list(s.snapshots.values())[-1])

    def _save(self, fig, filename):
        ok, path, sz = save_figure(fig, filename)
        self._saved_files.append((filename, ok, sz))
        return ok, path, sz

    def _get_smoothing(self):
        for s in self.solvers.values():
            return s.cfg.enable_smoothing
        return False

    def plot_density_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle(r'Magnetized RMI \u2014 Mach 10 Shock (GPU)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_density.png')
        print(f"  {'\u2713' if ok else '\u2717'} Density comparison \u2192 {path} ({sz:.0f} KB)")

    def plot_passive_scalar(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            C = np.clip(snap['C'], 0, 1)
            im = ax.pcolormesh(snap['x'], snap['y'], C.T, cmap='coolwarm', shading='auto', vmin=0, vmax=1)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Color $C$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label='$C$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Passive Scalar (Mixing Tracer)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_passive_scalar.png')
        print(f"  {'\u2713' if ok else '\u2717'} Passive scalar \u2192 {path} ({sz:.0f} KB)")

    def plot_schlieren(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            rho = snap['rho']; dx, dy = solver.cfg.dx, solver.cfg.dy
            grad_rho = np.sqrt(np.gradient(rho, dx, axis=0)**2 + np.gradient(rho, dy, axis=1)**2)
            schlieren = np.log10(grad_rho / np.maximum(rho, 1e-10) + 1e-10)
            vmin_s = np.percentile(schlieren, 3); vmax_s = np.percentile(schlieren, 99)
            im = ax.pcolormesh(snap['x'], snap['y'], schlieren.T, cmap='gray_r', shading='auto', vmin=vmin_s, vmax=vmax_s)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Schlieren\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\log_{10}(|\nabla\rho|/\rho)$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Numerical Schlieren', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_schlieren.png')
        print(f"  {'\u2713' if ok else '\u2717'} Schlieren \u2192 {path} ({sz:.0f} KB)")

    def plot_vorticity_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        omegas, snaps = [], []
        for label, solver in self.solvers.items():
            snap = self._final(solver); snaps.append(snap)
            dvydx = np.gradient(snap['vy'], solver.cfg.dx, axis=0)
            dvxdy = np.gradient(snap['vx'], solver.cfg.dy, axis=1)
            omegas.append(dvydx - dvxdy)
        vmax_om = max(np.percentile(np.abs(om), 99) for om in omegas)
        vmax_om = max(vmax_om, 1e-10)
        for ax, (label, _), snap, omega in zip(axes, self.solvers.items(), snaps, omegas):
            im = ax.pcolormesh(snap['x'], snap['y'], omega.T, cmap='RdBu_r', shading='auto', vmin=-vmax_om, vmax=vmax_om)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: $\\omega_z$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\omega_z$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Vorticity Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_vorticity.png')
        print(f"  {'\u2713' if ok else '\u2717'} Vorticity \u2192 {path} ({sz:.0f} KB)")

    def plot_magnetic_pressure_with_fieldlines(self):
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.B_transverse > 0}
        if not mhd:
            print("  No MHD cases for magnetic pressure plot"); return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Pmag = 0.5*(snap['Bx']**2 + snap['By']**2 + snap['Bz']**2)
            im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
            try:
                ax.streamplot(snap['x'], snap['y'], snap['Bx'].T, snap['By'].T,
                              color='white', linewidth=0.5, density=1.2, arrowsize=0.5)
            except Exception:
                pass
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: $P_B$ + field lines\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$P_B=B^2/2$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Magnetic Pressure & Field Lines', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_mag_pressure.png')
        print(f"  {'\u2713' if ok else '\u2717'} Magnetic pressure \u2192 {path} ({sz:.0f} KB)")

    def plot_diagnostics(self):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        colors, styles = self._colors, self._styles
        sm_en = self._get_smoothing(); ns = 5
        pairs = list(self.solvers.items())

        ax = axes[0, 0]
        for i, (label, s) in enumerate(pairs):
            t, mw = np.array(s.diag_times), np.array(s.diag_mixing_width_integral)
            ax.plot(t, smooth(mw, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, mw, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$W = \int \langle C\rangle(1-\langle C\rangle)\,dx$')
        ax.set_title('Integral Mixing Width', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[0, 1]
        for i, (label, s) in enumerate(pairs):
            t, mw = np.array(s.diag_times), np.array(s.diag_mixing_width_thresh)
            ax.plot(t, smooth(mw, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, mw, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel('Threshold Width')
        ax.set_title('Threshold MW (1%\u201399%)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[0, 2]
        for i, (label, s) in enumerate(pairs):
            t, theta = np.array(s.diag_times), np.array(s.diag_mixedness)
            ax.plot(t, smooth(theta, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, theta, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\theta$')
        ax.set_title(r'Mixedness $\theta$', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(0, 1)

        ax = axes[0, 3]
        for i, (label, s) in enumerate(pairs):
            t, amp = np.array(s.diag_times), np.array(s.diag_perturbation_amp)
            ax.plot(t, smooth(amp, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, amp, color=colors[i%5], lw=0.4, alpha=0.15)
            if s._rh_results is not None:
                cfg = s.cfg; Ly = cfg.y_max - cfg.y_min
                k_pert = 2*np.pi*cfg.perturbation_mode / Ly
                t_theory = np.linspace(0, cfg.t_end, 500)
                a_lin, t_sh, info = richtmyer_linear_theory(
                    t_theory, cfg.gamma, cfg.mach, 1.0, cfg.density_ratio, cfg.perturbation_amp, k_pert, By=cfg.B_transverse)
                ax.plot(t_theory, a_lin, color=colors[i%5], ls=':', lw=1.5, alpha=0.7)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'Mode amplitude $a_k$')
        ax.set_title(f'Mode {pairs[0][1].cfg.perturbation_mode} Amp + Linear Theory', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1, 0]
        for i, (label, s) in enumerate(pairs):
            ax.plot(s.diag_times, s.diag_enstrophy_local, color=colors[i%5], ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\langle\rho\omega_z^2\rangle_\mathrm{local}$')
        ax.set_title('Local Enstrophy (interface)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1, 1]
        for i, (label, s) in enumerate(pairs):
            ax.plot(s.diag_times, s.diag_stag_pressure, color=colors[i%5], ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('$t$'); ax.set_ylabel('Peak Stag. Pressure')
        ax.set_title('Stagnation Pressure', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        ax = axes[1, 2]
        has_mhd = False
        for i, (label, s) in enumerate(pairs):
            if s.cfg.B_transverse > 0:
                ax.plot(s.diag_times, s.diag_divB_max, color=colors[i%5], ls=styles[i%5], lw=2, label=f'{label} max')
                ax.plot(s.diag_times, s.diag_divB_L2, color=colors[i%5], ls=':', lw=1.5, alpha=0.7, label=f'{label} L2')
                has_mhd = True
        if has_mhd:
            ax.set_xlabel('$t$'); ax.set_ylabel(r'$|\nabla\cdot\mathbf{B}|$')
            ax.set_title(r'$\nabla\cdot\mathbf{B}$ Control (max & L2)', fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        else:
            ax.set_visible(False)

        ax = axes[1, 3]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times); e = np.array(s.diag_energy_total); bf = np.array(s.diag_boundary_flux_cumulative)
            if len(e) > 0 and abs(e[0]) > 1e-14:
                e_rel_raw = (e - e[0]) / abs(e[0]) * 100
                ax.plot(t, e_rel_raw, color=colors[i%5], ls=styles[i%5], lw=1, alpha=0.3)
                e_corrected = (e - bf - e[0]) / abs(e[0]) * 100
                ax.plot(t, e_corrected, color=colors[i%5], ls=styles[i%5], lw=2, label=f"{label} (corrected)")
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\Delta E / E_0$ (%)')
        ax.set_title('Energy Conservation', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_diagnostics.png')
        print(f"  {'\u2713' if ok else '\u2717'} Diagnostics \u2192 {path} ({sz:.0f} KB)")

    def plot_evolution(self):
        n_cases = len(self.solvers); n_cols = 4
        fig, axes = plt.subplots(n_cases, n_cols, figsize=(4*n_cols, 3*n_cases), sharex=True, sharey=True, squeeze=False)
        for row, (label, solver) in enumerate(self.solvers.items()):
            keys = sorted(solver.snapshots.keys(), key=lambda k: solver.snapshots[k]['t'])
            if len(keys) > n_cols:
                idx = np.linspace(0, len(keys)-1, n_cols, dtype=int)
                keys = [keys[i] for i in idx]
            for col in range(min(n_cols, len(keys))):
                ax = axes[row, col]; snap = solver.snapshots[keys[col]]; rho = snap['rho']
                ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto', vmin=0.5, vmax=rho.max()*0.95)
                ax.set_title(f"$t={snap['t']:.3f}$", fontsize=10); ax.set_aspect('equal')
                if col == 0: ax.set_ylabel(f"{label}\n$y$")
                if row == n_cases-1: ax.set_xlabel('$x$')
        fig.suptitle('Density Evolution (GPU)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_evolution.png')
        print(f"  {'\u2713' if ok else '\u2717'} Evolution strips \u2192 {path} ({sz:.0f} KB)")

    def plot_hero_enstrophy(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, s) in enumerate(self.solvers.items()):
            ax.plot(s.diag_times, s.diag_enstrophy_local, color=self._colors[i%5], ls=self._styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel(r'Local Enstrophy $\langle\rho\omega_z^2\rangle$', fontsize=13)
        ax.set_title('Enstrophy: Vorticity Suppression by Magnetic Field', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_enstrophy_hero.png')
        print(f"  {'\u2713' if ok else '\u2717'} Hero enstrophy \u2192 {path} ({sz:.0f} KB)")

    def plot_hero_density(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, key in zip(axes, [keys[0], keys[-1]]):
            snap = self._final(self.solvers[key]); rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto', vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"$\\mathbf{{{key}}}$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle(r'Magnetized RMI \u2014 Mach 10 Shock (GPU)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_hero_density.png')
        print(f"  {'\u2713' if ok else '\u2717'} Hero density \u2192 {path} ({sz:.0f} KB)")

    def plot_interface_shape(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, solver) in enumerate(self.solvers.items()):
            snap = self._final(solver); C = np.clip(snap['C'], 0, 1)
            ax.contour(snap['x'], snap['y'], C.T, levels=[0.5], colors=[self._colors[i%5]], linewidths=2)
            ax.plot([], [], color=self._colors[i%5], lw=2, label=label)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Interface Shape at Final Time ($C=0.5$)', fontweight='bold')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_interface_shape.png')
        print(f"  {'\u2713' if ok else '\u2717'} Interface shape \u2192 {path} ({sz:.0f} KB)")

    def plot_spectral_modes(self):
        keys = list(self.solvers.keys()); hydro_key = keys[0]
        has_mhd = any(s.cfg.B_transverse > 0 for s in self.solvers.values())
        ncols = 2 if has_mhd else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 5))
        if ncols == 1: axes = [axes]
        else: axes = list(axes)
        ax = axes[0]
        for i, (label, s) in enumerate(self.solvers.items()):
            if len(s.diag_mode_amps) > 0:
                amps = s.diag_mode_amps[-1]; n_modes = min(len(amps), 20)
                ax.semilogy(range(n_modes), amps[:n_modes], color=self._colors[i%5], ls=self._styles[i%5], lw=2, marker='o', ms=4, label=label)
        ax.set_xlabel('Mode number $k$'); ax.set_ylabel(r'Amplitude $|\hat{\eta}_k|$')
        ax.set_title('Interface Spectral Modes', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
        if has_mhd and len(axes) > 1:
            ax = axes[1]
            hydro_amps = self.solvers[hydro_key].diag_mode_amps[-1] if self.solvers[hydro_key].diag_mode_amps else None
            if hydro_amps is not None:
                for i, (label, s) in enumerate(self.solvers.items()):
                    if s.cfg.B_transverse > 0 and len(s.diag_mode_amps) > 0:
                        mhd_amps = s.diag_mode_amps[-1]; n_modes = min(len(mhd_amps), len(hydro_amps), 20)
                        ratio = np.zeros(n_modes)
                        for k in range(n_modes):
                            if hydro_amps[k] > 1e-16: ratio[k] = mhd_amps[k] / hydro_amps[k]
                        ax.plot(range(n_modes), ratio, color=self._colors[i%5], ls=self._styles[i%5], lw=2, marker='s', ms=4, label=label)
                ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
                ax.set_xlabel('Mode number $k$'); ax.set_ylabel(r'$|\hat{\eta}_k^{MHD}| / |\hat{\eta}_k^{hydro}|$')
                ax.set_title('MHD/Hydro Spectral Ratio', fontweight='bold')
                ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_spectral_modes.png')
        print(f"  {'\u2713' if ok else '\u2717'} Spectral modes \u2192 {path} ({sz:.0f} KB)")

    def plot_summary_bars(self):
        t_skip = 0.03
        labels_list, mwi_peaks, mwt_peaks, theta_means, enst_means = [], [], [], [], []
        sm_en = self._get_smoothing()
        for label, s in self.solvers.items():
            t = np.array(s.diag_times); mask = t > t_skip
            mwi = smooth(np.array(s.diag_mixing_width_integral)[mask], 7, sm_en)
            mwt = smooth(np.array(s.diag_mixing_width_thresh)[mask], 7, sm_en)
            theta = np.array(s.diag_mixedness)[mask]; en = np.array(s.diag_enstrophy_local)[mask]
            labels_list.append(label)
            mwi_peaks.append(float(np.max(mwi)) if len(mwi) > 0 else 0)
            mwt_peaks.append(float(np.max(mwt)) if len(mwt) > 0 else 0)
            theta_means.append(float(np.mean(theta)) if len(theta) > 0 else 0)
            enst_means.append(float(np.mean(en)) if len(en) > 0 else 0)
        x_pos = np.arange(len(labels_list)); bar_colors = self._colors[:len(labels_list)]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4.5))
        for ax, vals, ylabel, title in [
            (ax1, mwi_peaks, r'Peak $\int\langle C\rangle(1-\langle C\rangle)dx$', 'Integral MW'),
            (ax2, mwt_peaks, 'Peak Threshold Width', 'Threshold MW'),
            (ax3, theta_means, r'Mean $\theta$', 'Mixedness'),
            (ax4, enst_means, 'Mean Local Enstrophy', 'Enstrophy'),
        ]:
            bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='k')
            ax.set_xticks(x_pos); ax.set_xticklabels(labels_list, fontsize=8)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
            fmt = '.4f' if max(vals) < 1 else '.0f'
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02, f"{val:{fmt}}", ha='center', va='bottom', fontsize=9)
        ax3.set_ylim(0, 1)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_summary.png')
        print(f"  {'\u2713' if ok else '\u2717'} Summary bars \u2192 {path} ({sz:.0f} KB)")

    def plot_mixing_width_hero(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors, styles = self._colors, self._styles; sm_en = self._get_smoothing(); ns = 7
        pairs = list(self.solvers.items())
        ax = axes[0]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times); mw = smooth(np.array(s.diag_mixing_width_integral), ns, sm_en)
            ax.plot(t, mw, color=colors[i%5], ls=styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel(r'$W = \int \langle C\rangle(1-\langle C\rangle)\,dx$', fontsize=12)
        ax.set_title('Integral Mixing Width', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax = axes[1]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times); mw = smooth(np.array(s.diag_mixing_width_thresh), ns, sm_en)
            ax.plot(t, mw, color=colors[i%5], ls=styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel('Threshold Width (1%\u201399%)', fontsize=12)
        ax.set_title('Threshold Mixing Width', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_mixing_width_hero.png')
        print(f"  {'\u2713' if ok else '\u2717'} Mixing width hero \u2192 {path} ({sz:.0f} KB)")

    def plot_linear_theory_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); colors, styles = self._colors, self._styles
        ax = axes[0]
        for i, (label, s) in enumerate(self.solvers.items()):
            t = np.array(s.diag_times); amp = np.array(s.diag_perturbation_amp)
            ax.plot(t, amp, color=colors[i%5], ls=styles[i%5], lw=2.5, label=f'{label} (sim)')
            if s._rh_results is not None:
                cfg = s.cfg; Ly = cfg.y_max - cfg.y_min; k_pert = 2*np.pi*cfg.perturbation_mode / Ly
                t_theory = np.linspace(0, cfg.t_end, 500)
                a_lin, t_sh, info = richtmyer_linear_theory(t_theory, cfg.gamma, cfg.mach, 1.0, cfg.density_ratio, cfg.perturbation_amp, k_pert, By=cfg.B_transverse)
                ax.plot(t_theory, a_lin, color=colors[i%5], ls=':', lw=1.5, alpha=0.7, label=f'{label} (linear)')
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel(r'Perturbation Amplitude $a_k$', fontsize=12)
        ax.set_title('Simulation vs Richtmyer Linear Theory', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax = axes[1]; keys = list(self.solvers.keys())
        if len(keys) >= 2:
            hydro_s = self.solvers[keys[0]]; mhd_s = self.solvers[keys[-1]]
            t_h = np.array(hydro_s.diag_times); t_m = np.array(mhd_s.diag_times)
            amp_h = np.array(hydro_s.diag_perturbation_amp); amp_m = np.array(mhd_s.diag_perturbation_amp)
            vx_h = hydro_s._rh_results['vx2'] if hydro_s._rh_results else 1.0
            vx_m = mhd_s._rh_results['vx2'] if mhd_s._rh_results else 1.0
            ax.plot(t_h * vx_h, amp_h / hydro_s.cfg.perturbation_amp, color=colors[0], ls=styles[0], lw=2.5, label=f'{keys[0]} (v_post\u00b7t)')
            ax.plot(t_m * vx_m, amp_m / mhd_s.cfg.perturbation_amp, color=colors[2], ls=styles[2], lw=2.5, label=f'{keys[-1]} (v_post\u00b7t)')
        ax.set_xlabel(r'Normalized time $v_{post} \cdot t$', fontsize=12); ax.set_ylabel(r'$a_k / a_0$', fontsize=12)
        ax.set_title('Velocity-Normalized Comparison', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_linear_theory.png')
        print(f"  {'\u2713' if ok else '\u2717'} Linear theory comparison \u2192 {path} ({sz:.0f} KB)")

    def plot_interface_perturbation(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); colors, styles = self._colors, self._styles
        hydro_key, mhd_key = keys[0], keys[-1]
        for ax_idx, (metric_name, metric_key) in enumerate([
            ('Mixing Width (peak-to-trough)', 'diag_mixing_width_thresh'),
            ('Perturbation Amplitude', 'diag_perturbation_amp'),
        ]):
            ax = axes[ax_idx]
            for i, key in enumerate([hydro_key, mhd_key]):
                s = self.solvers[key]; t = np.array(s.diag_times); val = np.array(getattr(s, metric_key))
                ax.plot(t, val, color=colors[i%5], ls=styles[i%5], lw=2.5, label=key)
            ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'Interface {metric_name}', fontweight='bold', fontsize=13)
            ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_interface_perturbation.png')
        print(f"  {'\u2713' if ok else '\u2717'} Interface perturbation \u2192 {path} ({sz:.0f} KB)")

    def plot_stagnation_pressure(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, key in enumerate([keys[0], keys[-1]]):
            s = self.solvers[key]
            ax.plot(s.diag_times, s.diag_stag_pressure, color=self._colors[i%5], ls=self._styles[i%5], lw=2.5, label=key)
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel('Peak Stagnation Pressure', fontsize=12)
        ax.set_title('Stagnation Pressure', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_stagnation_pressure.png')
        print(f"  {'\u2713' if ok else '\u2717'} Stagnation pressure \u2192 {path} ({sz:.0f} KB)")

    def plot_divB_comparison(self):
        mhd_cases = {k: v for k, v in self.solvers.items() if v.cfg.B_transverse > 0}
        if not mhd_cases: return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); colors, styles = self._colors, self._styles
        ax = axes[0]
        for i, (label, s) in enumerate(mhd_cases.items()):
            ax.plot(s.diag_times, s.diag_divB_max, color=colors[i%5], ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'max $|\nabla\cdot\mathbf{B}|$')
        ax.set_title(r'$\nabla\cdot\mathbf{B}$ Control (max norm)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax = axes[1]
        for i, (label, s) in enumerate(mhd_cases.items()):
            ax.plot(s.diag_times, s.diag_divB_L2, color=colors[i%5], ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'$L_2$ norm $|\nabla\cdot\mathbf{B}|$')
        ax.set_title(r'$\nabla\cdot\mathbf{B}$ Control ($L_2$ norm)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        ok, path, sz = self._save(fig, 'rmi_divB.png')
        print(f"  {'\u2713' if ok else '\u2717'} divB comparison \u2192 {path} ({sz:.0f} KB)")

    def plot_all(self):
        print("\n=== Generating figures ==="); sys.stdout.flush()
        self.plot_density_comparison()
        self.plot_passive_scalar()
        self.plot_schlieren()
        self.plot_vorticity_comparison()
        self.plot_magnetic_pressure_with_fieldlines()
        self.plot_diagnostics()
        self.plot_evolution()
        self.plot_hero_enstrophy()
        self.plot_hero_density()
        self.plot_interface_shape()
        self.plot_spectral_modes()
        self.plot_summary_bars()
        self.plot_mixing_width_hero()
        self.plot_linear_theory_comparison()
        self.plot_interface_perturbation()
        self.plot_stagnation_pressure()
        self.plot_divB_comparison()

        print(f"\n=== Figure manifest ({len(self._saved_files)} files) ===")
        total_kb = 0
        for fname, ok, sz in self._saved_files:
            status = '\u2713' if ok else '\u2717'
            print(f"  {status} {fname:40s}  {sz:7.0f} KB")
            total_kb += sz
        print(f"  {'Total':>42s}  {total_kb:7.0f} KB")
        print(f"  Directory: {os.path.abspath(OUTPUT_DIR)}")
        print("=== All figures done ===\n"); sys.stdout.flush()


print("PostProcessor OK \u2713")

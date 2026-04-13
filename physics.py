# physics.py — Math & GPU Engine
# CUDA kernels, variable conversions, MUSCL reconstruction,
# HLLD Riemann solver, Rankine-Hugoniot, and linear theory.

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# CuPy / CUDA setup
# ============================================================
try:
    import cupy as cp
    HAS_CUPY = True
    dev = cp.cuda.Device(0)
    dev.use()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    dev_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"CuPy CUDA acceleration: ENABLED ✓  (GPU: {dev_name})")
    print(f"  Compute capability: {props['major']}.{props['minor']}")
    print(f"  Total GPU memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
except ImportError:
    print("ERROR: CuPy is required for this GPU version.")
    print("Install with: pip install cupy-cuda12x  (or appropriate CUDA version)")
    sys.exit(1)

xp = cp  # alias: all array ops go through this

print(f"NumPy {np.__version__}, CuPy {cp.__version__}, Python {sys.version.split()[0]}")

from config import (
    NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)


# ============================================================
# CUDA Raw Kernels
# ============================================================
_hlld_kernel_code = r'''
extern "C" __global__
void hlld_flux_x_kernel(
    const double* __restrict__ WL,
    const double* __restrict__ WR,
    double* __restrict__ F,
    double* __restrict__ smax_arr,
    const int n_faces,
    const double gamma,
    const double ch,
    const double floor_rho,
    const double floor_pr
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_faces) return;

    const int NVAR = 10;
    double gm1 = gamma - 1.0;

    // Load left/right states
    double rhoL = WL[0*n_faces + idx]; if (rhoL < floor_rho) rhoL = floor_rho;
    double rhoR = WR[0*n_faces + idx]; if (rhoR < floor_rho) rhoR = floor_rho;
    double vxL = WL[1*n_faces + idx], vyL = WL[2*n_faces + idx], vzL = WL[3*n_faces + idx];
    double vxR = WR[1*n_faces + idx], vyR = WR[2*n_faces + idx], vzR = WR[3*n_faces + idx];
    double BxL = WL[4*n_faces + idx], ByL = WL[5*n_faces + idx], BzL = WL[6*n_faces + idx];
    double BxR = WR[4*n_faces + idx], ByR = WR[5*n_faces + idx], BzR = WR[6*n_faces + idx];
    double pL = WL[7*n_faces + idx]; if (pL < floor_pr) pL = floor_pr;
    double pR = WR[7*n_faces + idx]; if (pR < floor_pr) pR = floor_pr;
    double psiL = WL[8*n_faces + idx], psiR = WR[8*n_faces + idx];
    double CL_v = WL[9*n_faces + idx], CR_v = WR[9*n_faces + idx];

    // GLM Bx/psi
    double Bx_f, psi_f;
    if (ch > 0.0) {
        Bx_f = 0.5*(BxL + BxR) - 0.5/ch*(psiR - psiL);
        psi_f = 0.5*(psiL + psiR) - 0.5*ch*(BxR - BxL);
    } else {
        Bx_f = 0.5*(BxL + BxR);
        psi_f = 0.0;
    }
    double Bx2 = Bx_f * Bx_f;

    double BmL2 = Bx2 + ByL*ByL + BzL*BzL;
    double BmR2 = Bx2 + ByR*ByR + BzR*BzR;
    double ptL = pL + 0.5*BmL2;
    double ptR = pR + 0.5*BmR2;

    double aL2 = gamma*pL/rhoL;
    double aR2 = gamma*pR/rhoR;

    // Fast magnetosonic
    double tmpL = aL2 + BmL2/rhoL;
    double discL = tmpL*tmpL - 4.0*aL2*Bx2/rhoL;
    if (discL < 0.0) discL = 0.0;
    double cfL = sqrt(fmax(0.5*(tmpL + sqrt(discL)), 0.0));

    double tmpR = aR2 + BmR2/rhoR;
    double discR = tmpR*tmpR - 4.0*aR2*Bx2/rhoR;
    if (discR < 0.0) discR = 0.0;
    double cfR = sqrt(fmax(0.5*(tmpR + sqrt(discR)), 0.0));

    double SL = fmin(vxL - cfL, vxR - cfR);
    double SR = fmax(vxL + cfL, vxR + cfR);
    if (ch > 0.0) { SL = fmin(SL, -ch); SR = fmax(SR, ch); }

    double local_smax = fmax(fabs(SL), fabs(SR));
    smax_arr[idx] = local_smax;

    double dSL = SL - vxL;
    double dSR = SR - vxR;
    double den = dSR*rhoR - dSL*rhoL;
    if (fabs(den) < 1e-14) den = (den >= 0.0) ? 1e-14 : -1e-14;

    double SM = (dSR*rhoR*vxR - dSL*rhoL*vxL - ptR + ptL) / den;
    if (SM < SL + 1e-14) SM = SL + 1e-14;
    if (SM > SR - 1e-14) SM = SR - 1e-14;

    double ptS = ptL + rhoL*dSL*(SM - vxL);

    double dSL_SM = SL - SM; if (fabs(dSL_SM) < 1e-14) dSL_SM = 1e-14;
    double dSR_SM = SR - SM; if (fabs(dSR_SM) < 1e-14) dSR_SM = 1e-14;

    double rhoLS = fmax(rhoL*dSL/dSL_SM, floor_rho);
    double rhoRS = fmax(rhoR*dSR/dSR_SM, floor_rho);
    double sqLS = sqrt(rhoLS);
    double sqRS = sqrt(rhoRS);

    double eps_rel = 1e-8 * fmax(fabs(ptS), 1.0);
    double dnL = rhoL*dSL*dSL_SM - Bx2;
    double dnR = rhoR*dSR*dSR_SM - Bx2;
    if (fabs(dnL) < eps_rel) dnL = (dnL >= 0.0) ? eps_rel : -eps_rel;
    if (fabs(dnR) < eps_rel) dnR = (dnR >= 0.0) ? eps_rel : -eps_rel;

    double vyLS = vyL - Bx_f*ByL*(SM - vxL)/dnL;
    double vzLS = vzL - Bx_f*BzL*(SM - vxL)/dnL;
    double ByLS = ByL*(rhoL*dSL*dSL - Bx2)/dnL;
    double BzLS = BzL*(rhoL*dSL*dSL - Bx2)/dnL;

    double vyRS = vyR - Bx_f*ByR*(SM - vxR)/dnR;
    double vzRS = vzR - Bx_f*BzR*(SM - vxR)/dnR;
    double ByRS = ByR*(rhoR*dSR*dSR - Bx2)/dnR;
    double BzRS = BzR*(rhoR*dSR*dSR - Bx2)/dnR;

    double vBL = vxL*Bx_f + vyL*ByL + vzL*BzL;
    double vBLS = SM*Bx_f + vyLS*ByLS + vzLS*BzLS;
    double EL = pL/gm1 + 0.5*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL) + 0.5*BmL2;
    double ELS = ((SL - vxL)*EL - ptL*vxL + ptS*SM + Bx_f*(vBL - vBLS))/dSL_SM;

    double vBR = vxR*Bx_f + vyR*ByR + vzR*BzR;
    double vBRS = SM*Bx_f + vyRS*ByRS + vzRS*BzRS;
    double ER = pR/gm1 + 0.5*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR) + 0.5*BmR2;
    double ERS = ((SR - vxR)*ER - ptR*vxR + ptS*SM + Bx_f*(vBR - vBRS))/dSR_SM;

    // Double star
    double sBx = (Bx_f >= 0.0) ? 1.0 : -1.0;
    double sqLS_safe = fmax(sqLS, 1e-14);
    double sqRS_safe = fmax(sqRS, 1e-14);
    double SAL = SM - fabs(Bx_f)/sqLS_safe;
    double SAR = SM + fabs(Bx_f)/sqRS_safe;
    if (SAL < SL) SAL = SL; if (SAL > SM) SAL = SM;
    if (SAR < SM) SAR = SM; if (SAR > SR) SAR = SR;

    double dds = fmax(sqLS + sqRS, 1e-14);
    double vyDS = (sqLS*vyLS + sqRS*vyRS + (ByRS - ByLS)*sBx)/dds;
    double vzDS = (sqLS*vzLS + sqRS*vzRS + (BzRS - BzLS)*sBx)/dds;
    double ByDS = (sqLS*ByRS + sqRS*ByLS + sqLS*sqRS*(vyRS - vyLS)*sBx)/dds;
    double BzDS = (sqLS*BzRS + sqRS*BzLS + sqLS*sqRS*(vzRS - vzLS)*sBx)/dds;
    double vBDS = SM*Bx_f + vyDS*ByDS + vzDS*BzDS;
    double ELDS = ELS - sqLS*(vBLS - vBDS)*sBx;
    double ERDS = ERS + sqRS*(vBRS - vBDS)*sBx;

    // Compute flux
    double f[10];

    if (SL >= 0.0) {
        f[0] = rhoL*vxL;
        f[1] = rhoL*vxL*vxL + ptL - Bx2;
        f[2] = rhoL*vyL*vxL - Bx_f*ByL;
        f[3] = rhoL*vzL*vxL - Bx_f*BzL;
        f[4] = (ch > 0.0) ? psi_f : 0.0;
        f[5] = ByL*vxL - Bx_f*vyL;
        f[6] = BzL*vxL - Bx_f*vzL;
        f[7] = (EL + ptL)*vxL - Bx_f*vBL;
        f[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        f[9] = rhoL*CL_v*vxL;
    } else if (SR <= 0.0) {
        f[0] = rhoR*vxR;
        f[1] = rhoR*vxR*vxR + ptR - Bx2;
        f[2] = rhoR*vyR*vxR - Bx_f*ByR;
        f[3] = rhoR*vzR*vxR - Bx_f*BzR;
        f[4] = (ch > 0.0) ? psi_f : 0.0;
        f[5] = ByR*vxR - Bx_f*vyR;
        f[6] = BzR*vxR - Bx_f*vzR;
        f[7] = (ER + ptR)*vxR - Bx_f*vBR;
        f[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        f[9] = rhoR*CR_v*vxR;
    } else {
        // Need star states
        double FL[10], UL_c[10], ULS_c[10];
        FL[0] = rhoL*vxL;
        FL[1] = rhoL*vxL*vxL + ptL - Bx2;
        FL[2] = rhoL*vyL*vxL - Bx_f*ByL;
        FL[3] = rhoL*vzL*vxL - Bx_f*BzL;
        FL[4] = (ch > 0.0) ? psi_f : 0.0;
        FL[5] = ByL*vxL - Bx_f*vyL;
        FL[6] = BzL*vxL - Bx_f*vzL;
        FL[7] = (EL + ptL)*vxL - Bx_f*vBL;
        FL[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        FL[9] = rhoL*CL_v*vxL;

        UL_c[0]=rhoL; UL_c[1]=rhoL*vxL; UL_c[2]=rhoL*vyL; UL_c[3]=rhoL*vzL;
        UL_c[4]=Bx_f; UL_c[5]=ByL; UL_c[6]=BzL; UL_c[7]=EL; UL_c[8]=psiL; UL_c[9]=rhoL*CL_v;

        ULS_c[0]=rhoLS; ULS_c[1]=rhoLS*SM; ULS_c[2]=rhoLS*vyLS; ULS_c[3]=rhoLS*vzLS;
        ULS_c[4]=Bx_f; ULS_c[5]=ByLS; ULS_c[6]=BzLS; ULS_c[7]=ELS; ULS_c[8]=psi_f; ULS_c[9]=rhoLS*CL_v;

        double FLS[10];
        for (int v=0; v<10; v++) FLS[v] = FL[v] + SL*(ULS_c[v] - UL_c[v]);

        double FR[10], UR_c[10], URS_c[10];
        FR[0] = rhoR*vxR;
        FR[1] = rhoR*vxR*vxR + ptR - Bx2;
        FR[2] = rhoR*vyR*vxR - Bx_f*ByR;
        FR[3] = rhoR*vzR*vxR - Bx_f*BzR;
        FR[4] = (ch > 0.0) ? psi_f : 0.0;
        FR[5] = ByR*vxR - Bx_f*vyR;
        FR[6] = BzR*vxR - Bx_f*vzR;
        FR[7] = (ER + ptR)*vxR - Bx_f*vBR;
        FR[8] = (ch > 0.0) ? Bx_f*ch*ch : 0.0;
        FR[9] = rhoR*CR_v*vxR;

        UR_c[0]=rhoR; UR_c[1]=rhoR*vxR; UR_c[2]=rhoR*vyR; UR_c[3]=rhoR*vzR;
        UR_c[4]=Bx_f; UR_c[5]=ByR; UR_c[6]=BzR; UR_c[7]=ER; UR_c[8]=psiR; UR_c[9]=rhoR*CR_v;

        URS_c[0]=rhoRS; URS_c[1]=rhoRS*SM; URS_c[2]=rhoRS*vyRS; URS_c[3]=rhoRS*vzRS;
        URS_c[4]=Bx_f; URS_c[5]=ByRS; URS_c[6]=BzRS; URS_c[7]=ERS; URS_c[8]=psi_f; URS_c[9]=rhoRS*CR_v;

        double FRS[10];
        for (int v=0; v<10; v++) FRS[v] = FR[v] + SR*(URS_c[v] - UR_c[v]);

        // Double star states
        double ULDS[10], URDS[10];
        ULDS[0]=rhoLS; ULDS[1]=rhoLS*SM; ULDS[2]=rhoLS*vyDS; ULDS[3]=rhoLS*vzDS;
        ULDS[4]=Bx_f; ULDS[5]=ByDS; ULDS[6]=BzDS; ULDS[7]=ELDS; ULDS[8]=psi_f; ULDS[9]=rhoLS*CL_v;

        URDS[0]=rhoRS; URDS[1]=rhoRS*SM; URDS[2]=rhoRS*vyDS; URDS[3]=rhoRS*vzDS;
        URDS[4]=Bx_f; URDS[5]=ByDS; URDS[6]=BzDS; URDS[7]=ERDS; URDS[8]=psi_f; URDS[9]=rhoRS*CR_v;

        double FLDS[10], FRDS[10];
        for (int v=0; v<10; v++) FLDS[v] = FLS[v] + SAL*(ULDS[v] - ULS_c[v]);
        for (int v=0; v<10; v++) FRDS[v] = FRS[v] + SAR*(URDS[v] - URS_c[v]);

        if (SAL >= 0.0) {
            for (int v=0; v<10; v++) f[v] = FLS[v];
        } else if (SM >= 0.0) {
            for (int v=0; v<10; v++) f[v] = FLDS[v];
        } else if (SAR >= 0.0) {
            for (int v=0; v<10; v++) f[v] = FRDS[v];
        } else {
            for (int v=0; v<10; v++) f[v] = FRS[v];
        }

        if (ch > 0.0) {
            f[4] = psi_f;
            f[8] = ch*ch*Bx_f;
        }
    }

    // Store flux
    for (int v = 0; v < 10; v++) {
        F[v*n_faces + idx] = f[v];
    }
}
'''

_hlld_kernel = cp.RawKernel(_hlld_kernel_code, 'hlld_flux_x_kernel')


# ============================================================
# GPU Vectorized Operations
# ============================================================
def cons_to_prim(U, gamma):
    """Conservative to primitive — fully on GPU."""
    W = xp.empty_like(U)
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    ri = 1.0 / rho
    W[iRHO] = rho
    W[iVX] = U[MX] * ri
    W[iVY] = U[MY] * ri
    W[iVZ] = U[MZ] * ri
    W[iBX] = U[BX]
    W[iBY] = U[BY]
    W[iBZ] = U[BZ]
    W[iPSI] = U[PSI]
    W[iCLR] = U[RHOC] * ri
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    W[iPR] = xp.maximum((gamma - 1) * (U[EN] - KE - ME), FLOOR_PR)
    return W


def prim_to_cons(W, gamma):
    """Primitive to conservative — fully on GPU."""
    U = xp.empty_like(W)
    rho = xp.maximum(W[iRHO], FLOOR_RHO)
    U[RHO] = rho
    U[MX] = rho * W[iVX]
    U[MY] = rho * W[iVY]
    U[MZ] = rho * W[iVZ]
    U[BX] = W[iBX]
    U[BY] = W[iBY]
    U[BZ] = W[iBZ]
    U[PSI] = W[iPSI]
    U[RHOC] = rho * W[iCLR]
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    U[EN] = W[iPR] / (gamma - 1) + KE + ME
    return U


def muscl_x(W):
    """MUSCL reconstruction in x — GPU vectorized with van Leer limiter."""
    dm = W[:, 1:-1, :] - W[:, :-2, :]
    dp = W[:, 2:, :] - W[:, 1:-1, :]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, 1:-2, :] + 0.5 * d[:, :-1, :]
    WR = W[:, 2:-1, :] - 0.5 * d[:, 1:, :]
    WL[iRHO] = xp.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = xp.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = xp.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = xp.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def muscl_y(W):
    """MUSCL reconstruction in y — GPU vectorized with van Leer limiter."""
    dm = W[:, :, 1:-1] - W[:, :, :-2]
    dp = W[:, :, 2:] - W[:, :, 1:-1]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, :, 1:-2] + 0.5 * d[:, :, :-1]
    WR = W[:, :, 2:-1] - 0.5 * d[:, :, 1:]
    WL[iRHO] = xp.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = xp.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = xp.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = xp.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def hlld_flux_x(WL, WR, gamma, ch):
    """HLLD flux using CUDA kernel — one thread per interface."""
    n_faces = WL.shape[1]
    F = xp.empty_like(WL)
    smax_arr = xp.empty(n_faces, dtype=xp.float64)

    block = 256
    grid = (n_faces + block - 1) // block

    _hlld_kernel(
        (grid,), (block,),
        (WL, WR, F, smax_arr,
         np.int32(n_faces), np.float64(gamma), np.float64(ch),
         np.float64(FLOOR_RHO), np.float64(FLOOR_PR))
    )

    smax = float(xp.max(smax_arr))
    return F, smax


def hlld_flux_y(WL, WR, gamma, ch):
    """HLLD flux in y by coordinate rotation."""
    perm = list(range(NVAR))
    perm[int(iVX)], perm[int(iVY)] = int(iVY), int(iVX)
    perm[int(iBX)], perm[int(iBY)] = int(iBY), int(iBX)
    Fr, sm = hlld_flux_x(WL[perm], WR[perm], gamma, ch)
    return Fr[perm], sm


# ============================================================
# MHD Rankine-Hugoniot
# ============================================================
def mhd_rankine_hugoniot(gamma, M, rho1, p1, By1):
    """Solve MHD Rankine-Hugoniot (CPU — called once at init)."""
    M2 = M * M
    if abs(By1) < 1e-14:
        r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
        cs1 = np.sqrt(gamma * p1 / rho1)
        vs = M * cs1
        p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
        vx2 = vs * (1 - 1.0 / r)
        return rho1 * r, p2, vx2, 0.0, vs

    cs1_sq = gamma * p1 / rho1
    va1_sq = By1**2 / rho1
    cf1 = np.sqrt(cs1_sq + va1_sq)
    vs = M * cf1

    def get_p2(r):
        return p1 + rho1 * vs**2 * (1 - 1.0/r) + 0.5 * By1**2 * (1 - r**2)

    def energy_residual(r):
        p2_ = get_p2(r)
        lhs = 0.5*vs**2 + gamma/(gamma-1)*p1/rho1 + By1**2/rho1
        rhs = 0.5*vs**2/r**2 + gamma/(gamma-1)*p2_/(rho1*r) + r*By1**2/rho1
        return lhs - rhs

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    for _ in range(100):
        f = energy_residual(r)
        dr = 1e-6 * r
        fp = (energy_residual(r + dr) - energy_residual(r - dr)) / (2 * dr)
        if abs(fp) < 1e-30: break
        r_new = r - f / fp
        r_new = max(r_new, 1.001)
        r_new = min(r_new, (gamma + 1) / (gamma - 1) - 0.001)
        if abs(r_new - r) < 1e-12:
            r = r_new; break
        r = r_new

    rho2 = rho1 * r
    p2 = get_p2(r)
    By2 = r * By1
    vx2 = vs * (1 - 1.0 / r)
    return rho2, p2, vx2, By2, vs


# ============================================================
# Smoothing Helper
# ============================================================
def smooth(y, n=5, enabled=True):
    """Optional smoothing filter for diagnostics."""
    if not enabled:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * n:
        return y
    kernel = np.ones(n) / n
    yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(yp, kernel, mode='same')[n:-n]


# ============================================================
# Linear Theory
# ============================================================
def richtmyer_linear_theory(t_arr, gamma, mach, rho1, rho_heavy, a0, k, By=0.0):
    """Compute linear RMI growth (CPU)."""
    M2 = mach * mach
    if abs(By) < 1e-14:
        cs1 = np.sqrt(gamma * 1.0 / rho1)
        vs = mach * cs1
    else:
        cs1_sq = gamma * 1.0 / rho1
        va1_sq = By**2 / rho1
        cf1 = np.sqrt(cs1_sq + va1_sq)
        vs = mach * cf1

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)
    a0_post = a0 / r
    rho2_light = rho1 * r
    rho2_heavy = rho_heavy * r
    A_post = (rho2_heavy - rho2_light) / (rho2_heavy + rho2_light)
    delta_v = vs * (1.0 - 1.0 / r)
    x_shock_init = 1.2
    x_interface = 1.5
    t_shock = (x_interface - x_shock_init) / vs
    da_dt = A_post * k * a0_post * delta_v

    a_linear = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < t_shock:
            a_linear[i] = a0
        else:
            dt = t - t_shock
            if abs(By) < 1e-14:
                a_linear[i] = a0_post + da_dt * dt
            else:
                rho_avg = 0.5 * (rho2_light + rho2_heavy)
                vA = By / np.sqrt(rho_avg)
                omega_A = k * vA
                if omega_A > 1e-14:
                    a_linear[i] = a0_post * np.cos(omega_A * dt) + \
                                  (da_dt / omega_A) * np.sin(omega_A * dt)
                else:
                    a_linear[i] = a0_post + da_dt * dt

    info = {
        'A_post': A_post, 'da_dt': da_dt, 'delta_v': delta_v,
        'vs': vs, 't_shock': t_shock, 'a0_post': a0_post, 'r': r,
    }
    return np.abs(a_linear), t_shock, info

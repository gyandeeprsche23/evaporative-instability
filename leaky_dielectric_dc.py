"""
Two-layer leaky-dielectric film under a DC field: neutral stability D0^2(k).

Python cross-check / plotting companion of  leaky_dielectric_DC_neutral.wl
(which continues the notebook Leaky_dielectric_DC.nb, same symbols).

Geometry (as in the notebook)
    fluid 1 : -h1 < z < 0      wall z = -h1 held at potential 0
    fluid 2 :   0 < z < h2     wall z = +h2 held at potential D0
    interface z = zeta e^{ikx + st},  gravity along -z

Neutral state (s = 0, so W(0) = 0), everything proportional to zeta:
    potentials   psih_j'' - k^2 psih_j = 0
        psih1(-h1) = psih2(h2) = 0
        psih1(0) - psih2(0) = zeta (a2 - a1)                          (2.46)
        sig2 psih2'(0) - sig1 psih1'(0) + q0 W'(0) = 0                (2.48, d/dt = 0)
    velocities   (D^2 - k^2)^2 W_j = 0, no slip, W, W' continuous, W(0) = 0
        [mu (W'' + k^2 W)] = k^2 q0 (psih1(0) + a1 zeta)              (tangential)
    normal stress
        -[mu (W''' - 3 k^2 W')]/k^2 + eps0 [eps a psih'] + (rho2 - rho1) g zeta
            - gamma k^2 zeta = 0
    ([f] = f2 - f1 at z = 0).  With psih = D0 phi, q0 = D0 Q, a = D0 alpha the
    problem depends on D0 only through E = D0^2 and the normal stress becomes
    a QUADRATIC in E:   c2 E^2 + c1 E + c0 = 0.
"""
import numpy as np

EPS0 = 8.854187817e-12
G = 9.81

# name: bottom fluid (1) / top fluid (2).  Representative literature values --
# conductivities of oils in particular vary by an order of magnitude between
# batches; replace with measured values for your own fluids.
FLUIDS = {
    "water / air": dict(
        eps1=80.0, eps2=1.0, sig1=5.5e-6, sig2=1e-15,
        mu1=1.0e-3, mu2=1.8e-5, rho1=998.0, rho2=1.2, gamma=0.072),
    "castor oil / air": dict(
        eps1=4.5, eps2=1.0, sig1=4e-11, sig2=1e-15,
        mu1=0.69, mu2=1.8e-5, rho1=961.0, rho2=1.2, gamma=0.035),
    "silicone oil / air": dict(
        eps1=2.8, eps2=1.0, sig1=1e-12, sig2=1e-15,
        mu1=0.97, mu2=1.8e-5, rho1=970.0, rho2=1.2, gamma=0.021),
    "water / silicone oil (5 cSt)": dict(
        eps1=80.0, eps2=2.7, sig1=5.5e-6, sig2=1e-12,
        mu1=1.0e-3, mu2=4.6e-3, rho1=998.0, rho2=913.0, gamma=0.040),
    "silicone oil / castor oil": dict(
        eps1=2.8, eps2=4.5, sig1=1e-12, sig2=4e-11,
        mu1=0.97, mu2=0.69, rho1=970.0, rho2=961.0, gamma=0.0045),
}


def base_state(p):
    """Base state per unit D0: field gradients alpha1, alpha2 and charge Q."""
    den = p["sig2"] * p["h1"] + p["sig1"] * p["h2"]
    al1, al2 = p["sig2"] / den, p["sig1"] / den
    Q = EPS0 * (p["eps1"] * al1 - p["eps2"] * al2)
    return al1, al2, Q


def _phi_basis(k, h1, h2):
    """Scaled exponentials (all <= 1) and their z-derivatives at z = 0.
    phi1 = C1 e^{kz} + C2 e^{-k(z+h1)},  phi2 = C3 e^{k(z-h2)} + C4 e^{-kz}."""
    e1, e2 = np.exp(-k * h1), np.exp(-k * h2)
    val0 = np.array([1.0, e1, e2, 1.0])
    der0 = np.array([k, -k * e1, k * e2, -k])
    wall1 = np.array([e1, 1.0])          # phi1(-h1)
    wall2 = np.array([1.0, e2])          # phi2(+h2)
    return val0, der0, wall1, wall2


def _w_basis(k, h, sign):
    """Layer basis {e^{a z}, e^{-a (z + sign*h... )}} written so every entry <= 1.
    sign = -1 : layer 1 (wall at z=-h),  sign = +1 : layer 2 (wall at z=+h).
    Returns derivative tables d[n] (n=0..3) at z=0 and at the wall, for the
    four functions  f0 = e^{c z}, f1 = e^{-c (z - zw)}, f2 = z f0, f3 = (z-zw) f1
    with c = -sign*k (f0 decays towards the wall) and zw = sign*h."""
    c, zw = -sign * k, sign * h
    def tab(z):
        E0, E1 = np.exp(c * z), np.exp(-c * (z - zw))
        d = np.zeros((4, 4))
        for n in range(4):
            d[n, 0] = c ** n * E0
            d[n, 1] = (-c) ** n * E1
            d[n, 2] = (c ** n * z + n * c ** (n - 1)) * E0
            d[n, 3] = ((-c) ** n * (z - zw) + n * (-c) ** (n - 1)) * E1
        return d
    return tab(0.0), tab(zw)


def neutral_coeffs(k, p):
    """Coefficients (c2, c1, c0) of  c2 E^2 + c1 E + c0 = 0,  E = D0^2."""
    h1, h2 = p["h1"], p["h2"]
    al1, al2, Q = base_state(p)

    # ---- potentials: response to zeta = 1 and to X = W'(0) = 1 ----
    v0, d0, w1, w2 = _phi_basis(k, h1, h2)
    A = np.zeros((4, 4))
    A[0, :2] = w1
    A[1, 2:] = w2
    A[2] = [v0[0], v0[1], -v0[2], -v0[3]]
    A[3] = [-p["sig1"] * d0[0], -p["sig1"] * d0[1], p["sig2"] * d0[2], p["sig2"] * d0[3]]
    # equilibrate the charge row (conductivities can be 1e-15)
    sc = np.abs(A[3]).max()
    rhs_z = np.array([0, 0, al2 - al1, 0.0])
    rhs_X = np.array([0, 0, 0, -Q / sc])
    A[3] /= sc
    Ca, Cb = np.linalg.solve(A, rhs_z), np.linalg.solve(A, rhs_X)

    def phi_outputs(C, zeta):
        phi1_0 = C[0] * v0[0] + C[1] * v0[1]
        d1 = C[0] * d0[0] + C[1] * d0[1]
        d2 = C[2] * d0[2] + C[3] * d0[3]
        c = phi1_0 + al1 * zeta                       # tangential-field factor
        e = EPS0 * (p["eps2"] * al2 * d2 - p["eps1"] * al1 * d1)   # Maxwell normal stress
        return c, e
    c_a, e_a = phi_outputs(Ca, 1.0)
    c_b, e_b = phi_outputs(Cb, 0.0)

    # ---- velocities: response to unit tangential stress tau = 1 ----
    z1, wl1 = _w_basis(k, h1, -1)
    z2, wl2 = _w_basis(k, h2, +1)
    M = np.zeros((8, 8))
    M[0, :4], M[1, :4] = wl1[0], wl1[1]               # W1 = W1' = 0 at -h1
    M[2, 4:], M[3, 4:] = wl2[0], wl2[1]               # W2 = W2' = 0 at +h2
    M[4, :4] = z1[0]                                  # W1(0) = 0   (kinematic, s = 0)
    M[5, 4:] = z2[0]                                  # W2(0) = 0
    M[6, :4], M[6, 4:] = z1[1], -z2[1]                # W1'(0) = W2'(0)
    M[7, :4] = -p["mu1"] * (z1[2] + k ** 2 * z1[0])
    M[7, 4:] = p["mu2"] * (z2[2] + k ** 2 * z2[0])
    rhs = np.zeros(8)
    rhs[7] = 1.0
    scale = np.abs(M).max(axis=1)
    w = np.linalg.solve(M / scale[:, None], rhs / scale)
    m = z1[1] @ w[:4]                                 # W'(0) per unit tau
    visc = lambda mu, t, c: -mu * (t[3] @ c - 3 * k ** 2 * (t[1] @ c)) / k ** 2
    n_v = visc(p["mu2"], z2, w[4:]) - visc(p["mu1"], z1, w[:4])   # [-p + 2 mu W'] per unit tau

    # ---- close the loop and assemble the normal stress ----
    R = (p["rho2"] - p["rho1"]) * G - p["gamma"] * k ** 2
    beta = k ** 2 * Q * c_b * m
    c2 = -e_a * beta + e_b * m * k ** 2 * Q * c_a
    c1 = n_v * k ** 2 * Q * c_a + e_a - R * beta
    c0 = R
    return c2, c1, c0


def neutral_D0sq(k, p):
    """All positive real roots E = D0^2 of the neutral condition (sorted)."""
    c2, c1, c0 = neutral_coeffs(k, p)
    s = max(abs(c1), 1e-300)
    if abs(c2) * abs(c0) < 1e-13 * c1 ** 2:           # quadratic term negligible
        roots = np.array([-c0 / c1])
    else:
        roots = np.roots([c2, c1, c0])
    roots = roots[np.abs(roots.imag) < 1e-9 * np.abs(roots)]
    return np.sort(roots.real[roots.real > 0])


def neutral_curve(ks, p):
    """Lowest positive branch D0^2(k) (nan where no neutral voltage exists)."""
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        r = neutral_D0sq(k, p)
        if len(r):
            out[i] = r[0]
    return out


def perfect_conductor_dielectric(k, p):
    """Classical limit sig1 -> inf, sig2 -> 0 (check)."""
    return (p["gamma"] * k ** 2 + (p["rho1"] - p["rho2"]) * G) * p["h2"] ** 2 / (
        EPS0 * p["eps2"] * k / np.tanh(k * p["h2"]))


def perfect_dielectrics(k, p):
    """Classical limit of two perfect dielectrics, no interfacial charge (check)."""
    e1, e2, h1, h2 = p["eps1"], p["eps2"], p["h1"], p["h2"]
    t1, t2 = np.tanh(k * h1), np.tanh(k * h2)
    pe = EPS0 * e1 * e2 * (e1 - e2) ** 2 * k / ((e2 * h1 + e1 * h2) ** 2 * (e1 * t2 + e2 * t1))
    return (p["gamma"] * k ** 2 + (p["rho1"] - p["rho2"]) * G) / pe


# --------------------------------------------------------------------------
# Full time-dependent linear problem (inertia, charge relaxation, kinematic
# condition) as a generalized eigenproblem  A x = s B x  -- used to check that
# the s = 0 curve really is the stability boundary.
# --------------------------------------------------------------------------
def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (X - X.T + np.eye(N + 1))
    return D - np.diag(D.sum(axis=1)), x


def growth_rates(k, D0, p, N=32):
    """Eigenvalues s of the full linearised problem, sorted by real part (desc).
    Spurious modes of the collocation scheme are removed by keeping only the
    eigenvalues that agree between two resolutions."""
    a, b = _growth_rates(k, D0, p, N), _growth_rates(k, D0, p, N + 10)
    keep = [s for s in a if np.abs(b - s).min() < 5e-2 * abs(s) + 1e-12]
    return np.array(keep)


def _growth_rates(k, D0, p, N):
    """Solved in units of length h1, time mu1 h1/gamma, stress gamma/h1."""
    from scipy.linalg import eig
    L, T, P = p["h1"], p["mu1"] * p["h1"] / p["gamma"], p["gamma"] / p["h1"]
    al1, al2, Q = base_state(p)
    a1, a2, q0 = al1 * D0, al2 * D0, Q * D0
    Vs = max(abs(D0), 1e-30)                    # potential scale
    qs = EPS0 * Vs / L                          # charge scale
    kk, H1, H2 = k * L, 1.0, p["h2"] / L
    mu = [1.0, p["mu2"] / p["mu1"]]
    rho = [p["rho1"] * L ** 2 / (p["mu1"] * T), p["rho2"] * L ** 2 / (p["mu1"] * T)]
    Bo = (p["rho2"] - p["rho1"]) * G * L / P
    eb = EPS0 * Vs ** 2 / L ** 2 / P            # electric / capillary stress
    sg = [p["sig1"] * T / EPS0, p["sig2"] * T / EPS0]
    A1, A2, Q0 = a1 * L / Vs, a2 * L / Vs, q0 / qs

    Dx, x = cheb(N)
    n = N + 1
    D1 = Dx * 2 / H1                           # layer 1: z = H1 (x - 1)/2, x=1 -> z=0
    D2 = Dx * 2 / H2                           # layer 2: z = H2 (1 - x)/2, x=1 -> z=0
    D2 = -D2
    I = np.eye(n)
    nt = 2 * n + 6
    iW1, iW2 = slice(0, n), slice(n, 2 * n)
    iC, iz, iq = slice(2 * n, 2 * n + 4), 2 * n + 4, 2 * n + 5
    A = np.zeros((nt, nt))
    B = np.zeros((nt, nt))
    v0, d0, w1, w2 = _phi_basis(kk, H1, H2)
    ops = []
    for D, m_, r_ in ((D1, mu[0], rho[0]), (D2, mu[1], rho[1])):
        Dd2 = D @ D
        Lap = Dd2 - kk ** 2 * I
        ops.append((D, Dd2, Dd2 @ D, m_ * Lap @ Lap, r_ * Lap))
    for sl, (D, Dd2, Dd3, A4, B4) in zip((iW1, iW2), ops):
        rows = np.arange(2, n - 2) + sl.start
        A[rows, sl] = A4[2:n - 2]
        B[rows, sl] = B4[2:n - 2]
    (Da, D2a, D3a, _, _), (Db, D2b, D3b, _, _) = ops
    # node 0 is z = 0, node N is the wall
    def setrow(row, **kw):
        for key, val in kw.items():
            tgt = {"W1": iW1, "W2": iW2, "C": iC}.get(key)
            if tgt is None:
                A[row, {"z": iz, "q": iq}[key]] = val
            else:
                A[row, tgt] = val
    # layer 1 boundary rows: 0, 1, n-2, n-1 ; layer 2: n, n+1, 2n-2, 2n-1
    setrow(n - 1, W1=I[N]); setrow(n - 2, W1=Da[N])                 # no slip z=-h1
    setrow(2 * n - 1, W2=I[N]); setrow(2 * n - 2, W2=Db[N])         # no slip z=+h2
    setrow(0, W1=I[0], W2=-I[0])                                    # W continuous
    setrow(1, W1=Da[0], W2=-Db[0])                                  # W' continuous
    C_phi1_0 = np.array([v0[0], v0[1], 0, 0])
    C_d1 = np.array([d0[0], d0[1], 0, 0])
    C_d2 = np.array([0, 0, d0[2], d0[3]])
    # tangential stress
    setrow(n, W2=mu[1] * (D2b[0] + kk ** 2 * I[0]), W1=-mu[0] * (D2a[0] + kk ** 2 * I[0]),
           C=-kk ** 2 * eb * Q0 * C_phi1_0, z=-kk ** 2 * eb * Q0 * A1)
    # normal stress:  -p2 + p1 + 2[mu W'] + Bo z + eb [eps a psih'] - k^2 z = 0
    setrow(n + 1,
           W2=-mu[1] * (D3b[0] - 3 * kk ** 2 * Db[0]) / kk ** 2,
           W1=mu[0] * (D3a[0] - 3 * kk ** 2 * Da[0]) / kk ** 2,
           C=eb * (p["eps2"] * A2 * C_d2 - p["eps1"] * A1 * C_d1),
           z=Bo - kk ** 2)
    B[n + 1, iW2] = -rho[1] * Db[0] / kk ** 2
    B[n + 1, iW1] = rho[0] * Da[0] / kk ** 2
    # potentials
    r0 = 2 * n
    A[r0, iC] = [w1[0], w1[1], 0, 0]
    A[r0 + 1, iC] = [0, 0, w2[0], w2[1]]
    A[r0 + 2, iC] = [v0[0], v0[1], -v0[2], -v0[3]]; A[r0 + 2, iz] = -(A2 - A1)
    A[r0 + 3, iC] = -(p["eps1"] * C_d1 - p["eps2"] * C_d2); A[r0 + 3, iq] = 1.0   # Gauss
    # charge conservation: s q = q0 W'(0) + sig2 psi2' - sig1 psi1'
    A[r0 + 4, iW1] = Q0 * Da[0]
    A[r0 + 4, iC] = sg[1] * C_d2 - sg[0] * C_d1
    B[r0 + 4, iq] = 1.0
    # kinematic: s zeta = W(0)
    A[r0 + 5, iW1] = I[0]
    B[r0 + 5, iz] = 1.0
    s = eig(A, B, right=False)
    s = s[np.isfinite(s)]
    s = s[np.abs(s) < 1e8]
    return s[np.argsort(-s.real)] / T


# --------------------------------------------------------------------------
def _checks():
    base = dict(h1=1e-3, h2=1e-3)
    ks = np.array([50.0, 300.0, 1500.0, 6000.0])
    # (i) perfect conductor under perfect dielectric
    p = dict(FLUIDS["water / air"], **base)
    num = neutral_curve(ks, p)
    print("conductor/dielectric  rel.err:", np.abs(num / perfect_conductor_dielectric(ks, p) - 1).max())
    # (ii) no interfacial charge (eps1/eps2 = sig1/sig2) -> perfect dielectrics
    p = dict(FLUIDS["silicone oil / castor oil"], **base)
    p["sig1"] = p["sig2"] * p["eps1"] / p["eps2"]
    num = neutral_curve(ks, p)
    print("perfect dielectrics   rel.err:", np.abs(num / perfect_dielectrics(ks, p) - 1).max())
    # (iii) full eigenvalue problem: leading growth rate changes sign at D0^2
    for name in FLUIDS:
        p = dict(FLUIDS[name], **base)
        for k in (200.0, 2000.0):
            E = neutral_D0sq(k, p)
            if not len(E):
                print(f"{name:30s} k={k:6.0f}: no neutral voltage"); continue
            E = E[0]
            lo = growth_rates(k, np.sqrt(0.9 * E), p)[0]
            hi = growth_rates(k, np.sqrt(1.1 * E), p)[0]
            print(f"{name:30s} k={k:6.0f}  D0c={np.sqrt(E):9.1f} V   "
                  f"s(0.9 E)={lo:.3e}   s(1.1 E)={hi:.3e}")


def _plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#d1495b", "#3a9d6e", "#8e5bc9", "#d98b1f"]
    h = 1e-3
    ks = np.logspace(np.log10(20), np.log10(2e4), 500)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, f), c in zip(FLUIDS.items(), colors):
        p = dict(f, h1=h, h2=h)
        ax.loglog(ks, neutral_curve(ks, p), color=c, lw=2, label=name)
    ax.set_xlabel(r"wavenumber $k$  (m$^{-1}$)")
    ax.set_ylabel(r"neutral $D_0^{2}$  (V$^2$)")
    ax.set_title(r"DC neutral curves, $h_1=h_2=1$ mm (unstable above each curve)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("D0sq_vs_k_fluid_pairs.png", dpi=200)

    # effect of the conductivity ratio for one oil/oil pair
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    f = FLUIDS["silicone oil / castor oil"]
    S_list = [1e-3, 0.025, 0.2, f["eps1"] / f["eps2"], 3.0, 1e3]
    cm = plt.get_cmap("viridis")
    for j, S in enumerate(S_list):
        p = dict(f, h1=h, h2=h)
        p["sig1"] = S * p["sig2"]
        lab = rf"$\sigma_1/\sigma_2$ = {S:.3g}"
        if S == f["eps1"] / f["eps2"]:
            lab += r" $=\varepsilon_1/\varepsilon_2$ (no charge)"
        ax.loglog(ks, neutral_curve(ks, p), lw=2, color=cm(j / (len(S_list) - 1) * 0.9), label=lab)
    ax.set_xlabel(r"wavenumber $k$  (m$^{-1}$)")
    ax.set_ylabel(r"neutral $D_0^{2}$  (V$^2$)")
    ax.set_title("Silicone oil / castor oil: effect of conductivity ratio")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("D0sq_vs_k_conductivity_ratio.png", dpi=200)


if __name__ == "__main__":
    _checks()
    _plots()

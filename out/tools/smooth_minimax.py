"""Smooth minimax refit of the 41 rotation angles.

Maximises the verifier's synthesis margin M = min_x (tol(x) - |phase(x)-target(x)|)
over all x. Convex; solved by smoothing max_i g_i, g_i = |e_i| - tol_i, with a
log-sum-exp and L-BFGS using the analytic gradient.

  e(d) = r0 + 0.5 * B_active @ d      (turns; angles are half-turns)

Use --drop j to force rotation j to zero and refit the rest.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bundle/sol_decomp/builder'))
sys.path.insert(0, str(ROOT / 'bundle/verifiers/invsqrt'))
import numeric_model as pm  # noqa: E402
import verify_submission_invsqrt as vf  # noqa: E402


def load():
    spec = vf.ProblemSpecInvSqrt.for_n_p(10)
    xs = np.arange(1, spec.max_x + 1, dtype=np.uint64)
    target, tol = vf.reference_values(spec, xs)
    d, K2, R0, Y0, H, sign, flags = pm.load_data()
    codes, fixed = pm.features(xs, K2, R0, Y0, H, sign, flags)
    B = (((codes[:, None] >> np.arange(40, -1, -1, dtype=np.uint64)[None, :]) & 1)
         .astype(np.float64))
    angles = np.array([float(a) for a in d['arbitrary_rotation_half_turns']])
    tl = tol.astype(np.float64) / 2.0**64
    tgt = target.astype(np.float64) / 2.0**64
    full = (fixed.astype(np.float64) / 8.0 + (B @ angles) / 2.0) % 1.0
    r0 = (full - tgt + 0.5) % 1.0 - 0.5
    return mask_domain(B, tl, r0, angles)


def mask_domain(B, tl, r0, angles):
    return B, tl, r0, angles


def minimax(Ba, tl, r0, lam=1e-9, temps=(3e-11, 1e-11, 3e-12, 1e-12, 3e-13, 1e-13)):
    from scipy.optimize import minimize
    n = Ba.shape[1]
    d = np.zeros(n)
    Bt = Ba.T.copy()

    def fg(d, T):
        e = r0 + 0.5 * (Ba @ d)
        g = np.abs(e) - tl
        m = g.max()
        w = np.exp((g - m) / T)
        s = w.sum()
        w /= s
        val = m + T * np.log(s)
        grad = 0.5 * (Bt @ (w * np.sign(e)))
        return val, grad

    for T in temps:
        res = minimize(fg, d, args=(T,), jac=True, method='L-BFGS-B',
                       bounds=[(-lam, lam)] * n, options=dict(maxiter=200, maxfun=400,
                                                              ftol=1e-18, gtol=1e-18))
        d = res.x
    e = r0 + 0.5 * (Ba @ d)
    margin = tl - np.abs(e)
    i = int(np.argmin(margin))
    return d, dict(min_margin_turns=float(margin[i]), min_margin_x=i + 1,
                   ratio=float(abs(e[i]) / tl[i]), maxd=float(np.abs(d).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t0 = time.time()
    B, tl, r0, angles = load()
    margin = tl - np.abs(r0)
    i = int(np.argmin(margin))
    print(f'baseline margin {margin[i]:.6e} at x={i+1} ({time.time()-t0:.0f}s)')
    active = [j for j in range(41) if j != a.drop]
    x0 = angles.copy()
    if a.drop is not None:
        x0[a.drop] = 0.0
    # r0 currently includes the dropped angle's contribution; remove it.
    if a.drop is not None:
        r0 = (r0 - 0.5 * angles[a.drop] * B[:, a.drop])
        margin = tl - np.abs(r0)
        i = int(np.argmin(margin))
        print(f'post-drop start margin {margin[i]:.6e} at x={i+1}')
    Ba = B[:, active]
    d, res = minimax(Ba, tl, r0)
    print(json.dumps(res, indent=2))
    eps = np.pi * res['min_margin_turns'] / len(active)
    rotT = 0.56 * np.log2(1 / eps) + 5.3
    print(f'eps={eps:.4e} rotT={rotT:.4f} n_rot={len(active)} '
          f'rotT_total={len(active)*rotT:.3f}')
    x = x0.copy()
    x[active] = x0[active] + d
    if a.out:
        np.save(a.out, x)
        print('saved', a.out)


if __name__ == '__main__':
    main()

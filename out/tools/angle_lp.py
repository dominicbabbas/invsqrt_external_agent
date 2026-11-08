"""Exact numeric screening of angle refits and rotation deletion.

Loads the recovered integer phase model (bundle/sol_decomp/builder/numeric_model.py)
which reproduces the baseline's 41 arbitrary-rotation feature bits and 64-bit
phases on every x. Fits rotation angles (optionally with one or more rotations
forced to zero) to minimise max |error|/tol over the full domain.

This is a numeric model, not a built circuit. Any positive result still has to
be compiled (angle edits are one-line text edits; deletions remove a Z_POW line)
and verified with the official verifier.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bundle/sol_decomp/builder'))
sys.path.insert(0, str(ROOT / 'bundle/verifiers/invsqrt'))

import numeric_model as pm  # noqa: E402
import verify_submission_invsqrt as vf  # noqa: E402


def load_domain():
    spec = vf.ProblemSpecInvSqrt.for_n_p(10)
    xs = np.arange(1, spec.max_x + 1, dtype=np.uint64)
    target, tol = vf.reference_values(spec, xs)
    d, K2, R0, Y0, H, sign, flags = pm.load_data()
    codes, fixed = pm.features(xs, K2, R0, Y0, H, sign, flags)
    B = ((codes[:, None] >> np.arange(40, -1, -1, dtype=np.uint64)[None, :]) & 1).astype(np.float64)
    angles = np.array([float(a) for a in d['arbitrary_rotation_half_turns']], dtype=np.float64)
    return xs, target, tol, B, fixed, angles


def base_error_ratio(target, tol, phase):
    err = (phase - target).view(np.int64)
    used = np.abs(err).astype(np.float64)
    ratio = used / tol.astype(np.float64)
    return ratio, err


def phase_float(fixed, B, angles):
    # phase in turns, exact enough in float64 for screening (values mod 1, |err|~1e-11)
    return (fixed.astype(np.float64) / 8.0 + B @ angles) % 1.0


def fit(B, fixed, target, tol, active, x0, iters=30, topk=120000, lam=0.02):
    """Cutting-plane LP: minimise t s.t. |r0 + B[:,active]@d| <= t*tol.

    Adds all violated rows each round; a small |d| penalty keeps the LP from
    exploiting rows not yet in the cut set.
    """
    from scipy.optimize import linprog

    tgt = target.astype(np.float64) / 2.0**64
    tl = tol.astype(np.float64) / 2.0**64
    fixed_t = fixed.astype(np.float64) / 8.0
    full0 = (fixed_t + (B @ x0) / 2.0) % 1.0
    r0 = (full0 - tgt + 0.5) % 1.0 - 0.5
    Ba = B[:, active]
    a0 = x0[active].copy()
    r = r0
    na = len(active)
    c = np.zeros(na + 1)
    c[-1] = 1.0
    bounds = [(-lam, lam)] * na + [(0.0, None)]
    sel = np.argsort(-np.abs(r) / np.maximum(tl, 1e-300))[:topk]
    result = None
    tprev = None
    for it in range(iters):
        Bs = Ba[sel] / tl[sel, None]
        rs = r[sel] / tl[sel]
        A = np.vstack([
            np.hstack([Bs, -np.ones((len(sel), 1))]),
            np.hstack([-Bs, -np.ones((len(sel), 1))]),
        ])
        b = np.concatenate([-rs, rs])
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not result.success:
            return None
        d = result.x[:na]
        t = result.x[-1]
        rr = r + Ba @ d
        ratio = np.abs(rr) / np.maximum(tl, 1e-300)
        viol = np.flatnonzero(ratio > t * (1 + 1e-9))
        if len(viol) == 0:
            break
        # keep old cut set plus all violated rows, capped
        cand = np.unique(np.concatenate([sel, viol]))
        if len(cand) > topk:
            keep = np.argsort(-ratio[cand])[:topk]
            cand = cand[keep]
        sel = cand
        if tprev is not None and abs(t - tprev) < 1e-15:
            break
        tprev = t
    d = result.x[:na]
    x = x0.copy()
    x[active] = a0 + d
    full = (fixed_t + (B @ x) / 2.0) % 1.0
    err = (full - tgt + 0.5) % 1.0 - 0.5
    ratio = np.abs(err) / tl
    return dict(angles=x, min_margin=float(1 - ratio.max()), worst_ratio=float(ratio.max()),
                worst_x=int(np.argmax(ratio)) + 1, iters=it + 1, lam=lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop', type=int, default=None, help='index of rotation to force to zero')
    ap.add_argument('--fit-all', action='store_true', help='free LP refit of all angles')
    args = ap.parse_args()
    t0 = time.time()
    xs, target, tol, B, fixed, angles = load_domain()
    print(f'domain built in {time.time()-t0:.1f}s')
    tgt = target.astype(np.float64) / 2.0**64
    tl = tol.astype(np.float64) / 2.0**64
    full = (fixed.astype(np.float64) / 8.0 + (B @ angles) / 2.0) % 1.0
    ratio = np.abs((full - tgt + 0.5) % 1.0 - 0.5) / tl
    i = int(np.argmax(ratio))
    print(f'baseline model: worst ratio {ratio[i]:.6f} at x={i+1}, margin {1-ratio[i]:.3e}')

    active = list(range(41))
    x0 = angles.copy()
    if args.drop is not None:
        active.remove(args.drop)
        x0[args.drop] = 0.0
    res = fit(B, fixed, target, tol, active, x0)
    if res is None:
        print('LP failed')
        return
    print(json.dumps({k: res[k] for k in ('min_margin', 'worst_ratio', 'worst_x', 'iters')}, indent=2))
    np.save(ROOT / 'out/analysis' / f'angle_fit_{"all" if args.drop is None else "drop"+str(args.drop)}.npy', res['angles'])
    print('saved angles')


if __name__ == '__main__':
    main()

# Experiment log — invsqrt

All runs in `/workspaces/invsqrt`. Python 3.13.8, kickmix 1.0.0, numpy 2.x,
numba 0.68.1, scipy 1.18.1, 4 cores. Official verifier used unmodified with
`--shots-per-x 1 --batch-size 2044`.

## 2026-11-08 — session start

- Read prompt, verifier, ARCHITECTURE.md, decomp REPORT.md, literature survey.
- Confirmed kickmix preinstalled; installed `numba` and `scipy` (needed by the
  recovered model).
- Exported baseline to `out/analysis/baseline.kmx` (53,841 lines; 53,818 ops;
  first 23 lines are register declarations).

## Baseline reproduction (official)

```
python bundle/verifiers/invsqrt/verify_submission_invsqrt.py \
  bundle/baseline/invsqrt.kmb --shots-per-x 1 --batch-size 2044 --salt repro
```
PASS, score **95,368.69531123221**, Q62, C1538.2048, mean Toff 1243.1139, 41 rot,
eps 2.434041e-13, worst error/tol 0.9993238 at x=161. Wall 13.2s.
Saved `out/baseline_repro.json`.

Key structural fact: all 41 arbitrary rotations are `Z_POW q28 <angle>`; the 15
high and 26 low angles are one bit-serial phase stream on a single qubit. Angles
in the file equal the recovered `arbitrary_rotation_half_turns` exactly.

## Hypothesis 1: refit the 41 angles (exact model)

The recovered integer model reproduces the baseline: worst ratio 0.999324 at
x=161 (identical to the official run). So the model is an exact surrogate for
this binary's phase; any angle/rotation experiment can be screened at zero
simulation cost.

First LP maximised `max_x |error|/tol` (ratio). Result: 0.9737933, margin
1.05e-10 turns. Built by editing only the 41 `Z_POW` angle strings
(`out/analysis/failed_angle_ratio_fit/build.py`). Official run: PASS but score
**95,498.413** — worse. Cause: the verifier computes
`eps = pi * min_x(tol(x) - |error(x)|) / n_rot`, i.e. the **absolute** margin, not
the ratio. The ratio fit moved the binding row to a large-x, small-tolerance row.

Lesson: optimise absolute margin. Re-ran LP with that objective (after fixing a
factor-of-2: the model's angles are in half-turns, phase contribution is
angle/2).

Baseline absolute margin: **3.1765419e-12 turns** at x=2,539,141. The 10 smallest
absolute margins are equal to within ~1e-18 turns across x = 2.5M, 2.9M, 2.48M,
1.37e4, 8.5e5, ... — a minimax optimum signature. LP with the correct objective
recovers 3.1765150e-12 (2.7e-7 lower) → no gain. `out/analysis/angle_all_lp.log`.

## Hypothesis 2: delete one rotation (one-line edit + refit)

Deleting rotation j removes its `Z_POW` line; the bit computation/uncompute stays,
so the circuit remains a valid basis-restoring circuit and only the tracked phase
changes by `-angle_j * b_j(x)`.

- Weighted least-squares necessary condition (`out/tools/drop_screen.py`, 4m23s):
  residual/tol < 1 only for j=15 (0.259) and j=16 (0.526); j=17 is 1.04; j=18..40
  are 2..1e5; j=0..14 are 7e4..1.7e7. `out/analysis/drop_screen.json`.
- Exact smooth-minimax over all x (log-sum-exp + L-BFGS, analytic gradient,
  `out/tools/smooth_minimax.py`) for j=15: best margin **-3.3396e-12 turns** →
  infeasible. `out/analysis/angle_smooth_drop15.npy`.

Conclusion: all 41 rotations are necessary; a rotation cannot be removed even
with free re-aiming of the other 40.

## Hypothesis 3: structural width / Toffoli ledgers

Priced with the verifier formula. Notable rows:
- Baseline Q62,C1538.20 → 95,368.7.
- 128-leaf quadratic: assume -43 Toffoli (126 vs 169 AND), 41 rot, prior F=38
  row-truncated fit margin 9.153e-12 → C1486.44, score **92,159 (estimate)**.
- Q55 with baseline Toffoli and 41 rot → **84,601 (estimate)**, clears goal.
- Q58 with baseline Toffoli and 41 rot → 89,216 (strong, not goal).
- Q58 with 35 rot and Toffoli <=1257 clears goal.
- Cubic 32-leaf: measured Horner Q173/3720 → negative.

## Negative / discarded

- `out/analysis/failed_angle_ratio_fit/` is a verified **non-improvement**
  (95,498.413). Kept as evidence of the wrong objective, not a candidate.
- LP with a large trust region exploited rows outside the cut set and diverged;
  fixed by margin-seeded cuts and `|d| <= 1e-8` (half-turn units). The final
  all-angle result is stable at 3.1765e-12.

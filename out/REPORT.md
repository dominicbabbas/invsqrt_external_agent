NO CANDIDATE YET: baseline reproduced at 95,368.695; exact-model angle/rotation levers are provably saturated, and the only priced route to the goal is a Q55-Q58 repack that is not built.

## Summary

I reproduced the baseline with the official verifier, then used the decompiled
**exact integer phase model** (`bundle/sol_decomp/builder/numeric_model.py`), which
reproduces this baseline's 41 feature bits and 64-bit phase on all 3,133,452 x
(the reproduced worst case, 0.9993238 at x=161, matches to the last digit).

Two cheap, fully buildable levers were tested to exhaustion and are **negative**:

1. **Refitting the 41 rotation angles** (one-line text edits). The baseline is
   already at the minimax optimum for the verifier's actual objective
   (`min_x tol(x) - |error(x)|`): an exact LP over all x recovers the baseline
   margin to 2.7e-7 relative. A ratio-normalised refit was built and officially
   verified; it PASSES but scores **95,498.413**, i.e. 129.7 *worse*. The
   verifier splits synthesis error over `n_rot` using the *absolute* margin, and
   the baseline's absolute margins are equalised to ~1e-18 turns across dozens of
   x — the signature of a minimax optimum.
2. **Deleting any one of the 41 rotations** (a one-line `Z_POW` deletion plus
   angle refit). A weighted-least-squares necessary condition kills all but
   rotations 15 and 16; a smooth minimax over all x gives rotation 15 the best
   achievable margin **-3.34e-12 turns** (infeasible; baseline margin is
   +3.18e-12). Dropping rotation 16 is worse by the same screen. Every rotation is
   necessary.

The remaining route is structural width reduction. Priced with the verifier's own
formula:

| Target | Requirement to reach 87,429 |
|---|---|
| Q58, 35 rotations | Toffoli <= 1257 (baseline 1243; +14 allowed) |
| Q58, 41 rotations | Toffoli <= 1213 (needs -31) |
| Q55, 41 rotations | Toffoli <= 1295 (baseline 1243; +52 allowed) |
| Q52, 41 rotations | Toffoli <= 1387 |

So a **Q55 repack at the baseline Toffoli count (with all 41 rotations) scores
84,601**, which clears the goal. No such circuit exists; the prior floor estimate
is ~Q55, and no complete circuit near it has been built. This is the next step.

## Evidence log (newest first)

### Exact-model angle and rotation results (`partial-check`, exact over all x)
- Model matches baseline: worst ratio 0.9993238 at x=161, 41 features, identical
  worst-case x and value as the official run.
- Baseline verifier min absolute margin: **3.1765419e-12 turns** at x=2,539,141
  (`out/analysis/angle_all_lp.log`). Many large-x rows share this value to 7+
  digits.
- Full-domain angle LP maximising that margin: 3.1765150e-12 — **no gain**
  (2.7e-7 relative, ~0.01 score).
- Rotation-drop screen (`out/analysis/drop_screen.json`): LS residual/tol is
  <1 only for rotations 15 (0.259) and 16 (0.526); all others are >1 (rotations
  17-40) or ~1e5+ (0-14).
- Smooth minimax for drop-15: best margin **-3.3396e-12** at x=2,484,510
  (`out/analysis/angle_smooth_drop15.npy`) → infeasible.

### Officially verified non-improvement (`official-verify`)
- `out/analysis/failed_angle_ratio_fit/invsqrt.kmb`
  sha256 `5f344680f896bb9a366c56423854f782166cd48d4c9aeb73f3a006624b727d41`.
- Salt `refit-a`, `--shots-per-x 1 --batch-size 2044`: **PASS**, score
  **95,498.41297027543**, Q62, C1540.297, eps 1.8905e-13, rotation_count 41,
  worst error/tol 0.9737933 at x=221.
- Why it is worse: it maximised the *ratio* error, which moved the binding
  constraint to a small-tolerance large-x row; the verifier grades the *absolute*
  margin, which fell from 3.18e-12 to 2.47e-12 turns.
- JSON: `out/analysis/failed_angle_ratio_fit/verify_refit-a.json`. This is a failed
  experiment, **not** a candidate.

### Baseline reproduction (`official-verify`)
- `bundle/baseline/invsqrt.kmb` (sha256 `1c5b1ee2...`), `--shots-per-x 1
  --batch-size 2044 --salt repro`.
- PASS score **95,368.69531123221**, Q=62, C=1538.2048, mean Toffoli 1243.1139,
  41 rotations @ eps 2.434041e-13, worst error/tol 0.9993238 at x=161.
- JSON: `out/baseline_repro.json`. Wall 13.2s, 35.3s CPU.

## Architecture ledgers (priced, not built)

Cost model: `C = Toffoli + (t_base + n_rot*(0.56 log2(1/eps)+5.3))/4`,
`eps = pi*M/n_rot`, `score = Q*C`.

| Architecture | Assumptions | C | Score | Verdict |
|---|---|---:|---:|---|
| Baseline | Q62, 1243.1 Toff, 41 rot, M=3.1765e-12 | 1538.20 | **95,368.7** | reference |
| Angle refit (built) | same Q/Toff/41, ratio-optimal angles | 1540.30 | 95,498.4 | **worse (verified)** |
| Angle minimax | same Q/Toff/41, absolute-margin LP | 1538.20 | 95,368.7 | no gain |
| 1 rotation dropped | 40 rot, best achievable margin negative | — | infeasible | **impossible** |
| 128-leaf quadratic | Q62, 1243.1-43=1200.1 Toff, 41 rot, M=9.153e-12 (prior F=38 row-truncated fit + per-leaf retune) | 1486.44 | **92,159 (estimate)** | needs new shared-lookup + row kernel; unbuilt; width risk |
| Cubic, 32 leaves | measured Horner Q173 / 3720 Toff | ~1875+ | >300,000 | negative |
| Q55 width-first | Q55, 1243.1 Toff, 41 rot, M=3.1765e-12 | 1538.20 | **84,601 (estimate)** | would clear goal; datapath not built |
| Q58 width-first | Q58, same | 1538.20 | 89,216 | strong, not goal |

The 128-leaf row is a paper/numeric estimate only: its rotation count and Q are
assumed equal to the baseline, which is unproven. The Q55 row assumes a repack
that costs *no* Toffoli, which the prior one-qubit analysis suggests is optimistic
(+25 CCX per qubit when shaved locally); a *global* repack is the open question.

## What failed and why

- **Ratio-optimal angle refit** buys nothing because the scorer uses absolute
  margin; it is recorded as an official PASS at a worse score.
- **Rotation deletion** is not free: each of the 41 bits carries independent
  phase information, and removing one leaves no angle assignment inside
  tolerance anywhere in the domain (drop-15 best margin negative).
- **Local peephole / wire eviction** was already saturated (prompt §4) and was
  not repeated.

## Next step

Build a Q55-Q58 datapath with the same 41-rotation phase stream and Toffoli count
<= ~1295 (Q55) or <= ~1257 at 35 rotations (Q58). Concretely: a global schedule
that keeps 22 x + 17 derivative + 12 accumulator + 4 flags live, using
measurement-based recompute for the coefficient/derivative words rather than
storing them, and preserving the exact 41-feature phase model (which is
verified) so the rotation ledger above applies unchanged. The decomp
`parameterized.py` regenerates the *ancestor* (97,746.8), not this baseline, so
the missing piece is a generator for the 38-Toffoli-cheaper baseline arithmetic.

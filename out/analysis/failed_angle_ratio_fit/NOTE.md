# NOT A CANDIDATE — verified non-improvement

This circuit is the baseline with the 41 arbitrary rotation angles refitted to
minimise the **ratio** error/tol. It PASSES the official verifier at salt
`refit-a` but scores **95,498.413**, which is 129.7 **worse** than the baseline
95,368.695.

It is kept only as evidence that the correct objective is the verifier's
*absolute* margin `min_x(tol(x)-|error(x)|)`, not the ratio. Do not submit.

- KMB sha256: 5f344680f896bb9a366c56423854f782166cd48d4c9aeb73f3a006624b727d41
- verify_refit-a.json: PASS, Q62, C1540.297, eps 1.8905e-13, worst ratio 0.9737933
  at x=221, but absolute margin 2.47e-12 turns (baseline 3.18e-12).
- Generator: build.py (edits Z_POW angle strings on out/analysis/baseline.kmx).

# invsqrt_external_agent

Research output for the Verifiable Quantum Circuit Compilation `invsqrt` benchmark
(phase by 1/sqrt(x) for x in [1, 3133452]).

Status: **NO CANDIDATE YET** — the baseline was reproduced at 95,368.695 and the
cheap angle/rotation levers were proven saturated. See `out/REPORT.md` (first line
is the machine-readable status) and `out/EXPERIMENT_LOG.md`.

Contents:
- `out/REPORT.md` — status, claims and evidence levels, architecture ledgers.
- `out/EXPERIMENT_LOG.md` — timestamped hypotheses, runs, failures, run times.
- `out/baseline_repro.json` — official verifier output for the baseline.
- `out/analysis/` — exact-model angle/rotation screens and the baseline KMX dump.
- `out/analysis/failed_angle_ratio_fit/` — a verified non-improvement kept as evidence.
- `out/tools/` — exact numeric screening tools (angle LP, drop screen, minimax).
- `out/candidates/` — empty: no verified candidate beats the baseline.

No credentials or external account data are included.

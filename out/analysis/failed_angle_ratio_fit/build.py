"""Build candidate angle_refit_v1: the baseline circuit with the 41 arbitrary
Z_POW angles replaced by an exact-model minimax refit.

Angle-only edit. Diagonal Z rotations do not change X-basis measurement
statistics, so basis restoration/uncompute is unchanged; only the tracked
phase changes. Angles come from out/tools/angle_lp.py (exact integer model
checked with out/tools/exact_eval.py) and are inserted as 17-digit decimals.
"""
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'out/analysis/baseline.kmx'
ANGLES = ROOT / 'out/analysis/angle_fit_all.npy'
OUT = ROOT / 'out/analysis/failed_angle_ratio_fit'
OUT.mkdir(parents=True, exist_ok=True)

angles = np.load(ANGLES)
lines = SRC.read_text().splitlines(keepends=True)
idx = []
for i, line in enumerate(lines):
    p = line.split()
    if p and p[0] == 'Z_POW' and Fraction(p[2]) % Fraction(1, 4):
        idx.append(i)
assert len(idx) == 41 == len(angles), (len(idx), len(angles))
for k, i in enumerate(idx):
    p = lines[i].split()
    lines[i] = f'Z_POW {p[1]} {format(Decimal(float(angles[k])), "f")}\n'

out_kmx = OUT / 'invsqrt.kmx'
out_kmx.write_text(''.join(lines))

import kickmix as km  # noqa: E402
c = km.Circuit.from_file(str(out_kmx), format='kmx')
c.to_file(str(OUT / 'invsqrt.kmb'), format='kmb')
print('wrote', OUT / 'invsqrt.kmb', 'num_qubits', c.num_qubits, 'ops', len(c))

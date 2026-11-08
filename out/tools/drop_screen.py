"""Necessary-condition screen for deleting one rotation.

Dropping rotation j subtracts a_j*b_j(x) (turns) from the phase. Other angles
can only add a linear combination of the other 40 feature bits. Solve the
weighted least-squares proxy and report the resulting max |residual|/tol over
all x. If that exceeds 1, no angle assignment can keep the circuit inside
tolerance, so the drop is impossible.
"""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bundle/sol_decomp/builder'));sys.path.insert(0,str(ROOT/'bundle/verifiers/invsqrt'))
import numeric_model as pm, verify_submission_invsqrt as vf
spec=vf.ProblemSpecInvSqrt.for_n_p(10);xs=np.arange(1,spec.max_x+1,dtype=np.uint64)
target,tol=vf.reference_values(spec,xs)
d,K2,R0,Y0,H,sign,flags=pm.load_data()
codes,fixed=pm.features(xs,K2,R0,Y0,H,sign,flags)
B=((codes[:,None]>>np.arange(40,-1,-1,dtype=np.uint64)[None,:])&1).astype(np.float64)
ang=np.array([float(a) for a in d['arbitrary_rotation_half_turns']])
tl=tol.astype(np.float64)/2.0**64
tgt=target.astype(np.float64)/2.0**64
phase=(fixed.astype(np.float64)/8.0+(B@ang)/2.0)%1.0
cur=(phase-tgt+0.5)%1.0-0.5
w=1.0/np.maximum(tl,1e-300)
res=[]
for j in range(41):
    y=-ang[j]*B[:,j]/2.0      # phase change from deleting j (turns)
    o=[i for i in range(41) if i!=j]
    X=B[:,o]*0.5
    # weighted least squares: minimise sum ((y - Xc)/tl)^2
    Xw=X*w[:,None]; yw=y*w
    c,*_=np.linalg.lstsq(Xw,yw,rcond=None)
    r=y-X@c
    ratio=np.abs(r)/tl
    i=int(np.argmax(ratio))
    res.append(dict(j=j,angle=float(ang[j]),max_resid_ratio=float(ratio[i]),x=i+1,
                    max_resid_turns=float(abs(r[i])),tol_turns=float(tl[i])))
    print(f'j={j:2d} a={ang[j]:.4e} max_resid/tol={ratio[i]:.3e} at x={i+1}')
json.dump(res,open(ROOT/'out/analysis/drop_screen.json','w'),indent=1)

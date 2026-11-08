import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bundle/sol_decomp/builder'));sys.path.insert(0,str(ROOT/'bundle/verifiers/invsqrt'))
import numeric_model as pm, verify_submission_invsqrt as vf
f=sys.argv[1]
d,K2,R0,Y0,H,sign,flags=pm.load_data()
xs=np.arange(1,3133453,dtype=np.uint64)
spec=vf.ProblemSpecInvSqrt.for_n_p(10);target,tol=vf.reference_values(spec,xs)
codes,fixed=pm.features(xs,K2,R0,Y0,H,sign,flags)
angles=list(np.load(f))
print('max |angle|',max(abs(a) for a in angles),'min nonzero',min(abs(a) for a in angles if a))
hi,lo=pm.angle_units(angles)
phase=pm.phases_from_features(codes,fixed,hi,lo)
err=(phase-target).view(np.int64);used=np.abs(err).astype(np.float64)
ratio=used/tol.astype(np.float64)
i=int(np.argmax(ratio))
print(json.dumps(dict(worst_ratio=float(ratio[i]),worst_x=i+1,margin_units=int(tol[i]-used[i]),margin_turns=float((tol[i]-used[i])/2**64),eps_if_41=float(np.pi*((tol[i]-used[i])/2**64)/41)),indent=2))
# also report top-5
order=np.argsort(-ratio)[:5]
print('top5',[(int(o)+1,float(ratio[o])) for o in order])

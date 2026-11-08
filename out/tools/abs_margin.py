"""Maximise the verifier's min absolute margin (tol - |err|) over all x by
refitting the 41 rotation angles with an exact-model LP."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bundle/sol_decomp/builder'));sys.path.insert(0,str(ROOT/'bundle/verifiers/invsqrt'))
import numeric_model as pm, verify_submission_invsqrt as vf

def load():
    spec=vf.ProblemSpecInvSqrt.for_n_p(10);xs=np.arange(1,spec.max_x+1,dtype=np.uint64)
    target,tol=vf.reference_values(spec,xs)
    d,K2,R0,Y0,H,sign,flags=pm.load_data()
    codes,fixed=pm.features(xs,K2,R0,Y0,H,sign,flags)
    B=((codes[:,None]>>np.arange(40,-1,-1,dtype=np.uint64)[None,:])&1).astype(np.float64)
    angles=np.array([float(a) for a in d['arbitrary_rotation_half_turns']])
    tgt=target.astype(np.float64)/2.0**64; tl=tol.astype(np.float64)/2.0**64
    full=(fixed.astype(np.float64)/8.0+(B@angles)/2.0)%1.0
    err=(full-tgt+0.5)%1.0-0.5
    return target,tol,B,fixed,angles,err,tl,tgt

def fit(target,tol,B,fixed,angles,active,x0,iters=30,topk=150000,lam=1e-8):
    from scipy.optimize import linprog
    tl=tol.astype(np.float64)/2.0**64
    tgt=target.astype(np.float64)/2.0**64
    full0=(fixed.astype(np.float64)/8.0+(B@x0)/2.0)%1.0
    r0=(full0-tgt+0.5)%1.0-0.5     # turns
    u=r0/tl                       # normalised error
    Ba=B[:,active]
    a0=x0[active].copy()
    na=len(active)
    wref=1e-11  # fixed positive scale for the margin variable
    c=np.zeros(na+1);c[-1]=-1.0   # minimise -q, w = wref*q
    bounds=[(-lam,lam)]*na+[(None,None)]
    base_margin=(tl-np.abs(r0))
    sel=np.argsort(base_margin)[:topk]
    res=None
    for it in range(iters):
        M=0.5*Ba[sel]/tl[sel,None]; us=u[sel]
        qcol=(wref/tl[sel])[:,None]
        A=np.vstack([np.hstack([M,qcol]),
                     np.hstack([-M,qcol])])
        b=np.concatenate([1-us,1+us])
        res=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs')
        if not res.success: return None
        d=res.x[:na];q=res.x[-1];s=wref*q
        errn=(u+0.5*(Ba@d)/tl)*tl
        absmargin=tl-np.abs(errn)
        print(f'  it{it}: q={q:.6f} s={s:.4e} maxd={np.abs(d).max():.2e} actual_min={absmargin.min():.4e}')
        viol=np.flatnonzero(absmargin < s*(1-1e-9))
        if len(viol)==0: break
        cand=np.unique(np.concatenate([sel,viol]))
        if len(cand)>topk: cand=cand[np.argsort(absmargin[cand])[:topk]]
        sel=cand
    d=res.x[:na];x=x0.copy();x[active]=a0+d
    full=(fixed.astype(np.float64)/8.0+(B@x)/2.0)%1.0
    err=(full-tgt+0.5)%1.0-0.5
    margin=tl-np.abs(err)
    i=int(np.argmin(margin))
    return dict(angles=x,min_margin_turns=float(margin[i]),min_margin_x=i+1,
                worst_ratio=float(np.abs(err[i])/tl[i]),iters=it+1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--drop',type=int,default=None)
    ap.add_argument('--out',default=None);a=ap.parse_args()
    t0=time.time();target,tol,B,fixed,angles,err,tl,tgt=load()
    margin=tl-np.abs(err);i=int(np.argmin(margin))
    print(f'baseline exact min margin {margin[i]:.6e} turns at x={i+1} (ratio {abs(err[i])/tl[i]:.6f}); eps41={np.pi*margin[i]/41:.4e}')
    active=list(range(41));x0=angles.copy()
    if a.drop is not None: active.remove(a.drop);x0[a.drop]=0.0
    res=fit(target,tol,B,fixed,angles,active,x0)
    if res is None: print('LP failed');sys.exit(1)
    print(json.dumps({k:res[k] for k in ('min_margin_turns','min_margin_x','worst_ratio','iters')},indent=2))
    eps=np.pi*res['min_margin_turns']/len(active)
    print(f'eps={eps:.4e} rotT={0.56*np.log2(1/eps)+5.3:.4f} n_rot={len(active)}')
    if a.out: np.save(a.out,res['angles']);print('saved',a.out)

"""Exact symbolic certification of V11 basis restoration and 41 phase guards.

All 3,133,452 inputs and each independent HMR outcome are represented as BDDs.
The 41 non-Clifford Z_POW support functions are compared with the decompiled
integer phase model. Clifford/other phase is omitted from basis-support state.
"""
from pathlib import Path
import argparse,json,time,sys,gc,resource
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bundle/sol_decomp/tools'))
from bdd_boolean import BDD
W=ROOT
BASE_KMX=W/'out/analysis/inv_v11/invsqrt.kmx';p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--node-limit',type=int,default=1800000);p.add_argument('--phase-identities');p.add_argument('--relation-check');p.add_argument('--relation-line',type=int);p.add_argument('--source-kmx',default='out/analysis/inv_v11/invsqrt.kmx');p.add_argument('--source-kmb-sha256',default='95621d7ee3e13d91d27b2ee8627cd96d861a2ce696a0fa3afee2e1e01465316d');p.add_argument('--result-path',default='out/analysis/inv_v11/exact_cert_full.json');p.add_argument('--liveness-path',default='out/analysis/inv_v11/liveness_full.json');a=p.parse_args();BASE_KMX=W/a.source_kmx;t0=time.time();sys.setrecursionlimit(10000)
xs=np.arange(1,3133453,dtype=np.uint64) if a.full else np.unique(np.r_[np.arange(1,257,dtype=np.uint64),np.random.default_rng(431).integers(1,3133453,size=512,dtype=np.uint64)])
N=len(xs);M=(1<<N)-1;bdd=BDD(1,a.node_limit,0)
sys.path.insert(0,str(W/'bundle/sol_decomp/builder'))
import numeric_model as phase_model
model_data,K2,R0,Y0,H,sign,flags=phase_model.load_data()
model_codes,model_fixed=phase_model.features(xs.astype(np.uint64),K2,R0,Y0,H,sign,flags)
from fractions import Fraction
rotation_lines=[]
for line_no,line in enumerate(BASE_KMX.read_text().splitlines()[23:],24):
 p=line.split()
 if p and p[0]=='Z_POW' and Fraction(p[2]) % Fraction(1,4):rotation_lines.append((line_no,p[1],p[2]))
if len(rotation_lines)!=41:raise RuntimeError(f'expected 41 arbitrary rotations, found {len(rotation_lines)}')
rotation_line_index={line_no:i for i,(line_no,_,_) in enumerate(rotation_lines)}
expected_features=None
phase_identity_records=[]
if a.phase_identities:
 phase_identity_records=json.loads((W/a.phase_identities).read_text())['records']
phase_identities_by_line={}
for record in phase_identity_records:phase_identities_by_line.setdefault(record['line'],[]).append(record)
relation_data=json.loads((W/a.relation_check).read_text()) if a.relation_check else None
relation_record=relation_data.get('relation',relation_data) if relation_data else None
relation_check_result=None
class InputBasis:

    def __init__(self):
        self.rows = {}
        self.values = []
        self.cache = {}

    def decompose(self, v):
        old = self.cache.get(v)
        if old is not None:
            return old
        original = v
        mask = 0
        while v:
            pivot = v.bit_length() - 1
            row = self.rows.get(pivot)
            if row is None:
                idx = len(self.values)
                self.values.append(v)
                self.rows[pivot] = (v, idx)
                mask ^= 1 << idx
                break
            vec, idx = row
            v ^= vec
            mask ^= 1 << idx
        self.cache[original] = mask
        if len(self.cache) > 1000:
            self.cache.clear()
        return mask

def pack(v):
    return int.from_bytes(np.packbits(np.asarray(v, dtype=np.uint8), bitorder='little').tobytes(), 'little')
expected_features=[pack(((model_codes >> np.uint64(40-i))&np.uint64(1)).astype(np.uint8)) for i in range(41)]

def merge(*parts):
    out = {}
    for poly in parts:
        for guard, val in poly.items():
            v = out.get(guard, 0) ^ val
            if v:
                out[guard] = v
            elif guard in out:
                del out[guard]
    return out

def weighted(poly, guard):
    if not guard:
        return {}
    out = {}
    for g, v in poly.items():
        h = bdd.AND(g, guard)
        if h:
            z = out.get(h, 0) ^ v
            if z:
                out[h] = z
            elif h in out:
                del out[h]
    return out

def canonical(poly):
    tensor = {}
    for g, v in poly.items():
        mask = basis.decompose(v)
        while mask:
            bit = mask & -mask
            j = bit.bit_length() - 1
            mask ^= bit
            h = bdd.xor(tensor.get(j, 0), g)
            if h:
                tensor[j] = h
            elif j in tensor:
                del tensor[j]
    out = {}
    for j, g in tensor.items():
        z = out.get(g, 0) ^ basis.values[j]
        if z:
            out[g] = z
        elif g in out:
            del out[g]
    return out

def product(x, y):
    if not x or not y:
        return {}
    if x == y:
        return x.copy()
    out = {}
    for g, v in x.items():
        for h, w in y.items():
            value = v & w
            if not value:
                continue
            cond = bdd.AND(g, h)
            if cond:
                z = out.get(cond, 0) ^ value
                if z:
                    out[cond] = z
                elif cond in out:
                    del out[cond]
    if len(out) > 32:
        out = normalize_raw(out)
    return out

def normalize_raw(poly):
    bydata = {}
    for guard, val in poly.items():
        value = bdd.xor(bydata.get(val, 0), guard)
        if value:
            bydata[val] = value
        elif val in bydata:
            del bydata[val]
    out = {}
    for val, guard in bydata.items():
        value = out.get(guard, 0) ^ val
        if value:
            out[guard] = value
        elif guard in out:
            del out[guard]
    return out

def phase_tensor(poly):
    out = {}
    for g, v in poly.items():
        mask = basis.decompose(v)
        while mask:
            bit = mask & -mask
            j = bit.bit_length() - 1
            mask ^= bit
            h = bdd.xor(out.get(j, 0), g)
            if h:
                out[j] = h
            elif j in out:
                del out[j]
    return out

def eval_guard(edge, assignment):
    while not bdd.isterm(edge):
        node = edge >> 1
        c = edge & 1
        value = assignment.get(bdd.var[node], 0)
        edge = (bdd.hi[node] if value else bdd.lo[node]) ^ c
    return edge & 1

def witness(tensor):
    if not tensor:
        return None
    assignment = bdd.witness(next(iter(tensor.values())))['assignment']
    truth = 0
    for j, g in tensor.items():
        if eval_guard(g, assignment):
            truth ^= basis.values[j]
    assert truth
    ix = (truth & -truth).bit_length() - 1
    return dict(x=int(xs[ix]), input_index=ix, assignment=assignment)

def basis_gate(register, op, ids, cond):
    if op == 'X':
        register[ids[0]] = merge(register[ids[0]], {cond: M} if cond else {})
    elif op == 'CX':
        u, v = ids
        register[v] = merge(register[v], weighted(register[u], cond))
    elif op == 'CCX':
        u, v, z = ids
        register[z] = merge(register[z], weighted(product(register[u], register[v]), cond))
    elif op in ('SWAP', 'CSWAP'):
        u, v = ids[-2:]
        d = merge(register[u], register[v])
        d = product(d, register[ids[0]]) if op == 'CSWAP' else d
        d = weighted(d, cond)
        register[u] = merge(register[u], d)
        register[v] = merge(register[v], d)
basis=InputBasis();q=[{1:pack((xs>>j)&1)} if j<22 else {} for j in range(62)];q=[{g:v for g,v in d.items() if v} for d in q];initial=[v.copy() for v in q];b={};stack=[1];checks={};dirty=[];measurements=[];peak_live=0
ls=BASE_KMX.read_text().splitlines();source=list(enumerate(ls[23:],24));last={}
for n,line in source:
 for v in line.split():
  if v.startswith('b') and v[1:].isdigit():last[int(v[1:])]=n
expire={}
for bit,n in last.items():expire.setdefault(n,[]).append(bit)
def collect():
 global bdd,q,b,stack,peak_live
 old=bdd;new=BDD(1,a.node_limit,0);mapping={0:0}
 def move(edge):
  node=edge>>1;c=edge&1
  if node in mapping:return mapping[node]^c
  r=new.mk(old.var[node],move(old.lo[node]),move(old.hi[node]));mapping[node]=r;return r^c
 q=[{move(g):v for g,v in normalize_raw(d).items() if g} for d in q];b={j:move(e) for j,e in b.items()};stack=[move(e) for e in stack];peak_live=max(peak_live,len(new.var));bdd=new;gc.collect()
def same(actual,expected):
 diff=phase_tensor(merge(actual,expected));return None if not diff else witness(diff)
live_mask=(1<<22)-1
profile=[];boundaries={23:dict(live=list(range(22)),measurement_dependent=[])};rotation_checks=[];phase_identity_checks=[]
selected={23,261,523,15964,16303,19359,20844,23000,23828,25761,25957,32232,32278,35623,39000,53800,53801}
lastmask=live_mask
for n,line in source:
 p=line.split();op=p[0];cond=stack[-1]
 if 'if' in p:cond=bdd.AND(cond,b.get(int(p[-1][1:]),0))
 ids=[int(v[1:]) for v in p[1:] if v.startswith('q') and v[1:].isdigit()]
 for record in phase_identities_by_line.get(n,[]):
  if op!='CCZ' or record['instruction']!=line:raise RuntimeError(f'phase identity source mismatch at line {n}: {line}')
  u,v,z=ids
  actual=weighted(product(product(q[u],q[v]),q[z]),cond)
  expected={}
  for term in record['terms']:
   value={1:M}
   for wire in term:value=product(value,q[wire])
   expected=merge(expected,weighted(value,cond))
  mismatch=same(actual,expected)
  phase_identity_checks.append({'line':n,'instruction':line,'terms':record['terms'],'removed_mean':record['removed_mean'],'mismatch':mismatch})
 if op=='PUSH_CONDITION':stack.append(cond)
 elif op=='POP_CONDITION':stack.pop()
 elif op in ('X','CX','CCX','SWAP','CSWAP'):basis_gate(q,op,ids,cond)
 elif op in ('Z','CZ','CCZ','NEG'):pass
 elif op=='Z_POW':
  idx=rotation_line_index.get(n)
  if idx is not None:
   feature=weighted(q[ids[0]],cond)
   mismatch=same(feature,{1:expected_features[idx]})
   rotation_checks.append({'feature_index':idx,'line':n,'qubit':ids[0],'angle':rotation_lines[idx][2],'model_feature_bit':40-idx,'mismatch':mismatch})
 elif op in ('R','HMR'):
  z=ids[0]
  if op=='HMR':
   varidx=len(measurements);var=bdd.mk(varidx,0,1);measurements.append(dict(line=n,bit=int(p[2][1:]),qubit=z,var=varidx));old=b.get(int(p[2][1:]),0);b[int(p[2][1:])]=bdd.ite(cond,var,old)
  else:
   tensor=phase_tensor(weighted(q[z],cond))
   if tensor:dirty.append(dict(line=n,qubit=z,witness=witness(tensor)))
  q[z]=merge(q[z],weighted(q[z],cond))
 elif op.startswith('BIT_'):
  bit=int(p[1][1:]);old=b.get(bit,0)
  if op=='BIT_INVERT':b[bit]=bdd.xor(old,cond)
  elif op=='BIT_STORE0':b[bit]=bdd.ite(cond,0,old)
  elif op=='BIT_STORE1':b[bit]=bdd.ite(cond,1,old)
  else:raise ValueError(line)
 elif op=='INFO_TAG':pass
 else:raise ValueError(line)
 for bit in expire.get(n,[]):b.pop(bit,None)
 for j in set(ids):
  if len(q[j])>16:q[j]=canonical(q[j])
 if op in ('X','CX','CCX','SWAP','CSWAP','R','HMR'):
  for j in set(ids):
   if len(q[j])>1:q[j]=canonical(q[j])
   if q[j]:live_mask|=1<<j
   else:live_mask&=~(1<<j)
 if live_mask!=lastmask:
  profile.append([n,hex(live_mask),live_mask.bit_count()]);lastmask=live_mask
 if n in selected:
  qs=[canonical(d) for d in q]
  live=[j for j,d in enumerate(qs) if d]
  zero=[j for j,d in enumerate(qs) if not d]
  measurement_dependent=[j for j,d in enumerate(qs) if any(g!=1 for g in d)]
  rank=None
  if not measurement_dependent:
   vectors=[M]+[qs[j].get(1,0) for j in live]
   rows={}
   for value in vectors:
    while value:
     pivot=value.bit_length()-1
     if pivot in rows:value^=rows[pivot]
     else:rows[pivot]=value;break
   rank=len(rows)
  boundaries[n]=dict(live=live,zero=zero,measurement_dependent=measurement_dependent,input_function_rank_with_constant=rank)

 if relation_record and n==a.relation_line:
  relation={1:M} if relation_record.get('constant',False) else {}
  for name in relation_record.get('linear',[]):relation=merge(relation,q[int(name[1:])])
  for factor in relation_record.get('products',[]):
   left={1:M} if factor.get('A_const',False) else {}
   right={1:M} if factor.get('B_const',False) else {}
   for name in factor['A']:left=merge(left,q[int(name[1:])])
   for name in factor['B']:right=merge(right,q[int(name[1:])])
   relation=merge(relation,product(left,right))
  relation_check_result=dict(line=n,target_qubit=43,mismatch=same(q[43],relation))

 if n%5000==0 or len(bdd.var)>max(200000,3*peak_live):
  before=len(bdd.var);collect();print('PROGRESS',n,'before',before,'live',len(bdd.var),'qterms',sum(map(len,q)),'rank',len(basis.values),'seconds',time.time()-t0,flush=True)
checks['final_errors']=[dict(qubit=j,witness=w) for j in range(62) if (w:=same(q[j],initial[j])) is not None]
checks['rotation_feature_mismatches']=[r for r in rotation_checks if r['mismatch'] is not None]
checks['phase_rewrite_mismatches']=[r for r in phase_identity_checks if r['mismatch'] is not None]
result=dict(evidence='partial-check',scope='basis restoration, arbitrary-rotation feature guards, and optional Clifford phase rewrites over all independent HMR outcomes',source_kmx=str(BASE_KMX),source_kmb_sha256=a.source_kmb_sha256,source_kmx_sha256=__import__('hashlib').sha256(BASE_KMX.read_bytes()).hexdigest(),full_domain=a.full,inputs=N,measurements=len(measurements),checks=checks,rotation_feature_count=len(rotation_checks),rotation_feature_mismatches=len(checks['rotation_feature_mismatches']),phase_rewrite_count=len(phase_identity_checks),phase_rewrite_mismatches=len(checks['phase_rewrite_mismatches']),phase_rewrite_records=phase_identity_checks,rotation_records=rotation_checks,dirty_resets=dirty,peak_live_nodes=peak_live,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-t0)
result['optional_relation_check']=relation_check_result
out=W/a.result_path;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print('DONE',dict(evidence=result['evidence'],inputs=N,measurements=len(measurements),basis_errors=len(checks['final_errors']),rotation_mismatches=len(checks['rotation_feature_mismatches']),phase_rewrite_count=len(phase_identity_checks),phase_rewrite_mismatches=len(checks['phase_rewrite_mismatches']),dirty_resets=len(dirty),peak_rss_kb=result['peak_rss_kb'],seconds=result['seconds']),flush=True)

peaks=[]
for k,(line,mask,count) in enumerate(profile):
 if count==62:peaks.append([line,profile[k+1][0]-1 if k+1<len(profile) else len(ls)])
artifact=dict(evidence='partial-check',source_kmx=str(BASE_KMX),source_kmb_sha256=a.source_kmb_sha256,source_kmx_sha256=__import__('hashlib').sha256(BASE_KMX.read_bytes()).hexdigest(),scope='exact reachable basis support over every independent HMR outcome; phase excluded',inputs=N,full_domain=a.full,physical_qubits=62,final_basis_restore_errors=checks['final_errors'],dirty_resets=dirty,boundaries=boundaries,profile=profile,peak_intervals=peaks,seconds=time.time()-t0)
(W/(a.liveness_path if a.full else a.liveness_path.replace('.json','_sample.json'))).write_text(json.dumps(artifact,indent=2))
print('PEAKS',len(peaks),peaks,flush=True)

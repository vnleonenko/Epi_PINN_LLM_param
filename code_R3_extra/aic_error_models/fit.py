import numpy as np, json, math, itertools, sys
from scipy.optimize import least_squares
from core import *
from concurrent.futures import ProcessPoolExecutor
BLO={'beta':1e-4,'gamma':1e-4,'mu':1e-7}; BHI={'beta':5.0,'gamma':2.0,'mu':0.5}
SIG={'restricted':(1/14,0.5),'free':(1/60,100.0)}
def lo_hi(model,sb):
    if model=='SIRD': return np.log([1e-4,1e-4,1e-7]),np.log([5,2,0.5])
    a,b=SIG[sb]; return np.log([1e-4,a,1e-4,1e-7]),np.log([5,b,2,0.5])
def resid_vec(model,th,e0,em,w=None):
    r,traj=resid_series(model,th,e0)
    if not all(np.isfinite(x).all() for x in traj): return None
    if em=='one': return np.concatenate(r)
    if em=='log':
        return np.concatenate([np.log1p(np.maximum(traj[k],0))-np.log1p(OBS[s]) for k,s in enumerate('SIRD')])
    if em=='sep': return np.concatenate([r[k]*w[k] for k in range(4)])
def rss_list(model,th,e0):
    r,_=resid_series(model,th,e0); return [float(np.dot(x,x)) for x in r]
def crit(model,th,e0,em):
    if em=='sep': return sum(n*math.log(x) for x in rss_list(model,th,e0))
    v=resid_vec(model,th,e0,em); return len(v)*math.log(float(np.dot(v,v)))
def fit_one(args):
    model,sb,e0,em,th0=args
    lo,hi=lo_hi(model,sb); z=np.clip(np.log(th0),lo+1e-9,hi-1e-9)
    w=[1/np.std(OBS[s]) for s in 'SIRD']
    iters=6 if em=='sep' else 1
    for it in range(iters):
        def f(zz):
            v=resid_vec(model,np.exp(zz),e0,em,w)
            return np.full(1476,1e6) if v is None else v
        try:
            res=least_squares(f,z,bounds=(lo,hi),method='trf',xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=150)
        except Exception as e: return None
        z=res.x
        if em=='sep':
            rl=rss_list(model,np.exp(z),e0); w=[1/math.sqrt(x/n) for x in rl]
    th=np.exp(z)
    try: c=crit(model,th,e0,em)
    except Exception: return None
    return (model,sb,e0,em,list(th),c)
def starts(model,sb):
    bl=[0.01,0.1,0.5]; gl=[0.01,0.1,0.5]; ml=[1e-4,1e-2]
    if model=='SIRD':
        return [(b,g,m) for b in bl for g in gl for m in ml]+[(0.1416,0.118,0.0018),(0.1253,0.110,0.0073)]
    a,b_=SIG[sb]; sl=[a,math.sqrt(a*b_),b_]
    return [(b,s,g,m) for b in bl for s in sl for g in gl[:2] for m in ml]+[(0.459,max(a,min(b_,0.0714)),0.271,0.0217),(0.057,max(a,min(b_,0.5)),0.049,0.00215),(0.1107,max(a,min(b_,0.5)),0.0965,0.006)]
if __name__=='__main__':
    ems=sys.argv[1].split(',')
    I0=OBS['I'][0]; grid=[0,0.5,1,2,4,8]
    jobs=[]
    for em in ems:
        for th in starts('SIRD',None): jobs.append(('SIRD',None,0.0,em,th))
        for sb in ['restricted','free']:
            for g in grid:
                for th in starts('SEIR-D',sb): jobs.append(('SEIR-D',sb,g*I0,em,th))
    print(len(jobs),flush=True)
    out=[]
    with ProcessPoolExecutor(2) as ex:
        for k,r in enumerate(ex.map(fit_one,jobs,chunksize=4)):
            if r: out.append(r)
            if k%100==0: print(k,flush=True)
    json.dump(out,open(f'fits_{"_".join(ems)}.json','w'))

import numpy as np, json, math, sys, itertools
from scipy.optimize import least_squares
from core import sim, OBS, n
from concurrent.futures import ProcessPoolExecutor
I0=OBS['I'][0]
BPS={0:[()],2:[(55,182),(114,255),(110,200),(80,230)],3:[(44,186,285),(111,201,254),(110,160,255),(60,120,250)]}
def unpack(z,K,model):
    b=np.exp(z[:K+1]); bp=z[K+1:2*K+1]; i=2*K+1
    if model=='SEIR-D': s=math.exp(z[i]); i+=1
    else: s=1.0
    g,m=np.exp(z[i:i+2]); return b,bp,s,g,m
def resid(z,K,model,e0,obj):
    b,bp,s,g,m=unpack(z,K,model)
    Sx,I,R,D=sim(b,bp,s,g,m,e0,sird=(model=='SIRD'))
    pen=[]
    if K:
        d=np.diff(np.r_[0,bp,368]); pen=list(np.minimum(d-14,0)*1e5)
    if obj=='I': r=I-OBS['I']
    else: r=np.concatenate([Sx-OBS['S'],I-OBS['I'],R-OBS['R'],D-OBS['D']])
    if not np.all(np.isfinite(r)): r=np.full(r.size,1e7)
    return np.r_[r,pen]
def fit(a):
    K,model,e0r,obj,z0=a
    e0=e0r*I0
    lo=[math.log(1e-4)]*(K+1)+[14]*K; hi=[math.log(5)]*(K+1)+[354]*K
    if model=='SEIR-D': lo.append(math.log(1/14)); hi.append(math.log(0.5))
    lo+= [math.log(1e-4),math.log(1e-7)]; hi+=[math.log(2),math.log(0.5)]
    z0=np.clip(z0,np.array(lo)+1e-9,np.array(hi)-1e-9)
    try:
        r=least_squares(resid,z0,args=(K,model,e0,obj),bounds=(lo,hi),x_scale='jac',xtol=1e-10,ftol=1e-12,gtol=1e-10,max_nfev=400,diff_step=1e-3)
    except Exception as ex: return None
    rr=resid(r.x,K,model,e0,obj); m=n if obj=='I' else 4*n
    rss=float(np.dot(rr[:m],rr[:m])); pen=float(np.dot(rr[m:],rr[m:]))
    b,bp,s,g,mu=unpack(r.x,K,model)
    k=(K+1)+K+2+(2 if model=='SEIR-D' else 0)
    aic=m*math.log(rss/m)+2*k
    return dict(K=K,model=model,e0r=e0r,obj=obj,rss=rss,pen=pen,aic=aic,k=k,beta=list(b),bps=list(bp),sigma=s,gamma=g,mu=mu)
def jobs(obj):
    J=[]
    for K in [0,2,3]:
        for bp in BPS[K]:
            for bs in [0.12,0.2,0.3]:
                for gm in [(0.14,0.045),(0.11,0.005),(0.03,0.003)]:
                    base=[math.log(bs)]*(K+1)+list(bp)
                    J.append((K,'SIRD',0.0,obj,np.array(base+[math.log(gm[0]),math.log(gm[1])])))
                    for e0r in [0,0.5,1,2,4]:
                        for sg in [1/12,1/5,1/2.5]:
                            J.append((K,'SEIR-D',e0r,obj,np.array([math.log(bs*1.3)]*(K+1)+list(bp)+[math.log(sg),math.log(gm[0]),math.log(gm[1])])))
    return J
if __name__=='__main__':
    obj=sys.argv[1]; J=jobs(obj); print(len(J),flush=True)
    out=[]
    with ProcessPoolExecutor(2) as ex:
        for i,r in enumerate(ex.map(fit,J,chunksize=2)):
            if r: out.append(r)
            if i%50==0: print(i,flush=True)
    json.dump(out,open(f'fits_{obj}.json','w'))

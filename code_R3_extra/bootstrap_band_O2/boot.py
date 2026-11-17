from w2 import *
import json,sys
rng=np.random.default_rng(20261117)
n=51; L=7; Bn=1000
thm,_=fit(I[:n],Rc[:n],Dc[:n],n)
Im,Rm,Dm=sim(thm,n)
lg=lambda x:np.log1p(np.maximum(x,0))
fitted=[lg(Im),lg(Rm-Rc[0]),lg(Dm-Dc[0])]
obs=[lg(I[:n]),lg(Rc[:n]-Rc[0]),lg(Dc[:n]-Dc[0])]
res=[o-f for o,f in zip(obs,fitted)]
starts=np.arange(0,n-L+1)
thetas=[]
for b in range(Bn):
    idx=np.concatenate([np.arange(s,s+L) for s in rng.choice(starts,size=int(np.ceil(n/L)))])[:n]
    syn=[np.expm1(f+r[idx]) for f,r in zip(fitted,res)]
    sI=syn[0]; sR=syn[1]+Rc[0]; sD=syn[2]+Dc[0]
    th,_=fit(sI,sR,sD,n,x0=tuple(thm))
    thetas.append(th)
thetas=np.array(thetas)
np.save('thetas_O2.npy',thetas); json.dump({'thm':list(thm)},open('thm.json','w'))
print('mine',thm)
for k,nm in enumerate(['beta','gamma','mu']): print(nm,np.percentile(thetas[:,k],[2.5,97.5]))
r=thetas[:,0]-thetas[:,1]-thetas[:,2]; print('r',np.percentile(r,[2.5,97.5]))

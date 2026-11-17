import json,sys,math,numpy as np
from core import *
fits=[]
for f in sys.argv[1:]: fits+=json.load(open(f))
I0=OBS['I'][0]
def best(model,sb,e0,em):
    c=[x for x in fits if x[0]==model and x[1]==sb and abs(x[2]-e0)<1e-6 and x[3]==em]
    return min(c,key=lambda x:x[5]) if c else None
ems=sorted({x[3] for x in fits})
for em in ems:
    s=best('SIRD',None,0.0,em)
    print(f'== {em}: SIRD crit {s[5]:.2f} params {np.round(s[4],5)}')
    for sb in ['restricted','free']:
        for g in [0,0.5,1,2,4,8]:
            b=best('SEIR-D',sb,g*I0,em)
            if not b: continue
            d=b[5]-s[5]+2
            th=b[4]; lat=1/th[1]
            Sx,I,R,D=sim('SEIR-D',th,g*I0)
            ever=1-(Sx[-1]-0)/N  # approx (S+E)
            print(f'  {sb:10s} E0/I0={g:<4} dAIC={d:9.1f} latent={lat:6.2f} d  beta={th[0]:.4f} R0={th[0]/(th[2]+th[3]):.3f}')

import json,sys,math
obj=sys.argv[1]; F=json.load(open(f'fits_{obj}.json'))
F=[f for f in F if f['pen']<1]
best={}
for f in F:
    key=(f['K'],f['model'],f['e0r'])
    if key not in best or f['aic']<best[key]['aic']: best[key]=f
base={K:best[(K,'SIRD',0.0)]['aic'] for K in [0,2,3]}
for key in sorted(best,key=lambda k:(k[0],k[1],k[2])):
    f=best[key]
    print(key, 'RSS %.3g'%f['rss'],'AIC-SIRD(sameK) %+.1f'%(f['aic']-base[key[0]]),'latent %.1f'%(1/f['sigma']) if f['model']=='SEIR-D' else '', 'bps',[round(x,1) for x in f['bps']],'beta',[round(x,3) for x in f['beta']],'g %.3f m %.4f'%(f['gamma'],f['mu']))

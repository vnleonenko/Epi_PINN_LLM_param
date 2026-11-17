from w2 import *
import json
th=np.load('thetas_O2.npy'); thm=np.array(json.load(open('thm.json'))['thm'])
pub=np.array([0.2281,0.1773,0.00704]); delta=np.array([-0.01,0,0])
H=300
def simH(t): return sim(t,H)[0]
I1=simH(pub); Ia=simH(pub+delta)
print('Phase1 peak',int(I1.argmax()),int(I1.max()),' adjusted peak',int(Ia.argmax()),int(Ia.max()))
out={}
for name,cent in [('recentered',pub),('own',thm)]:
    traj=np.array([simH(cent+(t-thm)+delta) for t in th])
    traj1=np.array([simH(cent+(t-thm)) for t in th])
    lo,hi=np.percentile(traj,[2.5,97.5],axis=0); lo1,hi1=np.percentile(traj1,[2.5,97.5],axis=0)
    obs=I[51:134]; d=np.arange(51,134)
    cov=np.mean((obs>=lo[d])&(obs<=hi[d])); cov1=np.mean((obs>=lo1[d])&(obs<=hi1[d]))
    pk=np.argmax(traj,axis=1); ph=traj.max(axis=1)
    m=int(Ia.argmax())
    print(name,'coverage after origin adj %.3f phase1 %.3f'%(cov,cov1),'width at sim peak',int(hi[m]-lo[m]),'(%.0f%% of peak)'%(100*(hi[m]-lo[m])/Ia.max()),
          'peak day 95%',np.percentile(pk,[2.5,97.5]),'height',np.percentile(ph,[2.5,97.5]).astype(int),
          'obs peak in band?', lo[85]<=I[85]<=hi[85])
    # coverage before observed peak / after
    pre=(d<=85); print('  cov to peak %.3f after peak %.3f'%(np.mean(((obs>=lo[d])&(obs<=hi[d]))[pre]),np.mean(((obs>=lo[d])&(obs<=hi[d]))[~pre])))
    out[name]=dict(lo=lo.tolist(),hi=hi.tolist())
json.dump({'Ia':Ia.tolist(),'I1':I1.tolist(),'band':out,'obs':I.tolist()},open('band.json','w'))

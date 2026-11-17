import csv,ctypes,numpy as np,datetime as dt
from scipy.optimize import minimize
lib=ctypes.CDLL('./librk4.so'); P=ctypes.POINTER(ctypes.c_double)
lib.seird.argtypes=[ctypes.c_double]*9+[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[P]*5
N=6e6
rows={r['TIME.sk'][:10]:r for r in csv.DictReader(open('../work/data/SPb.COVID-19.united.csv'))}
d0=dt.date(2021,8,22); T=134
days=[(d0+dt.timedelta(k)).isoformat() for k in range(T)]
I=np.array([float(rows[d]['ACTIVE.sk']) for d in days])
Rc=np.cumsum([float(rows[d]['RECOVERED.sk']) for d in days]); Dc=np.cumsum([float(rows[d]['DEATHS.sk']) for d in days])
def sim(th,n=T,init=None):
    b,g,m=th
    I0,R0,D0=(I[0],Rc[0],Dc[0]) if init is None else init
    out=[np.empty(n) for _ in range(5)]
    lib.seird(b,1.0,g,m,N-I0-R0-D0,0.0,I0,R0,D0,n,N,1,*[o.ctypes.data_as(P) for o in out])
    return out[2],out[3],out[4]
def loss(th,obsI,obsR,obsD,n):
    Im,Rm,Dm=sim(th,n)
    if not np.all(np.isfinite(Im)): return 1e9
    a=np.mean((np.log1p(np.maximum(Im,0))-np.log1p(obsI))**2)
    b=np.mean((np.log1p(np.maximum(Rm-Rc[0],0))-np.log1p(obsR-Rc[0]))**2)
    c=np.mean((np.log1p(np.maximum(Dm-Dc[0],0))-np.log1p(obsD-Dc[0]))**2)
    return a+0.5*b+0.5*c
B=[(0.01,0.99),(0.001,0.5),(0.0001,0.1)]
def fit(obsI,obsR,obsD,n,x0=(0.2,0.15,0.005)):
    r=minimize(loss,x0,args=(obsI,obsR,obsD,n),method='L-BFGS-B',bounds=B)
    return r.x,r

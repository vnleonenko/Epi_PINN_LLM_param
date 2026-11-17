import ctypes,numpy as np,csv,os
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'libpw.so'))
P=ctypes.POINTER(ctypes.c_double)
lib.seird_pw.argtypes=[ctypes.c_int,P,P]+[ctypes.c_double]*8+[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[P]*5
rows=list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data.csv'))))
OBS={k:np.array([float(r[k]) for r in rows]) for k in 'SIRD'}
n=len(rows); N=6e6
def sim(betas,bps,sigma,gamma,mu,e0=0.0,sird=True):
    out=[np.empty(n) for _ in range(5)]; ptr=[o.ctypes.data_as(P) for o in out]
    b=np.ascontiguousarray(betas,float); bp=np.ascontiguousarray(sorted(bps) if len(bps) else [1e9],float)
    lib.seird_pw(len(bps),b.ctypes.data_as(P),bp.ctypes.data_as(P),sigma if not sird else 1.0,gamma,mu,
                 OBS['S'][0]-e0,e0,OBS['I'][0],OBS['R'][0],OBS['D'][0],n,N,1 if sird else 0,*ptr)
    S,E,I,R,D=out; return S+E,I,R,D

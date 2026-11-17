import ctypes,numpy as np,csv
lib=ctypes.CDLL('./librk4.so')
P=ctypes.POINTER(ctypes.c_double)
lib.seird.argtypes=[ctypes.c_double]*9+[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[P]*5
rows=list(csv.DictReader(open('data.csv')))
OBS={k:np.array([float(r[k]) for r in rows]) for k in 'SIRD'}
n=len(rows); N=6e6
def sim(model,th,e0=0.0):
    out=[np.empty(n) for _ in range(5)]
    ptr=[o.ctypes.data_as(P) for o in out]
    if model=='SIRD':
        b,g,m=th; s=1.0; sird=1; e0=0.0
    else:
        b,s,g,m=th; sird=0
    lib.seird(b,s,g,m,OBS['S'][0]-e0,e0,OBS['I'][0],OBS['R'][0],OBS['D'][0],n,N,sird,*ptr)
    S,E,I,R,D=out
    return S+E,I,R,D   # observed S compared with S+E
def resid_series(model,th,e0=0.0):
    Sx,I,R,D=sim(model,th,e0)
    return [Sx-OBS['S'],I-OBS['I'],R-OBS['R'],D-OBS['D']], (Sx,I,R,D)

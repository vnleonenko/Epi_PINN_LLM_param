import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from core import sim,OBS,n
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
I0=OBS['I'][0]; t=np.arange(n)
def best(obj,K,model):
    F=[f for f in json.load(open(f'fits_{obj}.json')) if f['pen']<1 and f['K']==K and f['model']==model]
    return min(F,key=lambda f:f['aic'])
col={0:'#7f7f7f',2:'#1f77b4',3:'#d95f02'}; lab={0:'constant $\\beta$',2:'two breakpoints',3:'three breakpoints'}
fig,axs=plt.subplots(2,2,figsize=(7.5,5.6),sharex=True)
summary={}
for j,obj in enumerate(['I','four']):
    ax=axs[0,j]; bx=axs[1,j]
    ax.scatter(t,OBS['I'],s=3,color='k',zorder=1,label='Observed')
    for K in [0,2,3]:
        for model,ls in [('SIRD','-'),('SEIR-D','--')]:
            f=best(obj,K,model)
            Sx,I,R,D=sim(f['beta'],f['bps'],f['sigma'],f['gamma'],f['mu'],f['e0r']*I0,sird=(model=='SIRD'))
            ax.plot(t,I,color=col[K],ls=ls,lw=1.3)
            bp=np.r_[0,sorted(f['bps']),368]; R0=np.array(f['beta'])/(f['gamma']+f['mu'])
            for k in range(len(R0)): bx.plot([bp[k],bp[k+1]],[R0[k],R0[k]],color=col[K],ls=ls,lw=1.6)
            summary[f'{obj}_{K}_{model}']=dict(gm=f['gamma']+f['mu'],e0r=f['e0r'],latent=(1/f['sigma'] if model=='SEIR-D' else None),bps=f['bps'],R0=list(R0))
    for a in (ax,bx): a.axvspan(110,150,color='#f2e6c9',zorder=0,lw=0)
    for s in ['top','right']: ax.spines[s].set_visible(False); bx.spines[s].set_visible(False)
    ax.set_ylim(0,130000); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:'%d'%v))
    bx.set_xlabel('Day'); bx.set_xlim(0,368)
axs[0,0].set_title('(a) Fitted to $I$ alone',loc='left',fontsize=9); axs[0,1].set_title('(b) Fitted to $S$, $I$, $R$, $D$',loc='left',fontsize=9)
axs[1,0].set_title('(c) $\\beta/(\\gamma+\\mu)$, fits of (a)',loc='left',fontsize=9); axs[1,1].set_title('(d) $\\beta/(\\gamma+\\mu)$, fits of (b)',loc='left',fontsize=9)
axs[0,0].set_ylabel('Active cases $I$'); axs[1,0].set_ylabel('$\\beta/(\\gamma+\\mu)$')
from matplotlib.lines import Line2D
h=[Line2D([],[],color=col[K],lw=1.6) for K in [0,2,3]]+[Line2D([],[],color='k',lw=1.3,ls='-'),Line2D([],[],color='k',lw=1.3,ls='--'),Line2D([],[],marker='o',ms=3,color='k',ls='')]
l=[lab[K] for K in [0,2,3]]+['SIRD','SEIR-D','Observed']
fig.legend(h,l,loc='lower center',ncol=6,frameon=False,fontsize=8,bbox_to_anchor=(0.5,0.0))
fig.tight_layout(rect=(0,0.05,1,1))
fig.savefig('fig_piecewise.pdf'); fig.savefig('fig_piecewise.png',dpi=150)
json.dump(summary,open('fig_summary.json','w'),indent=1)
for k,v in summary.items(): print(k, 'gm %.3f'%v['gm'], v['e0r'], v['latent'] and round(v['latent'],1), [round(x,1) for x in v['bps']], [round(x,2) for x in v['R0']])

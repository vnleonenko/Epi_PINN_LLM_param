import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
x0,y0=59.648432,439.976567; kx=1.58984/0.4004; ky=0.0014747751
P=json.load(open('paths.json')); mean=np.array(P['110'])
md=(mean[:,0]-x0)/kx; mI=(y0-mean[:,1])/ky
m=np.load('mcd_band.npy'); B=json.load(open('../boot/band.json'))
obs=np.array(B['obs']); lo=np.array(B['band']['recentered']['lo']); hi=np.array(B['band']['recentered']['hi'])
Ia=np.array(B['Ia']); I1=np.array(B['I1']); t=np.arange(len(Ia))
d=np.arange(51,134); o=obs[51:134]
mlo=np.interp(d,m[:,0],m[:,1]); mhi=np.interp(d,m[:,0],m[:,2])
cm=np.mean((o>=mlo)&(o<=mhi)); cb=np.mean((o>=lo[d])&(o<=hi[d]))
fig,axs=plt.subplots(1,2,figsize=(7.5,3.4),sharey=True)
C_red='#c0392b'; C_org='#e67e22'; C_blue='#1565c0'; C_mcd='#90caf9'; C_bt='#c7d3e3'
for k,ax in enumerate(axs):
    if k==0:
        ax.fill_between(m[:,0],m[:,1],m[:,2],color=C_mcd,alpha=0.55,lw=0,label='MC Dropout band (2.5–97.5%)')
        ax.plot(md,mI,color=C_blue,lw=1.4,label='PINN mean (Phase 3)')
        ax.set_title('(a) Monte Carlo Dropout band (Phase-3 PINN)',fontsize=9,loc='left')
        cov=cm
    else:
        ax.fill_between(t,lo,hi,color='#f5cba7',alpha=0.8,lw=0,label='Bootstrap band (2.5–97.5%)')
        ax.set_title('(b) Bootstrap band (SIRD)',fontsize=9,loc='left')
        cov=cb
    ax.plot(t,I1,color=C_red,lw=1.4,label='Initial forecast (Phase 1)')
    ax.plot(t,Ia,color=C_org,lw=1.8,ls='--',label='Adjusted forecast (SIRD)')
    ax.scatter(np.arange(0,51),obs[:51],s=6,color='k',zorder=5,label='Known data')
    ax.scatter(d,o,s=8,facecolor='white',edgecolor='k',lw=0.6,zorder=5,label='Unknown data')
    ax.axvline(50,color='gray',ls=':',lw=1)
    ax.text(0.03,0.95,'Observations after\norigin in band: %d%%'%round(100*cov),transform=ax.transAxes,ha='left',va='top',fontsize=8)
    ax.set_xlim(0,160); ax.set_ylim(0,150000); ax.set_xticks(range(0,160,20)); ax.set_xlabel('Days')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:'%d'%v))
axs[0].set_ylabel('Infected (I)')
h0,l0=axs[0].get_legend_handles_labels(); h1,l1=axs[1].get_legend_handles_labels()
order=[l0.index(x) for x in ['Initial forecast (Phase 1)','Adjusted forecast (SIRD)','PINN mean (Phase 3)','MC Dropout band (2.5–97.5%)','Known data','Unknown data']]
hs=[h0[i] for i in order]+[h1[l1.index('Bootstrap band (2.5–97.5%)')]]
ls=[l0[i] for i in order]+['Bootstrap band (2.5–97.5%)']
fig.legend(hs,ls,loc='lower center',ncol=4,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.12,1,1))
fig.savefig('fig_bands_compare.pdf'); fig.savefig('fig_bands_compare.png',dpi=150)
print(cm,cb)

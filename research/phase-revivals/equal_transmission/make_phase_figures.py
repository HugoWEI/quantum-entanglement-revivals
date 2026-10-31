import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase_equal_experiment import capacity_evolution,peak,scattering,gaussian_weights,smooth_curve

HERE=Path(__file__).resolve().parent
L=48;w=.775;m=2.5;t=np.arange(0,36.0001,.05)
# first-revival curves and exact reflection control
fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True)
a=capacity_evolution(L,m,1.2,.35,t);b=capacity_evolution(L,m,.35,1.2,t)
am=capacity_evolution(L,-m,1.2,.35,t);bm=capacity_evolution(L,-m,.35,1.2,t)
sa=smooth_curve(t,a);sb=smooth_curve(t,b)
axs[0].plot(t,sa,label='Defect (1.20, 0.35)')
axs[0].plot(t,sb,label='Defect (0.35, 1.20)')
axs[0].set_xlim(22.5,31);axs[0].set_ylabel('Smoothed modular capacity');axs[0].legend();axs[0].set_title('Equal transmission amplitudes, unequal revival times')
axs[1].plot(t,a-b,label='Initial staggered mass +2.5')
axs[1].plot(t,am-bm,label='Initial staggered mass −2.5')
axs[1].axhline(0,lw=.75);axs[1].set_ylabel('Raw capacity difference');axs[1].set_xlabel('Time (hopping units)');axs[1].legend()
fig.savefig(HERE/'equal_T_capacity_and_parity.png',dpi=180);plt.close(fig)
# nonlinear sign reversal scan and linear-response prediction
DS=np.array([.0,.02,.025,.05,.075,.1,.12,.14,.16,.17533285,.18,.2,.25,.3,.35,.425])
shifts=[];peakpairs=[]
for d in DS:
    pa=peak(t,capacity_evolution(L,m,w+d,w-d,t),L)['peak']
    pb=peak(t,capacity_evolution(L,m,w-d,w+d,t),L)['peak']
    shifts.append(pa-pb);peakpairs.append([pa,pb])
with (HERE/'linear_response_L48_w0.775.json').open() as f:r=json.load(f)
slope=r['response_coefficient_shift_per_delta']
fig,ax=plt.subplots(figsize=(8.5,5.5),constrained_layout=True)
ax.plot(DS,shifts,marker='o',label='Exact finite-chain first-peak difference')
ax.plot(DS,slope*DS,linestyle='--',label='Unfitted first-order prediction')
ax.axhline(0,lw=.7);ax.axvline(.17533285,linestyle=':',lw=1,label='Nonzero crossover ≈ 0.17533')
ax.set_xlabel('Defect asymmetry δ, with potentials (0.775±δ)');ax.set_ylabel('First-peak time difference');ax.set_title('Nonperturbative sign reversal despite fixed transmission on swapping');ax.legend()
fig.savefig(HERE/'nonlinear_phase_peak_shift.png',dpi=180);plt.close(fig)
# scattering phase with probability unchanged
kk=np.linspace(.15,np.pi-.15,601)
ta,ra=scattering(kk,1.2,.35);tb,rb=scattering(kk,.35,1.2)
fig,axs=plt.subplots(2,1,figsize=(8.5,7),sharex=True,constrained_layout=True)
axs[0].plot(kk,abs(ta)**2,label='Transmission |t|², both orientations');axs[0].plot(kk,abs(ra)**2,label='Reflection |r|², both orientations')
axs[0].legend();axs[0].set_ylabel('Probability')
axs[1].plot(kk,np.unwrap(np.angle(ra/rb)),label='arg[rL(u,v)/rL(v,u)]')
axs[1].set_xlabel('Momentum k');axs[1].set_ylabel('Relative reflection phase [rad]');axs[1].legend();axs[1].set_title('Reflection-phase contrast without changing scattering probabilities')
fig.savefig(HERE/'equal_transmission_scattering_phase.png',dpi=180);plt.close(fig)
(HERE/'nonlinear_scan.json').write_text(json.dumps({'L':L,'m':m,'w':w,'dt':.05,'delta':DS.tolist(),'shift':shifts,'peaks':peakpairs,'linear_slope':slope},indent=2))
print('Generated three figures and sign reversal scan')
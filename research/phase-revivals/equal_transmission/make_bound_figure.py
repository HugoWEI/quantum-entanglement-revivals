import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase_equal_experiment import smooth_curve
HERE=Path(__file__).resolve().parent
source=np.load(HERE/'equal_transmission_L48_dt0.05.npz')
Q=np.load(HERE/'bound_free_curve_L48.npz')
t=Q['time']
c1=source['staggered_m2.5_u1.2_v0.35'];c2=source['staggered_m2.5_u0.35_v1.2']
b1=Q['forward'];b2=Q['reverse']
fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True)
axs[0].plot(t,smooth_curve(t,c1),label='Original preparation, forward')
axs[0].plot(t,smooth_curve(t,c2),label='Original preparation, reverse')
axs[0].set_ylabel('Smoothed capacity');axs[0].set_title('Original staggered-mass preparation')
axs[0].legend()
axs[1].plot(t,smooth_curve(t,b1),label='Common bound-free preparation, forward')
axs[1].plot(t,smooth_curve(t,b2),label='Common bound-free preparation, reverse')
axs[1].set_title('Same projected initial state for both orientations; no occupied bound modes')
axs[1].set_ylabel('Smoothed capacity');axs[1].set_xlabel('Time (hopping units)');axs[1].legend()
axs[1].set_xlim(22.5,31)
fig.savefig(HERE/'bound_free_phase_control.png',dpi=180);plt.close(fig)
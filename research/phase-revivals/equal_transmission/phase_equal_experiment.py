"""Equal-transmission / unequal-reflection-phase controlled capacity experiment.

Exact postquench finite-chain diagonalization is used as an INDEPENDENT diagnostic,
NOT evidence for a closed-form phase-to-peak law. Gaussian smoothing is the
normalized discrete kernel on the declared full time grid.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.special import logsumexp

OUT=Path(__file__).resolve().parent

def hop(L):
    h=np.zeros((L,L),float);i=np.arange(L-1);h[i,i+1]=h[i+1,i]=-1;return h

def hin(L,m,mode='staggered'):
    j=np.arange(L)
    if mode=='staggered':mass=m*(-1.)**j
    elif mode=='mirror':mass=m*(-1.)**np.minimum(j,L-1-j)
    else:raise ValueError(mode)
    return hop(L)+np.diag(mass)

def occupied(L,m,mode='staggered'):
    e,V=np.linalg.eigh(hin(L,m,mode));return V[:,:L//2]

def defect_h(L,u,v):
    h=hop(L);h[L//2-1,L//2-1]=u;h[L//2,L//2]=v;return h

def f(p):
    p=np.clip(p,0,1);ans=np.zeros_like(p)
    mask=(p>0)&(p<1);v=p[mask];ans[mask]=v*(1-v)*np.log((1-v)/v)**2
    return ans

def capacity_evolution(L,m,u,v,times,mode='staggered'):
    Q=occupied(L,m,mode)
    E,V=np.linalg.eigh(defect_h(L,u,v));A=V.T@Q
    out=np.zeros(len(times))
    for k,t in enumerate(times):
        XA=V[:L//2]@(np.exp(-1j*E*t)[:,None]*A)
        eig=np.linalg.eigvalsh(XA.conj().T@XA)
        out[k]=sum(f(eig))
    return out

def scattering(k,u,v):
    z=np.exp(1j*np.asarray(k))
    den=1-z*z+(u+v)*z+u*v*z*z
    return (1-z*z)/den, -z*(u+v*z*z+u*v*z)/den

def gaussian_weights(t, x, sigma=2):
    a=-.5*((t-x)/sigma)**2;a-=np.max(a);w=np.exp(a);w/=sum(w)
    d=(t-x)/sigma**2;mu=w@d
    return w,w*(d-mu)

def smooth_curve(t,c,sigma=2):
    # untruncated normalized Gaussian on a discrete time grid
    zz=(t[:,None]-t[None,:])/sigma
    w=np.exp(-.5*zz*zz);w/=w.sum(axis=1,keepdims=True)
    return w@c

def peak(t,c,L,lo=None,hi=None,sigma=2):
    if lo is None:lo=.4*L
    if hi is None:hi=min(1.3*L,t[-1]-2)
    z=smooth_curve(t,c,sigma)
    inds,_=find_peaks(z)
    indices=[int(i) for i in inds if lo<t[i]<hi]
    if not indices:return {'peak':None,'height':None,'prominence':None}
    good=[]
    for ix in indices:
        rad=max(2,int(.15*L/(t[1]-t[0])))
        left=z[max(ix-rad,0):ix+1];right=z[ix:min(ix+rad+1,len(z))]
        prom=float(min(z[ix]-left.min(),z[ix]-right.min()))
        if prom>=.08:good.append((ix,prom))
    if not good:return {'peak':None,'height':None,'prominence':None}
    ix,pr=good[0]
    def derivative(x):return gaussian_weights(t,x,sigma)[1]@c
    peakx=brentq(derivative,t[ix-1],t[ix+1],xtol=1e-12)
    height=gaussian_weights(t,peakx,sigma)[0]@c
    return {'peak':float(peakx),'height':float(height),'prominence':pr}

def run(L=48,dt=.05,m=2.5,extra=False):
    T=.75*L if L<=48 else .76*L
    times=np.arange(0,T+.001*dt,dt)
    # Standard orientation, hold-out and symmetric controls
    defects=[(1.2,.35),(.35,1.2),(.8,.6),(.6,.8),(.5,.2),(.2,.5)]
    modes=['staggered']
    if extra:modes+=['mirror']
    cases=[];start=time.time()
    for mode in modes:
      masses=[m,-m] if mode=='staggered' else [m]
      for mass in masses:
        for u,v in defects:
            curve=capacity_evolution(L,mass,u,v,times,mode)
            pk=peak(times,curve,L)
            cases.append({'L':L,'dt':dt,'m':mass,'mode':mode,'u':u,'v':v,**pk,'curve':curve})
            print(f'L={L} {mode} m={mass:+g} {u:g},{v:g}: peak {pk["peak"]} height {pk["height"]}',flush=True)
    # wave number scattering diagnostics
    ks=np.linspace(.15,np.pi-.15,401)
    ta,ra=scattering(ks,*defects[0]);tb,rb=scattering(ks,*defects[1]);
    k0=np.pi/2
    t0,r0=scattering(k0,*defects[0]);t1,r1=scattering(k0,*defects[1]);
    symmetry=[]
    def get(mode,m,u,v):return next(x for x in cases if x['mode']==mode and x['m']==m and x['u']==u and x['v']==v)
    for u,v in defects[::2]:
        a=get('staggered',m,u,v);b=get('staggered',m,v,u)
        am=get('staggered',-m,u,v);bm=get('staggered',-m,v,u)
        symmetry.append({'pair':[u,v],'peak_delta_mplus':a['peak']-b['peak'], 'peak_delta_mminus':am['peak']-bm['peak'],
            'parity_max_curve_residual':float(max(np.max(abs(a['curve']-bm['curve'])),np.max(abs(b['curve']-am['curve']))))})
    if extra:
        for u,v in defects[::2]:
            a=get('mirror',m,u,v);b=get('mirror',m,v,u)
            symmetry.append({'pair':[u,v],'mirror_state_curve_max_difference':float(np.max(abs(a['curve']-b['curve']))), 'mirror_peak_delta':float(a['peak']-b['peak']) if a['peak'] and b['peak'] else None})
    output={'L':L,'dt':dt,'T':float(times[-1]),'m':m,'scattering':{'max_transmission_amplitude_error_swap':float(np.max(abs(ta-tb))),
      'max_transmission_probability_error_swap':float(np.max(abs(abs(ta)**2-abs(tb)**2))),
      'max_reflection_probability_error_swap':float(np.max(abs(abs(ra)**2-abs(rb)**2))),
      'k_pi_over_2':{'phase_difference_radians':float(np.angle(r0/r1)),'T':float(abs(t0)**2), 'R':float(abs(r0)**2), 'r_forward_phase':float(np.angle(r0)), 'r_reverse_phase':float(np.angle(r1))}},
      'cases':[{k:v for k,v in x.items() if k!='curve'} for x in cases],'comparisons':symmetry,'runtime_s':time.time()-start}
    np.savez_compressed(OUT/f'equal_transmission_L{L}_dt{dt}.npz',time=times,**{f'{x["mode"]}_m{x["m"]}_u{x["u"]}_v{x["v"]}':x['curve'] for x in cases})
    (OUT/f'results_L{L}_dt{dt}.json').write_text(json.dumps(output,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=48);p.add_argument('--dt',type=float,default=.05);p.add_argument('--extra',action='store_true')
    args=p.parse_args();x=run(L=args.L,dt=args.dt,extra=args.extra);print(json.dumps({'comparisons':x['comparisons'],'scattering':x['scattering'],'runtime':x['runtime_s']},indent=2))
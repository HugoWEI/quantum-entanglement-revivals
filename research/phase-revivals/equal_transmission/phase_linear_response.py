"""Parameter-free first-order asymmetry response at a symmetric two-site defect.

No asymmetric spectrum is used in predicted slope, only H(w,w) and its Frechet derivative.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from phase_equal_experiment import occupied,defect_h,f,gaussian_weights,peak,capacity_evolution

OUT=Path(__file__).resolve().parent

def fprime(p):
    p=np.clip(p,1e-12,1-1e-12)
    log=np.log1p(-p)-np.log(p)
    return (1-2*p)*log*log-2*log

def response(L,m,w,times):
    Q=occupied(L,m)
    H=defect_h(L,w,w)
    E,V=np.linalg.eigh(H)
    A=V.T@Q
    J=np.zeros((L,L));J[L//2-1,L//2-1]=1;J[L//2,L//2]=-1
    JV=V.T@J@V
    dE=E[:,None]-E[None,:]
    C0=np.zeros(len(times));C1=np.zeros(len(times))
    for idx,t in enumerate(times):
        phases=np.exp(-1j*E*t)
        dd=np.empty((L,L),complex)
        np.divide(phases[:,None]-phases[None,:],dE,out=dd,where=np.abs(dE)>1e-10)
        near=np.abs(dE)<=1e-10
        if np.any(near):dd[near]=(-1j*t*np.broadcast_to(phases[:,None],(L,L)))[near]
        X=V@(phases[:,None]*A)
        dX=V@((dd*JV)@A)
        X=X[:L//2];dX=dX[:L//2]
        G=X@X.conj().T
        p,U=np.linalg.eigh(G)
        C0[idx]=np.sum(f(p))
        dG=dX@X.conj().T+X@dX.conj().T
        C1[idx]=float(np.real(np.sum(fprime(p)*np.diag(U.conj().T@dG@U))))
    return C0,C1

def predict(L,m,w,deltas,dt=.05):
    T=.75*L if L<=48 else .76*L
    t=np.arange(0,T+.001*dt,dt)
    C0,C1=response(L,m,w,t)
    ref=peak(t,C0,L)
    if ref['peak'] is None:
        raise RuntimeError('no symmetric reference peak')
    tp=ref['peak']
    def smooth_deriv(x,c,order):
        wt,dwt=gaussian_weights(t,x,2.)
        if order==1:return dwt@c
        u=(t-x)/4;mu=wt@u
        wp=wt*(u-mu)
        # derivative w' as w*(u-meanu), derivative u=-1/4, derivative meanu=E[(u-mu)^2]-1/4
        return (wt*((u-mu)**2-np.sum(wt*(u-mu)**2)))@c
    c1der=smooth_deriv(tp,C1,1)
    c0second=smooth_deriv(tp,C0,2)
    slope=-2*c1der/c0second
    rows=[]
    for delta in deltas:
        plus=capacity_evolution(L,m,w+delta,w-delta,t)
        minus=capacity_evolution(L,m,w-delta,w+delta,t)
        pp=peak(t,plus,L);pm=peak(t,minus,L)
        true=pp['peak']-pm['peak']
        rows.append({'delta':delta,'predicted_shift':float(delta*slope),'observed_shift':float(true),
                     'error':float(true-delta*slope),'ratio_pred_to_obs':float(delta*slope/true) if abs(true)>1e-7 else None,
                     'plus_peak':pp['peak'],'minus_peak':pm['peak']})
        print(f'L={L} m={m} w={w} delta={delta} predicted {delta*slope:.7f} actual {true:.7f} err {true-delta*slope:.7f}',flush=True)
    result={'L':L,'m':m,'w':w,'dt':dt,'base_peak':tp,'response_coefficient_shift_per_delta':float(slope),
            'C1_derivative':float(c1der),'C0_second_derivative':float(c0second),'rows':rows}
    np.savez_compressed(OUT/f'linear_response_L{L}_w{w}.npz',t=t,C0=C0,C1=C1)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=48);p.add_argument('--w',type=float,default=.775);p.add_argument('--m',type=float,default=2.5);p.add_argument('--dt',type=float,default=.05)
    z=p.parse_args()
    rows=predict(z.L,z.m,z.w,[.025,.05,.1,.2,.425] if z.w==.775 else [.02,.05,.1,.2],z.dt)
    (OUT/f'linear_response_L{z.L}_w{z.w}.json').write_text(json.dumps(rows,indent=2))
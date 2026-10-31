"""Common-initial-state bound-mode ablation; same Q is used for both potential orders.

The projected initial state is nonlocal and NOT the staggered-mass ground state.
Therefore this is a counterfactual control, not a literal experiment at fixed original quench.
"""
import json, argparse
from pathlib import Path
import numpy as np
from phase_equal_experiment import occupied,defect_h,peak,f
HERE=Path(__file__).resolve().parent

def remove_bound_modes(L,m,u,v):
    Q=occupied(L,m)
    modes=[];data={}
    for pair in [(u,v),(v,u)]:
        E,W=np.linalg.eigh(defect_h(L,*pair))
        bound=np.where(abs(E)>2)[0]
        data[str(pair)]={'bound_eigenvalues':E[bound].tolist(),
                         'bound_occupations_original':[float(np.linalg.norm(W[:,i].T@Q)**2) for i in bound],
                         'bound_left_weights':[float(np.sum(W[:L//2,i]**2)) for i in bound]}
        modes.extend(W[:,i] for i in bound)
    if not modes: raise RuntimeError('No bound modes, no ablation to perform')
    V=np.column_stack(modes)
    U,s,_=np.linalg.svd(V,full_matrices=False)
    U=U[:,:np.count_nonzero(s>1e-9)]
    Z=Q-U@(U.T@Q)
    A,ss,B=np.linalg.svd(Z,full_matrices=False)
    Qp=A@B
    data['common_control']={'removed_subspace_dim':U.shape[1],
       'min_projection_singular_value':float(ss.min()),
       'max_bound_occupation_after':float(np.linalg.norm(U.T@Qp,2)**2),
       'same_projected_orbitals_for_both_defects':True,
       'initial_state_is_local_staggered_ground_state':False}
    return Qp,data

def capacity_from_orbitals(L,Q,u,v,t):
    E,V=np.linalg.eigh(defect_h(L,u,v))
    A=V.T@Q
    out=np.empty(len(t))
    for k,x in enumerate(t):
        XA=V[:L//2]@(np.exp(-1j*E*x)[:,None]*A)
        out[k]=np.sum(f(np.linalg.eigvalsh(XA.conj().T@XA)))
    return out

def run(L=48,dt=.05,m=2.5):
    T=.75*L if L<=48 else .76*L;t=np.arange(0,T+.001*dt,dt)
    rows=[]
    for u,v in [(1.2,.35),(.8,.6),(.5,.2)]:
        Qp,data=remove_bound_modes(L,m,u,v)
        ca=capacity_from_orbitals(L,Qp,u,v,t)
        cb=capacity_from_orbitals(L,Qp,v,u,t)
        pa,pb=peak(t,ca,L),peak(t,cb,L)
        row={'L':L,'m':m,'u':u,'v':v,'peak_forward':pa,'peak_reverse':pb,
             'shift':None if pa['peak'] is None or pb['peak'] is None else pa['peak']-pb['peak'],
             'bound_state_data':data}
        rows.append(row)
        print('L',L,'pair',(u,v),'bound-free peak shift',row['shift'],'original bound occupancies',
              [data[str(p)]['bound_occupations_original'] for p in [(u,v),(v,u)]],flush=True)
        if u==1.2:
            np.savez_compressed(HERE/f'bound_free_curve_L{L}.npz',time=t,forward=ca,reverse=cb)
    (HERE/f'bound_control_L{L}.json').write_text(json.dumps(rows,indent=2))
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--L',type=int,default=48);a=p.parse_args();run(a.L)
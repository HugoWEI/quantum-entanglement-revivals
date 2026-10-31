import unittest
import numpy as np
from phase_equal_experiment import scattering,capacity_evolution,occupied,defect_h,peak
from phase_linear_response import response

class EqualTransmissionTests(unittest.TestCase):
    def test_scattering_unitarity_and_probability_matching(self):
        ks=np.linspace(.2,2.9,101)
        for u,v in [(1.2,.35),(.5,.2),(.8,.6)]:
            t,r=scattering(ks,u,v)
            tt,rr=scattering(ks,v,u)
            self.assertLess(np.max(abs(t-tt)),1e-14)
            self.assertLess(np.max(abs(abs(r)**2-abs(rr)**2)),1e-14)
            self.assertLess(np.max(abs(abs(t)**2+abs(r)**2-1)),1e-13)
        self.assertGreater(abs(np.angle(scattering(np.pi/2,1.2,.35)[1]/scattering(np.pi/2,.35,1.2)[1])),2)

    def test_isospectral_reflection(self):
        for L in (12,24,48):
            self.assertLess(np.max(abs(np.linalg.eigvalsh(defect_h(L,1.2,.35))-
                                        np.linalg.eigvalsh(defect_h(L,.35,1.2)))),1e-12)

    def test_parity_duality_and_mirror_control(self):
        L=24;t=np.arange(0,19.01,.3)
        a=capacity_evolution(L,2.5,1.2,.35,t)
        b=capacity_evolution(L,-2.5,.35,1.2,t)
        self.assertLess(max(abs(a-b)),1e-11)
        s=capacity_evolution(L,2.5,1.2,.35,t,'mirror')
        ss=capacity_evolution(L,2.5,.35,1.2,t,'mirror')
        self.assertLess(max(abs(s-ss)),1e-11)

    def test_frechet_response_unfitted(self):
        L=24;w=.775;t=np.array([1.,3.1,8.2,10.7]);C0,C1=response(L,2.5,w,t)
        eps=1e-4
        Cp=capacity_evolution(L,2.5,w+eps,w-eps,t)
        Cm=capacity_evolution(L,2.5,w-eps,w+eps,t)
        self.assertLess(max(abs((Cp-Cm)/(2*eps)-C1)),2e-6)
        self.assertLess(max(abs((Cp+Cm)/2-C0)),2e-6)

    def test_peak_sign_reversal(self):
        L=48;t=np.arange(0,36.0001,.1);w=.775
        def difference(d):
            pa=peak(t,capacity_evolution(L,2.5,w+d,w-d,t),L)['peak']
            pb=peak(t,capacity_evolution(L,2.5,w-d,w+d,t),L)['peak']
            return pa-pb
        self.assertGreater(difference(.1),.05)
        self.assertLess(difference(.2),-.02)

if __name__=='__main__':unittest.main()
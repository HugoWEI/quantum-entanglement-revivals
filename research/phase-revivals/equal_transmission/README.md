# Equal-transmission, different-reflection-phase modular-capacity revivals

**Controlled research experiment — 31 October 2026.** This is a reproducible numerical result and symmetry analysis, **not a claim of a new physics law or publication-ready paper**. Unlike previous Airy accuracy studies, this experiment directly tests the question whether transmission probabilities alone determine the first modular-capacity revival.

## Physical model and strictly identical transmission

Even open chain of `L` sites; hopping = 1, ℏ = 1. For the post-quench single-particle Hamiltonian,

\[
h_f(u,v)=-\sum_{j=0}^{L-2}(|j\rangle\langle j+1|+\mathrm{h.c.})+u|a\rangle\langle a|+v|b\rangle\langle b|,
\quad a=L/2-1,\;b=L/2.
\]

The initial occupied orbitals are the lowest `L/2` eigenvectors of the uniform hopping Hamiltonian with alternating mass `m(-1)^j`. The reported observable is the left-half Gaussian modular capacity

\[
C_A(t)=\operatorname{Tr}f(G_A(t)),\qquad f(p)=p(1-p)\log^2[(1-p)/p], \quad G_A=X_A X_A^\dagger.
\]

For an infinite tight-binding lattice with the two adjacent potentials, set `z=exp(ik)`, `E=-2cos(k)`, and use the same left incoming wave and asymptotic origin for both orientations. Matching at the two impurity sites gives

\[
t(k;u,v)=\frac{1-z^2}{1-z^2+(u+v)z+uvz^2},\quad
r_L(k;u,v)=-\frac{z(u+vz^2+uvz)}{1-z^2+(u+v)z+uvz^2}.
\]

Thus **the complex transmission amplitude itself is exactly invariant** under `(u,v) -> (v,u)` at every momentum, and so is the reflection probability. But left-reflection phase generally changes. At `k=π/2`, `(u,v)=(1.20,0.35)`, `T=0.816509828737`, reflection-phase difference `arg[r_L(u,v)/r_L(v,u)] = -2.223731327885 rad`. Reflection phases should always be specified relative to a common scattering-origin convention.

## Strict parity-control selection rule

Let `R` reverse site indices. Then `R h_f(u,v)R=h_f(v,u)`. For even `L`, the staggered-mass preparation transforms as `R h_i(m)R=h_i(-m)`. Because the evolving Gaussian many-body state is pure, the modular capacity is the same on either half. Therefore, for all times,

\[
\boxed{C_A(t;u,v,m)=C_A(t;v,u,-m).}
\]

If the *initial* Hamiltonian is reflection-symmetric (the additional `mirror` option), all capacities are identical under swapping the two impurity potentials. These controls are important: the observed phase-sensitive contrast requires initial-state reflection asymmetry. **Swapped defects are parity-related, so this test alone does not isolate an independently new phase-only law.** Differences can involve how reflection-phase-dependent scattering states and bound-state profiles overlap the initially occupied subspace.

## Direct exact finite-chain benchmark

No Airy surrogate is used to establish the effect. Each finite post-quench Hamiltonian is diagonalized independently. The capacity is evaluated from `L/2` occupied Gaussian modes. The **operational peak** is the first qualifying local maximum of a normalized discrete Gaussian convolution with `σ_t=2` on `t∈[0,0.75L]` (`0.76L` at L64/80), `dt=0.05`; `prominence≥0.08` within a radius `.15L`, candidate window `[0.4L,1.3L]`. A continuous peak position is obtained as the root of the derivative of that *discrete* smoother. The truncation to the finite sampled time window is part of the observable convention.

| L | `t_peak(1.2,.35)` | `t_peak(.35,1.2)` | Peak difference (forward − reverse) |
|---:|---:|---:|---:|
|48|26.422680|27.311945|−0.889265|
|64|36.476575|37.049139|−0.572564|
|80|45.609595|46.884406|−1.274811|

Mirror-mass control: reversal `m→−m` flips the sign of the peak difference, while full raw capacity curves agree after the combined parity transformation within `3.1e−11` at L≤80. An explicitly reflection-symmetric *initial* mass produces raw capacity curves invariant under the defect swap within `6.5e−12` at L48. At those parameters, some symmetric-preparation curves have no qualifying first maximum under the chosen prominence filter; no peak location is asserted for them.

Additional held-out defect orientations at `L48`: `(0.80,0.60)` versus reverse gives `−0.011829`, while `(0.50,0.20)` versus reverse gives `+0.074433`. At `L64`, the same shifts are `+0.012225` and `+0.006104`, respectively. The shift depends nontrivially on `L`, potentials and preparation.

## Nonlinear sign reversal at *fixed mean potential*

Set `(u,v)=(w+δ,w−δ)` and its reverse, with `w=0.775` and staggered mass `m=2.5` at L48. The exact first-peak difference is **positive for small `δ`**, crosses zero at the nonzero `δ≈0.175333` and is negative beyond. At `δ=0.425`, the peak difference reaches `−0.889265`. The first qualifying peaks stay distinguishable, retain the required prominence, and have different heights even at the time-crossing. The crossover is a numerical root, **not a certified numerical interval or analytic critical point**.

## Parameter-free linear-response prediction and its limitation

Perturb the symmetric Hamiltonian `h_f(w,w)` by `δD`, where `D=|a><a|−|b><b|`. With `U_δ=e^{-ih_δt}` and initial projector `P_i`, compute `∂_δ U|_0` exactly from the divided difference of the matrix exponential. This gives `C(t,δ)=C_0(t)+δ C_1(t)+O(δ²)` and a parameter-free small-asymmetry peak-difference formula:

\[
\boxed{\Delta t_{\mathrm{peak}}(\delta)
=-2\delta\frac{\partial_t\overline C_1(t_0)}{\partial_t^2\overline C_0(t_0)}+O(\delta^3),}
\]

where `t0` is the unperturbed peak for the *same discrete smoother*. The odd nature follows by swapping `δ→−δ` if both peaks remain simple.

For `L48,w=.775`, predicted and actual shifts: `δ=.025`: `+.039225` vs `+.037932`; `δ=.05`: `+.078449` vs `+.067419`; `δ=.10`: `+.156898` vs `+.074556`; `δ=.425`: `+.666817` vs `−.889265`. At large asymmetry, **the perturbative formula fails even in sign**. There is no fitted parameter, and this failure is a useful falsification of a naive Wigner-delay/first-order description, but not a new general theorem.

## Reproduction

```bash
python -m pip install numpy scipy matplotlib
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_equal_transmission
OPENBLAS_NUM_THREADS=1 python phase_equal_experiment.py --L 48 --dt .05 --extra
OPENBLAS_NUM_THREADS=1 python phase_equal_experiment.py --L 64 --dt .05
OPENBLAS_NUM_THREADS=1 python phase_equal_experiment.py --L 80 --dt .05
OPENBLAS_NUM_THREADS=1 python phase_linear_response.py --L 48 --w .775
OPENBLAS_NUM_THREADS=1 python make_phase_figures.py
```

Five regression tests cover equal complex transmission, scattering unitarity, parity duality and symmetric control, direct Fréchet linear-response verification, and sign reversal. The `.npz` contain raw curves; JSON files contain peak and diagnostic values. The sample set is a finite numerical study, not a thermodynamic scaling argument.

## Existing work, novelty boundary, next experiment

Capizzi & Eisler, *SciPost Phys.* **14**, 070 (2023), https://doi.org/10.21468/SciPostPhys.14.4.070, already derive coherent reflected/transmitted quasiparticle revivals (their Sec. 3.4 and Eq. 64), emphasize sensitivity to reflection symmetry of the initial staggering (Appendix C), and explicitly identify momentum-dependent scattering as a challenging extension in their Discussion. We therefore **do not claim that coherent revivals, parity-sensitive interference, or phase dependence are new**. Capacity dynamics are also studied independently in the literature (e.g. harmonic chains: Phys. Rev. D 108, 125013 (2023)).

The sign-reversal crossover might be scientifically informative if we can derive it from a **controlled, gauge-invariant path-phase interference expression**, validate its dependence on gap and chain size, separate any localized bound-state contributions, and compare directly with the published coherent-quasiparticle ansatz. To avoid a trivial mirror-equivalence result, a further experiment should seek **non-parity-related isotransmission defects** or an experimentally well-defined phase-controlled family. Until then, this is a solid proof-of-principle computational result, not a publication-ready discovery.
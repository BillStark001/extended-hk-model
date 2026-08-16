# Joint alpha/q linear spectrum

This note records the Section 3.6 extension from the scalar opinion spectrum to
the complete opinion--edge tangent sector of the periodic pair closure. It does
not define or solve a finite-system committor.

## Moving translation-invariant base

Let the periodic opinion domain have length (L), let
(ho_0=1/L), and write (s=y-x). The old scalar calculation used

\[
E_0(x,y)=\bar k\rho_0^2.
\]

This is invariant when (q=0), but it is not stationary when (q>0): it has
both discordant edges and concordant rewiring candidates. The lowest closed
base is instead

\[
E_b(x,y,t)=\bar k\rho_0 n_b(s;d(t)),
\]

with initial discordant-neighbor mass (d_0) and

\[
n_b(s;d)=H_\epsilon(s)L^{-1}
 +(d_0-d)g(s)
 +\frac{d}{d_0}[1-H_\epsilon(s)]L^{-1}.
\]

Here (g) is the normalized concordant recommendation gain. Under the
fixed-out-degree finite-list closure,

\[
\dot d=-\frac{q C_{\rm rec}}{\bar k}
       \left[1-(1-d)^{\bar k}\right],
\]

where (C_{\rm rec}) is the probability that the finite recommendation list
contains at least one concordant item. Thus (q) changes both the tangent
operator directly and the base state on which it acts.

## Complete source-Fourier sector

For source mode (m), with (kappa_m=2\pi m/L), use

\[
\delta\rho=\rho_0 r_m e^{i\kappa_mx},
\qquad
\delta E=\bar k\rho_0e^{i\kappa_mx}
            [n_b(s)r_m+h_m(s)],
\qquad
\int h_m(s)\,ds=0.
\]

The (n_b r_m) term enforces the fixed-out-degree row constraint. Linearizing
the complete periodic density/edge right-hand side gives

\[
\frac{d}{dt}
\begin{pmatrix}r_m\\h_m\end{pmatrix}
=A_m[d(t);\alpha,q]
\begin{pmatrix}r_m\\h_m\end{pmatrix}.
\]

Because opinion transport is linear in (alpha), rewiring flux is linear in
(q), and diffusion is independent of both, the discretized matrix is

\[
A_m(d;\alpha,q)=A_m^{(0)}(d)
 +\alpha A_m^{(\alpha)}(d)
 +q A_m^{(q)}(d).
\]

The implementation constructs these three bases by centered Fréchet
differentiation of the full pair RHS. A direct-Jacobian regression gives a
relative reconstruction error below (10^{-8}) (typically about
(5\times10^{-11})).

## Two spectra are required

For a frozen value of (d), the instantaneous joint spectrum is

\[
\sigma_m(d;\alpha,q)=\sigma[A_m(d;\alpha,q)].
\]

These eigenvalues are local diagnostics. Except at a genuine equilibrium, they
are not autonomous long-time growth exponents.

Along the moving base, define the time-ordered propagator

\[
U_m(T)=\mathcal T\exp\left(
\int_0^T A_m[d(t);\alpha,q]dt
\right).
\]

Its singular spectrum is computed in the physical norm

\[
\lVert(r,h)\rVert^2
=|r|^2+\int|h(s)|^2ds.
\]

The finite-time rates are

\[
\gamma_{m,j}(T)=T^{-1}\log\sigma_j[U_m(T)].
\]

The command uses a fixed (T=100) by default. This horizon is a Section 3.6
diagnostic chosen by the CLI; it is not imported from a first-passage or
terminal-outcome calculation.

## Current numerical result

The default scan uses a 64-point relative-edge grid, modes 1--20, three base
snapshots, (T=100), and 40 time slices. It covers Random, Opinion, and
OpinionM9 at

\[
\alpha\in\{0.005,0.05,0.3\},\qquad
q\in\{0,0.005,0.05,0.3\}.
\]

At (alpha=0.05), increasing (q) from 0 to 0.3 changes the fastest
finite-time joint branch as follows:

| recommender | (q=0) | (q=0.3) |
|---|---:|---:|
| Random | mode 11, 0.0662 | mode 11, 0.0516 |
| Opinion | mode 6, 0.0489 | mode 7, 0.0347 |
| OpinionM9 | mode 6, 0.0495 | mode 7, 0.0353 |

Thus the net effect of fast rewiring along the evolving base is to reduce the
fastest joint rate in these cases. This does not mean that (q) is a scalar
damping term: frozen operators at the same (d) show branch-dependent shifts,
and joint edge directions can remain amplifying while a rho-only seed decays.
For example, Random at ((\alpha,q)=(0.05,0.3)) gives

| mode | leading joint rate | rho-only seed rate |
|---:|---:|---:|
| 4 | 0.0121 | -0.00951 |
| 8 | 0.0265 | -0.00171 |

This is the mechanism missing from the scalar (q=0) spectrum.

The 20/40/80-slice check changes the selected leading rates by at most
(4.7\times10^{-5}). A Random 64/96-grid comparison gives median differences
around (10^{-3}) or smaller. For (alpha\le0.05) and modes 1--12, the
maximum 64/96 finite-time difference is 0.0055. High modes and
(alpha=0.3) are less converged; cells whose fastest mode lies at the mode-20
cutoff are hatched and must not be interpreted as a resolved global maximum.

## Reproduction

From the repository root:

```sh
PYTHONPATH=src:. python -m theory.mesoscopic.joint_spectrum
```

The complete frozen eigenvalues, finite-time singular values, summaries,
validation table, figures, and source-hashed metadata are written under
`outputs/theory/mesoscopic/joint_spectrum/`.

## Scope boundary

- The theory is complete within the Random/Opinion pair-closure tangent state
  ((r_m,h_m)), not within the exact microscopic graph state.
- Structure/top-k recommendation requires wedge/community perturbations and is
  intentionally excluded rather than silently reduced to Random.
- The calculation contains deterministic drift only. Event covariance blocks
  (Q_{\rho\rho},Q_{\rho E},Q_{EE}), joint first passage, community-count
  mapping, and all Section 3.7 code are absent.
- A positive singular rate says that some unit-norm joint perturbation is
  amplified. It does not state the probability with which a finite random
  graph supplies that perturbation.

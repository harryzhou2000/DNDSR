# AUSM numerical fluxes in Euler

Select `eulerSettings.riemannSolverType` as `AUSMPWPlus`, `AUSMPlusUp`, or
`AUSMPlusUp2`. The same names work for the existing auxiliary/wall flux selections.
The default selection and all Roe/HLL formulas are unchanged. These fluxes are in
the Euler module; this change does not add a CUDA EulerP implementation.

## Sources and definition boundaries

- Kim, Kim and Rho (2001), *Methods for the Accurate Computations of Hypersonic
  Flows: I. AUSMPW+ Scheme*, JCP 174, 38–80,
  [DOI](https://doi.org/10.1006/jcph.2001.6873).
  Original §3.1.2–3.1.3, equations (24)–(31), printed pp. 47–49.
  The later [ICCFD10 author-group restatement, §3.1](https://www.iccfd.org/iccfd10/papers/ICCFD10-016-Paper.pdf)
  also helps resolve a printing discrepancy in the original negative-flow branch.
  Its two-phase and AUSMPW+_N modifications are **not** used.
- Liou (2006), *A sequel to AUSM, Part II: AUSM+-up for all speeds*, JCP 214,
  137–170, [DOI](https://doi.org/10.1016/j.jcp.2005.09.020).
  Equations (18)–(20), (24), (29)–(30), and the final all-speed algorithm
  (69)–(77), especially printed pp. 144 and 155–156.
- Kitamura and Shima (2013), *Towards shock-stable and accurate hypersonic heating
  computations: A new pressure flux for AUSM-family schemes*, JCP 245, 62–83,
  [DOI](https://doi.org/10.1016/j.jcp.2013.02.046).
  Section 4.1, printed p. 73, and Appendix A, equations (A.2) and (A.4a–j),
  printed pp. 81–82. The cubic pressure function is (2.3f), printed p. 67.

The user supplied all three original journal PDFs on 2026-10-30. Their decisive
equations were checked both as extracted text and as rendered PDF pages. Initial
implementation used the 2018 and 2010 author-written restatements; the subsequent
original-paper audit independently compared 624 stationary 3-D face fluxes.
The unstructured stencil, moving grid and varying-thermodynamics treatments below
remain explicitly identified extensions.

## Shared variables and split polynomials

The left/right conservative states are $U_K=(\rho_K,\rho_K\mathbf v_K,E_K,\ldots)$,
where $K\in\{L,R\}$ and $E_K$ is energy **per volume**. Let $\mathbf n$ be a unit normal from left to right,
$\mathbf v_g$ the face-grid velocity, $u_K=(\mathbf v_K-\mathbf v_g)\cdot\mathbf n$,
$p_K$ pressure and $H_K=(E_K+p_K)/\rho_K$ total specific enthalpy. Write
$M_K=u_K/a_f$ using the interface sound-speed estimate $a_f$ defined per scheme.
The sign $s=+1$ denotes a positive split and $s=-1$ a negative split.

For $|M|\ge1$,

$$\mathcal M_s(M)=\tfrac12(M+s|M|),\qquad
\mathcal P_s(M;\alpha)=\tfrac12(1+s\operatorname{sign}M).$$

For $|M|<1$,

$$\mathcal M_{2,s}(M)=\frac{s}{4}(M+s)^2,$$
$$\mathcal M_{4,s}(M)=\mathcal M_{2,s}(M)+\frac{s}{8}(M^2-1)^2,$$
$$\mathcal P_s(M;\alpha)=\frac14(M+s)^2(2-sM)+s\alpha M(M^2-1)^2.$$

The numerical flux is assembled as

$$F=c_L U_L+c_R U_R+
(0,p_f\mathbf n,c_Lp_L+c_Rp_R+p_f\mathbf v_g\cdot\mathbf n).$$

This equation applies to the first $d+2$ gas components only; extra entries are
untouched, exactly as in the existing Roe/HLL kernels. Thus $c_K$ has velocity
units and gas energy transports total enthalpy. The grid-pressure-work
term makes identical states yield $F_{\mathrm{physical}}-U\mathbf v_g\cdot\mathbf n$.

## AUSMPW+

For a constant perfect-gas heat-capacity ratio $\gamma$, define static enthalpy
$h_K=\gamma p_K/[(\gamma-1)\rho_K]$ and

$$H_n=\tfrac12(h_L+h_R)+\tfrac14(u_L^2+u_R^2),\qquad
a_*^2=\frac{2(\gamma-1)}{\gamma+1}H_n.$$

Use $a_f=a_*^2/\max(a_*,|u_L|)$ if $u_L+u_R\ge0$, otherwise
$a_f=a_*^2/\max(a_*,|u_R|)$. Tangential kinetic energy is excluded from $H_n$.

Set $m^+=\mathcal M_{2,+}(M_L)$, $m^-=\mathcal M_{2,-}(M_R)$ and

$$p_f=\mathcal P_+(M_L;0)p_L+\mathcal P_-(M_R;0)p_R,$$
$$w=1-\min(p_L/p_R,p_R/p_L)^3,$$
$$q=\min\left(1,\frac{p_{\min,\mathrm{stencil}}}{\min(p_L,p_R)}\right)^2,
\qquad f_K=\begin{cases}(p_K/p_f-1)q,&p_f>0,\\0,&p_f=0.\end{cases}$$

The transverse factor $q$ is squared. There is no extra subsonic cutoff on $f_K$.
The pressure-polynomial coefficient is fixed to the more dissipative cubic value
$\alpha_p=0$. It must not be confused with LVR's penalty fraction. The original
paper explicitly allows $0\le\alpha_p\le3/16$ below equation (28), and identifies
zero as the more robust endpoint. Its summary (24) shows $3/16$, so zero is an
explicitly permitted selection, not a uniquely prescribed constant. Our initial
$3/16$ choice produced a growing acoustic mode near physical Mach 0.93 and large
Sod oscillations in independent O1/O2 and DNDSR tests. That result rejects this
coefficient for our tested configuration; it does not establish that the paper
disallows $3/16$. Both phases remain identifiable in the study.
For $m^++m^-\ge0$,

$$c_L=a_f\{m^++m^-[(1-w)(1+f_R)-f_L]\},\qquad
c_R=a_fm^-w(1+f_R).$$

For $m^++m^-<0$,

$$c_L=a_fm^+w(1+f_L),\qquad
c_R=a_f\{m^-+m^+[(1-w)(1+f_L)-f_R]\}.$$

The last line following original equation (24), printed p. 47, includes an
additional $+f_L$ in the negative-flow bracket. Taken literally, that expression
violates invariance under swapping L/R and reversing the normal. The expression
above is the symmetric counterpart of the positive-flow branch and agrees with
the same authors' later 2018 restatement, equation (18). This is a documented
interpretation of an apparent printing error, not a claim of a published erratum.

The published structured 2-D stencil uses four transverse neighbour pressures.
For an unstructured cell $i$, this implementation instead forms $p_{\min,i}$ as
the minimum **cell-average** pressure over $i$ and its face neighbours, then uses
$p_{\min,\mathrm{stencil}}=\min(p_{\min,L},p_{\min,R})$. At a physical boundary
only the interior neighbourhood is used. This is a geometry-independent extension,
not a claim of exact equivalence to the published transverse-only stencil.

`EvaluateDt` obtains both side pressures in its existing face loop. Its cell loop
reduces face minima into component 1 of `deltaLambdaCell`, next to component 0's
H-correction value. The existing persistent pull exchanges both together. RHS reads
the cached minimum at the same refresh times as dLambda. **No additional halo call**
is introduced. Without PW+, this array remains one component wide and no pressure
face buffer is allocated. A standalone two-state kernel can omit the stencil;
its default $q=1$ is the 1-D form, not a substitute for the 2-D neighbourhood.

## AUSM+-up

Use each side's sensible total enthalpy in the grid frame,
$H_{g,K}=h_K+|\mathbf v_K-\mathbf v_g|^2/2$, and set

$$a_{*,K}^2=\frac{2(\gamma-1)}{\gamma+1}H_{g,K},\qquad
a_f=\min\left(\frac{a_{*,L}^2}{\max(a_{*,L},u_L)},
\frac{a_{*,R}^2}{\max(a_{*,R},-u_R)}\right).$$

The signed denominators are intentional: Liou first gives absolute values in
equation (28), then replaces them by equation (30) to avoid an expansion shock.
The final algorithm in §3.3 selects equation (30), which is the form used here.
Define $\bar\rho=(\rho_L+\rho_R)/2$,
$\bar M^2=(M_L^2+M_R^2)/2$,
$M_0=\sqrt{\min(1,\max(\bar M^2,M_\infty^2))}$ and $f_a=M_0(2-M_0)$.
The setting `ausmReferenceMach` supplies $M_\infty>0$ (default 0.1); values above
one are harmless because of the clamp. It does not affect PW+ or Roe/HLL.
This default is an implementation choice, not a universally prescribed
freestream Mach number. For strong-pressure-wave/shock-tube use, Liou recommends
the basic $f_a=1$ form on printed p. 156; setting `ausmReferenceMach=1` selects it.
With the published $K_p=1/4$, $K_u=3/4$ and $\sigma=1$,

$$M_f=\mathcal M_{4,+}(M_L)+\mathcal M_{4,-}(M_R)
-\frac{K_p}{f_a}\max(1-\sigma\bar M^2,0)
\frac{p_R-p_L}{\bar\rho a_f^2}.$$

If $M_f\ge0$, $(c_L,c_R)=(a_fM_f,0)$, otherwise $(0,a_fM_f)$.
Set $\alpha=3(-4+5f_a^2)/16$,
$P_L=\mathcal P_+(M_L;\alpha)$ and $P_R=\mathcal P_-(M_R;\alpha)$. Then

$$p_f=P_Lp_L+P_Rp_R-K_uP_LP_R(\rho_L+\rho_R)f_aa_f(u_R-u_L).$$

## AUSM+-up2

Keep the preceding **mass flux**, including its cutoff and pressure diffusion.
Only replace the pressure flux. Set $P_L=\mathcal P_+(M_L;0)$,
$P_R=\mathcal P_-(M_R;0)$ and
$V=\sqrt{(|\mathbf v_L-\mathbf v_g|^2+|\mathbf v_R-\mathbf v_g|^2)/2}$. Use

$$p_f=\frac{p_L+p_R}{2}+\frac{P_L-P_R}{2}(p_L-p_R)
+V(P_L+P_R-1)\bar\rho a_f.$$

Do not append the old $K_u$ term: this replaces the complete pressure flux.
The replacement includes the cubic $\alpha=0$ pressure functions of SLAU2
(2.3f), rather than retaining the adaptive up pressure polynomial. Section 4.1
explicitly selects exactly the SLAU2 pressure (3.5); the adaptive coefficient
listed in Appendix (A.4g) is relevant to the parent up pressure, not this replacement.
The velocity magnitude $V$ uses all velocity components, not only the face-normal
velocity.

## Integration and limits of the evidence

`Gas.hpp` dispatches to `AUSM.hpp`; no Roe-average computation is needed for the
AUSM gas flux. Roe-specific entropy-fix, increment-flux and mean-eigenvalue settings
do not alter AUSM's published coefficients. The existing scalar-face path is used
even when batched Roe is enabled. The primary/auxiliary rotation framework remains
available. Euler's existing scalar Rusanov transport, two-equation turbulent normal
stress and viscous/source terms remain unchanged, including for AUSM selections.
This preserves the Roe integration contract; it is not a new mass-consistent
AUSM species-transport implementation.

For a thermodynamic closure with local pressure gamma $\gamma_{e,K}$, acoustic
gamma $\gamma_{a,K}$ and base energy density $B_K$, pressure uses
$p_K=(\gamma_{e,K}-1)(E_K-\rho_K|\mathbf v_K|^2/2-B_K)$. The sonic estimate uses
$h_K=\gamma_{e,K}p_K/[(\gamma_{e,K}-1)\rho_K]$ and replaces
$2(\gamma-1)/(\gamma+1)$ on each side by
$2\gamma_{a,K}(\gamma_{e,K}-1)/[\gamma_{e,K}(\gamma_{a,K}+1)]$.
PW+ averages these two side estimates. This frozen-thermodynamics extension preserves
identical-state consistency and full base-energy transport; it is not the original
variable-composition sonic construction and has not been validated for reacting flow.
In particular, this does not implement Kim et al.'s reacting-gas Prandtl relation
(89), printed p. 62. The existing passive-variable treatment also differs from the
common AUSM convection coefficients for species in their equation (81). Building
the EX models verifies compilation, not equivalence to that reacting-flow method.

The dispatcher uses the supplied **cell-mean** states for its returned wave
magnitudes and facial states for the flux, preserving the Roe calling convention.
The returned wave magnitudes are $|\bar u-a_{\max}|$, $|\bar u|$ and
$|\bar u+a_{\max}|$, where $\bar u=(u_L+u_R)/2$ and
$a_{\max}=\max(\sqrt{\gamma_{a,L}p_L/\rho_L},\sqrt{\gamma_{a,R}p_R/\rho_R})$.
They feed the existing approximate implicit preconditioner. They are not an exact
AUSM flux Jacobian. The timestep estimate retains the existing physical acoustic
bound. Flux positivity/robustness still depends on reconstruction, time integration
and the nonlinear solve; the implementation does not replace their existing guards.

The initial bounded CFD validation tests AUSMPW+ only. AUSM+-up/up2 receive algebraic
tests, not claims of cylinder or hypersonic validation. See the independent
`workspace/ausm_fluxes` study for exact build/configuration states and numerical evidence.

# Multispecies Viscous Transport {#multispecies_viscous_transport}

This guide specifies the temperature-gradient closure used by the extended
Euler solvers for multispecies flow. It applies whether chemistry is enabled
or not: composition-dependent thermodynamics and species diffusion are a
multispecies transport concern, not a reaction-only concern.

@tableofcontents

## Scope and API

The solver routes viscous flux evaluation through
`PhysicsProperties::viscousFluxIdealGas()`. For a single-species gas, that
method delegates completely to `Gas::ViscousFlux_IdealGas()`. For a
multispecies gas, it retains the bare helper for viscous stress and viscous
work only, then adds mixture heat conduction and mixture-averaged species
diffusion in the physics layer.

Do not call `Gas::ViscousFlux_IdealGas()` directly for multispecies flow. Its
temperature-gradient formula assumes a single calorically perfect gas; it
cannot represent composition derivatives of mixture internal energy. The bare
interface documents this restriction in `src/Euler/Gas.hpp`.

## Temperature gradient from conservative variables

For a mixture with mass fractions $Y_k$, define the mixture specific internal
energy

$$
e(T,Y)=\sum_k Y_k e_k(T),
$$

where $e_k(T)$ is the species specific internal energy on the same energy
reference as the conservative state. At frozen composition,

$$
c_v=\left.\frac{\partial e}{\partial T}\right|_Y.
$$

Differentiating the conserved internal-energy density gives

$$
\nabla(\rho e)
= e\nabla\rho + \rho c_v\nabla T
+ \rho\sum_{k=1}^{N_s-1}(e_k-e_{N_s})\nabla Y_k.
$$

The last species is inferred from $\sum_kY_k=1$, so only $N_s-1$ transported
mass fractions appear. Therefore the exact temperature gradient used in the
code is

$$
\nabla T =
\frac{
\nabla(\rho e)-e\nabla\rho
-\sum_{k=1}^{N_s-1}(e_k-e_{N_s})\rho\nabla Y_k
}{\rho c_v}.
$$

For each spatial direction, the conservative reconstruction supplies

$$
\nabla(\rho e)=\nabla(\rho E)
-\mathbf{u}\mathbin{\cdot}\nabla(\rho\mathbf{u})
+\tfrac12|\mathbf{u}|^2\nabla\rho,
\qquad
\rho\nabla Y_k=\nabla(\rho Y_k)-Y_k\nabla\rho.
$$

`PhysicsProperties::reactiveTemperatureGradient()` implements these two
identities directly from the reconstructed conservative gradient. Although its
historical name contains `reactive`, the closure is valid for any ideal
multispecies mixture represented by the configured `ChemicalSource`.

## Why the previous closure was wrong

The pressure relation can be written with a state-dependent equivalent gamma,

$$
p=(\gamma_{\mathrm{eq}}-1)\rho e.
$$

Differentiating it while holding $\gamma_{\mathrm{eq}}$ fixed omits its
temperature and composition derivatives. That shortcut is only valid for a
calorically perfect, fixed-composition gas. In a premixed flame, the omitted
composition contribution can materially bias conductive heat flux and hence
the flame-speed prediction.

The corrected implementation does not differentiate $\gamma_{\mathrm{eq}}$.
It differentiates the mixture internal-energy identity above, using species
internal energies and the thermodynamic mixture $c_v$, which includes the
correct temperature dependence at frozen composition.

## Flux composition and signs

With the solver convention

$$
F_{\mathrm{total}}=F_{\mathrm{inviscid}}-F_{\mathrm{viscous}},
$$

the multispecies wrapper adds conductive heat flux

$$
q_n=k\nabla T\mathbin{\cdot}\mathbf{n},
$$

and obtains mixture-averaged species fluxes $J_k$ from the transport model.
The species slots store $-J_k\mathbin{\cdot}\mathbf{n}$ and the energy slot
contains the corresponding enthalpy transport
$-\sum_k h_kJ_k\mathbin{\cdot}\mathbf{n}$. The diffusion implementation also
accounts for the composition dependence of the mixture gas constant in the
conductive-gradient conversion used by that path.

At an adiabatic wall the normal conductive contribution is suppressed. At an
impermeable wall the species-diffusion boundary treatment enforces zero normal
species flux according to the boundary contract.

## Verification and use

The unit coverage is in `test/cpp/Euler/test_PhysicsProperties.cpp`. For a
reactive or multispecies case, verify that the active evaluator calls
`PhysicsProperties::viscousFluxIdealGas()` and compare a resolved laminar-flame
speed against an independent Cantera reference. A speed comparison alone is
not a proof of the local closure, but it is a sensitive end-to-end regression
for conduction--diffusion--chemistry coupling.

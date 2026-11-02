# Native coupled DITR time equations

`Solver/FieldDITR.hpp` provides the U2R2 (mask0) equations used by the active
Euler `ImplicitHermite3SimpleJacobianDualStep::Step`. It is a synchronous native
CPU field component, separate from accepted-state publication and history.
Cauchy integration is pending. U2R1/U3R1 and the legacy extra temporal multigrid
acceleration are not implemented in this component.

For the initial, intermediate and endpoint states `U0,Um,Ue`, rates `F=dU/dt`
and intermediate fraction `a`, it requires both original residuals to vanish:

```text
H = (c0 U0+c1 Ue-Um)/dt + c2 F0+c3 Fe
I = (U0-Ue)/dt + w0 F0+w1 Fm+w2 Fe
c = [1-3a²+2a³, 3a²-2a³, a-2a²+a³, -a²+a³]
w = [1/2-1/(6a), -1/(6a(a-1)), 1/2+1/(6(a-1))]
```

The sequential correction solves the transformed intermediate equation
`H+theta*I=0` with the endpoint frozen, then refreshes `Fm` and solves `I=0`
with the intermediate state frozen. Default `a=.55,theta=.9146` follows the
legacy coefficients. The inner stage callback may solve its equation to a
chosen tolerance; the collective stopping callback must still test **both**
original equations after the updated rates are evaluated. Convergence of an
individual frozen-stage equation is insufficient.

The intermediate state has independent retained storage and an explicit
`StageFactory` produces its rate/solve/reset callbacks. The factory is invoked
lazily during Prepare, outside local-only allocation/construction. This lets a
distributed frontend construct its own persistent halo, reconstruction,
auxiliary and linear workspace for that allocation. Reusing an endpoint binding
implicitly would transfer the wrong data. Optional reset runs per trial (for
example to cold-start an intermediate polynomial). Factory/reset may neither
rebind nor alter the intermediate means. Rate callbacks may update ghosts but
must preserve owned inputs. Stage solvers write the provided state and preserve
their forcing. Exact ArrayStorage pins reject same-allocation/different-view
rebinding; these checks are outside the hot numerical loops.

Caller serializes access, agrees controls/callback order on all participating
ranks, retains callback resources and coordinates local errors before MPI.
Callbacks complete synchronously and must not retain temporary raw views.
Prepare changes only trial/scratch. False means cap/failed-stage rejection and
the object can be reused with a fresh trial. An exception can leave partial
scratch and marks this component failed; restore the physical trial and rebuild
the component before retry. It neither advances clocks nor commits accepted
history. Native GPU execution is explicitly rejected until kernels/adapters
exist; there is no CPU fallback.

Tests use fixed numeric coefficients and an independent dense2x2 solve for a
nonautonomous scalar problem, then test both original residuals, repeated
preparation, caps/retry, aliasing and callback mutation/rebinding. A separate
nonautonomous convergence sequence verifies the expected asymptotic order.
For `a=.55`, an independent exact coupled solve gives errors6.38437e-7,
9.58222e-8,1.29855e-8,1.68644e-9 on8/16/32/64steps: apparent orders2.736,
2.883,2.945 approach3. The earlier8→16 order test failed its2.8 bound even
though the implementation matched that independent oracle. The test now uses
32→64 and the stronger2.9 bound. At `a=.5`, Simpson symmetry yields order4.
These are building-block results, not a claim of connected Cauchy DITR support.

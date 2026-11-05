# Configuration registration across native and Python libraries

`DNDS_DECLARE_CONFIG` defines ordinary C++ value types and their JSON,
schema and validation operations. `ConfigSectionBuilder<T>` describes their
fields; `ConfigRegistry<T>` owns the resulting immutable metadata. The registry
is infrastructure, separate from a solver's numerical state or configuration
precedence rules.

Registration is lazy and transactional. The first caller builds a temporary
descriptor under `std::call_once`; successful registration publishes it in one
step. A throwing registration leaves no partial publication and may be retried.
The pending descriptor is thread-local. Recursion into the same type fails
explicitly; registering a different nested section remains supported. After
publication, reading metadata is safe concurrently.

## Shared-library boundary

Each linked module retains its own complete registry metadata. On compilers
with ELF-style visibility, `DNDS_CONFIG_LOCAL` keeps the registry, section
builder and generated registration functions hidden. Configuration values and
their JSON format remain public and unchanged. Callers in a native shared
library and a Python extension can independently serialize the same value.
No registry lookup or registration check is added to numerical kernels.

Keep registration callbacks local together with their pending descriptor and
once flag. Mixing default-visible registration callbacks with hidden metadata
is unsafe: a weak `std::call_once` trampoline can be selected from a different
module and populate that module's descriptor, while the caller publishes an
empty descriptor. Depending on the call order, valid keys can then be rejected
or a populated configuration can serialize as `null`. Exporting only the
registry template does not fix this, because a hidden consumer can still have
local metadata and a preemptable registration trampoline. Hiding registration
helpers alone is also insufficient with libstdc++'s default-visible trampoline.
The registry therefore calls `std::call_once` with a `void (*)(void*)` callback
and an opaque request pointer. This standard-library instantiation contains no
type-specific metadata access: it invokes the actual hidden local registration
function. The request borrows the callback only for the synchronous once call;
it allocates no heap storage. Successful publication, retry and recursion rules
remain unchanged.

Do not compensate by accepting `null`, suppressing unknown-key checks or
substituting defaults. The hidden-consumer regression
`dnds_cauchy_config_shared` calls the exported native resolver, serializes its
nondefault P3 controls in a hidden consumer, and resolves that JSON again.
The Python equation-operator test also checks the actual resolved P3 controls
before constructing a stage solver and a spatial FAS hierarchy. Existing
`dnds_config` tests cover concurrent publication, recursion and retry.

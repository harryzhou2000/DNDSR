#pragma once

#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"

namespace DNDS::Cauchy
{
    class Domain;
}

// Binding resource holders only; no alternative solver orchestration.
namespace DNDS::Cauchy::Bindings
{
    namespace py = pybind11;
    namespace R = Runtime;
    using Field = Algebra::DofField<real>;

    /// Binding-only resource owner, not another numerical controller. Keep the
    /// originating Domain Python object alive as well: its Simulation may own
    /// the mesh communicator. Numerical methods below call native operators.
    struct FiniteVolumeOperators
    {
        FiniteVolume fv;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        explicit FiniteVolumeOperators(const Domain &domain);
    };
    // Numerical workspace definitions deliberately remain opaque. Complete
    // holders let pybind retain them without exposing their implementation.
    struct Workspace
    {
        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction;
        std::shared_ptr<FiniteVolume::Workspace> value;
    };
    struct MeanWorkspace
    {
        std::shared_ptr<FiniteVolume::MeanWorkspace> value;
    };
    /// Native correction controller with sharing plan/state bindings. Runtime
    /// lifetime is checked through its managed stream, never an engine address.
    /// Python keep_alive remains for compatibility with the frontend owner chain.
    struct NonlinearStageSolver
    {
        std::optional<FiniteVolumeOperators> owner;
        std::optional<EquationDofs> state;
        R::ExecutionStream stream;
        DomainSettings settings;
        std::unique_ptr<NonlinearSolver> value;
    };
    constexpr const char *Collective =
        "Synchronous native CPU collective on the domain communicator, from an MPI-authorized controller thread. All ranks follow the same call order. Join external producers and release conflicting views; serialize this workspace. Core/halo overlap uses the configured policy. No CUDA fallback.";
    constexpr const char *Local =
        "Synchronous CPU operation. Caller supplies ready inputs, releases conflicting views, retains communicator owners and coordinates rank-local failures before the next collective. No implicit halo exchange or device fallback.";

}

#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include <pybind11/stl.h>

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
        explicit FiniteVolumeOperators(const Domain &domain)
            : fv(domain.Discretization()), workers(domain.ChemicalWorkers())
        {
        }
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
    py::object ShareOptional(const std::optional<R::Array<real>> &array)
    {
        return array ? py::cast(R::PythonArray::Share(*array)) : py::none();
    }
    std::map<std::string, Field> PhysicalFields(const EquationDofs &state)
    {
        std::map<std::string, Field> result;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            result.emplace(state.Fields().Name(iField), state.PhysicalField(iField));
        return result;
    }
    constexpr const char *Collective =
        "Synchronous native CPU collective on the domain communicator, from an MPI-authorized controller thread. All ranks follow the same call order. Join external producers and release conflicting views; serialize this workspace. Core/halo overlap uses the configured policy. No CUDA fallback.";
    constexpr const char *Local =
        "Synchronous CPU operation. Caller supplies ready inputs, releases conflicting views, retains communicator owners and coordinates rank-local failures before the next collective. No implicit halo exchange or device fallback.";

    void BindFiniteVolumeOperators(py::module_ &m)
    {
        py::class_<EquationDofs>(
            m, "EquationDofs",
            "Logical flow/RANS/species binding over shared Runtime fields. Copies retain allocations and immutable metadata. Clone copies owned values; refresh ghosts afterward. No implicit regrouping or payload flattening.")
            .def_property_readonly("components", &EquationDofs::ComponentNumber)
            .def_property_readonly("space", &EquationDofs::Space)
            .def_property_readonly(
                "fields", &PhysicalFields,
                "Named physical DOF fields sharing owned/ghost arrays. Physical names and logical block offsets are separate.")
            .def_property_readonly("field_names",
                                   [](const EquationDofs &state)
        {
            std::vector<std::string> result;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                result.push_back(state.Fields().Name(iField));
            return result;
        }, "Physical communication order; do not substitute sorted dictionary order.")
            .def_property_readonly("blocks",
                                   [](const EquationDofs &state)
        {
            std::map<std::string, std::vector<std::string>> result;
            for (const auto &block : state.Schema().blocks)
                for (const auto &component : block.components)
                    result[block.name].push_back(component.name);
            return result;
        })
            .def("locate_block",
                 [](const EquationDofs &state, const std::string &name)
        {
            const auto location = state.LocateBlock(name);
            return std::make_tuple(state.Fields().Name(location.physicalField), location.firstComponent,
                                   location.componentNumber);
        }, py::arg("name"), "Physical field name, component offset and width. Metadata grants no access permission.")
            .def(
                "resolve_components",
                [](const EquationDofs &state, const std::vector<std::string> &names)
        {
            return state.Schema().ResolveComponents(names);
        }, py::arg("names"),
                "Resolve unique logical component names or block.component names before numerical execution; empty selects all.")
            .def("clone", &EquationDofs::Clone, py::call_guard<py::gil_scoped_release>(), Local)
            .def(
                "zeros_like", &EquationDofOperations::NewVector, py::call_guard<py::gil_scoped_release>(),
                "Allocate independent zeroed storage with the same logical binding/layout. Zero residual scratch need not be an admissible physical state.")
            .def_static(
                "from_fields",
                [](const EquationDofs &prototype, const std::vector<std::pair<std::string, Field>> &fields,
                   const std::vector<std::tuple<std::string, std::string, index>> &bindings)
        {
            Algebra::DofBundle bundle(prototype.Space());
            for (const auto &[name, field] : fields)
                bundle.Add(name, field);
            std::vector<EquationDofBinding> mapping;
            for (const auto &[block, field, first] : bindings)
                mapping.push_back({block, field, first});
            return EquationDofs(prototype.Schema(), std::move(bundle), std::move(mapping));
        },
                py::arg("prototype"), py::arg("fields"), py::arg("bindings"),
                "Share new physical fields with the prototype schema/row space. Ordered (name,field) and (block,field,offset) lists must cover all components without gaps/overlap. No value copy or new numerical capability.")
            .def("copy_from", &EquationDofOperations::Copy, py::arg("source"), py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("fill", &EquationDofOperations::Fill, py::arg("value"), py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("scale", &EquationDofOperations::Scale, py::arg("value"), py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("axpy", &EquationDofOperations::Axpy, py::arg("alpha"), py::arg("source"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("linear_combination", &EquationDofOperations::LinearCombination, py::arg("alpha"), py::arg("x"),
                 py::arg("beta"), py::arg("y"), py::call_guard<py::gil_scoped_release>(), Local);
        py::class_<AuxiliaryRequest>(m, "AuxiliaryRequest")
            .def(py::init<>())
            .def_readwrite("mean_thermodynamics", &AuxiliaryRequest::meanThermodynamics)
            .def_readwrite("face_spectral_radius", &AuxiliaryRequest::faceSpectralRadius)
            .def_readwrite("face_wave_jump", &AuxiliaryRequest::faceWaveJump)
            .def_readwrite("face_pressure_minimum", &AuxiliaryRequest::facePressureMinimum)
            .def_readwrite("cell_spectral_sum", &AuxiliaryRequest::cellSpectralSum)
            .def_readwrite("cell_wave_jump", &AuxiliaryRequest::cellWaveJump)
            .def_readwrite("cell_pressure_minimum", &AuxiliaryRequest::cellPressureMinimum)
            .def_property("derived_quantities",
                          [](const AuxiliaryRequest &request)
        {
            std::vector<DerivedQuantity> result;
            for (int q = 0; q < DerivedQuantityNumber; ++q)
                if (request.derived.Has(DerivedQuantity(q)))
                    result.push_back(DerivedQuantity(q));
            return result;
        }, [](AuxiliaryRequest &request, const std::vector<DerivedQuantity> &quantities)
        {
            DerivedRequest derived;
            for (const auto quantity : quantities)
            {
                const int q = int(quantity);
                DNDS_check_throw_info(q >= 0 && q < DerivedQuantityNumber, "unknown derived quantity");
                derived.enabled[q] = true;
            }
            request.derived = derived;
        });
        py::class_<AuxiliaryValues>(
            m, "AuxiliaryValues",
            "Optional explicit outputs. Debug arrays share allocations; external writes invalidate publication. No automatic refresh/exchange.")
            .def_property_readonly("mean_thermodynamics",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.meanThermodynamics);
        })
            .def_property_readonly("face_spectral_radius",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.faceSpectralRadius);
        })
            .def_property_readonly("face_wave_jump",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.faceWaveJump);
        })
            .def_property_readonly("face_pressure_minimum",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.facePressureMinimum);
        })
            .def_readonly("cell_spectral_sum", &AuxiliaryValues::cellSpectralSum)
            .def_readonly("cell_wave_jump", &AuxiliaryValues::cellWaveJump)
            .def_readonly("cell_pressure_minimum", &AuxiliaryValues::cellPressureMinimum)
            .def_readonly("derived_owned", &AuxiliaryValues::derivedOwned)
            .def_readonly("derived_ghost", &AuxiliaryValues::derivedGhost);
        py::class_<MeanGradients>(m, "MeanGradients",
                                  "Bound explicit gradient scratch; external writes invalidate auxiliary provenance.")
            .def_readonly("fields", &MeanGradients::fields);
        py::class_<Workspace>(
            m, "Workspace",
            "Native FV workspace tied to exact input allocations. Serialize use and drain work before release.");
        py::class_<MeanWorkspace>(m, "MeanWorkspace",
                                  "Separate zero-reconstruction physical-step scratch; never the ordinary VR guess.");
        py::enum_<DiagonalStorage>(m, "DiagonalStorage")
            .value("Entries", DiagonalStorage::Entries)
            .value("Block", DiagonalStorage::Block);
        py::enum_<OffDiagonalStorage>(m, "OffDiagonalStorage")
            .value("Assembled", OffDiagonalStorage::Assembled)
            .value("DirectEvaluated", OffDiagonalStorage::DirectEvaluated)
            .value("SemiMatrixFree", OffDiagonalStorage::DirectEvaluated);
        py::enum_<LinearRegion>(m, "LinearRegion")
            .value("All", LinearRegion::All)
            .value("Core", LinearRegion::Core)
            .value("HaloDependent", LinearRegion::HaloDependent);
        py::class_<JacobianPreparation>(m, "JacobianPreparation")
            .def(py::init<>())
            .def_readwrite("physical_mass", &JacobianPreparation::physicalMass)
            .def_readwrite("residual_scale", &JacobianPreparation::residualScale)
            .def_readwrite("roe_dissipation", &JacobianPreparation::roeDissipation)
            .def_readwrite("update_components", &JacobianPreparation::updateComponents);
        py::class_<LinearProblemContext::DofSGSIteration>(
            m, "DofSGSIteration",
            "Staged two-buffer SGS. Join input halo before dependent lower work. Upper reverses order. Finish after both scans; abandonment invalidates unfinished output.")
            .def("lower", &LinearProblemContext::DofSGSIteration::Lower, py::arg("region") = LinearRegion::All,
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("upper", &LinearProblemContext::DofSGSIteration::Upper, py::arg("region") = LinearRegion::All,
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("finish", &LinearProblemContext::DofSGSIteration::Finish, py::call_guard<py::gil_scoped_release>(),
                 Local);
        py::class_<LinearProblemContext, std::shared_ptr<LinearProblemContext>>(
            m, "LinearProblem",
            "Frozen approximate matrix. Explicitly prepare coefficients, inverses and factors. Serialize changes with all actions/solves.")
            .def_property_readonly("diagonal",
                                   [](const LinearProblemContext &p)
        {
            return R::PythonArray::Share(p.Diagonal());
        })
            .def_property_readonly("core", &LinearProblemContext::Core)
            .def_property_readonly("halo_dependent", &LinearProblemContext::HaloDependent)
            .def("invert_diagonal", &LinearProblemContext::InvertDiagonal, py::call_guard<py::gil_scoped_release>(),
                 Local)
            .def("apply",
                 py::overload_cast<const EquationDofs &, EquationDofs &, LinearRegion>(&LinearProblemContext::Apply,
                                                                                       py::const_),
                 py::arg("input"), py::arg("output"), py::arg("region") = LinearRegion::All,
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("begin_sgs",
                 py::overload_cast<const EquationDofs &, const EquationDofs &, EquationDofs &, EquationDofs &, bool>(
                     &LinearProblemContext::BeginSGS, py::const_),
                 py::arg("rhs"), py::arg("input"), py::arg("forward"), py::arg("output"),
                 py::arg("middle_exchange") = false, py::call_guard<py::gil_scoped_release>(), Local)
            .def(
                "begin_sgs_zero",
                py::overload_cast<const EquationDofs &, EquationDofs &, EquationDofs &, bool>(
                    &LinearProblemContext::BeginSGSZero, py::const_),
                py::arg("rhs"), py::arg("forward"), py::arg("output"), py::arg("middle_exchange") = false,
                "Dedicated statically selected zero-start SGS. No old input allocation or read; exchange forward before Upper only if middle_exchange is enabled.",
                py::call_guard<py::gil_scoped_release>(), Local)
            .def("apply_sgs",
                 py::overload_cast<const EquationDofs &, const EquationDofs &, EquationDofs &, EquationDofs &>(
                     &LinearProblemContext::ApplySGS, py::const_),
                 py::arg("rhs"), py::arg("input"), py::arg("forward"), py::arg("output"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("prepare_ilu", &LinearProblemContext::PrepareILUStructure, py::arg("fill") = 0,
                 py::arg("core_first") = true, py::arg("local_parts") = std::vector<index>{},
                 py::arg("ordering") = "natural", py::call_guard<py::gil_scoped_release>(), Local)
            .def("factorize_ilu", &LinearProblemContext::FactorizeILU, py::call_guard<py::gil_scoped_release>(), Local)
            .def("apply_ilu",
                 py::overload_cast<const EquationDofs &, EquationDofs &, EquationDofs &>(
                     &LinearProblemContext::ApplyILU, py::const_),
                 py::arg("rhs"), py::arg("forward"), py::arg("output"), py::call_guard<py::gil_scoped_release>(),
                 Local);
        py::class_<LinearSolveSettings>(m, "LinearSolveSettings")
            .def(py::init<>())
            .def_readwrite("method", &LinearSolveSettings::method)
            .def_readwrite("preconditioner", &LinearSolveSettings::preconditioner)
            .def_readwrite("stopping", &LinearSolveSettings::stopping)
            .def_readwrite("max_iterations", &LinearSolveSettings::maxIterations)
            .def_readwrite("subspace", &LinearSolveSettings::subspace)
            .def_readwrite("sgs_sweeps", &LinearSolveSettings::sgsSweeps)
            .def_readwrite("jacobi_local_sweeps", &LinearSolveSettings::jacobiLocalSweeps)
            .def_readwrite("ilu_fill", &LinearSolveSettings::iluFill)
            .def_readwrite("ilu_preconditioner_corrections", &LinearSolveSettings::iluPreconditionerCorrections)
            .def_readwrite("ilu_core_first", &LinearSolveSettings::iluCoreFirst)
            .def_readwrite("ilu_ordering", &LinearSolveSettings::iluOrdering)
            .def_readwrite("exchange_between_sweeps", &LinearSolveSettings::exchangeBetweenSweeps)
            .def_readwrite("relative_tolerance", &LinearSolveSettings::relativeTolerance)
            .def_readwrite("absolute_tolerance", &LinearSolveSettings::absoluteTolerance);
        py::class_<LinearExecution>(m, "LinearExecution")
            .def(py::init<>())
            .def_readwrite("overlap", &LinearExecution::overlap)
            .def_readwrite("collective_checks", &LinearExecution::collectiveChecks);
        py::enum_<Linear::LinearStop>(m, "LinearStop")
            .value("Converged", Linear::LinearStop::Converged)
            .value("IterationLimit", Linear::LinearStop::IterationLimit)
            .value("ArnoldiBreakdown", Linear::LinearStop::ArnoldiBreakdown)
            .value("PreconditionerBreakdown", Linear::LinearStop::PreconditionerBreakdown)
            .value("NonFinite", Linear::LinearStop::NonFinite)
            .value("FixedCount", Linear::LinearStop::FixedCount);
        py::class_<Linear::LinearIterationReport>(m, "LinearIterationReport")
            .def_readonly("stop", &Linear::LinearIterationReport::stop)
            .def_readonly("converged", &Linear::LinearIterationReport::converged)
            .def_readonly("usable", &Linear::LinearIterationReport::usable)
            .def_readonly("iterations", &Linear::LinearIterationReport::iterations)
            .def_readonly("initial_true_residual", &Linear::LinearIterationReport::initialTrueResidual)
            .def_readonly("true_residual", &Linear::LinearIterationReport::trueResidual)
            .def_readonly("operator_applications", &Linear::LinearIterationReport::operatorApplications)
            .def_readonly("inner_product_batches", &Linear::LinearIterationReport::innerProductBatches)
            .def_readonly("jacobi_sweeps", &Linear::LinearIterationReport::jacobiSweeps)
            .def_readonly("jacobi_input_exchanges", &Linear::LinearIterationReport::jacobiInputExchanges)
            .def_readonly("preconditioner_applications", &Linear::LinearIterationReport::preconditionerApplications);
        py::class_<DofLinearSolver>(
            m, "DofLinearSolver",
            "Native SGS/left-GMRES/ILU orchestration. Matching collective calls and producer completion are required. Solution aliases reusable scratch; clone before another solve.")
            .def("solve",
                 [](DofLinearSolver &solver, const EquationDofs &rhs, const EquationDofs *initialGuess,
                    const std::vector<real> &componentScales)
        {
            return solver.Solve(
                rhs, initialGuess ? OptionalRef<const EquationDofs>(*initialGuess) : OptionalRef<const EquationDofs>{},
                componentScales);
        },
                 py::arg("rhs"), py::arg("initial_guess") = nullptr, py::arg("component_scales") = std::vector<real>{},
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def_property_readonly("solution", [](const DofLinearSolver &solver)
        {
            return solver.Solution();
        });
        py::class_<NonlinearSolveControl>(m, "NonlinearSolveControl")
            .def(py::init<>())
            .def_readwrite("correction_limit", &NonlinearSolveControl::correctionLimit)
            .def_readwrite("initial_pseudo_cfl", &NonlinearSolveControl::initialPseudoCFL)
            .def_readwrite("update_components", &NonlinearSolveControl::updateComponents);
        py::class_<NonlinearStageSolver>(
            m, "NonlinearStageSolver",
            "Bound native nonlinear controller, sharing the production CorrectOnce/solve logic. Retains state, workspace and engine. The caller supplies frozen normalized resOther and physical time; no time-method history lives here.")
            .def_property_readonly("state",
                                   [](const NonlinearStageSolver &solver)
        {
            return *solver.state;
        })
            .def("solve",
                 [](NonlinearStageSolver &solver, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage, NonlinearSolveControl control)
        {
            return solver.value->Solve(*solver.state, source, mass, rhsScale, time, stage, control);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::arg("control") = NonlinearSolveControl{},
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "correct_once",
                [](NonlinearStageSolver &solver, const EquationDofs &source, real mass, real rhsScale, real time,
                   int stage, NonlinearSolveControl control)
        {
            return solver.value->CorrectOnce(*solver.state, source, mass, rhsScale, time, stage, control);
        },
                py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                py::arg("stage") = 0, py::arg("control") = NonlinearSolveControl{},
                py::call_guard<py::gil_scoped_release>(),
                "Collective one-correction native block. No nonlinear convergence loop or acceptance decision; the caller controls CFL/stopping and owns rollback.")
            .def("evaluate_equation",
                 [](NonlinearStageSolver &solver, EquationDofs &output, real mass, real rhsScale, real time)
        {
            solver.value->EvaluateEquation(*solver.state, output, mass, rhsScale, time);
        }, py::arg("output"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("apply_correction",
                 [](NonlinearStageSolver &solver, const EquationDofs &correction, real scale,
                    const std::vector<index> &components)
        {
            return solver.value->ApplyCorrection(*solver.state, correction, scale, components);
        },
                 py::arg("correction"), py::arg("scale") = 1, py::arg("update_components") = std::vector<index>{},
                 py::call_guard<py::gil_scoped_release>(), Collective);
        py::class_<PolynomialMultigrid>(
            m, "PolynomialMultigrid",
            "Native same-cell spatial p-FAS. Retains its finest stage solver and mesh. Apply one cycle inside a caller's nonlinear loop, with other-stage source terms frozen; no temporal-pMG.")
            .def("apply_cycle",
                 [](PolynomialMultigrid &multigrid, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage, real pseudoCFL)
        {
            return multigrid.ApplyCycle(source, mass, rhsScale, time, stage, pseudoCFL);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::arg("pseudo_cfl") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "scratch_pool_statistics", &PolynomialMultigrid::ScratchPoolStatistics,
                "Reserved DoF payload bytes and peak simultaneous borrowed slots; excludes persistent/reconstruction workspaces.")
            .def("solve",
                 [](PolynomialMultigrid &multigrid, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage)
        {
            return multigrid.Solve(source, mass, rhsScale, time, stage);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::call_guard<py::gil_scoped_release>(), Collective);
        py::class_<FiniteVolumeOperators>(
            m, "FiniteVolumeOperators",
            "Configured native equation-state resources. Retains Domain/communicator and shares immutable FV plans plus worker capacity, not time history. Missing profiles still reject. Strang domains provide source-free transport.")
            .def(py::init<const Domain &>(), py::arg("domain"), py::keep_alive<1, 2>())
            .def_property_readonly("spatial_settings_json",
                                   [](const FiniteVolumeOperators &operators)
        {
            return Json(operators.fv.Settings()).dump();
        }, "Read-only resolved spatial controls. Source splitting is already reflected; these are not a new configuration to apply.")
            .def_property_readonly("metrics",
                                   [](const FiniteVolumeOperators &op) -> const CFV::FiniteVolumePlan &
        {
            return op.fv.Metrics();
        }, py::return_value_policy::reference_internal)
            .def("build_workspace",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return Workspace{op.fv.BuildReconstructionState(u), op.fv.BuildWorkspace(u, op.workers)};
        }, py::arg("state"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>(),
                 Collective)
            .def("build_mean_workspace",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return MeanWorkspace{op.fv.BuildMeanWorkspace(u, op.workers)};
        }, py::arg("state"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>(),
                 Collective)
            .def("evaluate",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, EquationDofs &r, Workspace &w, R::EagerEngine &e,
                    const R::ExecutionStream &s, real t)
        {
            op.fv.Evaluate(u, r, *w.reconstruction, *w.value, e, s, t);
        },
                 py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "build_face_flux_buffer",
                [](const FiniteVolumeOperators &op, const EquationDofs &u)
        {
            return R::PythonArray::Share(op.fv.BuildFaceFluxBuffer(u.ComponentNumber()));
        }, py::arg("state"),
                "Allocate caller-owned integrated negative outward face flux, in AuxiliaryFaces order and the local anchor cell frame. No source or halo.")
            .def("prepare_reconstruction",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, Workspace &w, R::EagerEngine &e,
                    const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareReconstruction(u, *w.reconstruction, *w.value, e, s, t);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("engine"), py::arg("stream"), py::arg("time") = 0,
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "evaluate_flux",
                [](const FiniteVolumeOperators &op, const EquationDofs &u, EquationDofs &r, Workspace &w,
                   R::PythonArray *faces, real t)
        {
            std::optional<R::Array<real>> buffer;
            if (faces)
                buffer.emplace(faces->AsArray<real>());
            py::gil_scoped_release release;
            op.fv.EvaluateFlux(u, r, *w.reconstruction, *w.value, CFV::ReconstructionRegion::All, t,
                               buffer ? OptionalRef<Runtime::Array<real>>(*buffer)
                                      : OptionalRef<Runtime::Array<real>>{});
        },
                py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("face_flux") = nullptr,
                py::arg("time") = 0,
                "Local flux-only overwrite using a completed evaluate publication at the same state/time. Input halos must be ready. Optional face array is shared and overwritten; release views first. No reconstruction or transfer. Caller coordinates failures.")
            .def(
                "add_source",
                [](const FiniteVolumeOperators &op, const EquationDofs &u, EquationDofs &r, Workspace &w, real t)
        {
            op.fv.AddSource(u, r, *w.reconstruction, *w.value, t);
        }, py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("time") = 0,
                py::call_guard<py::gil_scoped_release>(),
                "Local additive volume source using completed evaluate preparation at the same state/time. Includes rotation, RANS and chemistry. P0 retains physical source gradients. Output must be initialized and distinct; no halo exchange or reconstruction. Release views and coordinate failures.")
            .def("evaluate_mean",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, EquationDofs &r, MeanWorkspace &w,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.EvaluateMean(u, r, *w.value, e, s, t);
        },
                 py::arg("state"), py::arg("residual"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("reconstruction_guess",
                 [](const FiniteVolumeOperators &op, const Workspace &w)
        {
            return op.fv.ReconstructionGuess(*w.reconstruction);
        }, py::arg("workspace"), "Shared unlimited coefficient fields. Do not write during workspace/transfer use.")
            .def("flux_reconstruction",
                 [](const FiniteVolumeOperators &op, const Workspace &w)
        {
            return op.fv.FluxReconstruction(*w.reconstruction, *w.value);
        }, py::arg("workspace"))
            .def("reset_reconstruction",
                 [](const FiniteVolumeOperators &op, Workspace &w)
        {
            op.fv.ResetReconstruction(*w.reconstruction, *w.value);
        }, py::arg("workspace"), py::call_guard<py::gil_scoped_release>(), Local)
            .def("restore_reconstruction",
                 [](const FiniteVolumeOperators &op, Workspace &w, const EquationDofs &u)
        {
            op.fv.RestoreReconstructionGuess(*w.reconstruction, u);
        }, py::arg("workspace"), py::arg("guess"), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("build_auxiliary_values",
                 [](const FiniteVolumeOperators &op, AuxiliaryRequest request)
        {
            return op.fv.BuildAuxiliaryValues(request);
        }, py::arg("request"))
            .def("build_mean_gradients",
                 [](const FiniteVolumeOperators &op, EquationDofs &u)
        {
            return op.fv.BuildMeanGradients(u);
        }, py::arg("state"), py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_aux_values",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, AuxiliaryValues &a, MeanGradients &g,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareAuxValues(u, a, g, e, s, t, op.workers);
        },
                 py::arg("state"), py::arg("output"), py::arg("gradients"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_correction_aux",
                 [](const FiniteVolumeOperators &op, EquationDofs &u, Workspace &w, AuxiliaryValues &a,
                    R::EagerEngine &e, const R::ExecutionStream &s, real t)
        {
            op.fv.PrepareCorrectionAux(u, *w.reconstruction, *w.value, a, e, s, t);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("output"), py::arg("engine"), py::arg("stream"),
                 py::arg("time") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def("validate_aux_values",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const AuxiliaryValues &a, real t)
        {
            op.fv.ValidateAuxValues(u, a, t);
        }, py::arg("state"), py::arg("values"), py::arg("time") = 0, Local)
            .def("build_linear_problem",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, DiagonalStorage d, OffDiagonalStorage o)
        {
            return op.fv.BuildLinearProblem(u, d, o);
        }, py::arg("state"), py::arg("diagonal") = DiagonalStorage::Block,
                 py::arg("off_diagonal") = OffDiagonalStorage::Assembled, py::keep_alive<0, 1>(),
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("prepare_jacobian",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const AuxiliaryValues &a,
                    const R::PythonArray &inverseDt, LinearProblemContext &p, JacobianPreparation c, real t,
                    const Workspace *w)
        {
            // Dtype resolution uses Python. Freeze the native sharing
            // handle before releasing the GIL for numerical execution.
            const auto native = inverseDt.AsArray<real>();
            py::gil_scoped_release release;
            op.fv.PrepareJacobian(u, a, native, p, c, t, op.workers,
                                  w ? std::optional(FiniteVolume::JacobianContext{*w->reconstruction, *w->value})
                                    : std::nullopt);
        },
                 py::arg("state"), py::arg("auxiliary"), py::arg("inverse_pseudo_time"), py::arg("output"),
                 py::arg("controls") = JacobianPreparation{}, py::arg("time") = 0,
                 py::arg("source_workspace") = nullptr, Local)
            .def("build_linear_solver",
                 [](const FiniteVolumeOperators &op, std::shared_ptr<LinearProblemContext> p, const EquationDofs &u,
                    LinearSolveSettings settings, LinearExecution execution)
        {
            return std::make_unique<DofLinearSolver>(std::move(p), u, [fv = op.fv](Field &field)
            {
                return fv.BindHalo(field);
            }, settings, execution);
        },
                 py::arg("problem"), py::arg("prototype"), py::arg("settings") = LinearSolveSettings{},
                 py::arg("execution") = LinearExecution{}, py::keep_alive<0, 1>(),
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("build_stage_solver",
                 [](const FiniteVolumeOperators &operators, EquationDofs &state, Workspace &workspace,
                    R::EagerEngine &engine, const R::ExecutionStream &stream, const std::string &implicitJson)
        {
            auto solver = std::make_unique<NonlinearStageSolver>();
            const MPIInfo mpi(operators.fv.Metrics().Communicator());
            std::optional<Algebra::ComponentSelection> selection;
            MPI::LocalStage(mpi, [&]
            {
                // Reuse production schema/default/unknown-key validation.
                // Only implicit controls can be overridden; spatial and
                // physical settings come from the immutable native plan.
                Json settings = operators.fv.Settings();
                const auto overrides = Json::parse(implicitJson);
                DNDS_check_throw_info(overrides.is_object(), "implicit overrides must be a JSON object");
                settings["implicit"].merge_patch(overrides);
                solver->settings =
                    SimulationConfig::Resolve({{"defaults", settings}, {"time", {{"method", "bdf1"}}}}, mpi.size)
                        .Domains()
                        .front()
                        .settings;
                solver->owner.emplace(operators);
                solver->state.emplace(state);
                solver->stream = stream;
                selection.emplace(state.ComponentNumber(),
                                  state.Schema().ResolveComponents(solver->settings.implicit.residualComponents));
            });
            solver->value =
                std::make_unique<NonlinearSolver>(operators.fv, state, workspace.reconstruction, workspace.value,
                                                  engine, stream, solver->settings, *selection, operators.workers);
            return solver;
        },
                 py::arg("state"), py::arg("workspace"), py::arg("engine"), py::arg("stream"),
                 py::arg("implicit_json") = "{}", py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
                 py::keep_alive<0, 3>(), py::keep_alive<0, 4>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("build_multigrid",
                 [](const FiniteVolumeOperators &operators, ssp<Geom::UnstructuredMesh> mesh,
                    NonlinearStageSolver &solver)
        {
            MPI::LocalStage(MPIInfo(operators.fv.Metrics().Communicator()), [&]
            {
                DNDS_check_throw_info(mesh && &operators.fv.Mesh() == mesh.get() && solver.owner &&
                                          solver.owner->fv.SharesPlan(operators.fv),
                                      "p-FAS requires its original mesh and finest operator owner");
            });
            auto engine = solver.stream.LockEngine();
            return std::make_unique<PolynomialMultigrid>(std::move(mesh), operators.fv, *solver.state, *solver.value,
                                                         engine, solver.stream, solver.settings, operators.workers);
        },
                 py::arg("mesh"), py::arg("stage_solver"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
                 py::keep_alive<0, 3>(), py::call_guard<py::gil_scoped_release>(), Collective)
            .def("write_visualization",
                 [](const FiniteVolumeOperators &op, const EquationDofs &u, const Workspace &w,
                    const std::string &prefix, real t, int deflate)
        {
            op.fv.WriteVisualization(u, *w.reconstruction, *w.value, prefix, t, deflate, op.workers);
        },
                 py::arg("state"), py::arg("workspace"), py::arg("prefix"), py::arg("time") = 0,
                 py::arg("hdf_deflate_level") = 0, py::call_guard<py::gil_scoped_release>(), Collective);
    }
}

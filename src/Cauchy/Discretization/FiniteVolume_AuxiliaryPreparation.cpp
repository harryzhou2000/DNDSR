#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/MPI/Failure.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using AuxiliaryDetail::EquationAuxiliaryArrays;

    struct FiniteVolume::AuxiliaryPreparation::State
    {
        FiniteVolume owner;
        std::shared_ptr<const Data> data;
        EquationDofs source;
        AuxiliaryValues buffers;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        CFV::FiniteVolumeRead geometry;
        FiniteVolumeStencilStorage::ReadAccess stencilRead;
        EquationDofs::ReadScope owned;
        std::optional<EquationDofs::ReadScope> complete;
        std::optional<MeanGradients> gradients;
        std::optional<BoundaryInputs> boundary;
        std::optional<R::Array<real>::ReadLease> boundaryRead;
        RHSScalarView incoming;
        std::vector<R::Array<real>::ReadLease> gradientOwned, gradientGhost;
        R::MetadataBuffer<RHSFieldView> gradientViews;
        R::MetadataBuffer<EquationBlockLocation> locations;
        std::optional<WallDistanceRead> walls;
        WallDistanceView wallValues;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::array<std::optional<R::Array<real>::WriteLease>, 7> writes;
        std::array<RHSOutputView, 7> values;
        std::optional<R::Array<real>::WriteLease> scratchWrite;
        RHSOutputView scratch;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int independent;
        R::MetadataBuffer<index> vectorStarts;
        real time;
        RigidFrame frame;
        int phase = 0;
        bool started = false;
        bool boundaryPending = false, independentDone = false;

        State(const FiniteVolume &plan, const EquationDofs &state, const AuxiliaryValues &output, real atTime,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, AuxiliaryInputs inputs)
            : owner(plan), data(plan._data), source(state.NumericAlias()), buffers(output), workers(std::move(pool)),
              geometry(data->metrics.AcquireRead()), stencilRead(data->stencil.Read()), owned(source.ReadOwned()),
              independent(data->equation.HasMixture() ? data->equation.Mixture().SpeciesNumber() - 1 : 0),
              time(atTime), frame(data->motion.At(time)), boundaryPending(inputs.deferBoundary)
        {
            if (inputs.meanGradients)
            {
                gradients = *inputs.meanGradients;
                DNDS_check_throw_info(gradients->_data, "mean gradient input is uninitialized");
                gradients->_data->Check(source, gradients->fields, data, time, false);
                for (const auto &field : gradients->fields)
                    gradientOwned.push_back(field.Owned().Read());
                gradientViews = R::MetadataBuffer<RHSFieldView>::Generate(gradientOwned.size(), [&](index iField)
                                                                          { return RHSFieldView{gradientOwned[iField].View<R::UniformTableRows>(), {}}; });
            }
            if (inputs.boundary)
            {
                boundary = *inputs.boundary;
                boundaryRead.emplace(
                    boundary->QuadratureValues().CoupledField().Owned().Read());
                incoming = boundaryRead->View<R::UniformTableRows>();
            }
            if (data->equation.HasRANS() && data->equation.RANS().ComponentNumber() == 2)
            {
                DNDS_check_throw_info(gradients, "two-equation auxiliary transport requires explicit mean gradients");
                walls.emplace(data->wallDistance->AcquireRead());
                wallValues = walls->HostView();
            }
            index first = 0;
            locations = R::MetadataBuffer<EquationBlockLocation>::Generate(source.BlockNumber(),
                                                                           [&](index iBlock)
                                                                           { return source.LocateBlock(iBlock); });
            std::vector<index> vectors;
            for (const auto &block : source.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans")
                    ransFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectors.push_back(first + vector);
                first += index(block.components.size());
            }
            vectorStarts = R::MetadataBuffer<index>(vectors);
            // Validate/reserve the backend before any numerical publication.
            {
                auto evaluator = Evaluator();
            }
            const auto arrays = EquationAuxiliaryArrays(buffers);
            try
            {
                for (const auto *derived : {&buffers.derivedOwned, &buffers.derivedGhost})
                    if (*derived)
                        (*derived)->PreflightRefresh();
                for (int k = 0; k < 7; ++k)
                    if (arrays[k])
                    {
                        auto handle = *arrays[k];
                        writes[k].emplace(handle.Write());
                        values[k] = writes[k]->View<R::UniformTableRows>();
                    }
                if ((arrays[4] && !arrays[1]) || (arrays[5] && !arrays[2]) || (arrays[6] && !arrays[3]))
                {
                    scratchWrite.emplace(buffers._faceScratch->Write());
                    scratch = scratchWrite->View<R::UniformTableRows>();
                }
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy equation auxiliary preparation failed");
                Cancel();
                throw;
            }
        }
        void Cancel()
        {
            for (auto &write : writes)
                if (write)
                    write->CancelUnmodified();
            if (scratchWrite)
                scratchWrite->CancelUnmodified();
        }
        ~State()
        {
            if (!started)
                Cancel();
            else
            {
                for (auto &write : writes)
                    if (write)
                        write->Fail();
                if (scratchWrite)
                    scratchWrite->Fail();
            }
        }
        PointEvaluator Evaluator() const
        {
            return data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true, true, false});
        }
        void RefreshDerived(bool ghost)
        {
            auto &field = ghost ? buffers.derivedGhost : buffers.derivedOwned;
            if (!field)
                return;
            if (workers)
                RefreshDerivedFields(source, data->equation, *field, workers->Acquire(), true);
            else
                RefreshDerivedFields(source, data->equation, *field, true);
            (ghost ? derivedGhost : derivedOwned).emplace(field->ReadCurrent(source, data->equation));
        }
        CPU::AuxiliaryKernelRequest KernelInput(SpanView<const EquationBlockReadView> blocks) const
        {
            CPU::AuxiliaryKernelRequest request;
            request.stencil = stencilRead.View();
            request.metric = geometry.HostView();
            request.blocks = blocks;
            request.gradients = gradientViews.HostView();
            request.locations = locations.HostView();
            request.vectorStarts = vectorStarts.HostView();
            request.conditions = {data->boundaryConditions.data(), index(data->boundaryConditions.size())};
            request.wall = wallValues;
            request.hasWall = bool(walls);
            if (buffers._derivedThermodynamics)
                request.derived.emplace(MeanThermodynamicView::Derived(
                    *derivedOwned, derivedGhost ? OptionalRef<const DerivedFields::ReadScope>(*derivedGhost)
                                                : OptionalRef<const DerivedFields::ReadScope>{}));
            for (int k = 0; k < 7; ++k)
                request.enabled[k] = bool(writes[k]);
            request.values = values;
            request.scratch = scratch;
            request.useScratch = bool(scratchWrite);
            request.incoming = incoming;
            request.frame = frame;
            request.components = source.ComponentNumber();
            request.flowFirst = flowFirst;
            request.speciesFirst = speciesFirst;
            request.ransFirst = ransFirst;
            request.dimension = data->metrics.Dimension();
            request.independentSpecies = independent;
            request.time = time;
            return request;
        }
        void InterfaceIndependent()
        {
            DNDS_check_throw_info(phase == 0 && boundaryPending && !independentDone,
                                  "auxiliary independent phase requires a pending boundary");
            started = true;
            RefreshDerived(false);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, owned);
            const auto request = KernelInput(blocks.HostView());
            RequireNumericalSuccess(CPU::PrepareAuxiliaryMeans(request, data->equation, evaluator, 0, data->metrics.NumCell()), "core auxiliary means");
            RequireNumericalSuccess(CPU::PrepareAuxiliaryFaces(request, data->equation, data->boundaryPoint, evaluator,
                                                               request.stencil.auxiliaryInterfaceIndependentFaces),
                                    "interface-independent auxiliary faces");
            CPU::ReduceAuxiliaryCells(request, request.stencil.interfaceIndependentCells);
            independentDone = true;
        }
        void BindBoundary(const BoundaryInputs &input)
        {
            DNDS_check_throw_info(phase == 0 && boundaryPending,
                                  "auxiliary boundary is not pending");
            owner.ValidateBoundaryInputs(input, time);
            const auto arrays = EquationAuxiliaryArrays(buffers);
            for (const auto *array : arrays)
                if (array)
                    for (const auto *packet : {&input.QuadratureValues(), &input.FaceCenterValues()})
                        for (index iField = 0; iField < packet->PhysicalFieldNumber(); ++iField)
                            DNDS_check_throw_info(
                                !Algebra::Detail::Overlaps(array->Storage(),
                                                           packet->PhysicalField(iField).Owned().Storage()),
                                "auxiliary output aliases deferred boundary input");
            boundary = input;
            boundaryRead.emplace(boundary->QuadratureValues().CoupledField().Owned().Read());
            incoming = boundaryRead->View<R::UniformTableRows>();
            boundaryPending = false;
        }
        void Core()
        {
            DNDS_check_throw_info(phase == 0 && !boundaryPending,
                                  "auxiliary core already executed or interface is pending");
            started = true;
            if (!independentDone)
                RefreshDerived(false);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, owned);
            const auto request = KernelInput(blocks.HostView());
            if (!independentDone)
                RequireNumericalSuccess(CPU::PrepareAuxiliaryMeans(request, data->equation, evaluator, 0, data->metrics.NumCell()),
                                        "core auxiliary means");
            RequireNumericalSuccess(CPU::PrepareAuxiliaryFaces(request, data->equation, data->boundaryPoint, evaluator,
                                                               independentDone ? request.stencil.auxiliaryInterfaceFaces
                                                                               : request.stencil.auxiliaryCoreFaces),
                                    "core auxiliary faces");
            CPU::ReduceAuxiliaryCells(request, independentDone ? request.stencil.interfaceCoreCells
                                                               : request.stencil.coreCells);
            phase = 1;
        }
        void Dependent()
        {
            DNDS_check_throw_info(phase == 1, "equation auxiliary dependent phase requires core");
            complete.emplace(source.ReadWithGhost()); // An unfinished halo is retriable before any dependent write.
            if (gradients)
            {
                gradients->_data->Check(source, gradients->fields, data, time, true);
                for (const auto &field : gradients->fields)
                    gradientGhost.push_back(field.Ghost().Read());
                gradientViews = R::MetadataBuffer<RHSFieldView>::Generate(gradientViews.size(), [&](index iField)
                                                                          { return RHSFieldView{gradientViews[iField].owned, gradientGhost[iField].View<R::UniformTableRows>()}; });
            }
            RefreshDerived(true);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, *complete);
            const auto request = KernelInput(blocks.HostView());
            RequireNumericalSuccess(CPU::PrepareAuxiliaryMeans(request, data->equation, evaluator, data->metrics.NumCell(), data->metrics.NumCellProc()), "dependent auxiliary means");
            RequireNumericalSuccess(CPU::PrepareAuxiliaryFaces(request, data->equation, data->boundaryPoint, evaluator, request.stencil.auxiliaryDependentFaces), "dependent auxiliary faces");
            CPU::ReduceAuxiliaryCells(request, request.stencil.dependentCells);
            phase = 2;
        }
    };
    FiniteVolume::AuxiliaryPreparation
    FiniteVolume::AuxiliaryPreparation::BuildState(
        const FiniteVolume &plan, const EquationDofs &state,
        const AuxiliaryValues &output, real time,
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        AuxiliaryInputs inputs)
    {
        return AuxiliaryPreparation(std::make_unique<State>(
            plan, state, output, time, std::move(workers), inputs));
    }

    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation &FiniteVolume::AuxiliaryPreparation::operator=(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation::~AuxiliaryPreparation() = default;
    void FiniteVolume::AuxiliaryPreparation::ApplyInterfaceIndependent()
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        _state->InterfaceIndependent();
    }
    void FiniteVolume::AuxiliaryPreparation::BindBoundary(const BoundaryInputs &input)
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        _state->BindBoundary(input);
    }
    void FiniteVolume::AuxiliaryPreparation::Apply(CFV::ReconstructionRegion region)
    {
        DNDS_check_throw_info(_state, "equation auxiliary scope is finished or moved");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core || region == CFV::ReconstructionRegion::HaloDependent, "invalid auxiliary region");
        if (region != CFV::ReconstructionRegion::HaloDependent)
            _state->Core();
        if (region != CFV::ReconstructionRegion::Core)
            _state->Dependent();
    }
    void FiniteVolume::AuxiliaryPreparation::Finish(AuxiliaryValues &output)
    {
        DNDS_check_throw_info(_state && _state->phase == 2, "cannot publish incomplete equation auxiliary preparation");
        const auto a = EquationAuxiliaryArrays(_state->buffers), b = EquationAuxiliaryArrays(output);
        for (int k = 0; k < 9; ++k)
            DNDS_check_throw_info(bool(a[k]) == bool(b[k]) && (!a[k] || a[k]->SharedStorage() == b[k]->SharedStorage()), "equation auxiliary buffers were replaced");
        int k = 0;
        for (const auto *field : {&output.cellSpectralSum, &output.cellWaveJump, &output.cellPressureMinimum})
        {
            const std::optional<CFV::FVField> *original[]{&_state->buffers.cellSpectralSum, &_state->buffers.cellWaveJump, &_state->buffers.cellPressureMinimum};
            DNDS_check_throw_info(!*field || ((*field)->Space().Same(_state->source.Space()) && (*field)->Ghost().SharedStorage() == (*original[k])->Ghost().SharedStorage()), "equation auxiliary ghost/space changed");
            ++k;
        }
        for (auto &write : _state->writes)
            write.reset();
        _state->scratchWrite.reset();
        auto origin = std::make_shared<AuxiliaryValues::EquationOrigin>(_state->data, _state->source, output, _state->time,
                                                                        _state->boundary);
        if (_state->gradients)
            for (const auto &field : _state->gradients->fields)
                for (const auto *array : {&field.Owned(), &field.Ghost()})
                {
                    origin->gradientInputs.push_back(*array);
                    origin->gradientVersions.push_back({array->Version(), array->ShapeVersion()});
                }
        output._equationOrigin = std::move(origin);
        _state.reset();
    }

}

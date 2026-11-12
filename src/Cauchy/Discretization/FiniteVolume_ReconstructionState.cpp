#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;
    namespace
    {
        EquationDofs BindFields(const EquationDofs &prototype, Algebra::DofBundle fields, index modes)
        {
            auto schema = prototype.Schema();
            std::vector<EquationDofBinding> bindings;
            for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
            {
                const auto location = prototype.LocateBlock(iBlock);
                auto &block = schema.blocks[iBlock];
                bindings.push_back({block.name, prototype.Fields().Name(location.physicalField), location.firstComponent * modes});
                if (modes != 1)
                {
                    auto components = std::move(block.components);
                    block.components.clear();
                    block.vectorStarts.clear();
                    for (const auto &component : components)
                        for (index mode = 0; mode < modes; ++mode)
                            block.components.push_back({component.name + ":mode" + std::to_string(mode), component.units, "polynomial-coefficient"});
                }
            }
            if (modes != 1)
                schema.family = "polynomial:" + schema.family;
            return {std::move(schema), std::move(fields), std::move(bindings)};
        }
    }
    bool FiniteVolume::ReconstructionState::FinalizedCurrent() const
    {
        const auto current = [](const EquationDofOperations::Binding &binding, const EquationDofs &values)
        {
            if (binding.snapshot.PhysicalFieldNumber() != values.PhysicalFieldNumber())
                return false;
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &saved = binding.snapshot.PhysicalField(iField).Owned();
                const auto &value = values.PhysicalField(iField).Owned();
                if (saved.SharedStorage() != value.SharedStorage() ||
                    binding.versions[size_t(iField)] != value.Version())
                    return false;
            }
            return true;
        };
        if (!finalizedFrom || !finalizedPublication)
            return false;
        return current(*finalizedFrom, coefficients) &&
               current(*finalizedPublication, FinalizedValues());
    }
    void FiniteVolume::ReconstructionState::PublishFinalized()
    {
        finalizedFrom.emplace(coefficients);
        finalizedPublication.emplace(FinalizedValues());
    }
    const EquationDofs &FiniteVolume::ReconstructionState::FinalizedValues() const
    {
        return distinctFinalized ? fluxCoefficients : coefficients;
    }
    EquationDofs &FiniteVolume::ReconstructionState::FinalizedValues()
    {
        return distinctFinalized ? fluxCoefficients : coefficients;
    }
    EquationDofs &FiniteVolume::ReconstructionState::MutableFinalized()
    {
        DNDS_check_throw_info(distinctFinalized,
                              "unfiltered reconstruction has no mutable finalized scratch");
        return fluxCoefficients;
    }
    std::vector<R::TransferBinding> &FiniteVolume::ReconstructionState::FinalizedHalo()
    {
        return distinctFinalized ? fluxHalo : coefficientHalo;
    }

    EquationDofs FiniteVolume::ReconstructionState::BuildCoefficients(const Data &data, const EquationDofs &state)
    {
        // Direct P0/P1 publications consume means/physical gradients. Keep
        // only a managed alias for the common schedule; allocate no modal
        // payload and bind no coefficient transfer.
        if (!data.reconstruction)
            return state.NumericAlias();
        Algebra::DofBundle fields(state.Space());
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto &physical = state.PhysicalField(iField);
            fields.Add(state.Fields().Name(iField), data.reconstruction->BuildCoefficients(
                                                        index(physical.Width()),
                                                        FiniteVolumeDetail::FieldLayout(physical.Owned().Layout(), data.metrics.NumCell(),
                                                                                        index(physical.Width()) * data.reconstruction->NRecDOF())));
        }
        return BindFields(state, std::move(fields), data.reconstruction->NRecDOF());
    }
    FiniteVolume::ReconstructionState::ReconstructionState(std::shared_ptr<const Data> data, EquationDofs &state)
        : owner(std::move(data)), binding(state), coefficients(BuildCoefficients(*owner, state)),
          distinctFinalized(owner->reconstruction && ((owner->settings.numerics.limiter == "WBAP" &&
                                                       !(owner->settings.numerics.reconstruction == "gradient" &&
                                                         !owner->settings.numerics.positivity)) ||
                                                      bool(owner->limitedVRGeometry) || owner->settings.numerics.positivity)),
          fluxCoefficients(distinctFinalized ? BuildCoefficients(*owner, state)
                                             : coefficients.NumericAlias())
    {
        for (index iField = 0; iField < coefficients.PhysicalFieldNumber(); ++iField)
        {
            meanHalo.push_back(owner->BindTrackedHalo(state.PhysicalField(iField)));
            if (owner->reconstruction)
            {
                Algebra::Fill(coefficients.PhysicalField(iField), 0.);
                coefficientHalo.push_back(owner->BindHalo(coefficients.PhysicalField(iField)));
            }
            if (distinctFinalized)
            {
                Algebra::Fill(fluxCoefficients.PhysicalField(iField), 0.);
                fluxHalo.push_back(owner->BindHalo(fluxCoefficients.PhysicalField(iField)));
            }
        }
        if (owner->limitedVRGeometry)
        {
            limitedVRHistory.emplace(R::RowLayout::Dense(owner->metrics.NumCell(), 2));
            auto write = limitedVRHistory->Write();
            const auto values = write.View<R::FixedRows<2>>();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                values(iCell, 0) = values(iCell, 1) = 0;
        }
        PublishFinalized();
    }
    FiniteVolume::FinalizedReconstruction::FinalizedReconstruction(
        std::shared_ptr<const Data> owner, const EquationDofs &source,
        const EquationDofs &values, bool distinct)
        : _owner(std::move(owner)), _source(source.NumericAlias()),
          _values(values.NumericAlias()), _distinct(distinct)
    {
        for (index iField = 0; iField < _source.PhysicalFieldNumber(); ++iField)
            _sourceVersions.push_back(_source.PhysicalField(iField).Owned().Version());
        for (index iField = 0; iField < _values.PhysicalFieldNumber(); ++iField)
            _valueVersions.push_back(_values.PhysicalField(iField).Owned().Version());
    }
    void FiniteVolume::FinalizedReconstruction::Validate() const
    {
        DNDS_check_throw_info(_owner &&
                                  _sourceVersions.size() == size_t(_source.PhysicalFieldNumber()) &&
                                  _valueVersions.size() == size_t(_values.PhysicalFieldNumber()),
                              "invalid finalized reconstruction publication");
        for (index iField = 0; iField < _source.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(_sourceVersions[size_t(iField)] ==
                                      _source.PhysicalField(iField).Owned().Version(),
                                  "warm reconstruction changed after finalization");
        for (index iField = 0; iField < _values.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(_valueVersions[size_t(iField)] ==
                                      _values.PhysicalField(iField).Owned().Version(),
                                  "finalized reconstruction changed after publication");
    }
    const EquationDofs &FiniteVolume::FinalizedReconstruction::Values() const
    {
        Validate();
        return _values;
    }
    const EquationDofs &FiniteVolume::ReconstructionGuess(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data,
                              "foreign equation reconstruction state");
        return state.coefficients;
    }
    FiniteVolume::ReconstructionStorageInfo FiniteVolume::InspectReconstructionStorage(
        const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data,
                              "foreign equation reconstruction state");
        ReconstructionStorageInfo result;
        result.polynomialPlan = bool(_data->reconstruction);
        result.distinctFinalized = state.distinctFinalized;
        result.aliasesEquationState = !result.polynomialPlan;
        for (index iField = 0; result.aliasesEquationState &&
                               iField < state.coefficients.PhysicalFieldNumber();
             ++iField)
            result.aliasesEquationState =
                state.coefficients.PhysicalField(iField).Owned().SharedStorage() ==
                state.binding.snapshot.PhysicalField(iField).Owned().SharedStorage();
        if (result.polynomialPlan)
        {
            result.coefficientFields = state.coefficients.PhysicalFieldNumber() *
                                       (state.distinctFinalized ? 2 : 1);
            result.coefficientTransferBindings = index(state.coefficientHalo.size() +
                                                       state.fluxHalo.size());
            const auto accumulate = [&](const EquationDofs &values)
            {
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    result.coefficientOwnedBytes = CheckedSize::Add(
                        result.coefficientOwnedBytes,
                        values.PhysicalField(iField).Owned().Storage().Bytes());
                    result.coefficientGhostBytes = CheckedSize::Add(
                        result.coefficientGhostBytes,
                        values.PhysicalField(iField).Ghost().Storage().Bytes());
                }
            };
            accumulate(state.coefficients);
            if (state.distinctFinalized)
                accumulate(state.fluxCoefficients);
        }
        return result;
    }
    void FiniteVolume::InterfaceDonorPublication::Validate() const
    {
        DNDS_check_throw_info(_owner &&
                                  _meanVersions.size() ==
                                      size_t(_means.PhysicalFieldNumber()),
                              "invalid interface donor publication");
        _owner->equation.ValidateStateSchema(_means);
        for (index iField = 0; iField < _means.PhysicalFieldNumber(); ++iField)
        {
            _owner->metrics.ValidateField(
                _means.PhysicalField(iField),
                _means.PhysicalField(iField).Width());
            DNDS_check_throw_info(
                _meanVersions[size_t(iField)] ==
                    _means.PhysicalField(iField).Owned().Version(),
                "interface donor mean changed after publication");
        }
        if (_kind == Kind::Variational)
        {
            DNDS_check_throw_info(_owner->reconstruction && _coefficients &&
                                      _reconstructionVersions.size() ==
                                          size_t(_coefficients->PhysicalFieldNumber()),
                                  "variational interface donor is incomplete");
            for (index iField = 0; iField < _coefficients->PhysicalFieldNumber();
                 ++iField)
            {
                DNDS_check_throw_info(
                    _coefficients->Fields().Name(iField) ==
                            _means.Fields().Name(iField) &&
                        _coefficients->PhysicalField(iField).Width() ==
                            _means.PhysicalField(iField).Width() *
                                _owner->reconstruction->NRecDOF(),
                    "variational interface donor field grouping differs");
                _owner->metrics.ValidateField(
                    _coefficients->PhysicalField(iField),
                    _coefficients->PhysicalField(iField).Width());
                DNDS_check_throw_info(
                    _reconstructionVersions[size_t(iField)] ==
                        _coefficients->PhysicalField(iField).Owned().Version(),
                    "interface donor coefficients changed after publication");
            }
        }
        if (_kind == Kind::Gradient)
        {
            DNDS_check_throw_info(
                _gradients.size() == size_t(_means.PhysicalFieldNumber()) &&
                    _reconstructionVersions.size() == _gradients.size(),
                "gradient interface donor is incomplete");
            for (index iField = 0; iField < index(_gradients.size()); ++iField)
            {
                _owner->metrics.ValidateField(
                    _gradients[size_t(iField)],
                    _means.PhysicalField(iField).Width() *
                        _owner->metrics.Dimension());
                DNDS_check_throw_info(
                    _reconstructionVersions[size_t(iField)] ==
                        _gradients[size_t(iField)].Owned().Version(),
                    "interface donor gradient changed after publication");
            }
        }
    }
    FiniteVolume::InterfaceDonorPublication FiniteVolume::PublishInterfaceDonor(
        const EquationDofs &state, const ReconstructionState &reconstruction,
        const Workspace &work, bool prepared) const
    {
        ValidateWorkspace(state, work);
        DNDS_check_throw_info(reconstruction.owner == _data,
                              "foreign interface reconstruction state");
        reconstruction.binding.Check(state);
        InterfaceDonorPublication result(_data, state.NumericAlias());
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            result._meanVersions.push_back(
                state.PhysicalField(iField).Owned().Version());
        if (_data->settings.numerics.reconstruction == "variational")
        {
            const auto &values = prepared ? reconstruction.FinalizedValues()
                                          : reconstruction.coefficients;
            if (prepared)
                DNDS_check_throw_info(reconstruction.FinalizedCurrent(),
                                      "interface donor reconstruction is stale");
            result._kind = InterfaceDonorPublication::Kind::Variational;
            result._coefficients = values.NumericAlias();
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                result._reconstructionVersions.push_back(
                    values.PhysicalField(iField).Owned().Version());
        }
        else if (prepared &&
                 _data->settings.numerics.reconstruction == "gradient")
        {
            DNDS_check_throw_info(
                work.nativeDirectGradient &&
                    work.directGradients.size() ==
                        size_t(state.PhysicalFieldNumber()) &&
                    work.directGradientPublications.size() ==
                        work.directGradients.size(),
                "interface donor gradient publication is unavailable");
            result._kind = InterfaceDonorPublication::Kind::Gradient;
            result._gradients = work.directGradients;
            for (index iField = 0; iField < index(work.directGradients.size());
                 ++iField)
            {
                const auto &field = work.directGradients[size_t(iField)].Owned();
                const auto &publication =
                    work.directGradientPublications[size_t(iField)][0];
                DNDS_check_throw_info(
                    publication.Current(
                        *_data, state.PhysicalField(iField).Owned(),
                        state.PhysicalField(iField).Owned(), field,
                        work.incoming, work.rhsTime),
                    "interface donor gradient publication is stale");
                result._reconstructionVersions.push_back(field.Version());
            }
        }
        result.Validate();
        return result;
    }
    FiniteVolume::InterfaceDonorPublication
    FiniteVolume::PublishVariationalInterfaceDonor(
        const EquationDofs &state, const EquationDofs &coefficients) const
    {
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(
            _data->reconstruction && state.Space().Same(Metrics().Space()) &&
                coefficients.Space().Same(state.Space()) &&
                coefficients.Device() == state.Device() &&
                coefficients.PhysicalFieldNumber() == state.PhysicalFieldNumber(),
            "checkpoint interface donor needs compatible variational fields");
        InterfaceDonorPublication result(_data, state.NumericAlias());
        result._kind = InterfaceDonorPublication::Kind::Variational;
        result._coefficients = coefficients.NumericAlias();
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            result._meanVersions.push_back(
                state.PhysicalField(iField).Owned().Version());
            result._reconstructionVersions.push_back(
                coefficients.PhysicalField(iField).Owned().Version());
        }
        result.Validate();
        return result;
    }
    void FiniteVolume::ValidateWorkspace(const EquationDofs &state, const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) &&
                                  state.ComponentNumber() == work.componentNumber &&
                                  state.PhysicalFieldNumber() == work.physicalFieldNumber,
                              "equation workspace schema or row space differs");
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(state.PhysicalField(iField).Width() == work.fields[iField].boundaryValues.Layout().Width(),
                                  "equation workspace physical field width differs");
    }
    bool FiniteVolume::HasRoeJacobianKernel(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        // Roe assembly is an operator-local flow5 leaf. It consumes the
        // canonical state/auxiliary views directly and does not require the
        // archived whole-flow reconstruction workspace.
        return !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
               work.componentNumber == 5 && work.physicalFieldNumber == 1;
    }
    index FiniteVolume::PrepareFaceCharacteristicMatrices(const CFV::FVField &state, index firstComponent,
                                                          R::Array<real> &forwardMatrices,
                                                          R::Array<real> &inverseMatrices) const
    {
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        index fallbacks = 0;
        {
            auto owned = state.Owned().Read(), ghost = state.Ghost().Read();
            const RHSFieldView u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            DNDS_check_throw_info(firstComponent >= 0 && firstComponent + 5 <= state.Width(),
                                  "flow characteristic range differs from its physical field");
            const auto loadFlow = [&](index iCell)
            {
                FlowState value;
                for (int j = 0; j < 5; ++j)
                    value[j] = u(iCell, firstComponent + j);
                return value;
            };
            auto forward = forwardMatrices.Write(), inverse = inverseMatrices.Write();
            const auto l = forward.View<R::TableRows>(), r = inverse.View<R::TableRows>();
            for (auto iFace : _data->stencilFaces)
            {
                const auto cells = mesh.FaceCells(iFace);
                if (cells[0] < 0 || cells[0] >= Metrics().NumCellProc())
                    continue;
                const auto left = loadFlow(cells[0]);
                auto right = left;
                if (cells[1] >= 0 && cells[1] < Metrics().NumCellProc())
                    right = FiniteVolumeDetail::RotateState(loadFlow(cells[1]), CFV::Detail::OtherFrameInCell(mesh, iFace, 0));
                const auto normal = metric.GetFaceNorm(metric.FaceQuadratureBegin(iFace));
                const auto matrices = _data->equation.PrepareCharacteristics(left, right, {normal[0], normal[1], normal[2]});
                fallbacks += matrices.componentwiseFallback;
                for (index j = 0; j < 25; ++j)
                    l(iFace, j) = matrices.forward[j], r(iFace, j) = matrices.inverse[j];
            }
        }
        return fallbacks;
    }
    index FiniteVolume::PrepareFaceCharacteristics(const CFV::FVField &state,
                                                   CFV::FaceCharacteristicTransform &transform,
                                                   index firstComponent) const
    {
        DNDS_check_throw_info(transform.first == firstComponent && transform.count == 5,
                              "flow characteristic range differs from its physical field");
        const auto fallbacks = PrepareFaceCharacteristicMatrices(
            state, firstComponent, transform.forward, transform.inverse);
        DNDS_check_throw_info(_data->reconstruction,
                              "modal characteristic transform requires variational reconstruction");
        _data->reconstruction->FinalizeFaceTransform(transform);
        return fallbacks;
    }
    index FiniteVolume::PrepareFaceCharacteristics(
        const CFV::FVField &state, CFV::GradientFaceCharacteristicTransform &transform,
        index firstComponent) const
    {
        DNDS_check_throw_info(transform.first == firstComponent && transform.count == 5,
                              "flow gradient characteristic range differs from its physical field");
        const auto fallbacks = PrepareFaceCharacteristicMatrices(
            state, firstComponent, transform.forward, transform.inverse);
        _data->gradient.FinalizeFaceTransform(transform);
        return fallbacks;
    }
    FiniteVolume::FinalizedReconstruction FiniteVolume::AcquireFinalizedReconstruction(
        const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        DNDS_check_throw_info(state.FinalizedCurrent(), "reconstruction has no current finalized publication");
        return {_data, state.coefficients, state.FinalizedValues(), state.distinctFinalized};
    }
    const EquationDofs &FiniteVolume::FluxReconstruction(const ReconstructionState &state, const Workspace &work) const
    {
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        DNDS_check_throw_info(state.FinalizedCurrent(), "reconstruction has no current finalized publication");
        return state.FinalizedValues();
    }
    void FiniteVolume::RestoreReconstructionGuess(ReconstructionState &state, const EquationDofs &guess,
                                                  OptionalRef<const Runtime::Array<real>> limitedVRHistory) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, guess, limitedVRHistory, {});
        restore.Commit();
        if (_data->settings.numerics.limiter == "WBAP" || _data->limitedVRGeometry ||
            _data->settings.numerics.positivity)
        {
            state.finalizedFrom.reset();
            state.finalizedPublication.reset();
        }
        else
            state.PublishFinalized();
    }
    void FiniteVolume::CopyReconstructionState(ReconstructionState &destination,
                                               const ReconstructionState &source) const
    {
        DNDS_check_throw_info(&destination != &source && destination.owner == _data && source.owner == _data,
                              "reconstruction state copy requires distinct states from this finite-volume plan");
        DNDS_check_throw_info(source.FinalizedCurrent(),
                              "source reconstruction has no current finalized polynomial");
        const auto history = source.limitedVRHistory
                                 ? OptionalRef<const R::Array<real>>(*source.limitedVRHistory)
                                 : OptionalRef<const R::Array<real>>{};
        auto restore = ReserveReconstructionRestoreImpl(destination, source.coefficients, history, {}, true,
                                                        OptionalRef<const EquationDofs>(source.FinalizedValues()));
        restore.Commit();
        destination.PublishFinalized();
    }
    void FiniteVolume::RestoreReconstructionState(ReconstructionState &state,
                                                  const EquationDofs &warm, const EquationDofs &finalized,
                                                  OptionalRef<const Runtime::Array<real>> limitedVRHistory) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, warm, limitedVRHistory, {}, true,
                                                        OptionalRef<const EquationDofs>(finalized));
        restore.Commit();
        state.PublishFinalized();
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRHistory(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        return state.limitedVRHistory;
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRAlpha(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.limitedVR ? std::optional<R::Array<real>>(work.limitedVR->alpha) : std::nullopt;
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestore(std::shared_ptr<ReconstructionState> state, const EquationDofs &guess,
                                                                                   OptionalRef<const Runtime::Array<real>> limitedVRHistory,
                                                                                   bool collectiveChecks) const
    {
        const auto validate = [&]
        { DNDS_check_throw_info(bool(state), "null equation reconstruction state"); };
        R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                           validate, "reconstruction workspace validation failed");
        auto &reference = *state;
        return ReserveReconstructionRestoreImpl(reference, guess, limitedVRHistory, std::move(state), collectiveChecks);
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestoreImpl(ReconstructionState &state, const EquationDofs &guess,
                                                                                       OptionalRef<const Runtime::Array<real>> limitedVRHistory,
                                                                                       std::shared_ptr<ReconstructionState> retained, bool collectiveChecks,
                                                                                       OptionalRef<const EquationDofs> finalized) const
    {
        const auto *historyInput = limitedVRHistory.Get();
        const auto stage = [&](const auto &run)
        {
            R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                               run, "reconstruction restore preparation failed");
        };
        struct Reservation
        {
            std::shared_ptr<ReconstructionState> owner;
            R::SessionAccess::Lease session;
            std::optional<EquationDofs::ReadScope> source, fluxSource;
            std::optional<EquationDofs::WriteScope> destination, fluxDestination;
            std::optional<R::Array<real>::ReadLease> historySource;
            std::optional<R::Array<real>::WriteLease> historyDestination;
            std::vector<EquationBlockArrayView<const real>> input, fluxInput;
            std::vector<EquationBlockArrayView<real>> output, fluxOutput;
            bool started = false;
            ~Reservation()
            {
                if (!started && destination)
                    destination->CancelUnmodified();
                if (!started && fluxDestination)
                    fluxDestination->CancelUnmodified();
                if (!started && historyDestination)
                    historyDestination->CancelUnmodified();
            }
        };
        std::shared_ptr<Reservation> reservation;
        stage([&]
              {
                reservation = std::make_shared<Reservation>();
                reservation->owner = std::move(retained);
                reservation->session = state.session.Acquire();
                DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
                state.coefficients.RequireCompatible(guess);
                EquationDofOperations::RequireDistinct(state.coefficients, guess);
                const auto &finalInput = state.distinctFinalized && finalized ? *finalized : guess;
                if (state.distinctFinalized)
                {
                    state.fluxCoefficients.RequireCompatible(guess);
                    EquationDofOperations::RequireDistinct(state.fluxCoefficients, guess);
                    state.fluxCoefficients.RequireCompatible(finalInput);
                    EquationDofOperations::RequireDistinct(state.coefficients, finalInput);
                    EquationDofOperations::RequireDistinct(state.fluxCoefficients, finalInput);
                }
                DNDS_check_throw_info(state.limitedVRHistory || !historyInput,
                                      "LVR history requires an active matching reconstruction state");
                if (historyInput)
                {
                    DNDS_check_throw_info(historyInput->GetDevice() == R::Device::Host() &&
                                              !historyInput->Layout().IsVariable() &&
                                              historyInput->RowCount() == Metrics().NumCell() &&
                                              historyInput->Layout().Width() == 2,
                                          "LVR history shape or placement differs");
                    for (index iField = 0; iField < guess.PhysicalFieldNumber(); ++iField)
                    {
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                         state.coefficients.PhysicalField(iField).Owned().Storage()),
                                              "LVR history aliases reconstruction output");
                        if (state.distinctFinalized)
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                             state.fluxCoefficients.PhysicalField(iField).Owned().Storage()),
                                                  "LVR history aliases finalized reconstruction output");
                    }
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(), state.limitedVRHistory->Storage()),
                                          "LVR history aliases its destination");
                    const auto history = historyInput->Read();
                    const auto values = history.View<R::UniformTableRows>();
                    const std::array<real, 2> caps{_data->settings.numerics.limitedVR.alphaMax,
                                                   _data->settings.numerics.limitedVR.contactAlphaMax};
                    for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            DNDS_check_throw_info(std::isfinite(values(iCell, j)) && values(iCell, j) >= 0 && values(iCell, j) <= caps[j],
                                                  "LVR history exceeds its valid branch cap");
                }
                reservation->source.emplace(guess.ReadOwned());
                if (state.distinctFinalized && finalized)
                    reservation->fluxSource.emplace(finalInput.ReadOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    const auto input = reservation->source->Block(iBlock);
                    for (index iCell = 0; iCell < input.owned.RowCount(); ++iCell)
                        for (index j = 0; j < input.ComponentNumber(); ++j)
                            DNDS_check_throw_info(std::isfinite(input(iCell, j)), "nonfinite equation reconstruction guess");
                    reservation->input.push_back(input.owned);
                    if (state.distinctFinalized)
                    {
                        const auto fluxInput = finalized ? reservation->fluxSource->Block(iBlock) : input;
                        for (index iCell = 0; iCell < fluxInput.owned.RowCount(); ++iCell)
                            for (index j = 0; j < fluxInput.ComponentNumber(); ++j)
                                DNDS_check_throw_info(std::isfinite(fluxInput(iCell, j)),
                                                      "nonfinite finalized equation reconstruction");
                        reservation->fluxInput.push_back(fluxInput.owned);
                    }
                }
                reservation->destination.emplace(state.coefficients.WriteOwned());
                if (state.distinctFinalized)
                    reservation->fluxDestination.emplace(state.fluxCoefficients.WriteOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    reservation->output.push_back(reservation->destination->Block(iBlock));
                    if (state.distinctFinalized)
                        reservation->fluxOutput.push_back(reservation->fluxDestination->Block(iBlock));
                }
                if (historyInput)
                    reservation->historySource.emplace(historyInput->Read());
                if (state.limitedVRHistory)
                    reservation->historyDestination.emplace(state.limitedVRHistory->Write()); });
        std::optional<ReconstructionRestore> restore;
        stage([&]
              { restore.emplace(ReconstructionRestore([reservation]() noexcept
                                                      {
            reservation->started = true;
            for (index iBlock = 0; iBlock < index(reservation->input.size()); ++iBlock)
            {
                const auto input = reservation->input[iBlock];
                const auto output = reservation->output[iBlock];
                for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                    for (index j = 0; j < output.ComponentNumber(); ++j)
                        output(iCell, j) = input(iCell, j);
                if (!reservation->fluxOutput.empty())
                {
                    const auto fluxInput = reservation->fluxInput[iBlock];
                    const auto fluxOutput = reservation->fluxOutput[iBlock];
                    for (index iCell = 0; iCell < fluxOutput.RowCount(); ++iCell)
                        for (index j = 0; j < fluxOutput.ComponentNumber(); ++j)
                            fluxOutput(iCell, j) = fluxInput(iCell, j);
                }
            }
            if (reservation->historyDestination)
            {
                const auto output = reservation->historyDestination->View<R::UniformTableRows>();
                if (reservation->historySource)
                {
                    const auto input = reservation->historySource->View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            output(iCell, j) = input(iCell, j);
                }
                else
                    for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            output(iCell, j) = 0;
            } })); });
        return std::move(*restore);
    }
    void FiniteVolume::ResetReconstruction(ReconstructionState &state, Workspace &work) const
    {
        auto stateSession = state.session.Acquire();
        auto session = work.session.Acquire();
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        // Reserve every destination before changing any value. In particular,
        // a retained LVR history reader must not allow coefficients to be
        // cleared first and only then report the conflict.
        std::optional<EquationDofs::WriteScope> coefficients, fluxCoefficients;
        std::vector<R::Array<real>::WriteLease> temporary;
        std::optional<R::Array<real>::WriteLease> previous, alpha;
        try
        {
            if (_data->reconstruction)
                coefficients.emplace(state.coefficients.WriteOwned());
            if (state.distinctFinalized)
                fluxCoefficients.emplace(state.fluxCoefficients.WriteOwned());
            temporary.reserve(work.fields.size());
            for (auto &field : work.fields)
                if (field.reconstruction)
                    temporary.emplace_back(field.reconstruction->temporary.Owned().Write());
            if (work.limitedVR)
            {
                previous.emplace(state.limitedVRHistory->Write());
                alpha.emplace(work.limitedVR->alpha.Write());
            }
        }
        catch (...)
        {
            if (coefficients)
                coefficients->CancelUnmodified();
            if (fluxCoefficients)
                fluxCoefficients->CancelUnmodified();
            for (auto &write : temporary)
                write.CancelUnmodified();
            if (previous)
                previous->CancelUnmodified();
            if (alpha)
                alpha->CancelUnmodified();
            throw;
        }
        if (coefficients)
        {
            const auto layout = coefficients->ComponentLayout();
            const auto values = layout.View();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = 0;
        }
        if (fluxCoefficients)
        {
            const auto layout = fluxCoefficients->ComponentLayout();
            const auto values = layout.View();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = 0;
        }
        for (auto &write : temporary)
        {
            const auto values = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.Width(); ++j)
                    values(iCell, j) = 0;
        }
        for (auto *write : {previous ? &*previous : nullptr, alpha ? &*alpha : nullptr})
            if (write)
            {
                const auto values = write->View<R::UniformTableRows>();
                for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (index j = 0; j < values.Width(); ++j)
                        values(iCell, j) = 0;
            }
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        state.finalizedFrom.reset();
        state.finalizedPublication.reset();
    }
    int FiniteVolume::LastReconstructionIterations(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.iterations;
    }
    real FiniteVolume::LastReconstructionUpdate(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.relativeUpdate;
    }
    void FiniteVolume::LimitReconstructionPP(const EquationDofs &state, ReconstructionState &reconstruction, Workspace &work) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        Metrics().Communication()->LocalStage([&]
                                              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(work.positivity, "equation reconstruction PP was not requested");
            DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
            work.positivity->Apply(state, reconstruction.coefficients, reconstruction.MutableFinalized(), true);
            reconstruction.PublishFinalized(); });
    }
    std::pair<index, real> FiniteVolume::PositivityReconstructionStats(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        auto result = work.positivity
                          ? work.positivity->LocalStatistics()
                          : std::pair<index, real>{work.directPPLimited,
                                                   work.directPPMinimum};
        result.first = MPI::AllreduceValue<MPI::Reduction::Sum>(result.first, Metrics().Communicator());
        result.second = MPI::AllreduceValue<MPI::Reduction::Min>(result.second, Metrics().Communicator());
        return result;
    }
}

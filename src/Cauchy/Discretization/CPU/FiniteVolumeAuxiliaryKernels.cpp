#include "Cauchy/Discretization/FiniteVolumeAuxiliaryInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Discretization/AuxiliaryReduction.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "DNDS/MPI/Failure.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using AuxiliaryDetail::EquationAuxiliaryArrays;

    void FiniteVolume::FillMeanGradientBoundaryCPU(
        const EquationDofs &state, MeanGradients &gradients,
        EquationPointEvaluator &evaluator, real time,
        RHSScalarView incoming) const
    {
        using namespace EquationFVDetail;
        DNDS_check_throw_info(gradients._data,
                              "mean gradient workspace is uninitialized");
        auto &work = *gradients._data;
        const auto read = state.ReadOwned();
        const auto blocks = BlockViews(state, read);
        std::vector<R::Array<real>::WriteLease> writes;
        std::vector<RHSOutputView> values;
        for (auto &array : work.boundary)
            writes.push_back(array.Write());
        for (const auto &write : writes)
            values.push_back(write.View<R::UniformTableRows>());
        index first = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        for (const auto &block : state.Schema().blocks)
        {
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
                speciesFirst = first;
            if (block.name == "rans")
                ransFirst = first;
            first += index(block.components.size());
        }
        const int independent = _data->equation.HasMixture() ? _data->equation.Mixture().SpeciesNumber() - 1 : 0;
        const auto *model = _data->equation.HasRANS() ? &_data->equation.RANS() : nullptr;
        std::optional<WallDistanceRead> walls;
        if (model && model->ComponentNumber() == 2)
            walls.emplace(_data->wallDistance->AcquireRead());
        const auto wall = walls ? walls->HostView() : WallDistanceView{};
        PointScratch mean(state.ComponentNumber()), point(state.ComponentNumber());
        const auto frame = _data->motion.At(time);
        for (index iPoint = 0; iPoint < _data->gradient.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto &sample = _data->samples[_data->boundaryEntries[_data->gradientBoundaryPoints[iPoint]]];
            Gather(blocks, sample.cell, mean);
            point.values = mean.values;
            _data->boundaryPoint.Evaluate(
                point.values, sample.boundary, sample.boundaryPoint,
                {sample.normal[0], sample.normal[1], sample.normal[2]},
                {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                evaluator, frame, time, BoundaryPointStateKind::FaceTrace,
                OptionalRef<const std::vector<real>>(mean.values),
                walls ? wall.cellMean(sample.cell, 0) : 0, incoming);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
            if (model && model->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                point.values[ransFirst] = 0;
#endif
            first = 0;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto location = state.LocateBlock(iBlock);
                for (index j = 0; j < location.componentNumber; ++j)
                    values[location.physicalField](iPoint, location.firstComponent + j) = point.values[first++];
            }
        }
    }

    struct FiniteVolume::AuxiliaryPreparation::State
    {
        std::shared_ptr<const Data> data;
        EquationDofs source;
        AuxiliaryValues buffers;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        CFV::FiniteVolumeRead geometry;
        EquationDofs::ReadScope owned;
        std::optional<EquationDofs::ReadScope> complete;
        std::optional<MeanGradients> gradients;
        std::optional<BoundaryInputs> boundary;
        std::optional<R::Array<real>::ReadLease> boundaryRead;
        RHSScalarView incoming;
        std::vector<R::Array<real>::ReadLease> gradientOwned, gradientGhost;
        std::vector<RHSFieldView> gradientViews;
        std::vector<EquationBlockLocation> locations;
        std::optional<WallDistanceRead> walls;
        WallDistanceView wallValues;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        std::array<std::optional<R::Array<real>::WriteLease>, 7> writes;
        std::array<RHSOutputView, 7> values;
        std::optional<R::Array<real>::WriteLease> scratchWrite;
        RHSOutputView scratch;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int independent;
        std::vector<index> vectorStarts;
        real time;
        RigidFrame frame;
        int phase = 0;
        bool started = false;

        State(std::shared_ptr<const Data> plan, const EquationDofs &state, const AuxiliaryValues &output, real atTime,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, AuxiliaryInputs inputs)
            : data(std::move(plan)), source(state.NumericAlias()), buffers(output), workers(std::move(pool)),
              geometry(data->metrics.AcquireRead()), owned(source.ReadOwned()),
              independent(data->equation.HasMixture() ? data->equation.Mixture().SpeciesNumber() - 1 : 0),
              time(atTime), frame(data->motion.At(time))
        {
            if (inputs.meanGradients)
            {
                gradients = *inputs.meanGradients;
                DNDS_check_throw_info(gradients->_data, "mean gradient input is uninitialized");
                gradients->_data->Check(source, gradients->fields, data, time, false);
                for (const auto &field : gradients->fields)
                    gradientOwned.push_back(field.Owned().Read());
                for (const auto &read : gradientOwned)
                    gradientViews.push_back({read.View<R::UniformTableRows>(), {}});
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
            for (index iBlock = 0; iBlock < source.BlockNumber(); ++iBlock)
                locations.push_back(source.LocateBlock(iBlock));
            for (const auto &block : source.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans")
                    ransFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectorStarts.push_back(first + vector);
                first += index(block.components.size());
            }
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
        EquationPointEvaluator Evaluator() const
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
        void LoadGradients(index iCell, EquationFVDetail::PointScratch &point) const
        {
            if (!gradients)
                return;
            index first = 0;
            const int dim = data->metrics.Dimension();
            for (const auto &location : locations)
                for (index j = 0; j < location.componentNumber; ++j, ++first)
                    for (int d = 0; d < dim; ++d)
                        point.gradient(first, d) = gradientViews[location.physicalField](iCell, (location.firstComponent + j) * dim + d);
        }
        std::array<real, 3> Properties(EquationFVDetail::PointScratch &point, EquationPointEvaluator &evaluator,
                                       real wallDistance = 1, bool turbulent = true) const
        {
            const auto flow = point.Flow(flowFirst);
            const auto result = evaluator.Evaluate(flow, point.Species(speciesFirst, independent));
            real diffusion = result.diffusionBound;
            if (ransFirst >= 0 && turbulent)
            {
                const auto &model = data->equation.RANS();
                const auto transport = model.Transport(EquationFVDetail::RANSPoint(point, flowFirst, ransFirst, model, result, wallDistance, 0));
                diffusion = std::max({diffusion, real(4. / 3.) * (result.transport.viscosity + transport.viscosity) / flow[0],
                                      (result.transport.conductivity + transport.conductivity) / (flow[0] * result.thermo.cv)});
            }
            return {result.thermo.pressure, result.thermo.soundSpeed, diffusion};
        }
        void Means(const std::vector<EquationBlockReadView> &blocks, index first, index last, EquationPointEvaluator &evaluator)
        {
            if (!writes[0])
                return;
            EquationFVDetail::PointScratch point(source.ComponentNumber());
            for (index iCell = first; iCell < last; ++iCell)
            {
                EquationFVDetail::Gather(blocks, iCell, point);
                LoadGradients(iCell, point);
                const auto result = Properties(point, evaluator, walls ? wallValues.cellMean(iCell, 0) : 1);
                for (int j = 0; j < 3; ++j)
                    values[0](iCell, j) = result[j];
            }
        }
        void Faces(const std::vector<EquationBlockReadView> &blocks, const std::vector<index> &faces, EquationPointEvaluator &evaluator)
        {
            const bool spectral = writes[1] || writes[4], wave = writes[2] || writes[5], pressure = writes[3] || writes[6];
            if (!spectral && !wave && !pressure)
                return;
            const auto metric = geometry.HostView();
            std::optional<MeanThermodynamicView> derived;
            if (buffers._derivedThermodynamics)
                derived.emplace(MeanThermodynamicView::Derived(
                    *derivedOwned,
                    derivedGhost
                        ? OptionalRef<const DerivedFields::ReadScope>(*derivedGhost)
                        : OptionalRef<const DerivedFields::ReadScope>{}));
            EquationFVDetail::PointScratch left(source.ComponentNumber()), right(source.ComponentNumber()), face(source.ComponentNumber());
            const auto *ransModel = data->equation.HasRANS() ? &data->equation.RANS() : nullptr;
            const auto prepared = [&](index iCell, auto &point)
            {
                if (derived && ransFirst < 0)
                    return std::array<real, 3>{(*derived)(iCell, 0), (*derived)(iCell, 1), (*derived)(iCell, 2)};
                if (writes[0])
                    return std::array<real, 3>{values[0](iCell, 0), values[0](iCell, 1), values[0](iCell, 2)};
                return Properties(point, evaluator, walls ? wallValues.cellMean(iCell, 0) : 1);
            };
            for (const auto iAux : faces)
            {
                const auto iFace = data->stencilFaces[iAux], first = data->faceEntries[iAux];
                const auto &anchor = data->samples[first];
                EquationFVDetail::Gather(blocks, anchor.cell, left);
                LoadGradients(anchor.cell, left);
                const auto leftProperties = prepared(anchor.cell, left);
                const Riemann::AcousticState leftData{{left.Flow(flowFirst), leftProperties[0]}, leftProperties[1]};
                real convection = 0, diffusion = 0, jump = 0, minimum = leftProperties[0];
                const auto count = metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                for (index q = 0; q < count; ++q)
                {
                    const auto &sample = data->samples[first + q];
                    if (sample.other == UnInitIndex)
                    {
                        right.values = left.values;
                        right.gradient = left.gradient;
                        // Euler's EvaluateDt uses the owned mean on both sides
                        // of an ordinary physical boundary. A coupling
                        // interface supplies its explicitly published remote
                        // point state.
                        if (sample.boundary.kind == BoundaryKind::Interface)
                            data->boundaryPoint.Evaluate(
                                right.values, sample.boundary,
                                sample.boundaryPoint,
                                {sample.normal[0], sample.normal[1],
                                 sample.normal[2]},
                                {sample.ownPoint[0], sample.ownPoint[1],
                                 sample.ownPoint[2]},
                                evaluator, frame, time,
                                BoundaryPointStateKind::FaceTrace, {}, 0,
                                incoming);
                    }
                    else
                    {
                        EquationFVDetail::Gather(blocks, sample.other, right);
                        LoadGradients(sample.other, right);
                        // Setup proves exact identity, including translational
                        // periodic images. Keep the full transform for every
                        // genuinely rotated component frame.
                        if (sample.changesFrame)
                            EquationFVDetail::RotatePoint(right, sample.rotation, vectorStarts);
                    }
                    const auto rightProperties = sample.other == UnInitIndex
                                                     ? (sample.boundary.kind == BoundaryKind::Interface
                                                            ? Properties(right, evaluator)
                                                            : leftProperties)
                                                     : prepared(sample.other, right);
                    const Riemann::AcousticState rightData{{right.Flow(flowFirst), rightProperties[0]}, rightProperties[1]};
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    if (spectral)
                    {
                        convection = std::max({convection, Riemann::SignalSpeed(leftData, normal, grid), Riemann::SignalSpeed(rightData, normal, grid)});
                        real bound = std::max(leftProperties[2], rightProperties[2]);
                        if (walls)
                        {
                            for (index j = 0; j < source.ComponentNumber(); ++j)
                                face.values[j] = .5 * (left.values[j] + right.values[j]);
                            face.gradient = .5 * (left.gradient + right.gradient);
                            if (sample.other == UnInitIndex)
                            {
                                face.values = left.values;
                            }
                            bound = std::max(bound, Properties(face, evaluator, wallValues.faces(sample.face, q))[2]);
                        }
                        diffusion = std::max(diffusion, bound * data->auxiliaryFaceArea[iAux] *
                                                            (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                    }
                    if (wave)
                        jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, normal));
                    if (pressure)
                        minimum = std::min(minimum, rightProperties[0]);
                }
                if (writes[1])
                {
                    values[1](iAux, 0) = convection;
                    values[1](iAux, 1) = diffusion;
                }
                if (writes[2])
                    values[2](iAux, 0) = jump;
                if (writes[3])
                    values[3](iAux, 0) = minimum;
                if (scratchWrite)
                {
                    scratch(iAux, 0) = convection + diffusion;
                    scratch(iAux, 1) = jump;
                    scratch(iAux, 2) = minimum;
                }
            }
        }
        void Cells(const std::vector<index> &cells)
        {
            if (!writes[4] && !writes[5] && !writes[6])
                return;
            const auto metric = geometry.HostView();
            const real rotationBound = Geom::tPoint{frame.angularVelocity[0], frame.angularVelocity[1], frame.angularVelocity[2]}.norm();
            FiniteVolumeDetail::ReduceAuxiliaryCells(*data, cells, metric, rotationBound, writes, values, scratch);
        }

        void Core()
        {
            DNDS_check_throw_info(phase == 0, "equation auxiliary core already executed");
            started = true;
            RefreshDerived(false);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, owned);
            Means(blocks, 0, data->metrics.NumCell(), evaluator);
            Faces(blocks, data->auxiliaryCoreFaces, evaluator);
            Cells(data->partition.Core());
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
                for (index iField = 0; iField < index(gradientViews.size()); ++iField)
                    gradientViews[iField].ghost = gradientGhost[iField].View<R::UniformTableRows>();
            }
            RefreshDerived(true);
            auto evaluator = Evaluator();
            const auto blocks = EquationFVDetail::BlockViews(source, *complete);
            Means(blocks, data->metrics.NumCell(), data->metrics.NumCellProc(), evaluator);
            Faces(blocks, data->auxiliaryDependentFaces, evaluator);
            Cells(data->partition.HaloDependent());
            phase = 2;
        }
    };
    FiniteVolume::AuxiliaryPreparation
    FiniteVolume::AuxiliaryPreparation::BuildState(
        std::shared_ptr<const Data> plan, const EquationDofs &state,
        const AuxiliaryValues &output, real time,
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
        AuxiliaryInputs inputs)
    {
        return AuxiliaryPreparation(std::make_unique<State>(
            std::move(plan), state, output, time, std::move(workers), inputs));
    }

    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    FiniteVolume::AuxiliaryPreparation::AuxiliaryPreparation(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation &FiniteVolume::AuxiliaryPreparation::operator=(AuxiliaryPreparation &&) noexcept = default;
    FiniteVolume::AuxiliaryPreparation::~AuxiliaryPreparation() = default;
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

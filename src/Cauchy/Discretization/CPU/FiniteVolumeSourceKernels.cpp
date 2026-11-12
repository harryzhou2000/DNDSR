#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    void FiniteVolume::Workspace::SampleSourceState(const EquationDofs &state, real time)
    {
        const PhaseTime timing(*this, SamplingPhase);
        sourceState.reset();
        sourceCoefficients.reset();
        if (sourceOffsets.empty())
            return;
        // P0 face convection/viscosity does not disable these derivatives.
        // Direct-FV sources consume physical gradients without a basis.
        if (ransFirst >= 0 && owner->settings.numerics.reconstruction != "variational")
            RefreshDirectGradients(state, false, time);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            if (ransFirst >= 0 && owner->settings.numerics.reconstruction == "variational")
            {
                CFV::PolynomialSampleOutputs outputs;
                outputs.values = &sourceValues[iField];
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    outputs.gradients[d] = &sourceGradients[iField][d];
                sourceSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
                continue;
            }
            if (owner->settings.numerics.reconstruction == "variational")
                sourceSampling->Evaluate(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceValues[iField]);
            else
            {
                // Euler's direct FV source uses the cell mean at its
                // center, together with the direct gradient. In a curved
                // or distorted cell the reference center need not be the
                // volume barycenter: do not add a reconstructed offset.
                const auto read = state.PhysicalField(iField).Owned().Read();
                auto write = sourceValues[iField].Write();
                const auto means = read.View<R::UniformTableRows>();
                const auto values = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < means.RowCount(); ++iCell)
                    for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                        for (index j = 0; j < means.Width(); ++j)
                            values(iPoint, j) = means(iCell, j);
            }
            if (ransFirst >= 0)
            {
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    if (owner->settings.numerics.reconstruction == "variational")
                        sourceSampling->Evaluate(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceGradients[iField][d], 1 + d);
                    else if (!sourceValuesAreMeans)
                    {
                        const auto read = directGradients[iField].Owned().Read();
                        auto write = sourceGradients[iField][d].Write();
                        const auto input = read.View<R::UniformTableRows>();
                        const auto output = write.View<R::UniformTableRows>();
                        const int dim = owner->metrics.Dimension();
                        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                            for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                                for (index j = 0; j < output.Width(); ++j)
                                    output(iPoint, j) = input(iCell, j * dim + d);
                    }
            }
        }
    }
    /// Opt-in source linearization at the already sampled RHS source
    /// quadrature. This is deliberately separate from the retained
    /// one-point approximate Jacobian and uses only the current published
    /// limited/PP polynomial values/gradients.
    bool FiniteVolume::Workspace::AddRHSQuadratureSourceJacobian(const EquationDofs &state,
                                                                 const R::Array<real>::WriteLease &diagonal,
                                                                 DiagonalStorage storage, real scale,
                                                                 EquationPointEvaluator &evaluator,
                                                                 OptionalRef<const MeanLinearizationOutput> linearization) const
    {
        if (!sourceSampling)
            return false;
        if (owner->settings.numerics.reconstruction != "variational")
            return false;
        // OptionalRef is an API/lifetime marker. Resolve it once before the
        // quadrature loop so its checked operator-> is not part of the hot
        // point path.
        const auto *linearizationOutput = linearization.Get();
        const auto means = state.ReadOwned();
        const auto blocks = BlockViews(state, means);
        const auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::optional<WallDistanceRead> wallRead;
        std::optional<R::Array<real>::ReadLease> lengthRead;
        if (ransFirst >= 0)
        {
            wallRead.emplace(owner->wallDistance->AcquireRead());
            lengthRead.emplace(owner->ransCellLength.Read());
        }
        const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>()
                                        : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
        const auto matrix = diagonal.View<R::TableRows>();
        const index nv = state.ComponentNumber();
        const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
        std::vector<EquationBlockLocation> locations;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            locations.push_back(state.LocateBlock(iBlock));
        PointScratch point(nv);
        // The linearization consumer still needs mean pressure and its
        // conservative derivative. Fill it separately from quadrature
        // source points so the high-order source path does not silently
        // replace the mean-state EOS contract.
        if (linearizationOutput)
        {
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                Gather(blocks, iCell, point, false);
                const auto properties = evaluator.Evaluate(
                    point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                    {ransFirst >= 0, false, true, false, false});
                linearizationOutput->Publish(iCell, point, properties);
            }
        }
        std::vector<R::Array<real>::ReadLease> sourceReads;
        sourceReads.reserve(sourceValues.size());
        for (const auto &field : sourceValues)
            sourceReads.push_back(field.Read());
        std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> gradientReads(sourceGradients.size());
        for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                gradientReads[iField][d].emplace(sourceGradients[iField][d].Read());
        const auto weightRead = sourceWeights.Read();
        const auto weights = weightRead.View<R::FixedRows<1>>();
        std::vector<RHSScalarView> samples;
        samples.reserve(sourceReads.size());
        for (const auto &read : sourceReads)
            samples.push_back(read.View<R::UniformTableRows>());
        std::vector<std::array<RHSScalarView, 3>> gradients(sourceGradients.size());
        for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                gradients[iField][d] = gradientReads[iField][d]->View<R::UniformTableRows>();
        const auto ransMean = ransFirst >= 0 ? means.Block("rans") : EquationBlockReadView{};
        const auto addTerms = [&](index iCell, const PointScratch &at,
                                  const EquationPointProperties &properties, real weight,
                                  real wallDistance)
        {
            if (ransFirst >= 0)
            {
                const auto &model = owner->equation.RANS();
                const auto input = RANSPoint(at, flowFirst, ransFirst, model, properties, wallDistance, lengths(iCell, 0));
                const auto destruction = model.ImplicitSourceDiagonal(input);
                for (int j = 0; j < model.ComponentNumber(); ++j)
                {
                    DNDS_check_throw_info(std::isfinite(destruction[j]), "nonfinite quadrature RANS source derivative");
                    const index row = ransFirst + j;
                    matrix(iCell, storage == DiagonalStorage::Block ? row + nv * row : row) +=
                        scale * weight * destruction[j];
                }
            }
            if (chemical)
                for (int j = 0; j < 5 + independent; ++j)
                {
                    const index column = j < 5 ? flowFirst + j : speciesFirst + j - 5;
                    for (int k = 0; k < independent; ++k)
                    {
                        const index row = speciesFirst + k;
                        const real derivative = properties.sourceJacobian(k, j);
                        DNDS_check_throw_info(std::isfinite(derivative), "nonfinite quadrature source derivative");
                        if (storage == DiagonalStorage::Block || row == column)
                            matrix(iCell, storage == DiagonalStorage::Block ? row + nv * column : row) -=
                                scale * weight * derivative;
                    }
                }
        };
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
        {
            for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
            {
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = locations[iBlock];
                    const auto values = samples[location.physicalField][iPoint];
                    for (index j = 0; j < location.componentNumber; ++j)
                        point.values[logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                    if (ransFirst >= 0)
                        for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        {
                            const auto values = gradients[location.physicalField][d][iPoint];
                            for (index j = 0; j < location.componentNumber; ++j)
                                point.gradient(logicalFirst[iBlock] + j, d) =
                                    values[location.firstComponent + j];
                        }
                }
                if (ransFirst >= 0)
                {
                    const auto &model = owner->equation.RANS();
                    RANSStateView{model.Kind(), model.ComponentNumber()}.RepairReconstruction(
                        ransMean[iCell], point.values.data() + ransFirst);
                }
                const auto properties = evaluator.Evaluate(
                    point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                    {ransFirst >= 0, false, false, chemical, chemical});
                const real wallDistance = ransFirst >= 0
                                              ? walls.cells(iCell, iPoint - sourceOffsets[iCell])
                                              : real(1);
                addTerms(iCell, point, properties, weights(iPoint, 0), wallDistance);
            }
        }
        return true;
    }

    /// Add the approximate source matrix to caller-owned output.
    /// The default is one mean point per cell; rhsQuadrature selects the
    /// explicit same-quadrature source linearization.
    /// The controller retains current means/coefficients and ready source
    /// gradients. Optional linearization outputs consume this same mean
    /// property evaluation; false means no samples existed and no outputs
    /// were filled. This entry neither samples RHS quadrature nor publishes
    /// source freshness; the production wrapper validates its provenance.
    bool FiniteVolume::Workspace::AddMeanSourceJacobian(const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
                                                        DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
                                                        OptionalRef<const MeanLinearizationOutput> linearization) const
    {
        if (sourceOffsets.empty())
            return false;
        if (owner->settings.implicit.sourceJacobianMode == "rhsQuadrature")
            return AddRHSQuadratureSourceJacobian(state, diagonal, storage, scale, evaluator, linearization);
        // Resolve the synchronous nullable borrow once; property sampling
        // below must not pay a checked OptionalRef dereference per cell.
        const auto *linearizationOutput = linearization.Get();
        // The default approximate source matrix is a one-point,
        // mean-state operator. The opt-in quadrature path is a separate
        // branch selected by the explicit implicit source mode.
        // RANS still consumes the current polynomial's center gradients.
        std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> gradientReads;
        // Direct gradients already contain every direction in one physical
        // field. Retain that allocation's read lease and select the direction
        // through indexing, rather than copying whole fields into scratch.
        // Polynomial center sampling keeps its separate per-direction arrays.
        struct SourceGradientView
        {
            RHSScalarView values;
            int componentStride = 1, direction = 0;
            real operator()(index iCell, index component) const
            {
                return values(iCell, component * componentStride + direction);
            }
        };
        std::vector<std::array<SourceGradientView, 3>> gradients;
        if (ransFirst >= 0)
        {
            gradientReads.resize(size_t(state.PhysicalFieldNumber()));
            gradients.resize(size_t(state.PhysicalFieldNumber()));
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                if (owner->settings.numerics.reconstruction != "variational")
                {
                    gradientReads[iField][0].emplace(directGradients[iField].Owned().Read());
                    const auto view = gradientReads[iField][0]->View<R::UniformTableRows>();
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        gradients[iField][d] = {view, owner->metrics.Dimension(), d};
                    continue;
                }
                CFV::PolynomialSampleOutputs outputs;
                // Workspace identity is const; shallow mutable handles
                // retain its existing scratch allocations for the write.
                auto scratch = sourceJacobianGradients[iField];
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    outputs.gradients[d] = &scratch[d];
                // One center, gradients only. The sampling facility owns
                // bounded layout specialization and generic fallback; no
                // unused sampled-value field is allocated for the matrix.
                sourceJacobianSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                {
                    gradientReads[iField][d].emplace(sourceJacobianGradients[iField][d].Read());
                    gradients[iField][d] = {gradientReads[iField][d]->View<R::UniformTableRows>(), 1, 0};
                }
            }
        }
        const auto means = state.ReadOwned();
        const auto blocks = BlockViews(state, means);
        const auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::optional<WallDistanceRead> wallRead;
        std::optional<R::Array<real>::ReadLease> lengthRead;
        if (ransFirst >= 0)
        {
            wallRead.emplace(owner->wallDistance->AcquireRead());
            lengthRead.emplace(owner->ransCellLength.Read());
        }
        const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
        const auto matrix = diagonal.View<R::TableRows>();
        const index nv = state.ComponentNumber();
        const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
        PointScratch point(nv);
        std::vector<EquationBlockLocation> locations;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            locations.push_back(state.LocateBlock(iBlock));
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
        {
            Gather(blocks, iCell, point, false);
            if (ransFirst >= 0)
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = locations[iBlock];
                    for (index j = 0; j < location.componentNumber; ++j)
                        for (int d = 0; d < owner->metrics.Dimension(); ++d)
                            point.gradient(logicalFirst[iBlock] + j, d) = gradients[location.physicalField][d](iCell, location.firstComponent + j);
                }
            const auto properties = evaluator.Evaluate(point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                                                       {ransFirst >= 0, false, bool(linearizationOutput), chemical, chemical});
            if (linearizationOutput)
                linearizationOutput->Publish(iCell, point, properties);
            const real volume = metric.GetCellVol(iCell);
            if (ransFirst >= 0)
            {
                const auto &model = owner->equation.RANS();
                const auto input = RANSPoint(point, flowFirst, ransFirst, model, properties, walls.cellMean(iCell, 0), lengths(iCell, 0));
                const auto destruction = model.ImplicitSourceDiagonal(input);
                for (int j = 0; j < model.ComponentNumber(); ++j)
                {
                    DNDS_check_throw_info(std::isfinite(destruction[j]), "nonfinite mean-state RANS destruction diagonal");
                    const index row = ransFirst + j;
                    matrix(iCell, storage == DiagonalStorage::Block ? row + nv * row : row) += scale * volume * destruction[j];
                }
            }
            if (chemical)
                for (int j = 0; j < 5 + independent; ++j)
                {
                    const index column = j < 5 ? flowFirst + j : speciesFirst + j - 5;
                    for (int k = 0; k < independent; ++k)
                    {
                        const index row = speciesFirst + k;
                        const real derivative = properties.sourceJacobian(k, j);
                        DNDS_check_throw_info(std::isfinite(derivative), "nonfinite mean-state source derivative");
                        if (storage == DiagonalStorage::Block || row == column)
                            matrix(iCell, storage == DiagonalStorage::Block ? row + nv * column : row) -= scale * volume * derivative;
                    }
                }
        }
        return true;
    }

    void FiniteVolume::Workspace::PublishSourceState(const EquationDofs &state, real time)
    {
        if (sourceOffsets.empty())
            return;
        sourceState.emplace(state.NumericAlias());
        sourceCoefficients.emplace(FluxCoefficients());
        sourceTime = time;
    }
    /// ValidateWorkspace has already fixed schema, binding and geometry.
    /// Source samples depend only on owned means, the published flux
    /// polynomial and physical time. Ghost refreshes do not change them.
    /// Reserve readers as well as checking generations: an active writer
    /// has not published its next generation yet and must still exclude us.
    bool FiniteVolume::Workspace::SourceSamplesCurrent(const EquationDofs &state, real time) const
    {
        if (sourceOffsets.empty())
            return true;
        if (!sourceState || !sourceCoefficients || sourceTime != time)
            return false;
        const auto means = state.ReadOwned(), coefficients = FluxCoefficients().ReadOwned();
        const auto unchanged = [](const EquationDofOperations::Binding &publication, const EquationDofs &values)
        {
            if (publication.snapshot.PhysicalFieldNumber() != values.PhysicalFieldNumber())
                return false;
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &saved = publication.snapshot.PhysicalField(iField).Owned();
                const auto &current = values.PhysicalField(iField).Owned();
                if (saved.SharedStorage() != current.SharedStorage() || saved.ShapeVersion() != current.ShapeVersion() ||
                    publication.versions[iField] != current.Version())
                    return false;
            }
            return true;
        };
        return unchanged(*sourceState, state) && sourceCoefficients->Current(FluxCoefficients());
    }
    /// High-order RHS source sampling supplies the opt-in quadrature
    /// source Jacobian or refreshes publication provenance only. The
    /// default Jacobian continues to sample its own center gradients.
    void FiniteVolume::Workspace::PrepareVolumeSource(const EquationDofs &state,
                                                      OptionalRef<EquationDofs> result,
                                                      EquationPointEvaluator &evaluator, real time,
                                                      OptionalRef<EquationDofs> lowOrder)
    {
        const PhaseTime timing(*this, SourcePhase);
        // Kinematic and constitutive sources share this additive entrance;
        // face-only calls never include either kind of source.
        const auto frame = owner->motion.At(time);
        if (result && frame.moving)
        {
            const auto read = state.ReadOwned();
            auto write = result->WriteOwned();
            std::optional<EquationDofs::WriteScope> lowWrite;
            if (lowOrder)
            {
                state.RequireCompatible(*lowOrder);
                EquationDofOperations::RequireDistinct(*result, *lowOrder);
                lowWrite.emplace(lowOrder->WriteOwned());
            }
            const auto flow = read.Block("flow");
            const auto output = write.Block("flow");
            const auto lowOutput = lowWrite ? lowWrite->Block("flow") : EquationBlockArrayView<real>{};
            const auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                FlowState mean{};
                for (int j = 0; j < 5; ++j)
                    mean[j] = flow(iCell, j);
                const auto source = frame.RotationSource(mean);
                for (int j = 0; j < 5; ++j)
                {
                    output(iCell, j) += metric.GetCellVol(iCell) * source[j];
                    if (lowWrite)
                        lowOutput(iCell, j) += metric.GetCellVol(iCell) * source[j];
                }
            }
        }
        if (sourceOffsets.empty())
            return;
        SampleSourceState(state, time);
        if (!result)
            return;
        std::vector<R::Array<real>::ReadLease> sources;
        for (const auto &field : sourceValues)
            sources.push_back(field.Read());
        std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> derivatives(sourceGradients.size());
        for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
            if (sourceValuesAreMeans)
                derivatives[iField][0].emplace(directGradients[iField].Owned().Read());
            else
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    derivatives[iField][d].emplace(sourceGradients[iField][d].Read());
        // Resolve physical layouts once, outside the quadrature loop.
        using SampleView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        std::vector<SampleView> samples;
        struct SourceGradientView
        {
            SampleView values;
            int componentStride = 1, direction = 0;
            real operator()(index iPoint, index component) const
            {
                return values(iPoint, component * componentStride + direction);
            }
        };
        std::vector<std::array<SourceGradientView, 3>> gradients(derivatives.size());
        for (const auto &read : sources)
            samples.push_back(read.View<R::UniformTableRows>());
        for (index iField = 0; iField < index(derivatives.size()); ++iField)
            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                gradients[iField][d] = sourceValuesAreMeans
                                           ? SourceGradientView{derivatives[iField][0]->View<R::UniformTableRows>(), owner->metrics.Dimension(), d}
                                           : SourceGradientView{derivatives[iField][d]->View<R::UniformTableRows>(), 1, 0};
        std::optional<WallDistanceRead> wallRead;
        if (owner->wallDistance)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto weightRead = sourceWeights.Read();
        const auto weights = weightRead.View<R::FixedRows<1>>();
        std::optional<R::Array<real>::ReadLease> lengthRead;
        if (ransFirst >= 0)
            lengthRead.emplace(owner->ransCellLength.Read());
        const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
        std::optional<EquationDofs::WriteScope> output;
        if (result)
            output.emplace(result->WriteOwned());
        const auto speciesOutput = result && speciesFirst >= 0 ? output->Block("species") : EquationBlockArrayView<real>{};
        const auto ransOutput = result && ransFirst >= 0 ? output->Block("rans") : EquationBlockArrayView<real>{};
        const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
        PointScratch point(state.ComponentNumber());
        const auto mean = state.ReadOwned();
        const auto ransMean = ransFirst >= 0 ? mean.Block("rans") : EquationBlockReadView{};
        std::vector<EquationBlockLocation> locations;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            locations.push_back(state.LocateBlock(iBlock));
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
            {
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = locations[iBlock];
                    // Resolve each borrowed row before traversing its
                    // components. The span retains physical element stride;
                    // no packing or AoS storage assumption is introduced.
                    const auto values = samples[location.physicalField][iPoint];
                    for (index j = 0; j < location.componentNumber; ++j)
                        point.values[logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                    if (ransFirst >= 0)
                        for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        {
                            const auto &gradient = gradients[location.physicalField][d];
                            const auto values = gradient.values[iPoint];
                            for (index j = 0; j < location.componentNumber; ++j)
                                point.gradient(logicalFirst[iBlock] + j, d) = values[(location.firstComponent + j) * gradient.componentStride + gradient.direction];
                        }
                }
                if (ransFirst >= 0)
                {
                    const auto &model = owner->equation.RANS();
                    RANSStateView{model.Kind(), model.ComponentNumber()}.RepairReconstruction(ransMean[iCell], point.values.data() + ransFirst);
                }
                const auto properties = evaluator.Evaluate(point.Flow(flowFirst), point.Species(speciesFirst, independent), 0, {ransFirst >= 0, false, false, chemical && result, false});
                if (chemical && result)
                    for (int k = 0; k < independent; ++k)
                        speciesOutput(iCell, k) += weights(iPoint, 0) * properties.chemicalSource[k];
                if (ransFirst >= 0)
                {
                    const auto &model = owner->equation.RANS();
                    const auto input = RANSPoint(point, flowFirst, ransFirst, model, properties, walls.cells(iCell, iPoint - sourceOffsets[iCell]), lengths(iCell, 0));
                    const auto source = result ? model.Source(input) : std::array<real, 2>{};
                    for (int j = 0; j < model.ComponentNumber(); ++j)
                    {
                        DNDS_check_throw_info(std::isfinite(source[j]), "nonfinite RANS source");
                        if (result)
                            ransOutput(iCell, j) += weights(iPoint, 0) * source[j];
                    }
                }
            }
    }
}

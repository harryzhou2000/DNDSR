#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    /// Immutable physical snapshot. Evaluated blocks own no MPI, worker, live
    /// input or FV owner. Geometry is retained only for evaluated storage;
    /// assembled filling traverses the existing FV samples directly.
    struct FiniteVolume::EquationJacobian final : OffDiagonalEvaluator
    {
        struct Point
        {
            index other, auxiliary;
            Vector3 normal, location;
            Eigen::Matrix3d rotation;
            real weight;
        };
        struct Geometry
        {
            std::weak_ptr<const Data> owner;
            std::shared_ptr<const Algebra::DofSpace::Partition> partition;
            std::vector<std::vector<index>> columns;
            std::vector<std::vector<std::vector<Point>>> rows;
        };
        std::shared_ptr<const Geometry> geometry;
        index components, flowFirst = -1, speciesFirst = -1;
        int independent;
        Eigen::MatrixXd means, pressureDerivative;
        std::vector<real> pressure;
        std::vector<std::array<real, 2>> radius;
        RigidFrame frame;
        real scale;
        std::vector<bool> active;
        EquationJacobian(const FiniteVolume &fv, const EquationDofs &state, LinearProblemContext &output, JacobianPreparation controls, real time)
            : components(state.ComponentNumber()), independent(fv._data->equation.HasMixture() ? fv._data->equation.Mixture().SpeciesNumber() - 1 : 0),
              means(output.HasAssembledOffDiagonal() ? 0 : components, output.HasAssembledOffDiagonal() ? 0 : fv.Metrics().NumCellProc()), pressureDerivative(components, fv.Metrics().NumCellProc()),
              pressure(size_t(fv.Metrics().NumCellProc())), radius(fv._data->stencilFaces.size()), frame(fv._data->motion.At(time)), scale(controls.residualScale)
        {
            const Algebra::ComponentSelection selection(components, controls.updateComponents);
            for (index j = 0; j < components; ++j)
                active.push_back(selection.Includes(j));
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                first += index(block.components.size());
            }
            if (output.HasAssembledOffDiagonal())
                return;
            const auto old = std::dynamic_pointer_cast<const EquationJacobian>(output.PreparedOffDiagonalEvaluator());
            if (old && old->geometry->owner.lock() == fv._data)
            {
                geometry = old->geometry;
                return;
            }
            auto prepared = std::make_shared<Geometry>();
            prepared->owner = fv._data;
            prepared->partition = output.Space().Pin();
            prepared->columns = output.Columns();
            prepared->rows.resize(output.Columns().size());
            auto read = fv.Metrics().AcquireRead();
            const auto metric = read.HostView();
            for (index iCell = 0; iCell < index(prepared->rows.size()); ++iCell)
            {
                auto &row = prepared->rows[iCell];
                const auto &columns = output.Columns()[iCell];
                row.resize(columns.size());
                for (index entry = fv._data->offsets[iCell]; entry < fv._data->offsets[iCell + 1]; ++entry)
                {
                    const auto &sample = fv._data->samples[entry];
                    if (sample.other == UnInitIndex || sample.other == iCell)
                        continue;
                    const auto found = std::find(columns.begin(), columns.end(), sample.other);
                    DNDS_check_throw_info(found != columns.end(), "equation Jacobian stencil is missing a neighbor");
                    row[found - columns.begin()].push_back({sample.other, sample.auxiliaryFace, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, metric.GetFaceWeight(sample.point)});
                }
            }
            geometry = std::move(prepared);
        }
        index Components() const override { return components; }
        auto RowSpaceIdentity() const -> std::shared_ptr<const Algebra::DofSpace::Partition> override { return geometry->partition; }
        const std::vector<std::vector<index>> &Columns() const override { return geometry->columns; }
        /// Add dF(rotated U)/dU minus frozen scalar dissipation, already
        /// weighted/scaled. Different periodic images remain separate terms.
        template <class Means>
        void Add(const Point &point, Eigen::Ref<Eigen::MatrixXd> output, const Means &values) const
        {
            FlowState flow{};
            for (int j = 0; j < 5; ++j)
                flow[j] = values(point.other, flowFirst + j);
            const Geom::tPoint momentum{flow[1], flow[2], flow[3]};
            const auto rotated = (point.rotation * momentum).eval();
            for (int d = 0; d < 3; ++d)
                flow[1 + d] = rotated[d];
            const auto grid = frame.GridVelocity(point.location);
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (flow[1 + d] / flow[0] - grid[d]) * point.normal[d];
            const real factor = .5 * scale * point.weight;
            const real lambda = radius[point.auxiliary][0] + radius[point.auxiliary][1];
            for (index column = 0; column < components; ++column)
            {
                if (!active[column])
                    continue;
                FlowState direction{};
                if (column >= flowFirst && column < flowFirst + 5)
                {
                    const auto j = column - flowFirst;
                    if (j >= 1 && j <= 3)
                        for (int d = 0; d < 3; ++d)
                            direction[1 + d] = point.rotation(d, j - 1);
                    else
                        direction[j] = 1;
                }
                // dp/dU is frozen in the stored neighbor's frame. Applying
                // its rotated direction to the flux directly composes J*R.
                const auto derivative = PrepareCompressibleFluxDerivative(flow, direction, pressure[point.other], pressureDerivative(column, point.other), point.normal, grid);
                real deltaSpeed = 0;
                for (int d = 0; d < 3; ++d)
                    deltaSpeed += (direction[1 + d] - flow[1 + d] / flow[0] * direction[0]) / flow[0] * point.normal[d];
                for (index row = 0; row < components; ++row)
                {
                    if (!active[row])
                        continue;
                    real value;
                    if (row >= flowFirst && row < flowFirst + 5)
                    {
                        const auto j = row - flowFirst;
                        value = derivative[j] - lambda * direction[j];
                    }
                    else
                        value = PrepareAdvectedScalarFluxDerivative(values(point.other, row), real(row == column), speed, deltaSpeed) - lambda * real(row == column);
                    output(row, column) += factor * value;
                }
            }
        }
        void EvaluateBlock(index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const override
        {
            output.setZero();
            struct FrozenMeans
            {
                const Eigen::MatrixXd &values;
                real operator()(index iCell, index j) const { return values(j, iCell); }
            } source{means};
            for (const auto &point : geometry->rows[iRow][iBlock])
                Add(point, output, source);
        }
    };

    std::shared_ptr<LinearProblemContext> FiniteVolume::BuildLinearProblem(const EquationDofs &prototype, DiagonalStorage diagonal, OffDiagonalStorage offDiagonal) const
    {
        Metrics().Validate();
        _data->partition.Validate();
        _data->equation.ValidateStateSchema(prototype);
        _data->equation.ResolveKernelProfile(prototype.Device());
        DNDS_check_throw_info(prototype.Space().Same(Metrics().Space()), "equation linear prototype row space differs");
        return std::make_shared<LinearProblemContext>(prototype.Space(), prototype.ComponentNumber(),
                                                      FiniteVolumeDetail::JacobianColumns(_data->offsets, _data->samples), _data->partition.Core(), diagonal, offDiagonal);
    }
    void FiniteVolume::PrepareJacobian(const EquationDofs &state, const AuxiliaryValues &auxiliary, const R::Array<real> &inversePseudoTime,
                                       LinearProblemContext &output, JacobianPreparation controls, real time, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                       const EquationWorkspace *sourceReconstruction) const
    {
        ValidateAuxValues(state, auxiliary, time);
        output.Validate();
        const index nv = state.ComponentNumber(), owned = Metrics().NumCell(), all = Metrics().NumCellProc();
        DNDS_check_throw_info(output.Space().Same(state.Space()) && output.Components() == nv && !controls.roeDissipation,
                              "equation Jacobian space/components differ or Roe profile is unavailable");
        DNDS_check_throw_info(std::isfinite(controls.physicalMass) && controls.physicalMass >= 0 && std::isfinite(controls.residualScale) && controls.residualScale >= 0, "invalid integrated equation Jacobian controls");
        DNDS_check_throw_info(inversePseudoTime.GetDevice() == R::Device::Host() && inversePseudoTime.RowCount() == owned && !inversePseudoTime.Layout().IsVariable() && inversePseudoTime.Layout().Width() == 1, "equation inverse pseudo-time shape/backend differs");
        DNDS_check_throw_info(auxiliary.faceSpectralRadius && _data->equation.HasMixture() == bool(workers), "equation Jacobian requires face signals and a matching worker backend");
        DNDS_check_throw_info(output.Columns() == FiniteVolumeDetail::JacobianColumns(_data->offsets, _data->samples), "equation Jacobian stencil differs from cell-face-cell incidence");
        const auto checkAlias = [&](const R::Array<real> &input)
        {
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Storage(), output.Diagonal().Storage()) &&
                                      (!output.HasAssembledOffDiagonal() || !Algebra::Detail::Overlaps(input.Storage(), output.OffDiagonal().Values().Storage())),
                                  "equation Jacobian output aliases input");
        };
        checkAlias(inversePseudoTime);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            for (const auto *input : {&state.PhysicalField(iField).Owned(), &state.PhysicalField(iField).Ghost()})
                checkAlias(*input);
        for (const auto *input : {auxiliary.meanThermodynamics ? &*auxiliary.meanThermodynamics : nullptr,
                                  auxiliary.faceSpectralRadius ? &*auxiliary.faceSpectralRadius : nullptr, auxiliary.faceWaveJump ? &*auxiliary.faceWaveJump : nullptr,
                                  auxiliary.facePressureMinimum ? &*auxiliary.facePressureMinimum : nullptr,
                                  auxiliary.cellSpectralSum ? &auxiliary.cellSpectralSum->Owned() : nullptr, auxiliary.cellWaveJump ? &auxiliary.cellWaveJump->Owned() : nullptr,
                                  auxiliary.cellPressureMinimum ? &auxiliary.cellPressureMinimum->Owned() : nullptr,
                                  auxiliary.derivedOwned ? &auxiliary.derivedOwned->Values() : nullptr, auxiliary.derivedGhost ? &auxiliary.derivedGhost->Values() : nullptr})
            if (input)
                checkAlias(*input);
        const bool chemical = _data->settings.equation.source == "chemical";
        const bool rans = _data->equation.HasRANS();
        DNDS_check_throw_info(_data->settings.equation.source == "none" || (chemical && _data->equation.HasMixture()), "unsupported equation Jacobian source");
        if (chemical || rans)
        {
            DNDS_check_throw_info(sourceReconstruction || (!rans && (_data->settings.numerics.reconstruction == "constant" || _data->equation.Mixture().SpeciesNumber() == 1)),
                                  "RANS or nonconstant chemical Jacobian requires current RHS source reconstruction");
            if (sourceReconstruction)
                ValidateEquationSourceJacobian(state, *sourceReconstruction, time);
        }
        auto evaluator = _data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {false, false, true, chemical, chemical});
        auto snapshot = std::make_shared<EquationJacobian>(*this, state, output, controls, time);
        const auto input = state.ReadWithGhost();
        const auto blocks = EquationFVDetail::BlockViews(state, input);
        const auto inputLayout = input.ComponentLayout();
        const auto inputValues = inputLayout.View();
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inversePseudoTime.Read();
        const auto radius = faceRead.View<R::UniformTableRows>(), tau = pseudo.View<R::UniformTableRows>();
        for (index iFace = 0; iFace < radius.RowCount(); ++iFace)
        {
            for (int j = 0; j < 2; ++j)
                DNDS_check_throw_info(std::isfinite(radius(iFace, j)) && radius(iFace, j) >= 0, "invalid equation face spectral radius");
            snapshot->radius[iFace] = {radius(iFace, 0), radius(iFace, 1)};
        }
        for (index iCell = 0; iCell < owned; ++iCell)
            DNDS_check_throw_info(std::isfinite(tau(iCell, 0)) && tau(iCell, 0) >= 0, "invalid equation inverse pseudo-time");
        std::optional<R::Array<real>::WriteLease> diagonal(output.Diagonal().Write());
        std::optional<Algebra::BlockMatrixStorage::WriteAccess> off;
        try
        {
            if (output.HasAssembledOffDiagonal())
                off.emplace(output.OffDiagonal().Write());
        }
        catch (...)
        {
            diagonal->CancelUnmodified();
            throw;
        }
        const auto d = diagonal->View<R::TableRows>();
        std::optional<Algebra::BlockMatrixView<real>> b;
        if (off)
            b.emplace(off->View());
        EquationFVDetail::PointScratch point(nv);
        Eigen::MatrixXd own(nv, nv), other(nv, nv);
        const auto rotation = snapshot->frame.NegativeSourceJacobian();
        const real rotationBound = Geom::tPoint{snapshot->frame.angularVelocity[0], snapshot->frame.angularVelocity[1], snapshot->frame.angularVelocity[2]}.norm();
        try
        {
            // Only O(N) scalar thermodynamic linearization data are frozen per
            // cell. Never cache a complete N*N flux matrix per cell/face.
            for (index iCell = 0; iCell < all; ++iCell)
            {
                EquationFVDetail::Gather(blocks, iCell, point);
                for (index j = 0; j < nv; ++j)
                {
                    DNDS_check_throw_info(std::isfinite(point.values[j]), "nonfinite equation Jacobian mean");
                    if (!output.HasAssembledOffDiagonal())
                        snapshot->means(j, iCell) = point.values[j];
                }
                const auto flow = point.Flow(snapshot->flowFirst);
                const bool source = chemical && !sourceReconstruction && iCell < owned && snapshot->independent > 0;
                const auto properties = evaluator.Evaluate(flow, point.Species(snapshot->speciesFirst, snapshot->independent), 0, {false, false, true, source, source});
                snapshot->pressure[iCell] = properties.thermo.pressure;
                for (index column = 0; column < nv; ++column)
                {
                    FlowState direction{};
                    if (column >= snapshot->flowFirst && column < snapshot->flowFirst + 5)
                        direction[column - snapshot->flowFirst] = 1;
                    struct SpeciesDirection
                    {
                        index first, column;
                        real operator[](int k) const { return real(column == first + k); }
                    } species{snapshot->speciesFirst, column};
                    snapshot->pressureDerivative(column, iCell) = PreparePrimitiveDerivative(flow, direction, properties, species).pressure;
                    DNDS_check_throw_info(std::isfinite(snapshot->pressureDerivative(column, iCell)), "nonfinite conservative EOS derivative");
                }
                if (iCell >= owned)
                    continue;
                own.setIdentity();
                own *= metric.GetCellVol(iCell) * (tau(iCell, 0) + controls.physicalMass);
                if (snapshot->frame.moving)
                {
                    if (output.DiagonalMode() == DiagonalStorage::Block)
                    {
                        for (int j = 0; j < 5; ++j)
                            for (int i = 0; i < 5; ++i)
                                own(snapshot->flowFirst + i, snapshot->flowFirst + j) += controls.residualScale * metric.GetCellVol(iCell) * rotation[i + 5 * j];
                    }
                    else
                        for (int i = 0; i < 5; ++i)
                            own(snapshot->flowFirst + i, snapshot->flowFirst + i) += controls.residualScale * metric.GetCellVol(iCell) * rotationBound;
                }
                if (source)
                    for (int j = 0; j < 5 + snapshot->independent; ++j)
                    {
                        const index column = j < 5 ? snapshot->flowFirst + j : snapshot->speciesFirst + j - 5;
                        for (int k = 0; k < snapshot->independent; ++k)
                        {
                            DNDS_check_throw_info(std::isfinite(properties.sourceJacobian(k, j)), "nonfinite chemical source derivative");
                            own(snapshot->speciesFirst + k, column) -= controls.residualScale * metric.GetCellVol(iCell) * properties.sourceJacobian(k, j);
                        }
                    }
                for (index j = 0; j < (output.DiagonalMode() == DiagonalStorage::Block ? nv * nv : nv); ++j)
                    d(iCell, j) = output.DiagonalMode() == DiagonalStorage::Block ? own.data()[j] : own(j, j);
            }
            if ((chemical || rans) && sourceReconstruction)
                AddEquationSourceJacobian(state, *sourceReconstruction, *diagonal, output.DiagonalMode(), controls.residualScale, evaluator);
            for (index iCell = 0; iCell < owned; ++iCell)
            {
                if (output.DiagonalMode() == DiagonalStorage::Block)
                    for (index j = 0; j < nv * nv; ++j)
                        own.data()[j] = d(iCell, j);
                else
                {
                    own.setZero();
                    for (index j = 0; j < nv; ++j)
                        own(j, j) = d(iCell, j);
                }
                if (b)
                    for (index j = 0; j < b->BlockNumber(iCell); ++j)
                        (*b)(iCell, j).setZero();
                for (index entry = _data->offsets[iCell]; entry < _data->offsets[iCell + 1]; ++entry)
                {
                    const auto &sample = _data->samples[entry];
                    const real weight = metric.GetFaceWeight(sample.point), lambda = radius(sample.auxiliaryFace, 0) + radius(sample.auxiliaryFace, 1);
                    own.diagonal().array() += .5 * controls.residualScale * weight * lambda;
                    if (sample.other == UnInitIndex)
                        continue; // Active Euler scalar-BC approximation.
                    if (!b && sample.other != iCell)
                        continue;
                    other.setZero();
                    snapshot->Add({sample.other, sample.auxiliaryFace, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, weight}, other, inputValues);
                    if (sample.other == iCell)
                        own += other;
                    else
                    {
                        const auto &columns = output.Columns()[iCell];
                        const auto found = std::find(columns.begin(), columns.end(), sample.other);
                        (*b)(iCell, found - columns.begin()) += other;
                    }
                }
                for (index j = 0; j < nv; ++j)
                    for (index i = 0; i < nv; ++i)
                        if (!snapshot->active[i] || !snapshot->active[j])
                            own(i, j) = real(i == j);
                DNDS_check_throw_info(own.allFinite(), "nonfinite equation approximate diagonal");
                for (index j = 0; j < (output.DiagonalMode() == DiagonalStorage::Block ? nv * nv : nv); ++j)
                    d(iCell, j) = output.DiagonalMode() == DiagonalStorage::Block ? own.data()[j] : own(j, j);
            }
        }
        catch (...)
        {
            diagonal->Fail();
            if (off)
                off->Fail();
            throw;
        }
        diagonal.reset();
        off.reset();
        if (!output.HasAssembledOffDiagonal())
            output.PublishOffDiagonalEvaluator(std::move(snapshot));
    }
}

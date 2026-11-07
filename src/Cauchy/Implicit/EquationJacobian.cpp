#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Implicit/EquationJacobianKernels.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    /// Immutable physical snapshot. Evaluated blocks own no MPI, worker, live
    /// input or FV owner. Geometry is retained only for evaluated storage;
    /// assembled filling traverses the existing FV samples directly.
    /// TODO(large-species Jacobian): an explicit approximation may omit
    /// flow5<->scalar derivatives and diagonalize extended diagonal blocks.
    /// Its off-diagonal storage can then be flow5 dense plus scalar entries;
    /// no EOS/source coupling may be discarded merely to fit that format.
    /// Keep coupled diagonals as an option, but compressed ILU factors then
    /// generally need projection (not algebraically equivalent). See the
    /// large-species TODO in docs/architecture/CauchyImplicit.md.
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
        index components, flowFirst = -1, speciesFirst = -1, kineticFirst = -1;
        int independent;
        Eigen::MatrixXd means, pressureDerivative;
        std::vector<real> pressure;
        std::vector<std::array<real, 2>> radius;
        RigidFrame frame;
        real scale;
        std::vector<bool> active;
        bool allComponents = true;
        EquationJacobian(const FiniteVolume &fv, const EquationDofs &state, LinearProblemContext &output, JacobianPreparation controls, real time)
            : components(state.ComponentNumber()), independent(fv._data->equation.HasMixture() ? fv._data->equation.Mixture().SpeciesNumber() - 1 : 0),
              means(output.HasAssembledOffDiagonal() ? 0 : components, output.HasAssembledOffDiagonal() ? 0 : fv.Metrics().NumCellProc()), pressureDerivative(components, fv.Metrics().NumCellProc()),
              pressure(size_t(fv.Metrics().NumCellProc())), radius(fv._data->stencilFaces.size()), frame(fv._data->motion.At(time)), scale(controls.residualScale)
        {
            const Algebra::ComponentSelection selection(components, controls.updateComponents);
            for (index j = 0; j < components; ++j)
            {
                active.push_back(selection.Includes(j));
                allComponents = allComponents && active.back();
            }
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans" && fv._data->equation.RANS().ComponentNumber() == 2)
                    kineticFirst = first;
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
                for (index entry = fv._data->jacobianOffsets[iCell]; entry < fv._data->jacobianOffsets[iCell + 1]; ++entry)
                {
                    const auto &sample = fv._data->jacobianSamples[entry];
                    if (sample.other == UnInitIndex || sample.other == iCell)
                        continue;
                    const auto found = std::find(columns.begin(), columns.end(), sample.other);
                    DNDS_check_throw_info(found != columns.end(), "equation Jacobian stencil is missing a neighbor");
                    row[found - columns.begin()].push_back({sample.other, sample.auxiliaryFace, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, metric.GetFaceArea(sample.face)});
                }
            }
            geometry = std::move(prepared);
        }
        index Components() const override { return components; }
        auto RowSpaceIdentity() const -> std::shared_ptr<const Algebra::DofSpace::Partition> override { return geometry->partition; }
        const std::vector<std::vector<index>> &Columns() const override { return geometry->columns; }
        /// Add dF(rotated U)/dU minus frozen scalar dissipation, already
        /// weighted/scaled by default. weighted=false returns the unweighted
        /// block for a local stationary-face hoist; the caller applies each
        /// original point's .5*scale*weight separately. Different periodic
        /// images remain separate terms.
        template <class Means>
        void Add(const Point &point, Eigen::Ref<Eigen::MatrixXd> output, const Means &values, bool weighted = true) const
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
            const real factor = weighted ? .5 * scale * point.weight : 1;
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
                auto derivative = PrepareCompressibleFluxDerivative(flow, direction, pressure[point.other], pressureDerivative(column, point.other), point.normal, grid);
                if (kineticFirst >= 0)
                {
                    const auto stress = TurbulentNormalFluxDerivative(flow, direction, values(point.other, kineticFirst), real(column == kineticFirst), point.normal);
                    for (int j = 0; j < 5; ++j)
                        derivative[j] += stress[j];
                }
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
        /// Directional form of Add: retain the prepared EOS pressure row,
        /// scalar dissipation, turbulent stress and periodic image, without
        /// constructing components-by-components temporary matrices.
        void ApplyBlock(index iRow, index iBlock, InputVector input,
                        Eigen::Ref<Eigen::VectorXd> output) const override
        {
            output.setZero();
            for (const auto &point : geometry->rows[iRow][iBlock])
            {
                FlowState flow{}, direction{};
                real dp = 0;
                for (index j = 0; j < components; ++j)
                    if (active[j])
                        dp += pressureDerivative(j, point.other) * input[j];
                for (int j = 0; j < 5; ++j)
                {
                    flow[j] = means(flowFirst + j, point.other);
                    direction[j] = active[flowFirst + j] ? input[flowFirst + j] : 0;
                }
                flow = FiniteVolumeDetail::RotateState(flow, point.rotation);
                direction = FiniteVolumeDetail::RotateState(direction, point.rotation);
                const auto grid = frame.GridVelocity(point.location);
                auto derivative = PrepareCompressibleFluxDerivative(flow, direction, pressure[point.other], dp, point.normal, grid);
                if (kineticFirst >= 0)
                {
                    const auto stress = TurbulentNormalFluxDerivative(flow, direction, means(kineticFirst, point.other),
                                                                      active[kineticFirst] ? input[kineticFirst] : 0, point.normal);
                    for (int j = 0; j < 5; ++j)
                        derivative[j] += stress[j];
                }
                real speed = 0, deltaSpeed = 0;
                for (int d = 0; d < 3; ++d)
                {
                    const real velocity = flow[1 + d] / flow[0];
                    speed += (velocity - grid[d]) * point.normal[d];
                    deltaSpeed += (direction[1 + d] - velocity * direction[0]) / flow[0] * point.normal[d];
                }
                const real factor = .5 * scale * point.weight;
                const real lambda = radius[point.auxiliary][0] + radius[point.auxiliary][1];
                for (index j = 0; j < components; ++j)
                    if (active[j])
                    {
                        const auto k = j - flowFirst;
                        const real value = k >= 0 && k < 5 ? derivative[k] - lambda * direction[k]
                                                           : PrepareAdvectedScalarFluxDerivative(means(j, point.other), input[j], speed, deltaSpeed) - lambda * input[j];
                        output[j] += factor * value;
                    }
            }
        }

        /// Closed-form convection block at a frozen mean, with the same EOS
        /// pressure row supplied by preparation. Compute d(v.n)/dU once rather
        /// than repeating conservative velocity conversions for every column.
        /// The pointwise directional implementation above remains the oracle;
        /// two-equation turbulent stress retains that generic path for now.
        /// No gamma closure, source approximation or eigenvalue change occurs.
        template <class Means>
        void AddFrozenFluxBlock(const Point &point, Eigen::Ref<Eigen::MatrixXd> output, const Means &values, bool weighted = true) const
        {
            if (kineticFirst >= 0)
            {
                Add(point, output, values, weighted);
                return;
            }
            FlowState flow{};
            for (int j = 0; j < 5; ++j)
                flow[j] = values(point.other, flowFirst + j);
            // Resolve the existing rank-two identity predicate before frame
            // products. Partial selections and every real rotation retain
            // the transformed path, including arbitrarily small rotations.
            const bool sameFrame = allComponents && (point.rotation.array() == Eigen::Matrix3d::Identity().array()).all();
            if (!sameFrame)
            {
                const Geom::tPoint momentum{flow[1], flow[2], flow[3]};
                const auto rotated = (point.rotation * momentum).eval();
                for (int d = 0; d < 3; ++d)
                    flow[1 + d] = rotated[d];
            }
            const auto grid = frame.GridVelocity(point.location);
            const real inverseDensity = 1 / flow[0];
            real normalVelocity = 0, gridNormalVelocity = 0, scalarSpeed = 0;
            for (int d = 0; d < 3; ++d)
            {
                const real velocity = flow[1 + d] / flow[0];
                normalVelocity += velocity * point.normal[d];
                gridNormalVelocity += grid[d] * point.normal[d];
                scalarSpeed += (velocity - grid[d]) * point.normal[d];
            }
            const real relativeVelocity = normalVelocity - gridNormalVelocity;
            FlowState speedDerivative{};
            speedDerivative[0] = -normalVelocity * inverseDensity;
            for (int j = 0; j < 3; ++j)
                if (sameFrame)
                    speedDerivative[1 + j] = point.normal[j] * inverseDensity;
                else
                {
                    for (int d = 0; d < 3; ++d)
                        speedDerivative[1 + j] += point.rotation(d, j) * point.normal[d];
                    speedDerivative[1 + j] *= inverseDensity;
                }
            const real factor = weighted ? .5 * scale * point.weight : 1;
            const real lambda = radius[point.auxiliary][0] + radius[point.auxiliary][1];
            // With no component rotation/projection, J = D + q h^T + t g^T:
            // h=d(v.n)/dU and g is the prepared conservative EOS pressure row.
            // Keep distinct flow/passive relative speeds, moving-grid work and
            // arbitrary logical block offsets. No gamma-equivalent closure.
            // Exact identity is intentional: a small physical rotation must
            // still take the fully transformed path below.
            if (sameFrame)
            {
                for (index column = 0; column < components; ++column)
                {
                    const real deltaSpeed = column >= flowFirst && column < flowFirst + 5
                                                ? speedDerivative[column - flowFirst]
                                                : 0;
                    const real dp = pressureDerivative(column, point.other);
                    output(flowFirst, column) += factor * (flow[0] * deltaSpeed + (column == flowFirst ? relativeVelocity - lambda : 0));
                    for (int d = 0; d < 3; ++d)
                        output(flowFirst + 1 + d, column) += factor * (flow[1 + d] * deltaSpeed + dp * point.normal[d] +
                                                                       (column == flowFirst + 1 + d ? relativeVelocity - lambda : 0));
                    output(flowFirst + 4, column) += factor * ((flow[4] + pressure[point.other]) * deltaSpeed + dp * normalVelocity +
                                                               (column == flowFirst + 4 ? relativeVelocity - lambda : 0));
                    const auto passive = [&](index begin, index end)
                    {
                        for (index row = begin; row < end; ++row)
                            output(row, column) += factor * (values(point.other, row) * deltaSpeed +
                                                             (row == column ? scalarSpeed - lambda : 0));
                    };
                    passive(0, flowFirst);
                    passive(flowFirst + 5, components);
                }
                return;
            }
            for (index column = 0; column < components; ++column)
            {
                if (!active[column])
                    continue;
                FlowState direction{};
                real deltaSpeed = 0;
                if (column >= flowFirst && column < flowFirst + 5)
                {
                    const index j = column - flowFirst;
                    deltaSpeed = speedDerivative[j];
                    if (j >= 1 && j <= 3)
                        for (int d = 0; d < 3; ++d)
                            direction[1 + d] = point.rotation(d, j - 1);
                    else
                        direction[j] = 1;
                }
                const real dp = pressureDerivative(column, point.other);
                FlowState derivative{};
                derivative[0] = direction[0] * relativeVelocity + flow[0] * deltaSpeed;
                for (int d = 0; d < 3; ++d)
                    derivative[1 + d] = direction[1 + d] * relativeVelocity + flow[1 + d] * deltaSpeed + dp * point.normal[d];
                derivative[4] = direction[4] * relativeVelocity + (flow[4] + pressure[point.other]) * deltaSpeed + dp * normalVelocity;
                for (index row = 0; row < components; ++row)
                {
                    if (!active[row])
                        continue;
                    const real value = row >= flowFirst && row < flowFirst + 5
                                           ? derivative[row - flowFirst] - lambda * direction[row - flowFirst]
                                           : PrepareAdvectedScalarFluxDerivative(values(point.other, row), real(row == column), scalarSpeed, deltaSpeed) - lambda * real(row == column);
                    output(row, column) += factor * value;
                }
            }
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
                                                      *_data->jacobianColumns, _data->partition.Core(), diagonal, offDiagonal);
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
        output.ValidateColumns(_data->jacobianColumns);
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
        auto evaluator = _data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {rans, false, true, chemical, chemical});
        auto snapshot = std::make_shared<EquationJacobian>(*this, state, output, controls, time);
        const auto input = state.ReadWithGhost();
        const auto blocks = EquationFVDetail::BlockViews(state, input);
        const auto inputLayout = input.ComponentLayout();
        const auto inputValues = inputLayout.View();
        // Only this assembled operator opts into the six-component AoS leaf.
        // Keep physical parent pitch when borrowing each row; every other
        // layout/grouping retains the original logical component accessor.
        const bool sixComponentShape = output.HasAssembledOffDiagonal() && nv == 6 && snapshot->flowFirst == 0 &&
                                       snapshot->allComponents && snapshot->kineticFirst < 0;
        const auto ownedTable = sixComponentShape ? Algebra::TryOwnedTable(inputValues.OwnedPart()) : std::nullopt;
        const auto ghostTable = sixComponentShape ? Algebra::TryGhostTable(inputValues.GhostPart()) : std::nullopt;
        const bool useSixComponentKernel = ownedTable && ghostTable;
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
                EquationFVDetail::Gather(blocks, iCell, point, false);
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
            // A block-shaped output is already the row's dense workspace.
            // Keep its write lease and use the centralized borrowed adapter;
            // diagonal-line storage retains the existing projection scratch.
            const auto addFaceBlock = [&](const EquationJacobian::Point &point, auto &block)
            {
                if (useSixComponentKernel && (point.rotation.array() == Eigen::Matrix3d::Identity().array()).all())
                {
                    const auto row = point.other < owned ? (*ownedTable)[point.other] : (*ghostTable)[point.other - owned];
                    Detail::AddFrozenFluxBlock6({row.data(), 6}, {snapshot->pressureDerivative.col(point.other).data(), 6},
                                                snapshot->pressure[point.other], point.normal, snapshot->frame.GridVelocity(point.location),
                                                .5 * snapshot->scale * point.weight, snapshot->radius[point.auxiliary][0] + snapshot->radius[point.auxiliary][1],
                                                {block.data(), 36});
                    return;
                }
                snapshot->AddFrozenFluxBlock(point, block, inputValues);
            };
            const auto assembleFaceRow = [&](index iCell, auto &own)
            {
                if (b)
                    for (index j = 0; j < b->BlockNumber(iCell); ++j)
                        (*b)(iCell, j).setZero();
                // First-order mean-state linearization: exactly one action
                // per face incidence, regardless of polynomial degree or
                // face curvature. Accurate geometry was integrated at setup.
                for (index entry = _data->jacobianOffsets[iCell]; entry < _data->jacobianOffsets[iCell + 1]; ++entry)
                {
                    const auto &sample = _data->jacobianSamples[entry];
                    const real weight = metric.GetFaceArea(sample.face), lambda = radius(sample.auxiliaryFace, 0) + radius(sample.auxiliaryFace, 1);
                    own.diagonal().array() += .5 * controls.residualScale * weight * lambda;
                    if (sample.other == UnInitIndex)
                        continue; // Active Euler scalar-BC approximation.
                    if (!b && sample.other != iCell)
                        continue;
                    const EquationJacobian::Point point{sample.other, sample.auxiliaryFace, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, weight};
                    if (sample.other == iCell)
                    {
                        other.setZero();
                        addFaceBlock(point, other);
                        own += other;
                    }
                    else
                    {
                        const auto &columns = output.Columns()[iCell];
                        const auto found = std::find(columns.begin(), columns.end(), sample.other);
                        auto destination = (*b)(iCell, found - columns.begin());
                        addFaceBlock(point, destination);
                    }
                }
                for (index j = 0; j < nv; ++j)
                    for (index i = 0; i < nv; ++i)
                        if (!snapshot->active[i] || !snapshot->active[j])
                            own(i, j) = real(i == j);
                DNDS_check_throw_info(own.allFinite(), "nonfinite equation approximate diagonal");
            };
            if (output.DiagonalMode() == DiagonalStorage::Block)
            {
                const auto matrices = Algebra::MakeEigenRowMatrices(d, nv, nv);
                for (index iCell = 0; iCell < owned; ++iCell)
                {
                    auto block = matrices[iCell];
                    assembleFaceRow(iCell, block);
                }
            }
            else
                for (index iCell = 0; iCell < owned; ++iCell)
                {
                    own.setZero();
                    for (index j = 0; j < nv; ++j)
                        own(j, j) = d(iCell, j);
                    assembleFaceRow(iCell, own);
                    for (index j = 0; j < nv; ++j)
                        d(iCell, j) = own(j, j);
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

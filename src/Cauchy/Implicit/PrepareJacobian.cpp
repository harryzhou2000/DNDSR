#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include <Eigen/Core>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    using Matrix = Eigen::Matrix<real, 5, 5>;

    /// Frozen physics with reusable geometry, independent of the FV owner's
    /// communicator lifetime. Only the weak owner key is used to reuse geometry
    /// at the next preparation. Block evaluation needs no mesh/Array leases.
    struct FiniteVolume::MatrixFreeJacobian final : OffDiagonalEvaluator
    {
        struct Point
        {
            index other, auxiliary, sample;
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
        EquationSystem equation;
        RigidFrame frame;
        std::vector<FlowState> means;
        std::vector<std::array<real, 2>> radius;
        std::vector<std::array<real, 3>> magnitudes;
        real scale;
        bool roe;
        bool selected = false;
        std::array<bool, 5> active{};
        MatrixFreeJacobian(const FiniteVolume &fv, const LinearProblemContext &output, JacobianPreparation controls, real time)
            : equation(fv._data->equation), frame(fv._data->motion.At(time)), scale(controls.residualScale), roe(controls.roeDissipation)
        {
            const Algebra::ComponentSelection selection(5, controls.updateComponents);
            selected = !selection.All();
            for (int j = 0; j < 5; ++j)
                active[j] = selection.Includes(j);
            auto old = std::dynamic_pointer_cast<const MatrixFreeJacobian>(output.PreparedOffDiagonalEvaluator());
            if (old && old->geometry->owner.lock() == fv._data)
                geometry = old->geometry;
            else
            {
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
                            continue; // These contributions belong to the diagonal.
                        const auto slot = std::find(columns.begin(), columns.end(), sample.other);
                        DNDS_check_throw_info(slot != columns.end(), "evaluated Jacobian stencil is missing an adjacent cell");
                        row[slot - columns.begin()].push_back({sample.other, sample.auxiliaryFace, entry, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, sample.rotation, metric.GetFaceWeight(sample.point)});
                    }
                }
                geometry = std::move(prepared);
            }
            if (roe)
                magnitudes.resize(fv._data->samples.size());
        }
        index Components() const override { return 5; }
        std::shared_ptr<const Algebra::DofSpace::Partition> RowSpaceIdentity() const override { return geometry->partition; }
        const std::vector<std::vector<index>> &Columns() const override { return geometry->columns; }
        void EvaluateBlock(index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const override
        {
            output.setZero();
            for (const auto &point : geometry->rows[iRow][iBlock])
            {
                const auto right = RotateState(means[point.other], point.rotation);
                Matrix dissipation = Matrix::Identity() * (radius[point.auxiliary][0] + radius[point.auxiliary][1]);
                if (roe)
                {
                    const auto basis = equation.PrepareCharacteristics(means[iRow], right, point.normal);
                    Eigen::Matrix<real, 5, 1> lambda;
                    const auto &signal = magnitudes[point.sample];
                    lambda << signal[0], signal[0], signal[0], signal[1], signal[2];
                    dissipation = Eigen::Map<const Matrix>(basis.inverse.data()) * lambda.asDiagonal() * Eigen::Map<const Matrix>(basis.forward.data());
                    dissipation.diagonal().array() += radius[point.auxiliary][1];
                }
                const auto derivative = equation.PrepareFluxJacobian(right, point.normal, frame.GridVelocity(point.location));
                const Matrix other = (.5 * scale * point.weight) * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
                Matrix rotation = Matrix::Identity();
                rotation.block<3, 3>(1, 1) = point.rotation;
                output.noalias() += other * rotation;
            }
            // Project after the periodic rotation: selected components refer
            // to the stored, mesh-bound equation frame on both endpoints.
            if (selected)
                for (int j = 0; j < 5; ++j)
                    for (int k = 0; k < 5; ++k)
                        if (!active[j] || !active[k])
                            output(j, k) = 0;
        }
    };

    std::shared_ptr<LinearProblemContext> FiniteVolume::BuildLinearProblem(const CFV::FVField &prototype, DiagonalStorage diagonal, OffDiagonalStorage offDiagonal) const
    {
        Metrics().ValidateField(prototype, 5);
        _data->partition.Validate();
        std::vector<std::vector<index>> columns(Metrics().NumCell());
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            auto &row = columns[iCell];
            for (index entry = _data->offsets[iCell]; entry < _data->offsets[iCell + 1]; ++entry)
            {
                const auto other = _data->samples[entry].other;
                if (other != UnInitIndex && other != iCell)
                    row.push_back(other);
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        return std::make_shared<LinearProblemContext>(prototype.Space(), 5, std::move(columns), _data->partition.Core(), diagonal, offDiagonal);
    }
    void FiniteVolume::PrepareJacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                                       const R::Array<real> &inversePseudoTime, LinearProblemContext &output,
                                       JacobianPreparation controls, real time) const
    {
        ValidateAuxValues(state, auxiliary, time);
        const Algebra::ComponentSelection selection(5, controls.updateComponents);
        std::array<bool, 5> active{};
        for (int j = 0; j < 5; ++j)
            active[j] = selection.Includes(j);
        output.Validate();
        DNDS_check_throw_info(output.Space().Same(state.Space()) && output.Components() == 5 &&
                                  (!controls.roeDissipation || output.DiagonalMode() == DiagonalStorage::Block),
                              "Jacobian space/components differ or Roe diagonal is not a block");
        DNDS_check_throw_info(std::isfinite(controls.physicalMass) && controls.physicalMass >= 0 &&
                                  std::isfinite(controls.residualScale) && controls.residualScale >= 0,
                              "invalid integrated Jacobian coefficients");
        DNDS_check_throw_info(inversePseudoTime.GetDevice() == R::Device::Host() && inversePseudoTime.RowCount() == Metrics().NumCell() &&
                                  !inversePseudoTime.Layout().IsVariable() && inversePseudoTime.Layout().Width() == 1,
                              "inverse pseudo-time must be an owned CPU scalar table");
        DNDS_check_throw_info(auxiliary.faceSpectralRadius, "Jacobian requires prepared face spectral radii");
        const auto &faceArray = *auxiliary.faceSpectralRadius;
        const auto checkAlias = [&](const R::Array<real> &source)
        {
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source.Storage(), output.Diagonal().Storage()) &&
                                      (!output.HasAssembledOffDiagonal() || !Algebra::Detail::Overlaps(source.Storage(), output.OffDiagonal().Values().Storage())),
                                  "Jacobian output aliases a frozen input");
        };
        checkAlias(state.Owned());
        checkAlias(state.Ghost());
        checkAlias(faceArray);
        checkAlias(inversePseudoTime);
        if (auxiliary.meanThermodynamics)
            checkAlias(*auxiliary.meanThermodynamics);
        for (const auto *derived : {&auxiliary.derivedOwned, &auxiliary.derivedGhost})
            if (*derived)
                checkAlias((*derived)->Values());
        for (const auto *field : {&auxiliary.cellWaveJump, &auxiliary.cellPressureMinimum})
            if (*field)
            {
                checkAlias((*field)->Owned());
                checkAlias((*field)->Ghost());
            }
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto frame = _data->motion.At(time);
        const auto sourceDerivative = frame.NegativeSourceJacobian();
        const real rotationBound = Geom::tPoint{frame.angularVelocity[0], frame.angularVelocity[1], frame.angularVelocity[2]}.norm();
        const auto owned = state.Owned().Read(), ghost = state.Ghost().Read(), face = faceArray.Read(), pseudo = inversePseudoTime.Read();
        const RHSFieldView mean{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        RHSAuxiliaryViews packet;
        std::array<std::optional<R::Array<real>::ReadLease>, 5> retained;
        std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
        if (auxiliary.meanThermodynamics)
        {
            retained[0].emplace(auxiliary.meanThermodynamics->Read());
            packet.meanThermodynamics = retained[0]->View<R::UniformTableRows>();
        }
        else if (auxiliary.derivedOwned && auxiliary.derivedGhost &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::Pressure) &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::SoundSpeed) &&
                 auxiliary.derivedOwned->Request().Has(DerivedQuantity::DiffusionBound))
        {
            const auto typed = EquationDofs::Coupled(_data->equation.Schema(), state);
            derivedOwned.emplace(auxiliary.derivedOwned->ReadCurrent(typed, _data->equation));
            derivedGhost.emplace(auxiliary.derivedGhost->ReadCurrent(typed, _data->equation));
            packet.meanThermodynamics = MeanThermodynamicView::Derived(*derivedOwned, &*derivedGhost);
        }
        const auto bind = [&](const auto &field, int slot, auto &view)
        {
            if (!field)
                return;
            retained[slot].emplace(field->Owned().Read());
            retained[slot + 1].emplace(field->Ghost().Read());
            view = RHSFieldView{retained[slot]->View<R::UniformTableRows>(), retained[slot + 1]->View<R::UniformTableRows>()};
        };
        bind(auxiliary.cellWaveJump, 1, packet.cellWaveJump);
        bind(auxiliary.cellPressureMinimum, 3, packet.cellPressureMinimum);
        DNDS_check_throw_info(!controls.roeDissipation || !_data->settings.numerics.riemann.useStencil ||
                                  ((!_data->flux.UsesTransverseSignal() || packet.cellWaveJump) && (!_data->flux.UsesPressureStencil() || packet.cellPressureMinimum)),
                              "Roe Jacobian requires the configured flux's prepared stencil signals");
        const auto radius = face.View<R::UniformTableRows>(), inverseTau = pseudo.View<R::UniformTableRows>();
        DNDS_check_throw_info(radius.RowCount() == index(_data->stencilFaces.size()) && radius.Width() == 2, "Jacobian face spectral shape differs");
        std::shared_ptr<MatrixFreeJacobian> evaluated;
        if (!output.HasAssembledOffDiagonal())
        {
            evaluated = std::make_shared<MatrixFreeJacobian>(*this, output, controls, time);
            evaluated->means.resize(state.Owned().RowCount() + state.Ghost().RowCount());
            for (index iCell = 0; iCell < index(evaluated->means.size()); ++iCell)
                evaluated->means[iCell] = Load(mean, iCell);
            evaluated->radius.resize(radius.RowCount());
            for (index iFace = 0; iFace < radius.RowCount(); ++iFace)
                evaluated->radius[iFace] = {radius(iFace, 0), radius(iFace, 1)};
        }
        std::optional<R::Array<real>::WriteLease> diagonal(output.Diagonal().Write());
        std::optional<Algebra::BlockMatrixStorage::WriteAccess> offDiagonal;
        try
        {
            if (output.HasAssembledOffDiagonal())
                offDiagonal.emplace(output.OffDiagonal().Write());
        }
        catch (...)
        {
            diagonal->CancelUnmodified();
            throw;
        }
        const auto d = diagonal->View<R::TableRows>();
        const auto blocks = offDiagonal ? std::optional(offDiagonal->View()) : std::nullopt;
        const auto prepared = [&](index iCell, const FlowState &u)
        {
            if (packet.meanThermodynamics)
                return Riemann::AcousticState{{u, (*packet.meanThermodynamics)(iCell, 0)}, (*packet.meanThermodynamics)(iCell, 1)};
            return _data->equation.PrepareFluxState(u);
        };
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            const real inverseStep = inverseTau(iCell, 0);
            DNDS_check_throw_info(std::isfinite(inverseStep) && inverseStep >= 0, "nonfinite or negative inverse pseudo-time");
            Matrix own = Matrix::Identity() * (metric.GetCellVol(iCell) * (inverseStep + controls.physicalMass));
            if (frame.moving)
            {
                if (output.DiagonalMode() == DiagonalStorage::Block)
                    own += (controls.residualScale * metric.GetCellVol(iCell)) * Eigen::Map<const Matrix>(sourceDerivative.data());
                else
                    // Scalar entries cannot represent the skew rotation block.
                    // Retain an explicit norm bound as a preconditioning term.
                    own.diagonal().array() += controls.residualScale * metric.GetCellVol(iCell) * rotationBound;
            }
            if (blocks)
                for (index j = 0; j < blocks->BlockNumber(iCell); ++j)
                    (*blocks)(iCell, j).setZero();
            const auto left = Load(mean, iCell);
            for (index entry = _data->offsets[iCell]; entry < _data->offsets[iCell + 1]; ++entry)
            {
                const auto &sample = _data->samples[entry];
                const auto iAux = sample.auxiliaryFace;
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]}, point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const auto gridVelocity = frame.GridVelocity(point);
                const real convection = radius(iAux, 0), diffusion = radius(iAux, 1);
                DNDS_check_throw_info(std::isfinite(convection) && convection >= 0 && std::isfinite(diffusion) && diffusion >= 0, "invalid face spectral radius in Jacobian");
                const real factor = .5 * controls.residualScale * metric.GetFaceWeight(sample.point);
                if (!controls.roeDissipation && sample.other == UnInitIndex)
                {
                    own.diagonal().array() += factor * (convection + diffusion);
                    continue; // Euler's scalar diagonal does not linearize BCs.
                }
                const auto right = sample.other == UnInitIndex ? sample.boundary.Exterior(_data->equation.Gas(), left, normal, point, time, &frame)
                                                               : RotateState(Load(mean, sample.other), sample.rotation);
                Matrix dissipation = Matrix::Identity() * (convection + diffusion);
                if (controls.roeDissipation)
                {
                    const auto characteristic = _data->equation.PrepareCharacteristics(left, right, normal);
                    DNDS_check_throw_info(!characteristic.componentwiseFallback, "Roe Jacobian requires a valid physical characteristic basis");
                    const auto leftData = prepared(iCell, left);
                    const auto rightData = sample.other == UnInitIndex ? _data->equation.PrepareFluxState(right) : prepared(sample.other, right);
                    const auto model = _data->equation.PrepareRiemann(_data->flux, leftData, rightData, leftData, rightData, normal, gridVelocity);
                    auto settings = _data->settings.numerics.riemann.KernelSettings();
                    if (packet.cellWaveJump)
                    {
                        real jump = (*packet.cellWaveJump)(iCell, 0);
                        if (sample.other != UnInitIndex)
                            jump = std::max(jump, (*packet.cellWaveJump)(sample.other, 0));
                        settings.dLambda = std::max(settings.dLambda, jump);
                    }
                    if (packet.cellPressureMinimum)
                    {
                        real minimum = (*packet.cellPressureMinimum)(iCell, 0);
                        if (sample.other != UnInitIndex)
                            minimum = std::min(minimum, (*packet.cellPressureMinimum)(sample.other, 0));
                        settings.pressureStencilMin = std::min(settings.pressureStencilMin, minimum);
                    }
                    const auto signals = _data->flux.Evaluate(model.Input(leftData, rightData, normal, gridVelocity), settings);
                    if (evaluated)
                        evaluated->magnitudes[entry] = {signals.contact, signals.acousticPlus, signals.acousticMinus};
                    Eigen::Matrix<real, 5, 1> magnitudes;
                    magnitudes << signals.contact, signals.contact, signals.contact, signals.acousticPlus, signals.acousticMinus;
                    dissipation = Eigen::Map<const Matrix>(characteristic.inverse.data()) * magnitudes.asDiagonal() * Eigen::Map<const Matrix>(characteristic.forward.data());
                    dissipation.diagonal().array() += diffusion;
                }
                own += factor * dissipation;
                if (evaluated && sample.other != UnInitIndex && sample.other != iCell)
                    continue; // Physical derivative is regenerated by the evaluator.
                const auto derivative = _data->equation.PrepareFluxJacobian(right, normal, gridVelocity);
                Matrix other = factor * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
                if (sample.other == UnInitIndex)
                {
                    if (controls.roeDissipation)
                    {
                        const auto boundary = sample.boundary.ExteriorJacobian(_data->equation.Gas(), left, normal, point, time, &frame);
                        own.noalias() += other * Eigen::Map<const Matrix>(boundary.data());
                    }
                }
                else
                {
                    Matrix rotation = Matrix::Identity();
                    rotation.block<3, 3>(1, 1) = sample.rotation;
                    other = (other * rotation).eval();
                    if (sample.other == iCell)
                        own += other;
                    else if (blocks)
                    {
                        const auto &row = output.Columns()[iCell];
                        const auto slot = std::find(row.begin(), row.end(), sample.other);
                        DNDS_check_throw_info(slot != row.end(), "Jacobian stencil is missing an adjacent cell");
                        (*blocks)(iCell, slot - row.begin()) += other;
                    }
                }
            }
            DNDS_check_throw_info(own.allFinite(), "nonfinite approximate diagonal block");
            if (!selection.All())
            {
                for (int j = 0; j < 5; ++j)
                    for (int k = 0; k < 5; ++k)
                        if (!active[j] || !active[k])
                            own(j, k) = j == k ? 1 : 0;
                if (blocks)
                    for (index entry = 0; entry < blocks->BlockNumber(iCell); ++entry)
                        for (int j = 0; j < 5; ++j)
                            for (int k = 0; k < 5; ++k)
                                if (!active[j] || !active[k])
                                    (*blocks)(iCell, entry)(j, k) = 0;
            }
            if (output.DiagonalMode() == DiagonalStorage::Block)
                for (int j = 0; j < 25; ++j)
                    d(iCell, j) = own.data()[j];
            else
                for (int j = 0; j < 5; ++j)
                    d(iCell, j) = own(j, j);
        }
        diagonal.reset();
        if (evaluated)
            output.PublishOffDiagonalEvaluator(std::move(evaluated));
    }
}

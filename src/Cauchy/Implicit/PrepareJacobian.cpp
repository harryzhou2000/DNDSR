#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include <Eigen/Core>
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    using Matrix = Eigen::Matrix<real, 5, 5>;

    std::shared_ptr<LinearProblemContext> FiniteVolume::BuildLinearProblem(const CFV::FVField &prototype, DiagonalStorage diagonal) const
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
        return std::make_shared<LinearProblemContext>(prototype.Space(), 5, std::move(columns), _data->partition.Core(), diagonal);
    }
    void FiniteVolume::PrepareJacobian(const CFV::FVField &state, const AuxiliaryValues &auxiliary,
                                       const R::Array<real> &inversePseudoTime, LinearProblemContext &output,
                                       JacobianPreparation controls, real time) const
    {
        ValidateAuxValues(state, auxiliary, time);
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
                                      !Algebra::Detail::Overlaps(source.Storage(), output.OffDiagonal().Values().Storage()),
                                  "Jacobian output aliases a frozen input");
        };
        checkAlias(state.Owned());
        checkAlias(state.Ghost());
        checkAlias(faceArray);
        checkAlias(inversePseudoTime);
        if (auxiliary.meanThermodynamics)
            checkAlias(*auxiliary.meanThermodynamics);
        for (const auto *field : {&auxiliary.cellWaveJump, &auxiliary.cellPressureMinimum})
            if (*field)
            {
                checkAlias((*field)->Owned());
                checkAlias((*field)->Ghost());
            }
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto owned = state.Owned().Read(), ghost = state.Ghost().Read(), face = faceArray.Read(), pseudo = inversePseudoTime.Read();
        const RHSFieldView mean{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        RHSAuxiliaryViews packet;
        std::array<std::optional<R::Array<real>::ReadLease>, 5> retained;
        if (auxiliary.meanThermodynamics)
        {
            retained[0].emplace(auxiliary.meanThermodynamics->Read());
            packet.meanThermodynamics = retained[0]->View<R::UniformTableRows>();
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
        auto diagonal = output.Diagonal().Write();
        std::optional<Algebra::BlockMatrixStorage::WriteAccess> offDiagonal;
        try
        {
            offDiagonal.emplace(output.OffDiagonal().Write());
        }
        catch (...)
        {
            diagonal.CancelUnmodified();
            throw;
        }
        const auto d = diagonal.View<R::TableRows>();
        const auto blocks = offDiagonal->View();
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
            for (index j = 0; j < blocks.BlockNumber(iCell); ++j)
                blocks(iCell, j).setZero();
            const auto left = Load(mean, iCell);
            for (index entry = _data->offsets[iCell]; entry < _data->offsets[iCell + 1]; ++entry)
            {
                const auto &sample = _data->samples[entry];
                const auto iAux = sample.auxiliaryFace;
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]}, point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const real convection = radius(iAux, 0), diffusion = radius(iAux, 1);
                DNDS_check_throw_info(std::isfinite(convection) && convection >= 0 && std::isfinite(diffusion) && diffusion >= 0, "invalid face spectral radius in Jacobian");
                const real factor = .5 * controls.residualScale * metric.GetFaceWeight(sample.point);
                if (!controls.roeDissipation && sample.other == UnInitIndex)
                {
                    own.diagonal().array() += factor * (convection + diffusion);
                    continue; // Euler's scalar diagonal does not linearize BCs.
                }
                const auto right = sample.other == UnInitIndex ? sample.boundary.Exterior(_data->equation.Gas(), left, normal, point, time)
                                                               : RotateState(Load(mean, sample.other), sample.rotation);
                Matrix dissipation = Matrix::Identity() * (convection + diffusion);
                if (controls.roeDissipation)
                {
                    const auto characteristic = _data->equation.PrepareCharacteristics(left, right, normal);
                    DNDS_check_throw_info(!characteristic.componentwiseFallback, "Roe Jacobian requires a valid physical characteristic basis");
                    const auto leftData = prepared(iCell, left);
                    const auto rightData = sample.other == UnInitIndex ? _data->equation.PrepareFluxState(right) : prepared(sample.other, right);
                    const auto model = _data->equation.PrepareRiemann(_data->flux, leftData, rightData, leftData, rightData, normal);
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
                    const auto signals = _data->flux.Evaluate(model.Input(leftData, rightData, normal), settings);
                    Eigen::Matrix<real, 5, 1> magnitudes;
                    magnitudes << signals.contact, signals.contact, signals.contact, signals.acousticPlus, signals.acousticMinus;
                    dissipation = Eigen::Map<const Matrix>(characteristic.inverse.data()) * magnitudes.asDiagonal() * Eigen::Map<const Matrix>(characteristic.forward.data());
                    dissipation.diagonal().array() += diffusion;
                }
                own += factor * dissipation;
                const auto derivative = _data->equation.PrepareFluxJacobian(right, normal);
                Matrix other = factor * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
                if (sample.other == UnInitIndex)
                {
                    if (controls.roeDissipation)
                    {
                        const auto boundary = sample.boundary.ExteriorJacobian(_data->equation.Gas(), left, normal, point, time);
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
                    else
                    {
                        const auto &row = output.Columns()[iCell];
                        const auto slot = std::find(row.begin(), row.end(), sample.other);
                        DNDS_check_throw_info(slot != row.end(), "Jacobian stencil is missing an adjacent cell");
                        blocks(iCell, slot - row.begin()) += other;
                    }
                }
            }
            DNDS_check_throw_info(own.allFinite(), "nonfinite approximate diagonal block");
            if (output.DiagonalMode() == DiagonalStorage::Block)
                for (int j = 0; j < 25; ++j)
                    d(iCell, j) = own.data()[j];
            else
                for (int j = 0; j < 5; ++j)
                    d(iCell, j) = own(j, j);
        }
    }
}

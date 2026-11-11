#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/Algebra/EigenView.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using Matrix = Eigen::MatrixXd;
    using ConstMap = Eigen::Map<const Matrix>;
    using Map = Eigen::Map<Matrix>;

    void VariationalReconstructionPlan::ApplyLinearOperator(const FVField &coefficients, FVField &output,
                                                            const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index n = NRecDOF();
        DNDS_check_throw_info(coefficients.Width() > 0 && coefficients.Width() % n == 0, "linear reconstruction width is not a mode multiple");
        const index components = coefficients.Width() / n;
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        Metrics().ValidateField(coefficients, n * components), Metrics().ValidateField(output, n * components);
        auto owned = coefficients.Owned().Read(), ghost = coefficients.Ghost().Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> x{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        auto write = output.Owned().Write();
        const auto y = write.View<R::UniformTableRows>();
        auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read();
        auto rotationRead = _data->neighborRotation.Read();
        auto rotationActiveRead = _data->neighborRotationActive.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        auto matrixRead = _data->rawMatrixB->Read(), diagonalRead = _data->localMatrix->Read();
        const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>();
        const auto matrices = matrixRead.View<R::TableRows>(), diagonals = diagonalRead.View<R::TableRows>();
        const auto xMatrices = Algebra::MakeEigenRowMatrices(x, n, components);
        const auto yMatrices = Algebra::MakeEigenRowMatrices(y, n, components);
        Matrix local(n, components), other(n, components);
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            local.noalias() = ConstMap(diagonals[iCell].data(), n, n) * xMatrices[iCell];
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index neighbor = neighbors(iEntry, 0);
                if (neighbor == UnInitIndex)
                    continue;
                other = xMatrices[neighbor];
                if (_data->hasRotatedPeriodicity && rotationActive(iEntry, 0))
                    for (auto first : periodicVectorStarts)
                    {
                        const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                        other.middleCols(first, 3) = (other.middleCols(first, 3) * rotation.transpose()).eval();
                    }
                local.noalias() -= ConstMap(matrices[iEntry].data(), n, n) * other;
            }
            yMatrices[iCell] = local;
        }
    }
    void VariationalReconstructionPlan::BuildRightHandSide(const FVField &mean, const ReconstructionBoundaryValues &boundary,
                                                           FVField &output, const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index n = NRecDOF(), components = mean.Width();
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        Metrics().ValidateField(mean, components), Metrics().ValidateField(output, n * components);
        DNDS_check_throw_info(boundary.plan == _identity && boundary.values.GetDevice() == R::Device::Host() &&
                                  boundary.values.Layout().Kind() == R::RowLayoutKind::Table && boundary.values.RowCount() == NumBoundaryQuadrature() &&
                                  boundary.values.Layout().Width() == components,
                              "linear boundary shape/plan mismatch");
        auto owned = mean.Owned().Read(), ghost = mean.Ghost().Read(), boundaryRead = boundary.values.Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        const auto bc = boundaryRead.View<R::TableRows>();
        auto write = output.Owned().Write();
        const auto y = write.View<R::UniformTableRows>();
        auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read(), boundaryOffsetRead = _data->boundaryOffsets.Read();
        auto rotationRead = _data->neighborRotation.Read();
        auto rotationActiveRead = _data->neighborRotationActive.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        auto vectorRead = _data->rawVectorB->Read(), injectionRead = _data->rawBoundaryInjection->Read();
        const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>(), boundaryOffsets = boundaryOffsetRead.View<R::FixedRows<1>>();
        const auto vectors = vectorRead.View<R::TableRows>(), injections = injectionRead.View<R::TableRows>();
        const auto yMatrices = Algebra::MakeEigenRowMatrices(y, n, components);
        Matrix local(n, components);
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            local.setZero();
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index neighbor = neighbors(iEntry, 0);
                const bool rotatedEntry = neighbor != UnInitIndex && _data->hasRotatedPeriodicity &&
                                          rotationActive(iEntry, 0);
                for (index component = 0; component < components; ++component)
                    if (neighbor != UnInitIndex)
                    {
                        real value = u(neighbor, component);
                        if (rotatedEntry)
                            for (auto first : periodicVectorStarts)
                                if (component >= first && component < first + 3)
                                {
                                    const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                                    value = 0;
                                    for (int d = 0; d < 3; ++d)
                                        value += rotation(component - first, d) * u(neighbor, first + d);
                                }
                        local.col(component).noalias() += Eigen::Map<const Eigen::VectorXd>(vectors[iEntry].data(), n) * (value - u(iCell, component));
                    }
                    else
                        for (index iPoint = boundaryOffsets(iEntry, 0); iPoint < boundaryOffsets(iEntry + 1, 0); ++iPoint)
                            local.col(component).noalias() += Eigen::Map<const Eigen::VectorXd>(injections[iPoint].data(), n) * (bc(iPoint, component) - u(iCell, component));
            }
            yMatrices[iCell] = local;
        }
    }
    void VariationalReconstructionPlan::ComputeResidual(const FVField &rhs, const FVField &coefficients, FVField &output,
                                                        const std::vector<index> &periodicVectorStarts) const
    {
        Require(ReconstructionOperation::LinearSystem);
        Metrics().ValidateField(rhs, coefficients.Width());
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(rhs.Owned().Storage(), output.Owned().Storage()), "residual output aliases RHS");
        ApplyLinearOperator(coefficients, output, periodicVectorStarts);
        auto read = rhs.Owned().Read();
        auto write = output.Owned().Write();
        const auto b = read.View<R::UniformTableRows>();
        const auto y = write.View<R::UniformTableRows>();
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            for (index mode = 0; mode < rhs.Width(); ++mode)
                y(iCell, mode) = b(iCell, mode) - y(iCell, mode);
    }
    void VariationalReconstructionPlan::ApplyLocalPreconditioner(const FVField &input, FVField &output) const
    {
        Require(ReconstructionOperation::LocalPreconditioner);
        auto geometry = Metrics().AcquireRead();
        Validate();
        const index n = NRecDOF();
        DNDS_check_throw_info(input.Width() > 0 && input.Width() % n == 0, "preconditioner width is not a mode multiple");
        Metrics().ValidateField(input, input.Width()), Metrics().ValidateField(output, input.Width());
        auto read = input.Owned().Read(), matrixRead = _data->localInverse->Read();
        auto write = output.Owned().Write();
        const auto x = read.View<R::UniformTableRows>();
        const auto matrix = matrixRead.View<R::TableRows>();
        const auto y = write.View<R::UniformTableRows>();
        const auto xMatrices = Algebra::MakeEigenRowMatrices(x, n, input.Width() / n);
        const auto yMatrices = Algebra::MakeEigenRowMatrices(y, n, input.Width() / n);
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            yMatrices[iCell].noalias() = ConstMap(matrix[iCell].data(), n, n) * xMatrices[iCell];
    }
}

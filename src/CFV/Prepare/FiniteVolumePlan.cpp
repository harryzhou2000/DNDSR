#include "CFV/Plans/ReconstructionData.hpp"
#include "Geom/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;

    void ReconstructionSettings::Validate() const
    {
        DNDS_check_throw_info(maxOrder >= 1 && maxOrder <= 3 && intOrder >= maxOrder + 1 && intOrder <= INT_ORDER_MAX &&
                                  std::isfinite(jacobiRelax) && jacobiRelax > 0 && jacobiRelax < 2 &&
                                  std::isfinite(bcWeight) && bcWeight > 0 &&
                                  std::isfinite(rankTolerance) && rankTolerance > 0 && rankTolerance < 1 &&
                                  std::isfinite(WBAP_nStd) && WBAP_nStd > 0,
                              "invalid finite-volume reconstruction settings");
        for (const auto weight : boundaryDerivativeWeights)
            DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid boundary derivative weight");
    }
    FiniteVolumePlan FiniteVolumePlan::ConstructMetrics(ssp<UnstructuredMesh> mesh, int intOrder)
    {
        DNDS_check_throw_info(mesh, "finite-volume metrics require a mesh");
        const auto mpi = mesh->getMPI();
        auto communication = std::make_shared<R::CommunicationContext>(mpi.comm);
        std::shared_ptr<Detail::FiniteVolumeData> data;
        // All local allocation/geometry failures agree before the halo collective.
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(intOrder >= 2 && intOrder <= INT_ORDER_MAX, "invalid FV integration order");
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3, "FV requires 2D/3D geometry");
            data = std::make_shared<Detail::FiniteVolumeData>(mesh, intOrder, communication);
            data->cell = R::Array<real>(R::RowLayout::Dense(view.NumCellProc(), 8));
            data->faceOffsets = R::Array<index>(R::RowLayout::Dense(view.NumFaceProc() + 1, 1));
            auto offsets = data->faceOffsets.Write();
            auto offset = offsets.View<R::FixedRows<1>>();
            index count = 0;
            for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            {
                offset(iFace, 0) = count;
                count += Quadrature(view.GetFaceElement(iFace), intOrder).GetNumPoints();
            }
            offset(view.NumFaceProc(), 0) = count;
            data->faceQuadrature = R::Array<real>(R::RowLayout::Dense(count, 7));
            auto cells = data->cell.Write();
            const auto cell = cells.View<R::FixedRows<8>>();
            tSmallCoords coords;
            for (index iCell = 0; iCell < view.NumCellProc(); ++iCell)
            {
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                Quadrature quadrature(element, intOrder), exact(element, INT_ORDER_MAX);
                real volume = 0.;
                tPoint bary = tPoint::Zero();
                for (int iG = 0; iG < exact.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(INT_ORDER_MAX).at(iG);
                    const real det = CellJacobianDet(view.Dimension(), coords, shape);
                    DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV cell has nonpositive Jacobian");
                    volume += det * exact.GetWeight(iG);
                }
                for (int iG = 0; iG < quadrature.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(intOrder).at(iG);
                    bary += PPhysicsCoordD01Nj(coords, shape) * CellJacobianDet(view.Dimension(), coords, shape) * quadrature.GetWeight(iG);
                }
                bary /= volume;
                const auto cent = PPhysicsCoordD01Nj(coords, Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(1).at(0));
                const tPoint box = (coords.colwise() - bary).array().abs().rowwise().maxCoeff();
                const real scale = box.head(view.Dimension()).maxCoeff();
                DNDS_check_throw_info(std::isfinite(volume) && scale > 0, "invalid FV cell geometry");
                cell(iCell, 0) = volume;
                for (int d = 0; d < 3; ++d) cell(iCell, 1 + d) = cent[d], cell(iCell, 4 + d) = bary[d];
                cell(iCell, 7) = scale;
            }
            auto points = data->faceQuadrature.Write();
            const auto point = points.View<R::FixedRows<7>>();
            for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            {
                view.GetCoordsOnFace(iFace, coords);
                const auto element = view.GetFaceElement(iFace);
                const Quadrature quadrature(element, intOrder);
                for (int iG = 0; iG < quadrature.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(intOrder).at(iG);
                    const tPoint position = PPhysicsCoordD01Nj(coords, shape);
                    const auto jacobian = ShapeJacobianCoordD01Nj(coords, shape);
                    tPoint normal = view.Dimension() == 2 ? FacialJacobianToNormVec<2>(jacobian) : FacialJacobianToNormVec<3>(jacobian);
                    const real det = normal.norm();
                    DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV face has zero measure");
                    normal /= det;
                    const index iPoint = offset(iFace, 0) + iG;
                    for (int d = 0; d < 3; ++d) point(iPoint, d) = position[d], point(iPoint, d + 3) = normal[d];
                    point(iPoint, 6) = det * quadrature.GetWeight(iG);
                }
            } });
        data->halo.emplace(data->cells.BuildRowTransferPlan());
        return FiniteVolumePlan(std::move(data));
    }
    void FiniteVolumePlan::Validate() const
    {
        DNDS_check_throw_info(_data && _data->stamp.Valid(), "finite-volume plan is stale; reconstruct metrics");
        _data->cells.Validate();
    }
    FiniteVolumeRead::FiniteVolumeRead(std::shared_ptr<const Detail::FiniteVolumeData> data)
        : _data(std::move(data)),
          _meshRead(std::make_shared<MeshReadAccess>(_data->mesh->AcquireRead(MeshReadRequirements::CellFlux()))),
          _cells(_data->cell.Read()), _quadrature(_data->faceQuadrature.Read()), _offsets(_data->faceOffsets.Read())
    {
        DNDS_check_throw_info(_data->stamp.Valid(), "finite-volume metrics changed before acquisition");
    }
    FiniteVolumeHostView FiniteVolumeRead::HostView() const
    {
        return {_cells.View<R::FixedRows<8>>(), _offsets.View<R::FixedRows<1>>(), _quadrature.View<R::FixedRows<7>>()};
    }
    FiniteVolumeRead FiniteVolumePlan::AcquireRead() const
    {
        Validate();
        return FiniteVolumeRead(_data);
    }
    index FiniteVolumePlan::NumCell() const { return _data->cells.NumOwned(); }
    index FiniteVolumePlan::NumCellProc() const { return _data->cells.NumLocal(); }
    index FiniteVolumePlan::NumFaceProc() const { return _data->faceOffsets.RowCount() - 1; }
    int FiniteVolumePlan::Dimension() const { return _data->dim; }
    const EntitySpace &FiniteVolumePlan::Cells() const { return _data->cells; }
    FVField FiniteVolumePlan::BuildField(index width) const
    {
        Validate();
        DNDS_check_throw_info(width > 0, "FV field width must be positive");
        FVField field(_data->dofs, width);
        Detail::Zero(field);
        return field;
    }
    void FiniteVolumePlan::ValidateField(const FVField &field, index width) const
    {
        Validate();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_data->dofs) && index(field.Width()) == width && field.Device() == R::Device::Host(),
                              "FV field belongs to another plan/order, has wrong width, or is not on CPU");
    }
    R::TransferBinding FiniteVolumePlan::BindHalo(FVField &field) const
    {
        MPI::LocalStage(_data->mesh->getMPI(), [&]
                        { ValidateField(field, field.Width()); });
        return R::PayloadPlan(*_data->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
            .Bind(field.Owned(), field.Ghost(), R::TransferTransport::Direct);
    }
    real FiniteVolumePlan::GetCellVol(index iCell) const
    {
        Validate();
        auto read = _data->cell.Read();
        return read.View<R::FixedRows<8>, true>()(iCell, 0);
    }
    tPoint FiniteVolumePlan::GetCellCent(index iCell) const
    {
        Validate();
        auto read = _data->cell.Read();
        const auto view = read.View<R::FixedRows<8>, true>();
        return {view(iCell, 1), view(iCell, 2), view(iCell, 3)};
    }
}

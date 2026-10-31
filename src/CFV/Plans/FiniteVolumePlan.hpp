#pragma once

#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Geometric.hpp"
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"

namespace DNDS::Geom
{
    class MeshReadAccess;
}

namespace DNDS::CFV
{
    /// CPU preparation settings. Degree is polynomial degree (degree 1 is O2).
    /// Initial functional is the isotropic legacy VFV family; unsupported
    /// anisotropic/GG penalties are not silently translated to this method.
    struct ReconstructionSettings
    {
        int maxOrder = 2, intOrder = 5;
        real jacobiRelax = 1., WBAP_nStd = 10.;
        bool SORInstead = false, HQMWeights = false;
        real bcWeight = 1., rankTolerance = 1e-13;
        std::array<real, 4> boundaryDerivativeWeights{1, 0, 0, 0};
        DNDS_DECLARE_CONFIG(ReconstructionSettings)
        {
            DNDS_FIELD(maxOrder, "Polynomial degree, 1..3");
            DNDS_FIELD(intOrder, "Integration degree, at least polynomial degree plus one");
            DNDS_FIELD(jacobiRelax, "Jacobi/SOR relaxation");
            DNDS_FIELD(SORInstead, "Use ordered in-place SOR");
            DNDS_FIELD(HQMWeights, "Legacy HQM_OPT derivative and HQM_SD geometry weights");
            DNDS_FIELD(bcWeight, "Dirichlet boundary value weight");
            DNDS_FIELD(boundaryDerivativeWeights, "Boundary weights by derivative order; nonzero derivatives impose homogeneous derivative targets");
            DNDS_FIELD(rankTolerance, "Relative pivot tolerance for local systems");
            DNDS_FIELD(WBAP_nStd, "WBAP center weight");
        }
        void Validate() const;
    };

    namespace Detail
    {
        struct FiniteVolumeData;
        struct ReconstructionData;
    }
    using FVField = Algebra::DofField<real>;

    /// Borrowed metric spans for numerical loops. Retain FiniteVolumeRead through
    /// every use. These initial pointers address host storage; device placement
    /// is a later backend responsibility, not implied by callable accessors.
    struct FiniteVolumeHostView
    {
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> quadrature;
        DNDS_DEVICE_CALLABLE real GetCellVol(index iCell) const { return cells(iCell, 0); }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellCent(index iCell) const
        {
            return {cells(iCell, 1), cells(iCell, 2), cells(iCell, 3)};
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellBary(index iCell) const
        {
            return {cells(iCell, 4), cells(iCell, 5), cells(iCell, 6)};
        }
        DNDS_DEVICE_CALLABLE index FaceQuadratureBegin(index iFace) const { return faceOffsets(iFace, 0); }
        DNDS_DEVICE_CALLABLE index FaceQuadratureEnd(index iFace) const { return faceOffsets(iFace + 1, 0); }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFacePoint(index iPoint) const
        {
            return {quadrature(iPoint, 0), quadrature(iPoint, 1), quadrature(iPoint, 2)};
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceNorm(index iPoint) const
        {
            return {quadrature(iPoint, 3), quadrature(iPoint, 4), quadrature(iPoint, 5)};
        }
        DNDS_DEVICE_CALLABLE real GetFaceWeight(index iPoint) const { return quadrature(iPoint, 6); }
    };
    /// Read scope pins metadata, values and the mesh generation. Construction
    /// validates once; HostView() copies only borrowed pointers/scalars.
    class FiniteVolumeRead
    {
        std::shared_ptr<const Detail::FiniteVolumeData> _data;
        std::shared_ptr<Geom::MeshReadAccess> _meshRead;
        Runtime::Array<real>::ReadLease _cells, _quadrature;
        Runtime::Array<index>::ReadLease _offsets;
        explicit FiniteVolumeRead(std::shared_ptr<const Detail::FiniteVolumeData> data);
        friend class FiniteVolumePlan;

    public:
        FiniteVolumeHostView HostView() const;
    };

    /// Immutable metric publication; copies share Runtime buffers and mesh/order
    /// identity. Mesh edits invalidate it. No solution or time-loop ownership.
    class FiniteVolumePlan
    {
        std::shared_ptr<const Detail::FiniteVolumeData> _data;
        explicit FiniteVolumePlan(std::shared_ptr<const Detail::FiniteVolumeData> data) : _data(std::move(data)) {}
        friend class VariationalReconstructionPlan;

    public:
        static FiniteVolumePlan ConstructMetrics(ssp<Geom::UnstructuredMesh> mesh, int intOrder = 5);
        void Validate() const;
        FiniteVolumeRead AcquireRead() const;
        index NumCell() const;
        index NumCellProc() const;
        index NumFaceProc() const;
        int Dimension() const;
        const Geom::EntitySpace &Cells() const;
        FVField BuildField(index width) const;
        /// Collective binding, retained by caller for persistent repeated pulls.
        Runtime::TransferBinding BindHalo(FVField &field) const;
        void ValidateField(const FVField &field, index width) const;
        real GetCellVol(index iCell) const;
        Geom::tPoint GetCellCent(index iCell) const;
    };

    /// Explicit Dirichlet values at the prepared boundary quadrature points.
    /// Refresh after changing a state-dependent BC. No callback is invoked by
    /// reconstruction. Each object belongs to exactly one reconstruction plan.
    struct ReconstructionBoundaryValues
    {
        Runtime::Array<real> values;
        std::shared_ptr<const Detail::ReconstructionData> plan;
    };
    struct BoundaryQuadraturePoint
    {
        index cell, face;
        int quadrature;
        Geom::tPoint point, normal;
    };
    /// Matrices act on column vectors in [first,first+count); all other fields
    /// are limited componentwise. Fill forward/inverse from CELL MEANS once per
    /// field stage. Geometry periodic transforms and characteristic transforms
    /// are different operations. Initially rotated periodic state is rejected.
    struct FaceCharacteristicTransform
    {
        index first = 0, count = 0;
        Runtime::Array<real> forward, inverse;
        std::shared_ptr<const Detail::ReconstructionData> plan;
        uint64_t forwardVersion = 0, inverseVersion = 0;
        uint64_t forwardIdentity = 0, inverseIdentity = 0;
    };

    /// Own iteration scratch once; kernels never grow per-cell vectors.
    struct ReconstructionWorkspace
    {
        FVField coefficients, temporary;
        Runtime::Array<real> scratch;
        index components;
        std::shared_ptr<const Detail::ReconstructionData> plan;
    };

    /// Composition of metrics, basis/moments and factored VR systems on Runtime
    /// storage. CPU kernels consume explicit arrays; no legacy VR owner exists.
    class VariationalReconstructionPlan
    {
        std::shared_ptr<const Detail::ReconstructionData> _data;
        explicit VariationalReconstructionPlan(std::shared_ptr<const Detail::ReconstructionData> data) : _data(std::move(data)) {}

    public:
        static VariationalReconstructionPlan ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings = {});
        const FiniteVolumePlan &Metrics() const;
        const ReconstructionSettings &Settings() const;
        int NRecDOF() const;
        void Validate() const;
        ReconstructionWorkspace BuildWorkspace(index components) const;
        ReconstructionBoundaryValues BuildBoundaryValues(index components) const;
        index NumBoundaryQuadrature() const;
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const;
        FaceCharacteristicTransform BuildFaceTransform(index first, index count) const;
        /// Validate finite mutually inverse matrices once after filling. Later
        /// writes invalidate this publication; no matrix validation in hot loops.
        void FinalizeFaceTransform(FaceCharacteristicTransform &transform) const;
        /// One local sweep. Caller refreshes mean/coefficient halos first.
        /// Jacobi writes temporary then copies owned output; SOR follows local
        /// cell order. No implicit MPI, allocation in the cell loop, or callbacks.
        void DoReconstructionIter(const FVField &mean, ReconstructionWorkspace &work,
                                  const ReconstructionBoundaryValues &boundary) const;
        /// One descending-order 3WBAP pass. Call degree..1 and exchange the
        /// coefficient halo after EACH pass, matching the legacy schedule.
        void DoLimiterWBAP_3Order(ReconstructionWorkspace &work, int order,
                                  const FaceCharacteristicTransform *transform = nullptr) const;
        /// Arbitrary physical point sampling from cell means plus zero-mean
        /// coefficients. Cell IDs are current local IDs, including ready ghosts.
        void Evaluate(const FVField &mean, const FVField &coefficients,
                      const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                      Runtime::Array<real> &values) const;
    };
}

#pragma once

#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Geometric.hpp"
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"
#include "DNDS/HardEigen.hpp"
#include "CFV/Math/ReconstructionFunctional.hpp"
#include <map>

namespace DNDS::Geom
{
    class MeshReadAccess;
}

namespace DNDS::CFV
{
    enum class ReconstructionRegion
    {
        All,
        Core,
        HaloDependent
    };
    /// CPU preparation settings. Degree is polynomial degree (degree 1 is O2).
    /// Initial functional is the isotropic legacy VFV family; unsupported
    /// anisotropic/GG penalties are not silently translated to this method.
    struct ReconstructionSettings
    {
        int maxOrder = 2, intOrder = 5;
        real jacobiRelax = 1., WBAP_nStd = 10.;
        bool SORInstead = false, HQMWeights = false;
        bool coreFirst = false;
        bool cacheDiffBase = true;
        ReconstructionFunctionalSettings functional;
        real bcWeight = 1., svdTolerance = 1e-13, svdAbsoluteTolerance = 0;
        std::array<real, 4> boundaryDerivativeWeights{1, 1, 1, 1};
        /// Optional zone-specific replacement for the unsquared per-order
        /// boundary weights. Higher orders impose homogeneous derivative data.
        std::map<int, std::array<real, 4>> boundaryZoneWeights;
        DNDS_DECLARE_CONFIG(ReconstructionSettings)
        {
            DNDS_FIELD(maxOrder, "Polynomial degree, 1..3");
            DNDS_FIELD(intOrder, "Integration degree, at least polynomial degree plus one");
            DNDS_FIELD(jacobiRelax, "Jacobi/SOR relaxation");
            DNDS_FIELD(SORInstead, "Use ordered in-place SOR");
            DNDS_FIELD(coreFirst, "Use conservative node-star core before dependent cells for ordered SOR");
            DNDS_FIELD(HQMWeights, "Legacy HQM_OPT derivative and HQM_SD geometry weights");
            DNDS_FIELD(functional, "Independent isotropic functional weights and length policy");
            DNDS_FIELD(cacheDiffBase, "Retain basis values and derivatives on face quadrature points");
            DNDS_FIELD(bcWeight, "Dirichlet boundary value weight");
            DNDS_FIELD(boundaryDerivativeWeights, "Boundary weights by derivative order; nonzero derivatives impose homogeneous derivative targets");
            DNDS_FIELD(boundaryZoneWeights, "Optional BC-zone per-order weight overrides");
            DNDS_FIELD(svdTolerance, "Relative small-mode SVD cutoff against sigma_max");
            DNDS_FIELD(svdAbsoluteTolerance, "Absolute SVD cutoff, zero by default");
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

    /// MPI-global owned-cell setup diagnostics; no ghost double counting.
    struct ReconstructionInverseDiagnostics
    {
        index cells = 0, filteredCells = 0, discardedModes = 0, rankZeroCells = 0;
        real maxCondition = 0;
        index maxConditionCell = UnInitIndex;
        std::array<index, 20> rankCounts{}; // Degree-three 3D has at most 19 modes.
    };

    /// Borrowed metric spans for numerical loops. Retain FiniteVolumeRead through
    /// every use. These initial pointers address host storage; device placement
    /// is a later backend responsibility, not implied by callable accessors.
    struct FiniteVolumeHostView
    {
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> quadrature;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<3>> radialMoment;
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
        Runtime::Array<real>::ReadLease _cells, _quadrature, _radialMoment;
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
        /// axisymmetric uses radius=y>=0 and measures r*dA/r*ds, without 2*pi.
        /// Equation source terms and cylindrical component connections remain
        /// with the equation adapter; this option changes integration only.
        static FiniteVolumePlan ConstructMetrics(ssp<Geom::UnstructuredMesh> mesh, int intOrder = 5, bool axisymmetric = false);
        void Validate() const;
        FiniteVolumeRead AcquireRead() const;
        index NumCell() const;
        index NumCellProc() const;
        index NumFaceProc() const;
        int Dimension() const;
        bool Axisymmetric() const;
        const Geom::EntitySpace &Cells() const;
        FVField BuildField(index width) const;
        /// Collective binding, retained by caller for persistent repeated pulls.
        Runtime::TransferBinding BindHalo(FVField &field) const;
        void ValidateField(const FVField &field, index width) const;
        MPI_Comm Communicator() const;
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
    /// are different operations. Declare vector triples in BuildWorkspace so
    /// the periodic component map precedes characteristic limiting.
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
        /// Non-overlapping triples of polar-vector components. Other components
        /// are scalars. Periodic rotations act on these triples, independently
        /// of the geometric dimension (three-component flow on 2D geometry).
        std::vector<index> periodicVectorStarts;
    };

    /// Borrowed cache pointers for owned-cell face stencils. All derivatives
    /// are in the selected parent's reference frame. Retain the read scope.
    struct ReconstructionBasisHostView
    {
        Runtime::ArrayView<const real, false, Runtime::TableRows> face;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        int modes;
        DNDS_DEVICE_CALLABLE const real *FaceMatrix(index iPoint, int side) const { return face[2 * iPoint + side].data(); }
        /// Column-major (modes+1)-by-modes matrix; derivative includes order0.
        DNDS_DEVICE_CALLABLE real FaceDerivative(index iPoint, int side, int derivative, int mode) const
        {
            return FaceMatrix(iPoint, side)[mode * (modes + 1) + derivative];
        }
    };
    /// Pins the geometry generation and Runtime cache allocations once outside
    /// a hot loop. Face queries must be in an owned-cell stencil and select a
    /// real parent. Generic/debug access can use checked GetCachedFaceBasis.
    class ReconstructionBasisRead
    {
        std::shared_ptr<const Detail::ReconstructionData> _data;
        FiniteVolumeRead _geometry;
        std::optional<Runtime::Array<real>::ReadLease> _basis;
        explicit ReconstructionBasisRead(std::shared_ptr<const Detail::ReconstructionData> data);
        friend class VariationalReconstructionPlan;

    public:
        ReconstructionBasisHostView HostView() const;
    };

    /// Composition of metrics, basis/moments and factored VR systems on Runtime
    /// storage. CPU kernels consume explicit arrays; no legacy VR owner exists.
    class VariationalReconstructionPlan
    {
        friend class PolynomialSamplingPlan;
        std::shared_ptr<const Detail::ReconstructionData> _data;
        explicit VariationalReconstructionPlan(std::shared_ptr<const Detail::ReconstructionData> data) : _data(std::move(data)) {}

    public:
        static VariationalReconstructionPlan ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings = {});
        const FiniteVolumePlan &Metrics() const;
        const ReconstructionSettings &Settings() const;
        int NRecDOF() const;
        ReconstructionInverseDiagnostics GetInverseDiagnostics() const;
        /// Owned-cell spectrum diagnostics from setup, without refactorization.
        HardEigen::SVDInverseDiagnostics GetCellInverseDiagnostics(index iCell) const;
        void Validate() const;
        ReconstructionWorkspace BuildWorkspace(index components, std::vector<index> periodicVectorStarts = {}) const;
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
        /// Writes selected rows of temporary only; Jacobi reads input without
        /// modifying it. SOR uses work.coefficients as its separate working
        /// owned iterate, and input's ghost rows as the frozen remote iterate.
        /// Core acquires no input ghost leases. Finish both phases before
        /// CommitReconstruction; never update a live direct MPI send buffer.
        void DoReconstructionIterTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                    const ReconstructionBoundaryValues &boundary,
                                    ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Copy the completed temporary into owned coefficients. The caller
        /// certifies every owned row was produced and all old sends completed.
        void CommitReconstruction(ReconstructionWorkspace &work) const;
        /// One descending-order 3WBAP pass. Call degree..1 and exchange the
        /// coefficient halo after EACH pass, matching the legacy schedule.
        void DoLimiterWBAP_3Order(ReconstructionWorkspace &work, int order,
                                  const FaceCharacteristicTransform *transform = nullptr) const;
        /// Out-of-place selected-row pass. Call Core and HaloDependent against
        /// the same immutable coefficients, then commit/exchange before the
        /// next descending order. Inactive rows copy input unchanged.
        void DoLimiterWBAP_3OrderTo(ReconstructionWorkspace &work, int order,
                                    const FaceCharacteristicTransform *transform = nullptr,
                                    ReconstructionRegion region = ReconstructionRegion::All,
                                    const Runtime::Array<real> *indicator = nullptr, real threshold = 0) const;
        /// Arbitrary physical point sampling from cell means plus zero-mean
        /// coefficients. Cell IDs are current local IDs, including ready ghosts.
        void Evaluate(const FVField &mean, const FVField &coefficients,
                      const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                      Runtime::Array<real> &values) const;
        /// Spatial derivative of the polynomial, in the cell reference frame.
        /// derivative is a central Taylor multi-index ordinal (0 means value).
        /// For derivative>0 the cell mean contributes zero.
        void EvaluateDerivative(const FVField &mean, const FVField &coefficients,
                                const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                                Runtime::Array<real> &values, int derivative) const;
        /// Diagnostic copy of one cached basis matrix, rows=derivative modes,
        /// columns=nonconstant polynomial modes, in that parent's own frame.
        /// Throws when caching was disabled, the parent is exterior, or the
        /// face lies outside every owned cell stencil.
        Eigen::MatrixXd GetCachedFaceBasis(index iFace, int side, int iQuadrature) const;
        ReconstructionBasisRead AcquireBasisRead() const;
        /// Unscaled assembled linear system A*x - sum(B*x_neighbor). Callers
        /// refresh coefficient ghosts explicitly. Distinct output is required.
        /// Geometry/BC weights are frozen with this plan; no relaxation enters
        /// these operators. General GMRES is valid; symmetry/SPD and nullspace
        /// conditions for PCG must be established by the caller.
        void ApplyLinearOperator(const FVField &coefficients, FVField &output,
                                 const std::vector<index> &periodicVectorStarts = {}) const;
        /// Affine mean/Dirichlet forcing in unscaled coordinates. Homogeneous
        /// perturbations use ApplyLinearOperator without injecting BC values.
        void BuildRightHandSide(const FVField &mean, const ReconstructionBoundaryValues &boundary, FVField &output,
                                const std::vector<index> &periodicVectorStarts = {}) const;
        void ComputeResidual(const FVField &rhs, const FVField &coefficients, FVField &output,
                             const std::vector<index> &periodicVectorStarts = {}) const;
        /// Cell-diagonal truncated SVD inverse, including its minimum-norm
        /// projection when the local matrix has a null space. Owned rows only.
        void ApplyLocalPreconditioner(const FVField &input, FVField &output) const;
    };
}

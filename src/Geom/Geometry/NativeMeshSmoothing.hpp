#pragma once

#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/RadialBasisFunction.hpp"
#include <functional>

namespace DNDS::Geom
{
    struct NativeSmoothingOptions
    {
        real RBFRadius = 1, MaxIncludedAngle = 15, RBFPower = 1, refDWall = 1e-3;
        int nIter = 60, nSearch = 30;
        RBF::RBFKernelType kernel = RBF::InversedDistanceA1;
    };
    enum class NativeSmoothingConstraint : uint8_t
    {
        Interior,
        OriginalBoundary,
        Prescribed,
        OtherBoundary
    };

    /// Immutable staged displacement problem. Explicit roles replace the old
    /// largeReal coordinate sentinels; source mesh topology remains native.
    class NativeSmoothingWorkspace
    {
        friend class NativeMeshSmoothing;
        std::shared_ptr<const UnstructuredMeshState> _source;
        MPIInfo _mpi;
        std::optional<EntitySpace> _nodes;
        Runtime::ArrayPair<real> _displacement;
        Runtime::ArrayPair<uint8_t> _constraints;
        std::shared_ptr<Runtime::Array<real>::ReadLease> _ownedRead, _ghostRead;
        std::shared_ptr<Runtime::Array<uint8_t>::ReadLease> _ownedRoles, _ghostRoles;
        index _totalMoved = 0;

    public:
        void Validate() const
        {
            DNDS_check_throw_info(_source && _nodes, "uninitialized native smoothing workspace");
            _source->Validate();
            _nodes->Validate();
        }
        void Validate(const MPIInfo &mpi) const
        {
            Validate();
            int comparison = MPI_UNEQUAL;
            MPI_Comm_compare(_mpi.comm, mpi.comm, &comparison);
            DNDS_check_throw_info(comparison == MPI_IDENT || comparison == MPI_CONGRUENT, "native smoothing workspace communicator differs");
        }
        const std::shared_ptr<const UnstructuredMeshState> &Source() const
        {
            Validate();
            return _source;
        }
        const Runtime::ArrayPair<real> &Displacement() const
        {
            Validate();
            return _displacement;
        }
        const Runtime::ArrayPair<uint8_t> &Constraints() const
        {
            Validate();
            return _constraints;
        }
        index TotalMoved() const
        {
            Validate();
            return _totalMoved;
        }
    };

    class NativeMeshSmoothing
    {
    public:
        static std::shared_ptr<const NativeSmoothingWorkspace> PrepareBoundary(
            std::shared_ptr<const UnstructuredMeshState> source, const MPIInfo &mpi,
            NativeSmoothingOptions options, const std::function<bool(t_index)> &selected);
        /// Established local RBF interpolation. The boundary spatial dataset is
        /// explicitly replicated, matching the legacy algorithm's scalability
        /// limit; topology/ghost construction stays sparse. Returns detached
        /// coordinates, retaining identity/topology and invalidating wall vectors.
        static std::shared_ptr<const UnstructuredMeshState> LocalRBF(
            const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi,
            NativeSmoothingOptions options);
        static std::shared_ptr<const UnstructuredMeshState> GlobalRBF(
            const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi,
            NativeSmoothingOptions options);
        static std::shared_ptr<const UnstructuredMeshState> Elasticity(
            const NativeSmoothingWorkspace &workspace, const MPIInfo &mpi,
            NativeSmoothingOptions options);
    };
}

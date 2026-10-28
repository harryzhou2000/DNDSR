#pragma once

#include "Geom/Unstructured/UnstructuredMeshState.hpp"

namespace DNDS::Geom
{
    /// Detached serial-output rows in current rank/global order. Only readerRank
    /// retains global payloads. No legacy mesh or topology arrays are allocated.
    /// Copies retain gathered buffers; geometry changes require another Build.
    struct NativeMeshSerialOutput
    {
        int dimension = -1, readerRank = 0;
        bool periodic = false;
        Periodicity periodicInfo;
        Runtime::Array<real> coordinates;
        Runtime::Array<index> cellNodes;
        Runtime::Array<ElemInfo> cellInfo;
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi;
        /// Maps-only compatibility for solver output fields and part labels.
        std::shared_ptr<GlobalOffsetsMapping> cellGlobal, nodeGlobal;
        std::shared_ptr<OffsetAscendIndexMapping> cellPull, nodePull;

        static NativeMeshSerialOutput Build(const UnstructuredMeshState &source, const MPIInfo &mpi, int readerRank = 0);
        /// For facade writers: reject stale geometry/topology/order while
        /// detached gathered arrays remain readable after source destruction.
        void ValidateSource(const UnstructuredMeshState &source) const;

        /// Attach existing field arrays before requesting their MPI schedule.
        /// Detached output payloads stay usable after source destruction, but
        /// field borrowing rejects a source order invalidated by publication.
        /// Caller associates the field with this source; equal row counts do
        /// not prove that an arbitrary external field has the correct identity.
        template <class TTransformer>
        void BorrowCellIndexing(TTransformer &transformer) const
        {
            BorrowIndexing(transformer, cellGlobal, cellPull, _cells, _cellStamp);
        }
        template <class TTransformer>
        void BorrowNodeIndexing(TTransformer &transformer) const
        {
            BorrowIndexing(transformer, nodeGlobal, nodePull, _nodes, _nodeStamp);
        }

    private:
        MPIInfo _mpi;
        index _cells = 0, _nodes = 0;
        Runtime::VersionStamp _cellStamp, _nodeStamp;
        std::weak_ptr<const void> _sourceSnapshot;
        template <class TTransformer>
        void BorrowIndexing(TTransformer &transformer, const std::shared_ptr<GlobalOffsetsMapping> &global,
                            const std::shared_ptr<OffsetAscendIndexMapping> &pull, index owned,
                            const Runtime::VersionStamp &stamp) const
        {
            stamp.Validate();
            DNDS_check_throw_info(global && pull && transformer.father && transformer.son,
                                  "attach field arrays before borrowing native serial-output indexing");
            DNDS_check_throw_info(transformer.father->Size() == owned, "serial-output field owned rows differ from source");
            for (MPI_Comm comm : {transformer.mpi.comm, transformer.father->getMPI().comm, transformer.son->getMPI().comm})
            {
                DNDS_check_throw_info(comm != MPI_COMM_NULL, "serial-output field communicator is null");
                int relation = MPI_UNEQUAL;
                MPI_Comm_compare(comm, _mpi.comm, &relation);
                DNDS_check_throw_info(relation == MPI_IDENT || relation == MPI_CONGRUENT, "serial-output field communicator differs from source");
            }
            auto globalCopy = std::make_shared<GlobalOffsetsMapping>(*global);
            auto pullCopy = std::make_shared<OffsetAscendIndexMapping>(*pull);
            transformer.pLGlobalMapping = std::move(globalCopy);
            transformer.pLGhostMapping = std::move(pullCopy);
            transformer.father->pLGlobalMapping = transformer.pLGlobalMapping;
        }
    };
}

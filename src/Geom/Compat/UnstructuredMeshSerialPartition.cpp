#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/SerialAdjReordering.hpp"
#include "Solver/Direct.hpp"
namespace DNDS::Geom
{
    std::size_t UnstructuredMesh::CoordinateHash()
    {
        auto read = AcquireRead();
        const auto view = read.HostView();
        auto rowsHash = [&](index first, index count)
        {
            std::size_t values = 0;
            for (index row = first; row < first + count; ++row)
                for (int component = 0; component < 3; ++component)
                    values ^= std::hash<real>{}(view.Coord(row)[component]);
            return array_hash<std::size_t, 3>{}({std::size_t(count), 0, values});
        };
        // Exact ArrayPair< ArrayEigenVector<3> > diagnostic contract: include
        // owned and appended ghost rows, then gather in communicator rank order.
        index local = array_hash<std::size_t, 2>{}({rowsHash(0, view.NumNode()),
                                                    rowsHash(view.NumNode(), view.NumNodeGhost())});
        std::vector<index> hashes(size_t(mpi.size));
        MPI::Allgather(&local, 1, DNDS_MPI_INDEX, hashes.data(), 1, DNDS_MPI_INDEX, mpi.comm);
        return vector_hash<index>{}(hashes);
    }
    tLocalMatStruct UnstructuredMesh::GetCell2CellFaceVLocal(bool local)
    {
        auto read = AcquireRead(MeshReadRequirements::CellFlux());
        return CurrentNativeState()->BuildLocalCellNeighbors(local);
    }
    static std::vector<index> put_perm_back_to_local_parts(const std::vector<index> &new2old, const std::vector<index> &localPartStarts)
    {
        auto fGetIPart = [&](index iCell) -> index
        {
            return std::lower_bound(localPartStarts.begin(), localPartStarts.end(), iCell, std::less_equal<index>()) -
                   localPartStarts.begin() - 1;
        };
        index nParts = localPartStarts.size() - 1;
        index N = new2old.size();
        std::vector<std::vector<index>> outB(nParts);
        for (index iPart = 0; iPart < nParts; iPart++)
            outB[iPart].reserve(localPartStarts[iPart + 1] - localPartStarts[iPart]);
        for (index v : new2old)
        {
            index iPart = fGetIPart(v);
            DNDS_assert_info(iPart >= 0 && iPart < nParts, fmt::format("iPart [{}], v [{}], N [{}]", iPart, v, N));
            outB[iPart].push_back(v);
        }
        std::vector<index> out;
        out.reserve(new2old.size());
        for (const auto &vec : outB)
        {
            out.insert(out.end(), vec.begin(), vec.end());
        }
        DNDS_assert(out.size() == new2old.size());
        return out;
    }
    static void check_permutations(const std::vector<index> &new2old, const std::vector<index> &old2new, const std::vector<index> &localPartStarts)
    {
        index N = new2old.size();
        DNDS_assert(N == old2new.size());
        if (N == 0) // special: natural order
            return;
        std::unordered_set<index> counted;
        counted.reserve(N);
        for (auto v : new2old)
        {
            DNDS_assert(v >= 0 && v < N);
            DNDS_assert(counted.count(v) == 0);
            counted.insert(v);
        }
        counted.clear();
        for (auto v : old2new)
        {
            DNDS_assert(v >= 0 && v < N);
            DNDS_assert(counted.count(v) == 0);
            counted.insert(v);
        }

        for (index i = 0; i < N; i++)
        {
            DNDS_assert(new2old[old2new[i]] == i);
            DNDS_assert(old2new[new2old[i]] == i);
        }

        if (localPartStarts.size() > 2)
        {
            int nParts = localPartStarts.size() - 1;
            for (int iPart = 0; iPart < nParts; iPart++)
            {
                index start = localPartStarts[iPart];
                index end = localPartStarts[iPart + 1];
                // std::cout << start << ", " << end << ", " << N << std::endl;
                DNDS_assert(start <= end);
                DNDS_assert(start >= 0);
                DNDS_assert(end <= N);
                for (index iCell = localPartStarts[iPart]; iCell < localPartStarts[iPart + 1]; iCell++)
                {
                    index n2o = new2old[iCell];
                    index o2n = old2new[iCell];
                    DNDS_assert(start <= n2o && n2o < end);
                    DNDS_assert(start <= o2n && o2n < end);
                }
            }
        }
    }
    void UnstructuredMesh::ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control)
    {
        if (!control.useDirectPrec)
            return;
        std::vector<index> partStarts;

        for (int part = 0; part < NLocalParts(); ++part)
            partStarts.push_back(LocalPartStart(part));
        partStarts.push_back(NumCell());
        _nativeState = _nativeState->WithLocalCellSetup(partStarts, GetCell2CellFaceVLocal(true));
        _nativeSmoothing.reset();

        const auto &graph = LocalCellNeighbors();
        auto &localFillOrderingNew2Old = symLU.localFillOrderingNew2Old;
        auto &localFillOrderingOld2New = symLU.localFillOrderingOld2New;
        if (!this->NumCell())
        {
            // Nonempty peers report bandwidth through the same two reductions
            // below. Empty owners must remain in that collective schedule.
            if (control.getOrderingCode() == 3 || control.getOrderingCode() == 4)
            {
                index oldBandwidth = 0, newBandwidth = 0;
                MPI::AllreduceOneIndex(oldBandwidth, MPI_MAX, mpi);
                MPI::AllreduceOneIndex(newBandwidth, MPI_MAX, mpi);
            }
            return;
        }
        if (control.getOrderingCode() == -1)
        {
            localFillOrderingNew2Old.reserve(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 == 0)
                    localFillOrderingNew2Old.push_back(i);
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 != 0)
                    localFillOrderingNew2Old.push_back(i);
            localFillOrderingOld2New.resize(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                localFillOrderingOld2New.at(localFillOrderingNew2Old.at(i)) = i;
        }
        else if (control.getOrderingCode() == 0)
        {
            // do nothing, natural order
        }
        else if (control.getOrderingCode() == 1) // Metis
        {

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling metis" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_Metis(graph);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): metis done" << std::endl;
        }
        else if (control.getOrderingCode() == 2) // MMD
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::minimum_degree_ordering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_BoostMMD(graph);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done" << std::endl;
        }
        else if (control.getOrderingCode() == 3) // RCM
        {
            index bandWidthOld{0}, bandWidthNew{0};
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::cuthill_mckee_ordering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_BoostRCM(graph, bandWidthOld, bandWidthNew);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << fmt::format("UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done, old BW [{}] new BW [{}] ", bandWidthOld, bandWidthNew) << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else if (control.getOrderingCode() == 4) // CorrectRCM
        {
            index bandWidthOld{0}, bandWidthNew{0};
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling CorrectRCM::CuthillMcKeeOrdering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_CorrectRCM(
                    graph.begin(), graph.end(), bandWidthOld, bandWidthNew);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << fmt::format("UnstructuredMesh::ObtainLocalFactFillOrdering(): CorrectRCM done, old BW [{}] new BW [{}] ", bandWidthOld, bandWidthNew) << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else
        {
            DNDS_assert_info(false, "No such ordering code");
        }
        if (this->NLocalParts() > 1 && control.getOrderingCode() != 0)
        {
            localFillOrderingNew2Old = put_perm_back_to_local_parts(localFillOrderingNew2Old, partStarts);
            for (index i = 0; i < localFillOrderingNew2Old.size(); i++)
                localFillOrderingOld2New[localFillOrderingNew2Old[i]] = i;
        }
        check_permutations(localFillOrderingNew2Old, localFillOrderingOld2New, partStarts);
    }
}

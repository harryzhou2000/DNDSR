#pragma once

#include "DNDS/Core/Offsets.hpp"
#include "DNDS/Core/Permutation.hpp"
#include "Geom/Geometry/Geometric.hpp"

#include "Geom/Transition/CorrectRCM.hpp"

#include "Geom/IO/Metis.hpp"

#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

namespace DNDS::Geom
{
    /// Compute the half-bandwidth of an adjacency range under an optional old->new permutation.
    inline index ComputeGraphBandwidth(
        tLocalMatStruct::const_iterator mat_begin, tLocalMatStruct::const_iterator mat_end,
        const std::vector<index> *old2New = nullptr, index offset = 0)
    {
        const index mat_size = size_t_to_signed<index>(mat_end - mat_begin);
        index bandwidth = 0;
        for (index iCell = 0; iCell < mat_size; ++iCell)
        {
            const index lhs = old2New ? (*old2New)[size_t(iCell)] : (iCell + offset);
            for (auto iCOther : mat_begin[iCell])
            {
                const index rhs = old2New ? old2New->at(size_t(iCOther - offset)) : iCOther;
                bandwidth = std::max(bandwidth, std::abs(lhs - rhs));
            }
        }
        return bandwidth;
    }

    /**
     * @brief Partition a (sub-)graph using Metis.
     *
     * Adjacency entries in [mat_begin, mat_end) are absolute indices offset by ind_offset.
     * This function internally subtracts ind_offset and filters out cross-sub-graph
     * references (entries outside [ind_offset, ind_offset + n_elem)) before building
     * the 0-based CSR that Metis requires.
     *
     * @param ind_offset  Absolute index of the first row in this sub-graph. For a full
     *                    graph this is 0 and no filtering occurs.
     */
    inline auto PartitionSerialAdj_Metis(
        tLocalMatStruct::const_iterator mat_begin, tLocalMatStruct::const_iterator mat_end, int nPart,
        index ind_offset = 0,
        std::string metisType = "KWAY", int metisNcuts = 3, int metisUfactor = 5, int metisSeed = 0)
    {
        idx_t nCell = METIS::indexToIdx(size_t_to_signed<index>(mat_end - mat_begin));
        if (nCell <= 1)
            return std::vector<idx_t>(size_t(nCell), 0);
        idx_t nCon{1};
        std::array<idx_t, METIS_NOPTIONS> options{};
        METIS_SetDefaultOptions(options.data());
        {
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM; //? could try shem?
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
            // options[METIS_OPTION_NO2HOP] = 0; // only available in metis 5.1.0
            options[METIS_OPTION_NCUTS] = std::max(metisNcuts, 1);
            options[METIS_OPTION_NITER] = 10;
            // options[METIS_OPTION_UFACTOR] = 30; // load imbalance factor, fow k-way
            options[METIS_OPTION_UFACTOR] = metisUfactor;
            options[METIS_OPTION_MINCONN] = 1;
            options[METIS_OPTION_CONTIG] = 1;       // ! forcing contigious partition now ? necessary?
            options[METIS_OPTION_SEED] = metisSeed; // ! seeding 0 for determined result
            options[METIS_OPTION_NUMBERING] = 0;
            // options[METIS_OPTION_DBGLVL] = METIS_DBG_TIME | METIS_DBG_IPART;
            // options[METIS_OPTION_DBGLVL] = METIS_DBG_TIME;
        }
        //! Build CSR with 0-based local indices: subtract ind_offset and skip
        //! entries outside [ind_offset, ind_offset + nCell) (cross-sub-graph edges).
        auto [xadj, adjncy] = BuildCSRFromRows<idx_t, idx_t>(
            mat_begin, mat_end,
            [&](index iCOther, idx_t &out)
            {
                const index iLocal = iCOther - ind_offset;
                if (iLocal >= 0 && iLocal < nCell)
                {
                    out = METIS::indexToIdx(iLocal);
                    return true;
                }
                return false;
            });

        // METIS rejects contiguous-part requests on disconnected input. Keep
        // the historical connected-graph option; local mesh subgraphs may
        // legitimately have several components or isolated cells.
        std::vector<uint8_t> visited(size_t(nCell), 0);
        std::vector<idx_t> pending{0};
        visited[0] = 1;
        for (size_t next = 0; next < pending.size(); ++next)
            for (idx_t edge = xadj[size_t(pending[next])]; edge < xadj[size_t(pending[next]) + 1]; ++edge)
            {
                const auto other = adjncy[size_t(edge)];
                if (!visited[size_t(other)])
                {
                    visited[size_t(other)] = 1;
                    pending.push_back(other);
                }
            }
        if (pending.size() != size_t(nCell))
            options[METIS_OPTION_CONTIG] = 0;

        idx_t objval;
        std::vector<idx_t> partOut(nCell);

        int ret{0};
        if (metisType == "RB")
            ret = METIS_PartGraphRecursive(
                &nCell, &nCon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
                &nPart, nullptr, nullptr, options.data(), &objval, partOut.data());
        else if (metisType == "KWAY")
            ret = METIS_PartGraphKway(
                &nCell, &nCon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
                &nPart, nullptr, nullptr, options.data(), &objval, partOut.data());

        DNDS_assert_info(ret == METIS_OK, fmt::format("Metis return not ok, [{}]", ret));

        return partOut;
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_Metis(const tLocalMatStruct &mat)
    {
        idx_t nCell = METIS::indexToIdx(size_t_to_signed<index>(mat.size()));
        idx_t nCon{1};
        std::array<idx_t, METIS_NOPTIONS> options{};
        METIS_SetDefaultOptions(options.data());
        {
            options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;
            options[METIS_OPTION_NSEPS] = 1;
            options[METIS_OPTION_NITER] = 10;
            options[METIS_OPTION_UFACTOR] = 30;
            options[METIS_OPTION_COMPRESS] = 0; // do not compress
            // options[METIS_OPTION_CCORDER] = 0; //use default?
            options[METIS_OPTION_SEED] = 0;    // ! seeding 0 for determined result
            options[METIS_OPTION_PFACTOR] = 0; // not removing large vertices
            options[METIS_OPTION_NUMBERING] = 0;
            // options[METIS_OPTION_DBGLVL] = METIS_DBG_TIME | METIS_DBG_IPART;
        }
        auto [xadj, adjncy] = BuildCSRFromRows<idx_t, idx_t>(
            mat,
            [](index iCOther, idx_t &out)
            {
                out = METIS::indexToIdx(iCOther);
                return true;
            });
        std::vector<idx_t> perm(static_cast<size_t>(nCell)), iPerm(static_cast<size_t>(nCell));

        int ret = METIS_NodeND(&nCell, xadj.data(), adjncy.data(), nullptr, options.data(), perm.data(), iPerm.data());
        DNDS_assert_info(ret == METIS_OK, fmt::format("Metis return not ok, [{}]", ret));

        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;

        localFillOrderingNew2Old.resize(nCell);
        localFillOrderingOld2New.resize(nCell);
        for (index i = 0; i < nCell; i++)
        {
            localFillOrderingNew2Old[i] = perm[i];
            localFillOrderingOld2New[i] = iPerm[i];
        }

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_BoostMMD(const tLocalMatStruct &mat)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        using namespace boost;
        using Graph = adjacency_list<vecS, vecS, directedS>;
        Graph cell2cellG(mat.size());
        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                add_edge(iCell, iCOther, cell2cellG);
        std::vector<index> supernodeSizes(mat.size(), 1), degree(mat.size(), 0);
        localFillOrderingNew2Old.resize(mat.size(), 0);
        localFillOrderingOld2New.resize(mat.size(), 0);
        boost::property_map<Graph, vertex_index_t>::type id = get(vertex_index, cell2cellG);
        minimum_degree_ordering(
            cell2cellG,
            make_iterator_property_map(degree.data(), id, degree[0]),
            localFillOrderingOld2New.data(),
            localFillOrderingNew2Old.data(),
            make_iterator_property_map(supernodeSizes.data(), id, supernodeSizes[0]),
            0,
            id);
        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_BoostRCM(const tLocalMatStruct &mat, index &bandWidthOld, index &bandWidthNew)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        using namespace boost;
        using Graph = adjacency_list<vecS, vecS, undirectedS, property<vertex_color_t, default_color_type, property<vertex_degree_t, int>>>;
        using Vertex = graph_traits<Graph>::vertex_descriptor;
        // typedef graph_traits<Graph>::vertices_size_type size_type;
        Graph cell2cellG(mat.size());
        const std::vector<std::vector<index>> &cell2cellFaceV = mat;
        bandWidthOld = ComputeGraphBandwidth(mat.begin(), mat.end());
        for (index iCell = 0; iCell < size_t_to_signed<index>(mat.size()); iCell++)
            for (auto iCOther : cell2cellFaceV[iCell])
                add_edge(iCell, iCOther, cell2cellG);
        localFillOrderingNew2Old.resize(mat.size(), 0);
        Vertex startVert = vertex(0, cell2cellG);
        cuthill_mckee_ordering(cell2cellG, startVert, localFillOrderingNew2Old.rbegin(),
                               get(vertex_color, cell2cellG), get(vertex_degree, cell2cellG));
        localFillOrderingOld2New = InvertPermutationChecked(
            localFillOrderingNew2Old, "The output of boost::cuthill_mckee_ordering is invalid!");
        bandWidthNew = ComputeGraphBandwidth(mat.begin(), mat.end(), &localFillOrderingOld2New);

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    template <class T>
    class OffsetIterator
    {
        const T *ptr_;
        T offset_;

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T *;
        using reference = T; // NOT a real reference! It's a temporary.

        OffsetIterator(const T *p, T off) : ptr_(p), offset_(off) {}

        // Dereference returns a transformed VALUE
        T operator*() const
        {
            return *ptr_ + offset_;
        }

        // Iterator movement
        OffsetIterator &operator++()
        {
            ++ptr_;
            return *this;
        }

        OffsetIterator operator++(int)
        {
            OffsetIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        // Comparison
        bool operator==(const OffsetIterator &other) const
        {
            return ptr_ == other.ptr_;
        }

        bool operator!=(const OffsetIterator &other) const
        {
            return !(*this == other);
        }
    };
    // Proxy wrapper class
    template <class T>
    class OffsetRange
    {
        const T *begin_;
        const T *end_;
        T offset_;

    public:
        // Constructor from a sub-range of a vector
        OffsetRange(typename std::vector<T>::const_iterator b, typename std::vector<T>::const_iterator e, T offset)
            : begin_(std::addressof(*b)), end_(std::addressof(*e)), offset_(offset) {}

        // Support for full vector
        explicit OffsetRange(const std::vector<const T> &vec, T offset)
            : begin_(vec.data()), end_(vec.data() + vec.size()), offset_(offset) {}

        [[nodiscard]] OffsetIterator<T> begin() const { return {begin_, offset_}; }
        [[nodiscard]] OffsetIterator<T> end() const { return {end_, offset_}; }
        [[nodiscard]] ptrdiff_t size() const { return end_ - begin_; }
    };

    inline std::pair<std::vector<index>, std::vector<index>> ReorderSerialAdj_CorrectRCM(
        tLocalMatStruct::const_iterator mat_begin, tLocalMatStruct::const_iterator mat_end,
        index &bandWidthOld, index &bandWidthNew,
        index offset = 0)
    {
        std::vector<index> localFillOrderingNew2Old, localFillOrderingOld2New;
        bandWidthOld = 0;

        // for (index iCell = 0; iCell < this->NumCell(); ++iCell)
        // {
        //     cell2cellFaceVLocal[iCell].resize(4);
        //     int x = iCell % 20, y = iCell / 20;
        //     cell2cellFaceVLocal[iCell][0] = mod(x - 1, 20) + y * 20;
        //     cell2cellFaceVLocal[iCell][1] = mod(x + 1, 20) + y * 20;
        //     cell2cellFaceVLocal[iCell][2] = x + mod(y - 1, 20) * 20;
        //     cell2cellFaceVLocal[iCell][3] = x + mod(y + 1, 20) * 20;
        // }

        index mat_size = mat_end - mat_begin;
        bandWidthOld = ComputeGraphBandwidth(mat_begin, mat_end, nullptr, offset);

        localFillOrderingOld2New.resize(mat_size, 0);
        auto graphFunctor = [&](index i)
        {
            DNDS_assert(i >= 0 && i < mat_size);
            return OffsetRange<index>(mat_begin[i].cbegin(), mat_begin[i].cend(), -offset); // mind that OffsetRange adds the offset
        }; // todo: need improvement in CorrectRCM: can pass a temporary functor and store
        auto graph = CorrectRCM::UndirectedGraphProxy(graphFunctor, size_t_to_signed<int64_t>(mat_size));
        int ret = graph.CheckAdj();
        (void)ret;
        CorrectRCM::CuthillMcKeeOrdering(
            graph,
            [&](index i) -> index &
            {
                return localFillOrderingOld2New.at(i);
            },
            0);
        for (auto &v : localFillOrderingOld2New)
            v = size_t_to_signed<index>(localFillOrderingOld2New.size()) - 1 - v;

        localFillOrderingNew2Old = InvertPermutationChecked(
            localFillOrderingOld2New, "The output of CorrectRCM::CuthillMcKeeOrdering is invalid!");
        bandWidthNew = ComputeGraphBandwidth(mat_begin, mat_end, &localFillOrderingOld2New, offset);

        return {localFillOrderingNew2Old, localFillOrderingOld2New};
    }

    /**
     * @brief Partition a sub-graph and optionally RCM-reorder within each partition.
     *
     * When used for second-level (inner) partitioning, pass full_mat_begin / full_n_elem
     * so that cross-sub-graph references in the full graph are updated after permutation.
     * See implementation in SerialAdjReordering.cpp for detailed documentation.
     */
    std::vector<index> ReorderSerialAdj_PartitionMetisC(
        tLocalMatStruct::iterator mat_begin, tLocalMatStruct::iterator mat_end,
        std::vector<index>::iterator i_new2old_begin, std::vector<index>::iterator i_new2old_end,
        int nParts,
        index ind_offset,
        bool do_rcm,
        index &bwOldM, index &bwNewM,
        tLocalMatStruct::iterator full_mat_begin = tLocalMatStruct::iterator{},
        index full_n_elem = 0);

}

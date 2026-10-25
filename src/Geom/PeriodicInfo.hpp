#pragma once

#include "DNDS/ArrayTransformer.hpp"
#include "DNDS/ArrayPair.hpp"
#include "PeriodicTypes.hpp"

namespace DNDS::Geom
{

    class NodePeriodicBitsRow // instead of std::vector<NodePeriodicBits> for building on raw buffer as a "mapping" object
    {
        NodePeriodicBits *_p_indices;
        rowsize _row_size;

    public:
        NodePeriodicBitsRow(NodePeriodicBits *ptr, rowsize siz) : _p_indices(ptr), _row_size(siz) {} // default actually

        // --- Special members (cppcoreguidelines-special-member-functions) ---
        // NodePeriodicBitsRow is a non-owning view (pointer + size).
        // Assignment copies the pointed-to contents, including for rvalues;
        // construction only copies the non-owning view.
        ~NodePeriodicBitsRow() = default;
        NodePeriodicBitsRow(const NodePeriodicBitsRow &) = default;
        NodePeriodicBitsRow(NodePeriodicBitsRow &&) = default;
        NodePeriodicBitsRow &operator=(NodePeriodicBitsRow &&r)
        {
            this->operator=(static_cast<const NodePeriodicBitsRow &>(r));
            return *this;
        }

        NodePeriodicBits &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < _row_size);
            return _p_indices[j];
        }

        NodePeriodicBits operator[](rowsize j) const
        {
            DNDS_assert(j >= 0 && j < _row_size);
            return _p_indices[j];
        }

        operator std::vector<NodePeriodicBits>() const // copies to a new std::vector<index>
        {
            return {_p_indices, _p_indices + _row_size};
        }

        void operator=(const std::vector<NodePeriodicBits> &r)
        {
            DNDS_assert(_row_size == r.size());
            std::copy(r.begin(), r.end(), _p_indices);
        }

        void operator=(const NodePeriodicBitsRow &r)
        {
            if (this == &r)
                return;
            DNDS_assert(_row_size == r.size());
            std::copy(r.cbegin(), r.cend(), _p_indices);
        }

        NodePeriodicBits bitandReduce()
        {
            NodePeriodicBits ret;
            ret.setP1True();
            ret.setP2True();
            ret.setP3True();
            for (auto &v : *this)
                ret = ret & v;
            return ret;
        }

        NodePeriodicBits *begin() { return _p_indices; }
        NodePeriodicBits *end() { return _p_indices + _row_size; } // past-end
        [[nodiscard]] const NodePeriodicBits *cbegin() const { return _p_indices; }
        [[nodiscard]] const NodePeriodicBits *cend() const { return _p_indices + _row_size; } // past-end
        [[nodiscard]] rowsize size() const { return _row_size; }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayNodePeriodicBits : public ParArray<NodePeriodicBits, _row_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<NodePeriodicBits, _row_size, _row_max, _align>;
        using t_base::t_base;

        NodePeriodicBitsRow operator[](index i)
        {
            DNDS_assert(i < this->Size()); //! disable past-end input
            return NodePeriodicBitsRow(t_base::operator[](i), t_base::RowSize(i));
        }

        NodePeriodicBits *rowPtr(index i) { return t_base::operator[](i); }
    };
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayNodePeriodicBitsPair = ArrayPair<ArrayNodePeriodicBits<_row_size, _row_max, _align>>;

} // namespace DNDS::Geom

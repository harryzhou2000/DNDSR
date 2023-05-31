#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    class AdjacencyRow // instead of std::vector<index> for building on raw buffer as a "mapping" object
    {
        index *__p_indices;
        rowsize __Row_size;

    public:
        AdjacencyRow(index *ptr, rowsize siz) : __p_indices(ptr), __Row_size(siz) {} // default actually

        index &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        operator std::vector<index>() const // copies to a new std::vector<index>
        {
            return std::vector<index>(__p_indices, __p_indices + __Row_size);
        }

        index *begin() { return __p_indices; }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacency : public ParArray<index, _row_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<index, _row_size, _row_max, _align>;
        using t_base::t_base;

        AdjacencyRow operator[](index i)
        {
            return AdjacencyRow(t_base::operator[](i), t_base::RowSize(i));
        }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayAdjacencyPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;
}
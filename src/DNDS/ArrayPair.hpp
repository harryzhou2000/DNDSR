#include "ArrayTransformer.hpp"
#include "ArrayDerived/ArrayAdjacency.hpp"

namespace DNDS
{
    template <class TArray = ParArray<real, 1>>
    struct ArrayPair
    {
        using t_self = ArrayPair<TArray>;
        std::shared_ptr<TArray> father;
        std::shared_ptr<TArray> son;
        using TTrans = typename ArrayTransformerType<TArray>::Type;
        TTrans trans;

        decltype(father->operator[](0)) operator[](index i) const
        {
            if (i >= 0 && i < father->Size())
                return father->operator[](i);
            else
                return son->operator[](i - father->Size());
        }

        decltype(father->operator()(0, 0)) operator()(index i, rowsize j)
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, j);
            else
                return son->operator()(i - father->Size(), j);
        }

        template <class... TOthers>
        auto operator()(index i, TOthers... aOthers)
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, aOthers...);
            else
                return son->operator()(i - father->Size(), aOthers...);
        }

        auto RowSize(index i)
        {
            if (i >= 0 && i < father->Size())
                return father->RowSize(i);
            else
                return son->RowSize(i - father->Size());
        }

        void ResizeRow(index i, rowsize rs)
        {
            if (i >= 0 && i < father->Size())
                father->ResizeRow(i, rs);
            else
                son->ResizeRow(i - father->Size(), rs);
        }

        template <class... TOthers>
        void ResizeRow(index i, TOthers... aOthers)
        {
            if (i >= 0 && i < father->Size())
                father->ResizeRow(i, aOthers...);
            else
                son->ResizeRow(i - father->Size(), aOthers...);
        }

        index Size()
        {
            return father->Size() + son->Size();
        }

        void TransAttach()
        {
            DNDS_assert(bool(father) && bool(son));
            trans.setFatherSon(father, son);
        }

        void CompressBoth()
        {
            father->Compress();
            son->Compress();
        }

        void CopyFather(t_self &R)
        {
            father->CopyData(*R.father);
        }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayAdjacencyPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;
}
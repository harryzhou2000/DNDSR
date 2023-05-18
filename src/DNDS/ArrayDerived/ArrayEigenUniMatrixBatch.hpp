#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <int a, int b>
    inline constexpr rowsize EigenSize_Mul_RowSize()
    {
        if constexpr (a >= 0 && b >= 0)
        {
            return a * b;
        }
        if constexpr (a == Eigen::Dynamic || b == Eigen::Dynamic)
        {
            return DynamicSize;
        }
        return DNDS_ROWSIZE_MIN;
    }

    template <int _n_row, int _n_col>
    class ArrayEigenUniMatrixBatch : public ParArray<real, NonUniformSize> // use CSR array
    {
        static_assert(_n_row >= 0 || _n_row == Eigen::Dynamic, "invalid _n_row");
        static_assert(_n_col >= 0 || _n_col == Eigen::Dynamic, "invalid _n_col");

    public:
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        using t_EigenMatrix = Eigen::Matrix<real, _n_row, _n_col>;
        using t_EigenMap = Eigen::Map<t_EigenMatrix>; // default no buffer align and stride

    private:
        int _row_dynamic = _n_row > 0 ? _n_row : 0;
        int _col_dynamic = _n_col > 0 ? _n_col : 0;
        int _m_size = this->Rows() * this->Cols();

    private:
        using t_base::Resize;
        using t_base::ResizeRow; // privatize basic resizing
        using t_base::operator();
        // void Resize(index, rowsize) = delete;
        // void ResizeRow(index iRow, rowsize nRowSize) = delete;
    public:
        /**
         * @brief resizes all matrices to be used;
         *
         * @param r
         * @param c
         */
        void ResizeMatrix(int r = -1, int c = -1)
        {
            if constexpr (_n_row >= 0)
                DNDS_assert(r == -1 || r == _n_row);
            if constexpr (_n_col >= 0)
                DNDS_assert(c == -1 || c == _n_col);
            if (r >= 0)
                _row_dynamic = r;
            if (c >= 0)
                _col_dynamic = c;
            // TODO: multiplication overflow detect
            _m_size = this->Rows() * this->Cols();
            this->t_base::Resize(0);
        }

        void Resize(index n_size, int r = -1, int c = -1)
        {
            this->ResizeMatrix(r, c);
            this->t_base::Resize(n_size);
        }

    public:
        int Rows() { return _n_row > 0 ? _n_row : _row_dynamic; }
        int Cols() { return _n_col > 0 ? _n_col : _col_dynamic; }
        int MSize()
        {
            if constexpr (_n_row >= 0 && _n_col >= 0)
                return _n_row * _n_col;
            else
                return _m_size;
        }

        void ResizeBatch(index i, rowsize b_size)
        {
            this->ResizeRow(i, b_size * MSize());
        }

        rowsize BatchSize(index i)
        {
            rowsize row_size_c = this->t_base::RowSize(i);
            DNDS_assert(row_size_c % MSize() == 0);
            return row_size_c / MSize();
        }

        t_EigenMap operator()(index i, rowsize j)
        {
            DNDS_assert(j >= 0 && j < this->BatchSize(i));
            if constexpr (_n_row >= 0 && _n_col >= 0)
                return t_EigenMap(this->t_base::operator[](i) + MSize() * j, Rows(), Cols());
        }

        // TODO: getting sub matrix ?
    };
}
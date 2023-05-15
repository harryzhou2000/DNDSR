#pragma once

#include <vector>
#include <iostream>

#include "Defines.hpp"

namespace DNDS
{
    static const rowsize NoAlign = -1;

    enum DataLayout
    {
        ErrorLayout,
        TABLE_StaticFixed,
        TABLE_Fixed,
        TABLE_Max,
        TABLE_StaticMax,
        CSR,
    };

    inline constexpr bool isTABLE(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_Fixed || lo == TABLE_Max || lo == TABLE_StaticMax;
    }
    inline constexpr bool isTABLE_Fixed(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_Fixed;
    }
    inline constexpr bool isTABLE_Max(DataLayout lo)
    {
        return lo == TABLE_Max || lo == TABLE_StaticMax;
    }
    inline constexpr bool isTABLE_Static(DataLayout lo)
    {
        return lo == TABLE_StaticFixed || lo == TABLE_StaticMax;
    }
    inline constexpr bool isTABLE_Dynamic(DataLayout lo)
    {
        return lo == TABLE_Fixed || lo == TABLE_Max;
    }

    /**
     * @brief 2D var-len data container template
     * @details
     * ## Array's types
     * |                         | _row_size>=0       | _row_size==DynamicSize | _row_size==NonUniformSize |
     * | ---                     |          ---       |                    --- |                       --- |
     * |_row_max>=0              |  TABLE_StaticFixed |  TABLE_Fixed           |   TABLE_StaticMax         |
     * |_row_max==DynamicSize    |  TABLE_StaticFixed |  TABLE_Fixed           |   TABLE_Max               |
     * |_row_max==NonUniformSize |  TABLE_StaticFixed |  TABLE_Fixed           |   CSR                     |
     *
     * @todo //TODO implement align feature
     *
     *
     * @tparam T
     * @tparam _row_size
     * @tparam _row_max
     * @tparam _align
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class Array
    {
    public:
        using value_type = T;
        using self_type = Array<T, _row_size, _row_max, _align>;
        static const rowsize al = _align;
        static const rowsize rs = _row_size;
        static const rowsize rm = _row_max;
        static const unsigned sizeof_T = sizeof(value_type);

        static_assert(sizeof_T <= (1024ull * 1024ull * 1024ull), "Row size larger than 1G");
        static_assert(std::is_trivially_copyable_v<value_type>, "Do not put in a non trivially copyable type");

        static_assert(al == NoAlign || al >= 1, "Align bad");

        static const rowsize s_T = al == NoAlign ? sizeof_T : (sizeof_T / al + 1) * al;
        static_assert(s_T >= sizeof_T && s_T - sizeof_T < (al == NoAlign ? 1 : al), "I1");

        static constexpr DataLayout _GetDataLayout()
        {
            if constexpr (rs != DynamicSize && rs != NonUniformSize && rs >= 0)
                return TABLE_StaticFixed;
            else if constexpr (rs == DynamicSize)
                return TABLE_Fixed;
            else if constexpr (rs == NonUniformSize)
            {
                if constexpr (rm == NonUniformSize)
                    return CSR;
                else if constexpr (rm == DynamicSize)
                    return TABLE_Max;
                else if constexpr (rm >= 0)
                    return TABLE_StaticMax;
                else
                    return ErrorLayout;
            }
            else
                return ErrorLayout;
        }
        static const DataLayout _dataLayout = _GetDataLayout();
        static_assert(_dataLayout != ErrorLayout, "Layout Error");

    public:
        //* compressed data
        using t_Data = std::vector<value_type>;
        //* uncompressed data (only for CSR)
        using t_DataUncompressed = std::vector<std::vector<value_type>>;

        //* non uniform data: CSR
        using t_RowStart = std::vector<index>;
        using t_pRowStart = std::shared_ptr<t_RowStart>;

        //* non uniform-with max data: TABLE
        using t_RowSizes = std::vector<rowsize>;
        using t_pRowSizes = std::shared_ptr<t_RowSizes>;

    private:
        t_pRowStart _pRowStart; // CSR   in number of T
        t_pRowSizes _pRowSizes; // TABLE in number of T
        t_Data _data;
        t_DataUncompressed _dataUncompressed;

        index _size = 0;             // in number of T
        index _row_size_dynamic = 0; // in number of T
    public:
        t_pRowStart getRowStart() { return _pRowStart; }
        t_pRowSizes getRowSizes() { return _pRowSizes; }

    public:
        // default constructor using default:
        Array() = default;
        // copy constructor using default;
        Array(const self_type &R) = default;

        // TODO: constructors
        // TODO: A indexer-copying build method:
        // TODO: for CSR: c->c, u->u
        // TODO: for intertype: CSR->Max, Max->CSR ...

        // read isze
        index Size() const { return _size; }

        rowsize RowSize() const // iRow is actually dummy here
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        rowsize RowSize(index iRow) const
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                return rs;
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax) // TABLE with Max
            {
                DNDS_assert_info(_pRowSizes, "_pRowSizes invalid"); // TABLE with Max must have a RowSizes
                return _pRowSizes->at(iRow);
            }
            else if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                {
                    index pDiff = _pRowStart->at(iRow + 1) - _pRowStart->at(iRow);
                    DNDS_assert(pDiff < INT32_MAX); // overflow
                    return static_cast<rowsize>(pDiff);
                }
                else
                {
                    auto rs_cur = _dataUncompressed.at(iRow).size();
                    return static_cast<rowsize>(rs_cur);
                }
            }
        }

        rowsize RowSizeMax() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return _dataLayout == TABLE_Max ? _row_size_dynamic : rm;
            else
                DNDS_assert_info(false, "invalid call");
        }

        rowsize RowSizeField() const
        {
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                return this->RowSizeMax();
            else if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_StaticFixed)
                return this->RowSize();
            else
                DNDS_assert_info(false, "invalid call");
        }

        rowsize RowSizeField(index iRow)
        {
            if constexpr (_dataLayout == CSR)
                return this->RowSize(iRow);
            else
                DNDS_assert_info(false, "invalid call");
        }

        bool
        IfCompressed() const
        {

            if constexpr (_dataLayout == CSR)
            {
                if (_size > 0)
                {
                    return bool(_pRowStart);
                }
                return false; // size 0 array is defined as un-compressed
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
                return false;
            }
        }

        void
        CSRDecompress()
        {
            if (!IfCompressed())
                return;
            _dataUncompressed.resize(_size);
            for (index i = 0; i < _size; i++)
            {
                // _dataUncompressed[i].resize( - _pRowStart->at(i));
                auto iterStart = _data.begin() + _pRowStart->at(i);
                auto iterEnd = _data.begin() + _pRowStart->at(i + 1);
                _dataUncompressed[i].assign(iterStart, iterEnd);
            }
            _data.clear();
            _pRowStart.reset();
        }

        void
        CSRCompress()
        {
            if (IfCompressed())
                return;
            _pRowStart = std::make_shared<
                typename decltype(_pRowStart)::element_type>(_size + 1, 0);
            _pRowStart->at(0) = 0;
            for (index i = 0; i < _size; i++)
            {
                index rsI = _pRowStart->at(i);
                index rsIP = rsI + static_cast<index>(_dataUncompressed.at(i).size());
                DNDS_assert(rsIP >= rsI);
                _pRowStart->at(i + 1) = rsIP;
            }
            _data.resize(_pRowStart->at(_size));
            for (index i = 0; i < _size; i++)
            {
                // // _dataUncompressed[i].resize( - _pRowStart->at(i));
                // auto iterStart = _data.begin() + _pRowStart->at(i);
                // auto iterEnd = _data.begin() + _pRowStart->at(i + 1);
                // _dataUncompressed[i].assign(iterStart, iterEnd);
                // _data.push_back(data)
                memcpy(_data.data() + _pRowStart->at(i), _dataUncompressed[i].data(),
                       sizeof(T) * _dataUncompressed[i].size());
                DNDS_assert(_pRowStart->at(i) + _dataUncompressed[i].size() <= _data.size());
                //! any better way?
            }
            _dataUncompressed.clear();
        }

        void Compress()
        {
            if constexpr (_dataLayout == CSR)
                CSRCompress();
        }
        void Decompress()
        {
            if constexpr (_dataLayout == CSR)
                CSRDecompress();
        }

        /**
         * @brief resize invalidates all data and aux, and resets the sizes info to 0 for max
         *
         * @param nSize
         * @param nRow_size_dynamic
         * @return std::enable_if_t<
         * _dataLayout == TABLE_Fixed ||
         * _dataLayout == TABLE_StaticFixed ||
         * _dataLayout == TABLE_Max ||
         * _dataLayout == TABLE_StaticMax,
         * void>
         */
        void
        Resize(index nSize, rowsize nRow_size_dynamic)
        {
            if constexpr (_dataLayout == CSR) // to un compressed
            {
                DNDS_assert_info(!IfCompressed(), "Need to decompress before auto resizing");
                _size = nSize;
                _dataUncompressed.resize(nSize, typename decltype(_dataUncompressed)::value_type(nRow_size_dynamic));
            }
            else
            {
                _size = nSize;
                if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_Max)
                    _data.resize(nSize * nRow_size_dynamic), _row_size_dynamic = nRow_size_dynamic;
                else if constexpr (_dataLayout == TABLE_StaticFixed)
                    _data.resize(nSize * rs);
                else if constexpr (_dataLayout == TABLE_StaticMax)
                    _data.resize(nSize * rm);

                if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                {
                    if (_pRowSizes.use_count() == 1)
                        _pRowSizes->resize(nSize, 0);
                    else
                        _pRowSizes = std::make_shared<
                            typename decltype(_pRowSizes)::element_type>(nSize, 0);
                }
            }
        }

        void
        Resize(index nSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_assert_info(!IfCompressed(), "Need to decompress before auto resizing");
                _size = nSize;
                _dataUncompressed.resize(nSize);
            }
            else if constexpr (_dataLayout == TABLE_StaticFixed)
            {
                _size = nSize;
                _data.resize(nSize * rs);
            }
            else if constexpr (_dataLayout == TABLE_StaticMax)
            {
                _size = nSize;
                _data.resize(nSize * rm);
                if (_pRowSizes.use_count() == 1)
                    _pRowSizes->resize(nSize, 0);
                else
                    _pRowSizes = std::make_shared<
                        typename decltype(_pRowSizes)::element_type>(nSize, 0);
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        template <class TFRowSize>
        void
        Resize(index nSize, TFRowSize &&FRowSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                _size = nSize;
                _pRowSizes.reset(), _dataUncompressed.clear(); //*directly to compressed
                _pRowStart = std::make_shared<typename decltype(_pRowStart)::element_type>(nSize + 1);
                _pRowStart->operator[](0) = 0;
                for (index i = 0; i < nSize; i++)
                    (*_pRowStart)[i + 1] = (*_pRowStart)[i] + FRowSize(i);
                _data.resize(_pRowStart->at(nSize));
            }
            static_assert(_dataLayout == CSR, "Only Non Uniform, CSR for now");
            static_assert(std::is_invocable_r_v<rowsize, TFRowSize, index>, "Call invalid");
        }

        /**
         * @brief resize one row
         * valid only for non-uniform
         *
         * @param iRow
         * @param nRowSize
         */
        void
        ResizeRow(index iRow, rowsize nRowSize)
        {
            if constexpr (_dataLayout == CSR)
            {
                DNDS_assert_info(!IfCompressed(), "Need to decompress before auto resizing row");
                DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
                _dataUncompressed.at(iRow).resize(nRowSize);
            }
            else if constexpr (_dataLayout == TABLE_Max ||
                               _dataLayout == TABLE_StaticMax)
            {
                DNDS_assert(nRowSize <= _dataLayout == TABLE_Max ? _row_size_dynamic : rm); //_row_size_dynamic is max now
                DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
                DNDS_assert_info(_pRowSizes, "_pRowSizes invalid");
                if (_pRowSizes.use_count() > 1) // shared
                    _pRowSizes = std::make_shared<
                        typename decltype(_pRowSizes)::element_type>(*_pRowSizes); // copy the row sizes
                _pRowSizes->at(iRow) = nRowSize;                                   // change
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        // TODO: Data reference query method and pointer query method
        // TODO: ? same-size compress for non-uniforms

        const T &at(index iRow, rowsize iCol) const
        {
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position i out of range");
            DNDS_assert_info(iCol < RowSize(iRow) && iCol >= 0, "query position i out of range");
            if constexpr (_dataLayout == TABLE_StaticFixed)
                return _data.at(iRow * rs + iCol);
            else if constexpr (_dataLayout == TABLE_StaticMax)
                return _data.at(iRow * rm + iCol);
            else if constexpr (_dataLayout == TABLE_Fixed)
                return _data.at(iRow * _row_size_dynamic + iCol);
            else if constexpr (_dataLayout == TABLE_Max)
                return _data.at(iRow * _row_size_dynamic + iCol);
            else if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                    return _data.at(_pRowStart->at(iRow) + iCol);
                else
                    return _dataUncompressed.at(iRow).at(iCol);
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        T &operator()(index iRow, rowsize iCol)
        {
            return const_cast<T &>(at(iRow, iCol));
        }

        const T &operator()(index iRow, rowsize iCol) const
        {
            return at(iRow, iCol);
        }

        T *operator[](index iRow)
        {
            DNDS_assert_info(iRow <= _size && iRow >= 0, "query position i out of range");
            if constexpr (_dataLayout == TABLE_StaticFixed)
                return _data.data() + iRow * rs;
            else if constexpr (_dataLayout == TABLE_StaticMax)
                return _data.data() + iRow * rm;
            else if constexpr (_dataLayout == TABLE_Fixed)
                return _data.data() + iRow * _row_size_dynamic;
            else if constexpr (_dataLayout == TABLE_Max)
                return _data.data() + iRow * _row_size_dynamic;
            else if constexpr (_dataLayout == CSR)
            {
                if (IfCompressed())
                    return _data.data() + _pRowStart->at(iRow);
                else
                {
                    DNDS_assert_info(iRow < _size, "past-the-end query forbidden for CSR uncompressed");
                    return _dataUncompressed.at(iRow).data();
                }
            }
            else
            {
                DNDS_assert_info(false, "invalid call");
            }
        }

        T *data()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_assert_info(this->IfCompressed(), "CSR must be compressed to get data pointer");
            return _data.data();
        }

        size_t DataSize()
        {
            if constexpr (_dataLayout == CSR)
                DNDS_assert_info(this->IfCompressed(), "CSR must be compressed to get data pointer");
            return _data.size();
        }

        friend std::ostream &operator<<(std::ostream &o, const Array<T, _row_size, _row_max, _align> &A)
        {
            for (index i = 0; i < A._size; i++)
            {
                for (index j = 0; j < A.RowSize(i); j++)
                    o << A(i, j) << "\t";
                o << std::endl;
            }
            return o;
        }
    };

}
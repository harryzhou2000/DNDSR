#pragma once

#include <vector>

#include "Defines.hpp"

namespace DNDS
{
    static const rowsize NoAlign = -1;
    /**
     * @brief Basic Serial 2-D Array class
     *
     * @tparam T
     * @tparam _row_size
     * @tparam _row_max
     */
    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class Array
    {
    public:
        using value_type = T;
        using self_type = Array<T, _row_size, _row_max, align>;
        static const rowsize al = _align;
        static const rowsize rs = _row_size;
        static const rowsize rm = _row_max;
        static const unsigned sizeof_T = sizeof(value_type);

        static_assert(sizeof_T <= (1024ull * 1024ull * 1024ull), "Row size larger than 1G");
        static_assert(std::is_trivially_copyable_v<value_type>, "Do not put in a non trivially copyable type");

        static_assert(al == NoAlign || al >= 1, "Align bad");

        static rowsize s_T = al == NoAlign ? sizeof_T : (sizeof_T / al + 1) * al;
        static_assert(s_T >= sizeof_T && s_T - sizeof_T < (al == NoAlign ? 1 : al), "I1");

        enum DataLayout
        {
            ErrorLayout,
            TABLE_StaticFixed,
            TABLE_Fixed,
            TABLE_Max,
            TABLE_StaticMax,
            CSR,
        };

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

        std::enable_if_t<
            _dataLayout == TABLE_Fixed || _dataLayout == TABLE_StaticFixed,
            rowsize>
        RowSize(index iRow = 0) const // iRow is actually dummy here
        {
            if constexpr (_dataLayout == TABLE_Fixed)
                return _row_size_dynamic;
            else
                return rs;
        }

        std::enable_if_t<_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax || _dataLayout == CSR, rowsize>
        RowSize(index iRow) const
        {
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax) // TABLE with Max
            {
                DNDS_assert_info(_pRowSizes, "_pRowSizes invalid"); // TABLE with Max must have a RowSizes
                return _pRowSizes->at(iRow);
            }
            else // CSR
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
                    return static_cast<rowsize>();
                }
            }
        }

        std::enable_if_t<_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax, rowsize>
        RowSizeMax() const
        {
            return _dataLayout == TABLE_Max ? _row_size_dynamic : rm;
        }

        std::enable_if_t<_dataLayout == CSR, bool>
        IfCompressed() const
        {
            if (_size > 0)
            {
                return bool(_pRowStart);
            }
            return false; // size 0 array is defined as un-compressed
        }

        std::enable_if_t<
            _dataLayout == CSR,
            void>
        CSRDecompress()
        {
            // TODO
        }

        std::enable_if_t<
            _dataLayout == CSR,
            void>
        CSRCompress()
        {
            // TODO
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
        std::enable_if_t<
            _dataLayout == TABLE_Fixed ||
                _dataLayout == TABLE_StaticFixed ||
                _dataLayout == TABLE_Max ||
                _dataLayout == TABLE_StaticMax,
            void>
        Resize(index nSize, rowsize nRow_size_dynamic)
        {
            _size = nSize;
            if constexpr (_dataLayout == TABLE_Fixed || _dataLayout == TABLE_Max)
                _data.resize(nSize * nRow_size_dynamic), _row_size_dynamic = nRow_size_dynamic;
            else if constexpr (_dataLayout == TABLE_StaticFixed)
                _data.resize(nSize * rs);
            else if constexpr (_dataLayout == TABLE_StaticMax)
                _data.resize(nSize * rm);
            else
                DNDS_assert(false);

            if constexpr (_dataLayout == TABLE_Max || _dataLayout == TABLE_StaticMax)
                _pRowSizes = std::make_shared<decltype(_pRowSizes)::element_type>(nSize, 0);
        }

        std::enable_if_t<_dataLayout == CSR, void>
        Resize(index nSize, rowsize nRow_size_Reserve = 0)
        {
            DNDS_assert_info(!IfCompressed(), "Need to decompress before auto resizing");
            _size = nSize;
            _dataUncompressed.resize(nSize, decltype(_dataUncompressed)::value_type{nSize});
        }

        // TODO: A rowsizes known resize function for CSR

        /**
         * @brief resizes only one row, does not disturb other rows
         *
         * @param iRow
         * @param nRowSize
         * @return std::enable_if_t<
         * _dataLayout == TABLE_Max ||
         * _dataLayout == TABLE_StaticMax,
         * void>
         */
        std::enable_if_t<
            _dataLayout == TABLE_Max ||
                _dataLayout == TABLE_StaticMax,
            void>
        ResizeRow(index iRow, rowsize nRowSize)
        {
            DNDS_assert(nRowSize <= _dataLayout == TABLE_Max ? _row_size_dynamic : rm); //_row_size_dynamic is max now
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
            DNDS_assert_info(_pRowSizes, "_pRowSizes invalid");
            if (_pRowSizes.use_count() > 1)                                                     // shared
                _pRowSizes = std::make_shared<decltype(_pRowSizes)::element_type>(*_pRowSizes); // copy the row sizes
            _pRowSizes->at(iRow) = nRowSize;                                                    // change
        }

        std::enable_if_t<_dataLayout == CSR, void>
        ResizeRow(index iRow, rowsize nRowSize)
        {
            DNDS_assert_info(!IfCompressed(), "Need to decompress before auto resizing row");
            DNDS_assert_info(iRow < _size && iRow >= 0, "query position out of range");
            _dataUncompressed.at(iRow).resize(nRowSize);
        }

        // TODO: Data reference query method and pointer query method
        // TODO: ? same-size compress for non-uniforms
    };

}
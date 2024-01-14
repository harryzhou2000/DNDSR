#pragma once

#include <type_traits>
#include <cstdint>
#include "Eigen/Core"
#include <array>
#include "DNDS/HardEigen.hpp"

#include "DNDS/Defines.hpp"
#include "Geometric.hpp"

namespace DNDS::Geom::Elem
{
    /**
     * Complying to [CGNS Element standard](https://cgns.github.io/CGNS_docs_current/sids/conv.html)
     *  !note that we use 0 based indexing (CGNS uses 1 based in the link)
     *
     */

    static const int CellNumNodeMax = 27; //!

    enum ElemType
    {
        UnknownElem = 0,
        Line2 = 1,
        Line3 = 8,

        Tri3 = 2,
        Tri6 = 9,
        Quad4 = 3,
        Quad9 = 10,

        Tet4 = 4,
        Tet10 = 11,
        Hex8 = 5,
        Hex27 = 12,
        Prism6 = 6,
        Prism18 = 13,
        Pyramid5 = 7,
        Pyramid14 = 14,

        ElemType_NUM = 15
    };

    enum ParamSpace
    {
        UnknownPSpace = 0,
        LineSpace = 1,

        TriSpace = 2,
        QuadSpace = 3,

        TetSpace = 4,
        HexSpace = 5,
        PrismSpace = 6,
        PyramidSpace = 7,

        ParamSpace_NUM = 8
    };

    inline constexpr std::array<std::array<t_index, 5>, ElemType_NUM>
    __Get_Dim_Order_NVNNNF()
    {
        std::array<std::array<t_index, 5>, ElemType_NUM> ret{
            std::array<t_index, 5>{invalid_index}};
        for (auto &i : ret)
            for (auto &j : i)
                j = invalid_index;

        using TR = std::array<t_index, 5>;

        //                                  d  o  nv nn nf
        ret[Line2] = TR{1, 1, 2, 2, 0};
        ret[Line3] = TR{1, 2, 2, 3, 0};

        ret[Tri3] = TR{2, 1, 3, 3, 3};
        ret[Tri6] = TR{2, 2, 3, 6, 3};
        ret[Quad4] = TR{2, 1, 4, 4, 4};
        ret[Quad9] = TR{2, 2, 4, 9, 4};

        ret[Tet4] = TR{3, 1, 4, 4, 4};
        ret[Tet10] = TR{3, 2, 4, 10, 4};
        ret[Hex8] = TR{3, 1, 8, 8, 6};
        ret[Hex27] = TR{3, 2, 8, 27, 6};
        ret[Prism6] = TR{3, 1, 6, 6, 5};
        ret[Prism18] = TR{3, 2, 6, 18, 5};
        ret[Pyramid5] = TR{3, 1, 5, 5, 5};
        ret[Pyramid14] = TR{3, 2, 5, 14, 5};
        return ret;
    }

    const auto Dim_Order_NVNNNF = __Get_Dim_Order_NVNNNF();

    inline constexpr ParamSpace ElemType_to_ParamSpace(const ElemType t)
    {
        if (t == Line2 || t == Line3)
            return LineSpace;
        if (t == Tri3 || t == Tri6)
            return TriSpace;
        if (t == Quad4 || t == Quad9)
            return QuadSpace;
        if (t == Tet4 || t == Tet10)
            return TetSpace;
        if (t == Hex8 || t == Hex27)
            return HexSpace;
        if (t == Prism6 || t == Prism18)
            return PrismSpace;
        if (t == Pyramid5 || t == Pyramid14)
            return PyramidSpace;
        return UnknownPSpace;
    }

    inline constexpr ElemType GetFaceType(ElemType t_v, t_index iFace)
    {
        if (t_v == Tri3)
            return Line2;
        if (t_v == Tri6)
            return Line3;
        if (t_v == Quad4)
            return Line2;
        if (t_v == Quad9)
            return Line3;
        if (t_v == Tet4)
            return Tri3;
        if (t_v == Tet10)
            return Tri6;
        if (t_v == Hex8)
            return Quad4;
        if (t_v == Hex27)
            return Quad9;
        if (t_v == Prism6)
        {
            if (iFace < 3)
                return Quad4;
            return Tri3;
        }
        if (t_v == Prism18)
        {
            if (iFace < 3)
                return Quad9;
            return Tri6;
        }
        if (t_v == Pyramid5)
        {
            if (iFace < 1)
                return Quad4;
            return Tri3;
        }
        if (t_v == Pyramid14)
        {
            if (iFace < 1)
                return Quad9;
            return Tri6;
        }
        return UnknownElem;
    }

    inline constexpr std::array<std::array<std::array<t_index, 10>, 6>, ElemType_NUM>
    __Get_FaceNodeList()
    {
        auto ret = std::array<std::array<std::array<t_index, 10>, 6>, ElemType_NUM>{};
        for (auto &i : ret)
            for (auto &j : i)
                for (auto &k : j)
                    k = invalid_index;
        using TF = std::array<t_index, 10>;
        ret[Tri3][0] = TF{0, 1};
        ret[Tri3][1] = TF{1, 2};
        ret[Tri3][2] = TF{2, 0};

        ret[Tri6][0] = TF{0, 1, 3};
        ret[Tri6][1] = TF{1, 2, 4};
        ret[Tri6][2] = TF{2, 0, 5};

        ret[Quad4][0] = TF{0, 1};
        ret[Quad4][1] = TF{1, 2};
        ret[Quad4][2] = TF{2, 3};
        ret[Quad4][3] = TF{3, 0};

        ret[Quad9][0] = TF{0, 1, 4};
        ret[Quad9][1] = TF{1, 2, 5};
        ret[Quad9][2] = TF{2, 3, 6};
        ret[Quad9][3] = TF{3, 0, 7};

        auto IndexToZeroBase = [&](ElemType t)
        {
            for (auto &j : ret[t])
                for (auto &k : j)
                    k = (k == invalid_index || k == 0)
                            ? invalid_index
                            : k - 1;
        };

        ret[Tet4][0] = TF{1, 3, 2};
        ret[Tet4][1] = TF{1, 2, 4};
        ret[Tet4][2] = TF{2, 3, 4};
        ret[Tet4][3] = TF{3, 1, 4};
        IndexToZeroBase(Tet4);

        ret[Tet10][0] = TF{1, 3, 2, 7, 6, 5};
        ret[Tet10][1] = TF{1, 2, 4, 5, 9, 8};
        ret[Tet10][2] = TF{2, 3, 4, 6, 10, 9};
        ret[Tet10][3] = TF{3, 1, 4, 7, 8, 10};
        IndexToZeroBase(Tet10);

        ret[Hex8][0] = TF{1, 4, 3, 2};
        ret[Hex8][1] = TF{1, 2, 6, 5};
        ret[Hex8][2] = TF{2, 3, 7, 6};
        ret[Hex8][3] = TF{3, 4, 8, 7};
        ret[Hex8][4] = TF{1, 5, 8, 4};
        ret[Hex8][5] = TF{5, 6, 7, 8};
        IndexToZeroBase(Hex8);

        ret[Hex27][0] = TF{1, 4, 3, 2, 12, 11, 10, 9, 21};
        ret[Hex27][1] = TF{1, 2, 6, 5, 9, 14, 17, 13, 22};
        ret[Hex27][2] = TF{2, 3, 7, 6, 10, 15, 18, 14, 23};
        ret[Hex27][3] = TF{3, 4, 8, 7, 11, 16, 19, 15, 24};
        ret[Hex27][4] = TF{1, 5, 8, 4, 13, 20, 16, 12, 25};
        ret[Hex27][5] = TF{5, 6, 7, 8, 17, 18, 19, 20, 26};
        IndexToZeroBase(Hex27);

        ret[Prism6][0] = TF{1, 2, 5, 4};
        ret[Prism6][1] = TF{2, 3, 6, 5};
        ret[Prism6][2] = TF{3, 1, 4, 6};
        ret[Prism6][3] = TF{1, 3, 2};
        ret[Prism6][4] = TF{4, 5, 6};
        IndexToZeroBase(Prism6);

        ret[Prism18][0] = TF{1, 2, 5, 4, 7, 11, 13, 10, 16};
        ret[Prism18][1] = TF{2, 3, 6, 5, 8, 12, 14, 11, 17};
        ret[Prism18][2] = TF{3, 1, 4, 6, 9, 10, 15, 12, 18};
        ret[Prism18][3] = TF{1, 3, 2, 9, 8, 7};
        ret[Prism18][4] = TF{4, 5, 6, 13, 14, 15};
        IndexToZeroBase(Prism18);

        ret[Pyramid5][0] = TF{1, 4, 3, 2};
        ret[Pyramid5][1] = TF{1, 2, 5};
        ret[Pyramid5][2] = TF{2, 3, 5};
        ret[Pyramid5][3] = TF{3, 4, 5};
        ret[Pyramid5][4] = TF{4, 1, 5};
        IndexToZeroBase(Pyramid5);

        ret[Pyramid14][0] = TF{1, 4, 3, 2, 9, 8, 7, 6, 14};
        ret[Pyramid14][1] = TF{1, 2, 5, 6, 11, 10};
        ret[Pyramid14][2] = TF{2, 3, 5, 7, 12, 11};
        ret[Pyramid14][3] = TF{3, 4, 5, 8, 13, 12};
        ret[Pyramid14][4] = TF{4, 1, 5, 9, 10, 13};
        IndexToZeroBase(Pyramid14);

        return ret;
    }
    const auto FaceNodeList = __Get_FaceNodeList();

    inline constexpr auto
    __Get_ElemElevationSpan_O1O2List()
    {
        auto ret = std::array<std::array<std::array<t_index, 28>, 28>, ElemType_NUM>{};
        for (auto &i : ret)
            for (auto &j : i)
                for (auto &k : j)
                    k = invalid_index;
        using TF = std::array<t_index, 28>;
        ret[Line2][0] = TF{0, 1};

        ret[Tri3][0] = TF{0, 1};
        ret[Tri3][1] = TF{1, 2};
        ret[Tri3][2] = TF{2, 0};

        ret[Quad4][0] = TF{0, 1};
        ret[Quad4][1] = TF{1, 2};
        ret[Quad4][2] = TF{2, 3};
        ret[Quad4][3] = TF{3, 0};
        ret[Quad4][4] = TF{0, 1, 2, 3};

        auto IndexToZeroBase = [&](ElemType t)
        {
            for (auto &j : ret[t])
                for (auto &k : j)
                    k = (k == invalid_index || k == 0)
                            ? invalid_index
                            : k - 1;
        };

        ret[Tet4][0] = TF{1, 2};
        ret[Tet4][1] = TF{2, 3};
        ret[Tet4][2] = TF{3, 1};
        ret[Tet4][3] = TF{1, 4};
        ret[Tet4][4] = TF{2, 4};
        ret[Tet4][5] = TF{3, 4};
        IndexToZeroBase(Tet4);

        ret[Hex8][0] = TF{1, 2};
        ret[Hex8][1] = TF{2, 3};
        ret[Hex8][2] = TF{3, 4};
        ret[Hex8][3] = TF{4, 1};
        ret[Hex8][4] = TF{1, 5};
        ret[Hex8][5] = TF{2, 6};
        ret[Hex8][6] = TF{3, 7};
        ret[Hex8][7] = TF{4, 8};
        ret[Hex8][8] = TF{5, 6};
        ret[Hex8][9] = TF{6, 7};
        ret[Hex8][10] = TF{7, 8};
        ret[Hex8][11] = TF{8, 5};
        ret[Hex8][12] = TF{1, 4, 3, 2};
        ret[Hex8][13] = TF{1, 2, 6, 5};
        ret[Hex8][14] = TF{2, 3, 7, 6};
        ret[Hex8][15] = TF{3, 4, 8, 7};
        ret[Hex8][16] = TF{1, 5, 8, 4};
        ret[Hex8][17] = TF{5, 6, 7, 8};
        ret[Hex8][18] = TF{1, 2, 3, 4, 5, 6, 7, 8};
        IndexToZeroBase(Hex8);

        ret[Prism6][0] = TF{1, 2};
        ret[Prism6][1] = TF{2, 3};
        ret[Prism6][2] = TF{3, 1};
        ret[Prism6][3] = TF{1, 4};
        ret[Prism6][4] = TF{2, 5};
        ret[Prism6][5] = TF{3, 6};
        ret[Prism6][6] = TF{4, 5};
        ret[Prism6][7] = TF{5, 6};
        ret[Prism6][8] = TF{6, 4};
        ret[Prism6][9] = TF{1, 2, 5, 4};
        ret[Prism6][10] = TF{2, 3, 6, 5};
        ret[Prism6][11] = TF{3, 1, 4, 6};
        IndexToZeroBase(Prism6);

        ret[Pyramid5][0] = TF{1, 2};
        ret[Pyramid5][1] = TF{2, 3};
        ret[Pyramid5][2] = TF{3, 4};
        ret[Pyramid5][3] = TF{4, 1};
        ret[Pyramid5][4] = TF{1, 5};
        ret[Pyramid5][5] = TF{2, 5};
        ret[Pyramid5][6] = TF{3, 5};
        ret[Pyramid5][7] = TF{4, 5};
        ret[Pyramid5][8] = TF{1, 4, 3, 2};
        IndexToZeroBase(Pyramid5);

        return ret;
    }
    const auto ElemElevationSpan_O1O2List = __Get_ElemElevationSpan_O1O2List();

    constexpr t_real ParamSpaceVol(ParamSpace ps)
    {
        if (ps == LineSpace)
            return 2;
        if (ps == TriSpace)
            return 0.5;
        if (ps == QuadSpace)
            return 2 * 2;
        if (ps == TetSpace)
            return 1. / 6;
        if (ps == HexSpace)
            return 2 * 2 * 2;
        if (ps == PrismSpace)
            return 0.5 * 2;
        if (ps == PyramidSpace)
            return 4. / 3;
        return 0;
    }

    constexpr inline int GetElemElevation_O1O2_NumNode(ElemType t)
    {
        switch (t)
        {
        case Line2:
            return 1;
        case Tri3:
            return 3;
        case Quad4:
            return 5;
        case Tet4:
            return 6;
        case Hex8:
            return 19;
        case Prism6:
            return 12;
        case Pyramid5:
            return 9;
        default:
            break;
        }
        return -1;
    }

    constexpr inline ElemType GetElemElevation_O1O2_NodeSpanType(ElemType t, t_index ine)
    {
        switch (t)
        {
        case Line2:
            return Line2;
        case Tri3:
            return Line2;
        case Quad4:
            if (ine < 4)
                return Line2;
            else
                return Quad4;
        case Tet4:
            return Line2;
        case Hex8:
            if (ine < 12)
                return Line2;
            else if (ine < 18)
                return Quad4;
            else
                return Hex8;
        case Prism6:
            if (ine < 9)
                return Line2;
            else
                return Quad4;
        case Pyramid5:
            if (ine < 8)
                return Line2;
            else
                return Quad4;
        default:
            break;
        }
        return UnknownElem;
    }

    constexpr inline ElemType GetElemElevation_O1O2_ElevatedType(ElemType t)
    {
        switch (t)
        {
        case Line2:
            return Line3;
        case Tri3:
            return Tri6;
        case Quad4:
            return Quad9;
        case Tet4:
            return Tet10;
        case Hex8:
            return Hex27;
        case Prism6:
            return Prism18;
        case Pyramid5:
            return Pyramid14;
        default:
            break;
        }
        return UnknownElem;
    }

    /**
     * @brief calculates shape func matrix,
     * where Di is d_xi, d_et, d_zt when diffOrder == 1
     *
     * @warning
     * ! assumes v to be in good shape and 0 - init
     *
     * @param t type
     * @param p position in param space
     * @param v result value
     */
    template <int diffOrder, class TPoint, class TArray>
    void ShapeFunc_DiNj(ElemType t, const TPoint &p, TArray &v)
    {
        static_assert(diffOrder == 0 || diffOrder == 1);
        t_real xi = p[0];
        t_real et = p[1];
        t_real zt = p[2];

        if (t == Line2)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = xi * (-1.0 / 2.0) + 1.0 / 2.0;
                v(0, 1) = xi / 2.0 + 1.0 / 2.0;
            }
            else
            {
                v(0, 0) = -1.0 / 2.0;
                v(0, 1) = 1.0 / 2.0;
            }
        }

        if (t == Line3)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (xi * (xi - 1.0)) / 2.0;
                v(0, 1) = (xi * (xi + 1.0)) / 2.0;
                v(0, 2) = -xi * xi + 1.0;
            }
            else
            {
                v(0, 0) = xi - 1.0 / 2.0;
                v(0, 1) = xi + 1.0 / 2.0;
                v(0, 2) = xi * -2.0;
            }
        }

        if (t == Tri3)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = -et - xi + 1.0;
                v(0, 1) = xi;
                v(0, 2) = et;
            }
            else
            {
                v(0, 0) = -1.0;
                v(0, 1) = 1.0;
                v(1, 0) = -1.0;
                v(1, 2) = 1.0;
            }
        }

        if (t == Tri6)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (et + xi - 1.0) * (et + xi - 1.0 / 2.0) * 2.0;
                v(0, 1) = xi * (xi - 1.0 / 2.0) * 2.0;
                v(0, 2) = et * (et - 1.0 / 2.0) * 2.0;
                v(0, 3) = xi * (et * 2.0 + xi * 2.0 - 2.0) * -2.0;
                v(0, 4) = et * xi * 4.0;
                v(0, 5) = et * (et + xi - 1.0) * -4.0;
            }
            else
            {
                v(0, 0) = et * 4.0 + xi * 4.0 - 3.0;
                v(0, 1) = xi * 4.0 - 1.0;
                v(0, 3) = et * -4.0 - xi * 8.0 + 4.0;
                v(0, 4) = et * 4.0;
                v(0, 5) = et * -4.0;
                v(1, 0) = et * 4.0 + xi * 4.0 - 3.0;
                v(1, 2) = et * 4.0 - 1.0;
                v(1, 3) = xi * -4.0;
                v(1, 4) = xi * 4.0;
                v(1, 5) = et * -8.0 - xi * 4.0 + 4.0;
            }
        }

        if (t == Quad4)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0);
                v(0, 1) = -(et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0);
                v(0, 2) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0);
                v(0, 3) = -(et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0);
            }
            else
            {
                v(0, 0) = et / 4.0 - 1.0 / 4.0;
                v(0, 1) = et * (-1.0 / 4.0) + 1.0 / 4.0;
                v(0, 2) = et / 4.0 + 1.0 / 4.0;
                v(0, 3) = et * (-1.0 / 4.0) - 1.0 / 4.0;
                v(1, 0) = xi / 4.0 - 1.0 / 4.0;
                v(1, 1) = xi * (-1.0 / 4.0) - 1.0 / 4.0;
                v(1, 2) = xi / 4.0 + 1.0 / 4.0;
                v(1, 3) = xi * (-1.0 / 4.0) + 1.0 / 4.0;
            }
        }

        if (t == Quad9)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (et * xi * (et - 1.0) * (xi - 1.0)) / 4.0;
                v(0, 1) = (et * xi * (et - 1.0) * (xi + 1.0)) / 4.0;
                v(0, 2) = (et * xi * (et + 1.0) * (xi + 1.0)) / 4.0;
                v(0, 3) = (et * xi * (et + 1.0) * (xi - 1.0)) / 4.0;
                v(0, 4) = et * (et - 1.0) * (xi - 1.0) * (xi + 1.0) * (-1.0 / 2.0);
                v(0, 5) = xi * (et - 1.0) * (et + 1.0) * (xi + 1.0) * (-1.0 / 2.0);
                v(0, 6) = et * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (-1.0 / 2.0);
                v(0, 7) = xi * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (-1.0 / 2.0);
                v(0, 8) = (et - 1.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0);
            }
            else
            {
                v(0, 0) = (et * (xi * 2.0 - 1.0) * (et - 1.0)) / 4.0;
                v(0, 1) = (et * (xi * 2.0 + 1.0) * (et - 1.0)) / 4.0;
                v(0, 2) = (et * (xi * 2.0 + 1.0) * (et + 1.0)) / 4.0;
                v(0, 3) = (et * (xi * 2.0 - 1.0) * (et + 1.0)) / 4.0;
                v(0, 4) = -et * xi * (et - 1.0);
                v(0, 5) = (et * et - 1.0) * (xi * 2.0 + 1.0) * (-1.0 / 2.0);
                v(0, 6) = -et * xi * (et + 1.0);
                v(0, 7) = (et * et - 1.0) * (xi * 2.0 - 1.0) * (-1.0 / 2.0);
                v(0, 8) = xi * (et * et - 1.0) * 2.0;
                v(1, 0) = (xi * (et * 2.0 - 1.0) * (xi - 1.0)) / 4.0;
                v(1, 1) = (xi * (et * 2.0 - 1.0) * (xi + 1.0)) / 4.0;
                v(1, 2) = (xi * (et * 2.0 + 1.0) * (xi + 1.0)) / 4.0;
                v(1, 3) = (xi * (et * 2.0 + 1.0) * (xi - 1.0)) / 4.0;
                v(1, 4) = (et * 2.0 - 1.0) * (xi * xi - 1.0) * (-1.0 / 2.0);
                v(1, 5) = -et * xi * (xi + 1.0);
                v(1, 6) = (et * 2.0 + 1.0) * (xi * xi - 1.0) * (-1.0 / 2.0);
                v(1, 7) = -et * xi * (xi - 1.0);
                v(1, 8) = et * (xi * xi - 1.0) * 2.0;
            }
        }

        if (t == Tet4)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = -et - xi - zt + 1.0;
                v(0, 1) = xi;
                v(0, 2) = et;
                v(0, 3) = zt;
            }
            else
            {
                v(0, 0) = -1.0;
                v(0, 1) = 1.0;
                v(1, 0) = -1.0;
                v(1, 2) = 1.0;
                v(2, 0) = -1.0;
                v(2, 3) = 1.0;
            }
        }

        if (t == Tet10)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (et + xi + zt - 1.0) * (et + xi + zt - 1.0 / 2.0) * 2.0;
                v(0, 1) = xi * (xi - 1.0 / 2.0) * 2.0;
                v(0, 2) = et * (et - 1.0 / 2.0) * 2.0;
                v(0, 3) = zt * (zt - 1.0 / 2.0) * 2.0;
                v(0, 4) = xi * (et * 2.0 + xi * 2.0 + zt * 2.0 - 2.0) * -2.0;
                v(0, 5) = et * xi * 4.0;
                v(0, 6) = et * (et + xi + zt - 1.0) * -4.0;
                v(0, 7) = zt * (et * 2.0 + xi * 2.0 + zt * 2.0 - 2.0) * -2.0;
                v(0, 8) = xi * zt * 4.0;
                v(0, 9) = et * zt * 4.0;
            }
            else
            {
                v(0, 0) = et * 4.0 + xi * 4.0 + zt * 4.0 - 3.0;
                v(0, 1) = xi * 4.0 - 1.0;
                v(0, 4) = et * -4.0 - xi * 8.0 - zt * 4.0 + 4.0;
                v(0, 5) = et * 4.0;
                v(0, 6) = et * -4.0;
                v(0, 7) = zt * -4.0;
                v(0, 8) = zt * 4.0;
                v(1, 0) = et * 4.0 + xi * 4.0 + zt * 4.0 - 3.0;
                v(1, 2) = et * 4.0 - 1.0;
                v(1, 4) = xi * -4.0;
                v(1, 5) = xi * 4.0;
                v(1, 6) = et * -8.0 - xi * 4.0 - zt * 4.0 + 4.0;
                v(1, 7) = zt * -4.0;
                v(1, 9) = zt * 4.0;
                v(2, 0) = et * 4.0 + xi * 4.0 + zt * 4.0 - 3.0;
                v(2, 3) = zt * 4.0 - 1.0;
                v(2, 4) = xi * -4.0;
                v(2, 6) = et * -4.0;
                v(2, 7) = et * -4.0 - xi * 4.0 - zt * 8.0 + 4.0;
                v(2, 8) = xi * 4.0;
                v(2, 9) = et * 4.0;
            }
        }

        if (t == Hex8)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = -(et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0);
                v(0, 1) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0);
                v(0, 2) = -(et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0);
                v(0, 3) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0);
                v(0, 4) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0);
                v(0, 5) = -(et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0);
                v(0, 6) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0);
                v(0, 7) = -(et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0);
            }
            else
            {
                v(0, 0) = (et / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
                v(0, 1) = ((et / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0)) / 2.0;
                v(0, 2) = (et / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
                v(0, 3) = ((et / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0)) / 2.0;
                v(0, 4) = ((et / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0)) / 2.0;
                v(0, 5) = (et / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(0, 6) = ((et / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0)) / 2.0;
                v(0, 7) = (et / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(1, 0) = (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
                v(1, 1) = ((xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0)) / 2.0;
                v(1, 2) = (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
                v(1, 3) = ((xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 - 1.0 / 2.0)) / 2.0;
                v(1, 4) = ((xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0)) / 2.0;
                v(1, 5) = (xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(1, 6) = ((xi / 2.0 + 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0)) / 2.0;
                v(1, 7) = (xi / 2.0 - 1.0 / 2.0) * (zt / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(2, 0) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
                v(2, 1) = ((et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0)) / 2.0;
                v(2, 2) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(2, 3) = ((et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0)) / 2.0;
                v(2, 4) = ((et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0)) / 2.0;
                v(2, 5) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (-1.0 / 2.0);
                v(2, 6) = ((et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0)) / 2.0;
                v(2, 7) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (-1.0 / 2.0);
            }
        }
        if (t == Hex27)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (et * xi * zt * (et - 1.0) * (xi - 1.0) * (zt - 1.0)) / 8.0;
                v(0, 1) = (et * xi * zt * (et - 1.0) * (xi + 1.0) * (zt - 1.0)) / 8.0;
                v(0, 2) = (et * xi * zt * (et + 1.0) * (xi + 1.0) * (zt - 1.0)) / 8.0;
                v(0, 3) = (et * xi * zt * (et + 1.0) * (xi - 1.0) * (zt - 1.0)) / 8.0;
                v(0, 4) = (et * xi * zt * (et - 1.0) * (xi - 1.0) * (zt + 1.0)) / 8.0;
                v(0, 5) = (et * xi * zt * (et - 1.0) * (xi + 1.0) * (zt + 1.0)) / 8.0;
                v(0, 6) = (et * xi * zt * (et + 1.0) * (xi + 1.0) * (zt + 1.0)) / 8.0;
                v(0, 7) = (et * xi * zt * (et + 1.0) * (xi - 1.0) * (zt + 1.0)) / 8.0;
                v(0, 8) = et * zt * (et - 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 9) = xi * zt * (et - 1.0) * (et + 1.0) * (xi + 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 10) = et * zt * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 11) = xi * zt * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 12) = et * xi * (et - 1.0) * (xi - 1.0) * (zt - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 13) = et * xi * (et - 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 14) = et * xi * (et + 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 15) = et * xi * (et + 1.0) * (xi - 1.0) * (zt - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 16) = et * zt * (et - 1.0) * (xi - 1.0) * (xi + 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 17) = xi * zt * (et - 1.0) * (et + 1.0) * (xi + 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 18) = et * zt * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 19) = xi * zt * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 20) = (zt * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0)) / 2.0;
                v(0, 21) = (et * (et - 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0)) / 2.0;
                v(0, 22) = (xi * (et - 1.0) * (et + 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0)) / 2.0;
                v(0, 23) = (et * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0)) / 2.0;
                v(0, 24) = (xi * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (zt - 1.0) * (zt + 1.0)) / 2.0;
                v(0, 25) = (zt * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt + 1.0)) / 2.0;
                v(0, 26) = -(et - 1.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (zt - 1.0) * (zt + 1.0);
            }
            else
            {
                v(0, 0) = (et * zt * (xi * 2.0 - 1.0) * (et - 1.0) * (zt - 1.0)) / 8.0;
                v(0, 1) = (et * zt * (xi * 2.0 + 1.0) * (et - 1.0) * (zt - 1.0)) / 8.0;
                v(0, 2) = (et * zt * (xi * 2.0 + 1.0) * (et + 1.0) * (zt - 1.0)) / 8.0;
                v(0, 3) = (et * zt * (xi * 2.0 - 1.0) * (et + 1.0) * (zt - 1.0)) / 8.0;
                v(0, 4) = (et * zt * (xi * 2.0 - 1.0) * (et - 1.0) * (zt + 1.0)) / 8.0;
                v(0, 5) = (et * zt * (xi * 2.0 + 1.0) * (et - 1.0) * (zt + 1.0)) / 8.0;
                v(0, 6) = (et * zt * (xi * 2.0 + 1.0) * (et + 1.0) * (zt + 1.0)) / 8.0;
                v(0, 7) = (et * zt * (xi * 2.0 - 1.0) * (et + 1.0) * (zt + 1.0)) / 8.0;
                v(0, 8) = et * xi * zt * (et - 1.0) * (zt - 1.0) * (-1.0 / 2.0);
                v(0, 9) = zt * (et * et - 1.0) * (xi * 2.0 + 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 10) = et * xi * zt * (et + 1.0) * (zt - 1.0) * (-1.0 / 2.0);
                v(0, 11) = zt * (et * et - 1.0) * (xi * 2.0 - 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(0, 12) = et * (xi * 2.0 - 1.0) * (zt * zt - 1.0) * (et - 1.0) * (-1.0 / 4.0);
                v(0, 13) = et * (xi * 2.0 + 1.0) * (zt * zt - 1.0) * (et - 1.0) * (-1.0 / 4.0);
                v(0, 14) = et * (xi * 2.0 + 1.0) * (zt * zt - 1.0) * (et + 1.0) * (-1.0 / 4.0);
                v(0, 15) = et * (xi * 2.0 - 1.0) * (zt * zt - 1.0) * (et + 1.0) * (-1.0 / 4.0);
                v(0, 16) = et * xi * zt * (et - 1.0) * (zt + 1.0) * (-1.0 / 2.0);
                v(0, 17) = zt * (et * et - 1.0) * (xi * 2.0 + 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 18) = et * xi * zt * (et + 1.0) * (zt + 1.0) * (-1.0 / 2.0);
                v(0, 19) = zt * (et * et - 1.0) * (xi * 2.0 - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(0, 20) = xi * zt * (et * et - 1.0) * (zt - 1.0);
                v(0, 21) = et * xi * (zt * zt - 1.0) * (et - 1.0);
                v(0, 22) = ((et * et - 1.0) * (xi * 2.0 + 1.0) * (zt * zt - 1.0)) / 2.0;
                v(0, 23) = et * xi * (zt * zt - 1.0) * (et + 1.0);
                v(0, 24) = ((et * et - 1.0) * (xi * 2.0 - 1.0) * (zt * zt - 1.0)) / 2.0;
                v(0, 25) = xi * zt * (et * et - 1.0) * (zt + 1.0);
                v(0, 26) = xi * (et * et - 1.0) * (zt * zt - 1.0) * -2.0;
                v(1, 0) = (xi * zt * (et * 2.0 - 1.0) * (xi - 1.0) * (zt - 1.0)) / 8.0;
                v(1, 1) = (xi * zt * (et * 2.0 - 1.0) * (xi + 1.0) * (zt - 1.0)) / 8.0;
                v(1, 2) = (xi * zt * (et * 2.0 + 1.0) * (xi + 1.0) * (zt - 1.0)) / 8.0;
                v(1, 3) = (xi * zt * (et * 2.0 + 1.0) * (xi - 1.0) * (zt - 1.0)) / 8.0;
                v(1, 4) = (xi * zt * (et * 2.0 - 1.0) * (xi - 1.0) * (zt + 1.0)) / 8.0;
                v(1, 5) = (xi * zt * (et * 2.0 - 1.0) * (xi + 1.0) * (zt + 1.0)) / 8.0;
                v(1, 6) = (xi * zt * (et * 2.0 + 1.0) * (xi + 1.0) * (zt + 1.0)) / 8.0;
                v(1, 7) = (xi * zt * (et * 2.0 + 1.0) * (xi - 1.0) * (zt + 1.0)) / 8.0;
                v(1, 8) = zt * (et * 2.0 - 1.0) * (xi * xi - 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(1, 9) = et * xi * zt * (xi + 1.0) * (zt - 1.0) * (-1.0 / 2.0);
                v(1, 10) = zt * (et * 2.0 + 1.0) * (xi * xi - 1.0) * (zt - 1.0) * (-1.0 / 4.0);
                v(1, 11) = et * xi * zt * (xi - 1.0) * (zt - 1.0) * (-1.0 / 2.0);
                v(1, 12) = xi * (et * 2.0 - 1.0) * (zt * zt - 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                v(1, 13) = xi * (et * 2.0 - 1.0) * (zt * zt - 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                v(1, 14) = xi * (et * 2.0 + 1.0) * (zt * zt - 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                v(1, 15) = xi * (et * 2.0 + 1.0) * (zt * zt - 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                v(1, 16) = zt * (et * 2.0 - 1.0) * (xi * xi - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(1, 17) = et * xi * zt * (xi + 1.0) * (zt + 1.0) * (-1.0 / 2.0);
                v(1, 18) = zt * (et * 2.0 + 1.0) * (xi * xi - 1.0) * (zt + 1.0) * (-1.0 / 4.0);
                v(1, 19) = et * xi * zt * (xi - 1.0) * (zt + 1.0) * (-1.0 / 2.0);
                v(1, 20) = et * zt * (xi * xi - 1.0) * (zt - 1.0);
                v(1, 21) = ((et * 2.0 - 1.0) * (xi * xi - 1.0) * (zt * zt - 1.0)) / 2.0;
                v(1, 22) = et * xi * (zt * zt - 1.0) * (xi + 1.0);
                v(1, 23) = ((et * 2.0 + 1.0) * (xi * xi - 1.0) * (zt * zt - 1.0)) / 2.0;
                v(1, 24) = et * xi * (zt * zt - 1.0) * (xi - 1.0);
                v(1, 25) = et * zt * (xi * xi - 1.0) * (zt + 1.0);
                v(1, 26) = et * (xi * xi - 1.0) * (zt * zt - 1.0) * -2.0;
                v(2, 0) = (et * xi * (zt * 2.0 - 1.0) * (et - 1.0) * (xi - 1.0)) / 8.0;
                v(2, 1) = (et * xi * (zt * 2.0 - 1.0) * (et - 1.0) * (xi + 1.0)) / 8.0;
                v(2, 2) = (et * xi * (zt * 2.0 - 1.0) * (et + 1.0) * (xi + 1.0)) / 8.0;
                v(2, 3) = (et * xi * (zt * 2.0 - 1.0) * (et + 1.0) * (xi - 1.0)) / 8.0;
                v(2, 4) = (et * xi * (zt * 2.0 + 1.0) * (et - 1.0) * (xi - 1.0)) / 8.0;
                v(2, 5) = (et * xi * (zt * 2.0 + 1.0) * (et - 1.0) * (xi + 1.0)) / 8.0;
                v(2, 6) = (et * xi * (zt * 2.0 + 1.0) * (et + 1.0) * (xi + 1.0)) / 8.0;
                v(2, 7) = (et * xi * (zt * 2.0 + 1.0) * (et + 1.0) * (xi - 1.0)) / 8.0;
                v(2, 8) = et * (xi * xi - 1.0) * (zt * 2.0 - 1.0) * (et - 1.0) * (-1.0 / 4.0);
                v(2, 9) = xi * (et * et - 1.0) * (zt * 2.0 - 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                v(2, 10) = et * (xi * xi - 1.0) * (zt * 2.0 - 1.0) * (et + 1.0) * (-1.0 / 4.0);
                v(2, 11) = xi * (et * et - 1.0) * (zt * 2.0 - 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                v(2, 12) = et * xi * zt * (et - 1.0) * (xi - 1.0) * (-1.0 / 2.0);
                v(2, 13) = et * xi * zt * (et - 1.0) * (xi + 1.0) * (-1.0 / 2.0);
                v(2, 14) = et * xi * zt * (et + 1.0) * (xi + 1.0) * (-1.0 / 2.0);
                v(2, 15) = et * xi * zt * (et + 1.0) * (xi - 1.0) * (-1.0 / 2.0);
                v(2, 16) = et * (xi * xi - 1.0) * (zt * 2.0 + 1.0) * (et - 1.0) * (-1.0 / 4.0);
                v(2, 17) = xi * (et * et - 1.0) * (zt * 2.0 + 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                v(2, 18) = et * (xi * xi - 1.0) * (zt * 2.0 + 1.0) * (et + 1.0) * (-1.0 / 4.0);
                v(2, 19) = xi * (et * et - 1.0) * (zt * 2.0 + 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                v(2, 20) = ((et * et - 1.0) * (xi * xi - 1.0) * (zt * 2.0 - 1.0)) / 2.0;
                v(2, 21) = et * zt * (xi * xi - 1.0) * (et - 1.0);
                v(2, 22) = xi * zt * (et * et - 1.0) * (xi + 1.0);
                v(2, 23) = et * zt * (xi * xi - 1.0) * (et + 1.0);
                v(2, 24) = xi * zt * (et * et - 1.0) * (xi - 1.0);
                v(2, 25) = ((et * et - 1.0) * (xi * xi - 1.0) * (zt * 2.0 + 1.0)) / 2.0;
                v(2, 26) = zt * (et * et - 1.0) * (xi * xi - 1.0) * -2.0;
            }
        }

        if (t == Prism6)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = (zt / 2.0 - 1.0 / 2.0) * (et + xi - 1.0);
                v(0, 1) = xi * (zt - 1.0) * (-1.0 / 2.0);
                v(0, 2) = et * (zt - 1.0) * (-1.0 / 2.0);
                v(0, 3) = -(zt / 2.0 + 1.0 / 2.0) * (et + xi - 1.0);
                v(0, 4) = xi * (zt / 2.0 + 1.0 / 2.0);
                v(0, 5) = et * (zt / 2.0 + 1.0 / 2.0);
            }
            else
            {
                v(0, 0) = zt / 2.0 - 1.0 / 2.0;
                v(0, 1) = zt * (-1.0 / 2.0) + 1.0 / 2.0;
                v(0, 3) = zt * (-1.0 / 2.0) - 1.0 / 2.0;
                v(0, 4) = zt / 2.0 + 1.0 / 2.0;
                v(1, 0) = zt / 2.0 - 1.0 / 2.0;
                v(1, 2) = zt * (-1.0 / 2.0) + 1.0 / 2.0;
                v(1, 3) = zt * (-1.0 / 2.0) - 1.0 / 2.0;
                v(1, 5) = zt / 2.0 + 1.0 / 2.0;
                v(2, 0) = et / 2.0 + xi / 2.0 - 1.0 / 2.0;
                v(2, 1) = xi * (-1.0 / 2.0);
                v(2, 2) = et * (-1.0 / 2.0);
                v(2, 3) = et * (-1.0 / 2.0) - xi / 2.0 + 1.0 / 2.0;
                v(2, 4) = xi / 2.0;
                v(2, 5) = et / 2.0;
            }
        }

        if (t == Prism18)
        {
            if constexpr (diffOrder == 0)
            {
                v(0, 0) = zt * (zt - 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0);
                v(0, 1) = xi * zt * (xi - 1.0 / 2.0) * (zt - 1.0);
                v(0, 2) = et * zt * (et - 1.0 / 2.0) * (zt - 1.0);
                v(0, 3) = zt * (zt + 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0);
                v(0, 4) = xi * zt * (xi - 1.0 / 2.0) * (zt + 1.0);
                v(0, 5) = et * zt * (et - 1.0 / 2.0) * (zt + 1.0);
                v(0, 6) = -xi * zt * (zt - 1.0) * (et * 2.0 + xi * 2.0 - 2.0);
                v(0, 7) = et * xi * zt * (zt - 1.0) * 2.0;
                v(0, 8) = et * zt * (zt - 1.0) * (et + xi - 1.0) * -2.0;
                v(0, 9) = (zt - 1.0) * (zt + 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0) * -2.0;
                v(0, 10) = xi * (xi - 1.0 / 2.0) * (zt - 1.0) * (zt + 1.0) * -2.0;
                v(0, 11) = et * (et - 1.0 / 2.0) * (zt - 1.0) * (zt + 1.0) * -2.0;
                v(0, 12) = -xi * zt * (zt + 1.0) * (et * 2.0 + xi * 2.0 - 2.0);
                v(0, 13) = et * xi * zt * (zt + 1.0) * 2.0;
                v(0, 14) = et * zt * (zt + 1.0) * (et + xi - 1.0) * -2.0;
                v(0, 15) = xi * (zt - 1.0) * (zt + 1.0) * (et * 2.0 + xi * 2.0 - 2.0) * 2.0;
                v(0, 16) = et * xi * (zt - 1.0) * (zt + 1.0) * -4.0;
                v(0, 17) = et * (zt - 1.0) * (zt + 1.0) * (et + xi - 1.0) * 4.0;
            }
            else
            {
                v(0, 0) = (zt * (zt - 1.0) * (et * 4.0 + xi * 4.0 - 3.0)) / 2.0;
                v(0, 1) = (zt * (xi * 4.0 - 1.0) * (zt - 1.0)) / 2.0;
                v(0, 3) = (zt * (zt + 1.0) * (et * 4.0 + xi * 4.0 - 3.0)) / 2.0;
                v(0, 4) = (zt * (xi * 4.0 - 1.0) * (zt + 1.0)) / 2.0;
                v(0, 6) = zt * (zt - 1.0) * (et + xi * 2.0 - 1.0) * -2.0;
                v(0, 7) = et * zt * (zt - 1.0) * 2.0;
                v(0, 8) = et * zt * (zt - 1.0) * -2.0;
                v(0, 9) = -(zt * zt - 1.0) * (et * 4.0 + xi * 4.0 - 3.0);
                v(0, 10) = -(xi * 4.0 - 1.0) * (zt * zt - 1.0);
                v(0, 12) = zt * (zt + 1.0) * (et + xi * 2.0 - 1.0) * -2.0;
                v(0, 13) = et * zt * (zt + 1.0) * 2.0;
                v(0, 14) = et * zt * (zt + 1.0) * -2.0;
                v(0, 15) = (zt * zt - 1.0) * (et + xi * 2.0 - 1.0) * 4.0;
                v(0, 16) = et * (zt - 1.0) * (zt + 1.0) * -4.0;
                v(0, 17) = et * (zt - 1.0) * (zt + 1.0) * 4.0;
                v(1, 0) = (zt * (zt - 1.0) * (et * 4.0 + xi * 4.0 - 3.0)) / 2.0;
                v(1, 2) = (zt * (et * 4.0 - 1.0) * (zt - 1.0)) / 2.0;
                v(1, 3) = (zt * (zt + 1.0) * (et * 4.0 + xi * 4.0 - 3.0)) / 2.0;
                v(1, 5) = (zt * (et * 4.0 - 1.0) * (zt + 1.0)) / 2.0;
                v(1, 6) = xi * zt * (zt - 1.0) * -2.0;
                v(1, 7) = xi * zt * (zt - 1.0) * 2.0;
                v(1, 8) = zt * (zt - 1.0) * (et * 2.0 + xi - 1.0) * -2.0;
                v(1, 9) = -(zt * zt - 1.0) * (et * 4.0 + xi * 4.0 - 3.0);
                v(1, 11) = -(et * 4.0 - 1.0) * (zt * zt - 1.0);
                v(1, 12) = xi * zt * (zt + 1.0) * -2.0;
                v(1, 13) = xi * zt * (zt + 1.0) * 2.0;
                v(1, 14) = zt * (zt + 1.0) * (et * 2.0 + xi - 1.0) * -2.0;
                v(1, 15) = xi * (zt - 1.0) * (zt + 1.0) * 4.0;
                v(1, 16) = xi * (zt - 1.0) * (zt + 1.0) * -4.0;
                v(1, 17) = (zt * zt - 1.0) * (et * 2.0 + xi - 1.0) * 4.0;
                v(2, 0) = zt * (et + xi - 1.0) * (et + xi - 1.0 / 2.0) + (zt - 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0);
                v(2, 1) = (xi * (xi * 2.0 - 1.0) * (zt * 2.0 - 1.0)) / 2.0;
                v(2, 2) = (et * (et * 2.0 - 1.0) * (zt * 2.0 - 1.0)) / 2.0;
                v(2, 3) = zt * (et + xi - 1.0) * (et + xi - 1.0 / 2.0) + (zt + 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0);
                v(2, 4) = (xi * (xi * 2.0 - 1.0) * (zt * 2.0 + 1.0)) / 2.0;
                v(2, 5) = (et * (et * 2.0 - 1.0) * (zt * 2.0 + 1.0)) / 2.0;
                v(2, 6) = xi * (zt * 2.0 - 1.0) * (et + xi - 1.0) * -2.0;
                v(2, 7) = et * xi * (zt * 2.0 - 1.0) * 2.0;
                v(2, 8) = et * (zt * 2.0 - 1.0) * (et + xi - 1.0) * -2.0;
                v(2, 9) = (zt - 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0) * -2.0 - (zt + 1.0) * (et + xi - 1.0) * (et + xi - 1.0 / 2.0) * 2.0;
                v(2, 10) = xi * zt * (xi * 2.0 - 1.0) * -2.0;
                v(2, 11) = et * zt * (et * 2.0 - 1.0) * -2.0;
                v(2, 12) = xi * (zt * 2.0 + 1.0) * (et + xi - 1.0) * -2.0;
                v(2, 13) = et * xi * (zt * 2.0 + 1.0) * 2.0;
                v(2, 14) = et * (zt * 2.0 + 1.0) * (et + xi - 1.0) * -2.0;
                v(2, 15) = xi * zt * (et + xi - 1.0) * 8.0;
                v(2, 16) = et * xi * zt * -8.0;
                v(2, 17) = et * zt * (et + xi - 1.0) * 8.0;
            }
        }

        if (t == Pyramid5)
        {
            if constexpr (diffOrder == 0)
            {
                // v(0, 0) = -(et / 2.0 - 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt - 1.0);
                // v(0, 1) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt - 1.0);
                // v(0, 2) = -(et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0) * (zt - 1.0);
                // v(0, 3) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0) * (zt - 1.0);
                // v(0, 4) = zt;

                // rational: has a divisor
                if (std::abs(1 - zt) < 1e-15)
                    zt -= DNDS::signP(1 - zt) * 1e-15;
                v(0, 0) = ((et + zt - 1.0) * (xi + zt - 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(0, 1) = ((et + zt - 1.0) * (xi - zt + 1.0)) / (zt * 4.0 - 4.0);
                v(0, 2) = ((et - zt + 1.0) * (xi - zt + 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(0, 3) = ((xi + zt - 1.0) * (et - zt + 1.0)) / (zt * 4.0 - 4.0);
                v(0, 4) = zt;
            }
            else
            {
                // v(0, 0) = (et / 2.0 - 1.0 / 2.0) * (zt - 1.0) * (-1.0 / 2.0);
                // v(0, 1) = ((et / 2.0 - 1.0 / 2.0) * (zt - 1.0)) / 2.0;
                // v(0, 2) = (et / 2.0 + 1.0 / 2.0) * (zt - 1.0) * (-1.0 / 2.0);
                // v(0, 3) = ((et / 2.0 + 1.0 / 2.0) * (zt - 1.0)) / 2.0;
                // v(1, 0) = (xi / 2.0 - 1.0 / 2.0) * (zt - 1.0) * (-1.0 / 2.0);
                // v(1, 1) = ((xi / 2.0 + 1.0 / 2.0) * (zt - 1.0)) / 2.0;
                // v(1, 2) = (xi / 2.0 + 1.0 / 2.0) * (zt - 1.0) * (-1.0 / 2.0);
                // v(1, 3) = ((xi / 2.0 - 1.0 / 2.0) * (zt - 1.0)) / 2.0;
                // v(2, 0) = (et - 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                // v(2, 1) = (et / 2.0 - 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0);
                // v(2, 2) = -(et / 2.0 + 1.0 / 2.0) * (xi / 2.0 + 1.0 / 2.0);
                // v(2, 3) = (et / 2.0 + 1.0 / 2.0) * (xi / 2.0 - 1.0 / 2.0);
                // v(2, 4) = 1.0;

                // rational: has a divisor
                if (std::abs(1 - zt) < 1e-15)
                    zt -= DNDS::signP(1 - zt) * 1e-15;
                v(0, 0) = ((et + zt - 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(0, 1) = (et + zt - 1.0) / (zt * 4.0 - 4.0);
                v(0, 2) = ((et - zt + 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(0, 3) = (et - zt + 1.0) / (zt * 4.0 - 4.0);
                v(1, 0) = ((xi + zt - 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(1, 1) = (xi - zt + 1.0) / (zt * 4.0 - 4.0);
                v(1, 2) = ((xi - zt + 1.0) * (-1.0 / 4.0)) / (zt - 1.0);
                v(1, 3) = (xi + zt - 1.0) / (zt * 4.0 - 4.0);
                v(2, 0) = (1.0 / pow(zt - 1.0, 2.0) * (zt * 2.0 + et * xi - zt * zt - 1.0)) / 4.0;
                v(2, 1) = 1.0 / pow(zt - 1.0, 2.0) * (zt * -2.0 + et * xi + zt * zt + 1.0) * (-1.0 / 4.0);
                v(2, 2) = (1.0 / pow(zt - 1.0, 2.0) * (zt * 2.0 + et * xi - zt * zt - 1.0)) / 4.0;
                v(2, 3) = 1.0 / pow(zt - 1.0, 2.0) * (zt * -2.0 + et * xi + zt * zt + 1.0) * (-1.0 / 4.0);
                v(2, 4) = 1.0;
            }
        }

        if (t == Pyramid14)
        {
            if constexpr (diffOrder == 0)
            {
                // v(0, 0) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (xi - 1.0)) / 8.0;
                // v(0, 1) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (xi + 1.0)) / 8.0;
                // v(0, 2) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et + 1.0) * (xi + 1.0)) / 8.0;
                // v(0, 3) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et + 1.0) * (xi - 1.0)) / 8.0;
                // v(0, 4) = zt * (zt * 2.0 - 1.0);
                // v(0, 5) = et * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (xi - 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                // v(0, 6) = xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (et + 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                // v(0, 7) = et * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0) * (-1.0 / 4.0);
                // v(0, 8) = xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (-1.0 / 4.0);
                // v(0, 9) = zt * (zt * 2.0 - 2.0) * (et - 1.0 / 2.0) * (xi - 1.0 / 2.0) * -2.0;
                // v(0, 10) = zt * (zt * 2.0 - 2.0) * (et - 1.0 / 2.0) * (xi + 1.0 / 2.0) * 2.0;
                // v(0, 11) = zt * (zt * 2.0 - 2.0) * (et + 1.0 / 2.0) * (xi + 1.0 / 2.0) * -2.0;
                // v(0, 12) = zt * (zt * 2.0 - 2.0) * (et + 1.0 / 2.0) * (xi - 1.0 / 2.0) * 2.0;
                // v(0, 13) = ((zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * (et - 1.0) * (et + 1.0) * (xi - 1.0) * (xi + 1.0)) / 2.0;

                // !rational: has a divisor
                if (std::abs(1 - zt) < 1e-15)
                    zt -= DNDS::signP(1 - zt) * 1e-15;
                v(0, 0) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (xi + zt - 1.0)) / 8.0;
                v(0, 1) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (xi - zt + 1.0)) / 8.0;
                v(0, 2) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et - zt + 1.0) * (xi - zt + 1.0)) / 8.0;
                v(0, 3) = (et * xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (xi + zt - 1.0) * (et - zt + 1.0)) / 8.0;
                v(0, 4) = zt * (zt * 2.0 - 1.0);
                v(0, 5) = et * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (xi + zt - 1.0) * (xi - zt + 1.0) * (-1.0 / 4.0);
                v(0, 6) = xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (et - zt + 1.0) * (xi - zt + 1.0) * (-1.0 / 4.0);
                v(0, 7) = et * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (xi + zt - 1.0) * (et - zt + 1.0) * (xi - zt + 1.0) * (-1.0 / 4.0);
                v(0, 8) = xi * (zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (xi + zt - 1.0) * (et - zt + 1.0) * (-1.0 / 4.0);
                v(0, 9) = zt * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 2.0) * (et + zt - 1.0) * (xi + zt - 1.0) * (-1.0 / 2.0);
                v(0, 10) = (zt * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 2.0) * (et + zt - 1.0) * (xi - zt + 1.0)) / 2.0;
                v(0, 11) = zt * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 2.0) * (et - zt + 1.0) * (xi - zt + 1.0) * (-1.0 / 2.0);
                v(0, 12) = (zt * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 2.0) * (xi + zt - 1.0) * (et - zt + 1.0)) / 2.0;
                v(0, 13) = ((zt * 2.0 - 1.0) * (zt * 2.0 - 2.0) * 1.0 / pow(zt - 1.0, 4.0) * (et + zt - 1.0) * (xi + zt - 1.0) * (et - zt + 1.0) * (xi - zt + 1.0)) / 2.0;
            }
            else
            {
                // v(0, 0) = (et * (xi * 2.0 - 1.0) * (et - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(0, 1) = (et * (xi * 2.0 + 1.0) * (et - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(0, 2) = (et * (xi * 2.0 + 1.0) * (et + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(0, 3) = (et * (xi * 2.0 - 1.0) * (et + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(0, 5) = -et * xi * (et - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0);
                // v(0, 6) = (et * et - 1.0) * (xi * 2.0 + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * (-1.0 / 2.0);
                // v(0, 7) = -et * xi * (et + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0);
                // v(0, 8) = (et * et - 1.0) * (xi * 2.0 - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * (-1.0 / 2.0);
                // v(0, 9) = zt * (zt * 2.0 - 2.0) * (et - 1.0 / 2.0) * -2.0;
                // v(0, 10) = zt * (zt * 2.0 - 2.0) * (et - 1.0 / 2.0) * 2.0;
                // v(0, 11) = zt * (zt * 2.0 - 2.0) * (et + 1.0 / 2.0) * -2.0;
                // v(0, 12) = zt * (zt * 2.0 - 2.0) * (et + 1.0 / 2.0) * 2.0;
                // v(0, 13) = xi * (et * et - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * 2.0;
                // v(1, 0) = (xi * (et * 2.0 - 1.0) * (xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(1, 1) = (xi * (et * 2.0 - 1.0) * (xi + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(1, 2) = (xi * (et * 2.0 + 1.0) * (xi + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(1, 3) = (xi * (et * 2.0 + 1.0) * (xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0)) / 4.0;
                // v(1, 5) = (et * 2.0 - 1.0) * (xi * xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * (-1.0 / 2.0);
                // v(1, 6) = -et * xi * (xi + 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0);
                // v(1, 7) = (et * 2.0 + 1.0) * (xi * xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * (-1.0 / 2.0);
                // v(1, 8) = -et * xi * (xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0);
                // v(1, 9) = zt * (zt * 2.0 - 2.0) * (xi - 1.0 / 2.0) * -2.0;
                // v(1, 10) = zt * (zt * 2.0 - 2.0) * (xi + 1.0 / 2.0) * 2.0;
                // v(1, 11) = zt * (zt * 2.0 - 2.0) * (xi + 1.0 / 2.0) * -2.0;
                // v(1, 12) = zt * (zt * 2.0 - 2.0) * (xi - 1.0 / 2.0) * 2.0;
                // v(1, 13) = et * (xi * xi - 1.0) * (zt * -3.0 + (zt * zt) * 2.0 + 1.0) * 2.0;
                // v(2, 0) = (et * xi * (zt * 4.0 - 3.0) * (et - 1.0) * (xi - 1.0)) / 4.0;
                // v(2, 1) = (et * xi * (zt * 4.0 - 3.0) * (et - 1.0) * (xi + 1.0)) / 4.0;
                // v(2, 2) = (et * xi * (zt * 4.0 - 3.0) * (et + 1.0) * (xi + 1.0)) / 4.0;
                // v(2, 3) = (et * xi * (zt * 4.0 - 3.0) * (et + 1.0) * (xi - 1.0)) / 4.0;
                // v(2, 4) = zt * 4.0 - 1.0;
                // v(2, 5) = et * (xi * xi - 1.0) * (zt * 4.0 - 3.0) * (et - 1.0) * (-1.0 / 2.0);
                // v(2, 6) = xi * (et * et - 1.0) * (zt * 4.0 - 3.0) * (xi + 1.0) * (-1.0 / 2.0);
                // v(2, 7) = et * (xi * xi - 1.0) * (zt * 4.0 - 3.0) * (et + 1.0) * (-1.0 / 2.0);
                // v(2, 8) = xi * (et * et - 1.0) * (zt * 4.0 - 3.0) * (xi - 1.0) * (-1.0 / 2.0);
                // v(2, 9) = -(et * 2.0 - 1.0) * (xi * 2.0 - 1.0) * (zt * 2.0 - 1.0);
                // v(2, 10) = (et * 2.0 - 1.0) * (xi * 2.0 + 1.0) * (zt * 2.0 - 1.0);
                // v(2, 11) = -(et * 2.0 + 1.0) * (xi * 2.0 + 1.0) * (zt * 2.0 - 1.0);
                // v(2, 12) = (et * 2.0 + 1.0) * (xi * 2.0 - 1.0) * (zt * 2.0 - 1.0);
                // v(2, 13) = (et * et - 1.0) * (xi * xi - 1.0) * (zt * 4.0 - 3.0);

                // !rational: has a divisor
                if (std::abs(1 - zt) < 1e-15)
                    zt -= DNDS::signP(1 - zt) * 1e-15;

                v(0, 0) = (et * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et + zt - 1.0) * (xi * 2.0 + zt - 1.0)) / 4.0;
                v(0, 1) = (et * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi * 2.0 - zt + 1.0) * (et + zt - 1.0)) / 4.0;
                v(0, 2) = (et * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi * 2.0 - zt + 1.0) * (et - zt + 1.0)) / 4.0;
                v(0, 3) = (et * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et - zt + 1.0) * (xi * 2.0 + zt - 1.0)) / 4.0;
                v(0, 5) = -et * xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et + zt - 1.0);
                v(0, 6) = (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi * 2.0 - zt + 1.0) * (zt * 2.0 + et * et - zt * zt - 1.0) * (-1.0 / 2.0);
                v(0, 7) = -et * xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et - zt + 1.0);
                v(0, 8) = (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi * 2.0 + zt - 1.0) * (zt * 2.0 + et * et - zt * zt - 1.0) * (-1.0 / 2.0);
                v(0, 9) = -(zt * (et + zt - 1.0)) / (zt - 1.0);
                v(0, 10) = (zt * (et + zt - 1.0)) / (zt - 1.0);
                v(0, 11) = -(zt * (et - zt + 1.0)) / (zt - 1.0);
                v(0, 12) = (zt * (et - zt + 1.0)) / (zt - 1.0);
                v(0, 13) = xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (zt * 2.0 + et * et - zt * zt - 1.0) * 2.0;
                v(1, 0) = (xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi + zt - 1.0) * (et * 2.0 + zt - 1.0)) / 4.0;
                v(1, 1) = (xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et * 2.0 + zt - 1.0) * (xi - zt + 1.0)) / 4.0;
                v(1, 2) = (xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi - zt + 1.0) * (et * 2.0 - zt + 1.0)) / 4.0;
                v(1, 3) = (xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi + zt - 1.0) * (et * 2.0 - zt + 1.0)) / 4.0;
                v(1, 5) = (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et * 2.0 + zt - 1.0) * (zt * 2.0 + xi * xi - zt * zt - 1.0) * (-1.0 / 2.0);
                v(1, 6) = -et * xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi - zt + 1.0);
                v(1, 7) = (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (et * 2.0 - zt + 1.0) * (zt * 2.0 + xi * xi - zt * zt - 1.0) * (-1.0 / 2.0);
                v(1, 8) = -et * xi * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (xi + zt - 1.0);
                v(1, 9) = -(zt * (xi + zt - 1.0)) / (zt - 1.0);
                v(1, 10) = (zt * (xi - zt + 1.0)) / (zt - 1.0);
                v(1, 11) = -(zt * (xi - zt + 1.0)) / (zt - 1.0);
                v(1, 12) = (zt * (xi + zt - 1.0)) / (zt - 1.0);
                v(1, 13) = et * (zt * 2.0 - 1.0) * 1.0 / pow(zt - 1.0, 3.0) * (zt * 2.0 + xi * xi - zt * zt - 1.0) * 2.0;
                v(2, 0) = et * xi * 1.0 / pow(zt - 1.0, 4.0) * (zt * -2.0 - et * xi - et * zt * 2.0 - xi * zt * 2.0 + et * (zt * zt) * 2.0 + xi * (zt * zt) * 2.0 + zt * zt + et * xi * zt * 4.0 + 1.0) * (-1.0 / 4.0);
                v(2, 1) = (et * xi * 1.0 / pow(zt - 1.0, 4.0) * (zt * -2.0 + et * xi - et * zt * 2.0 + xi * zt * 2.0 + et * (zt * zt) * 2.0 - xi * (zt * zt) * 2.0 + zt * zt - et * xi * zt * 4.0 + 1.0)) / 4.0;
                v(2, 2) = et * xi * 1.0 / pow(zt - 1.0, 4.0) * (zt * -2.0 - et * xi + et * zt * 2.0 + xi * zt * 2.0 - et * (zt * zt) * 2.0 - xi * (zt * zt) * 2.0 + zt * zt + et * xi * zt * 4.0 + 1.0) * (-1.0 / 4.0);
                v(2, 3) = (et * xi * 1.0 / pow(zt - 1.0, 4.0) * (zt * -2.0 + et * xi + et * zt * 2.0 - xi * zt * 2.0 - et * (zt * zt) * 2.0 + xi * (zt * zt) * 2.0 + zt * zt - et * xi * zt * 4.0 + 1.0)) / 4.0;
                v(2, 4) = zt * 4.0 - 1.0;
                v(2, 5) = et * 1.0 / pow(zt - 1.0, 4.0) * (et + zt * 8.0 - (xi * xi) * (zt * zt) * 2.0 - et * zt * 2.0 + et * (xi * xi) + et * (zt * zt) + (xi * xi) * zt * 2.0 - (zt * zt) * 1.2E+1 + (zt * zt * zt) * 8.0 - (zt * zt * zt * zt) * 2.0 - et * (xi * xi) * zt * 4.0 - 2.0) * (-1.0 / 2.0);
                v(2, 6) = xi * 1.0 / pow(zt - 1.0, 4.0) * (xi - zt * 8.0 - xi * zt * 2.0 + (et * et) * xi - (et * et) * zt * 2.0 + xi * (zt * zt) + (zt * zt) * 1.2E+1 - (zt * zt * zt) * 8.0 + (zt * zt * zt * zt) * 2.0 + (et * et) * (zt * zt) * 2.0 - (et * et) * xi * zt * 4.0 + 2.0) * (-1.0 / 2.0);
                v(2, 7) = et * 1.0 / pow(zt - 1.0, 4.0) * (et - zt * 8.0 + (xi * xi) * (zt * zt) * 2.0 - et * zt * 2.0 + et * (xi * xi) + et * (zt * zt) - (xi * xi) * zt * 2.0 + (zt * zt) * 1.2E+1 - (zt * zt * zt) * 8.0 + (zt * zt * zt * zt) * 2.0 - et * (xi * xi) * zt * 4.0 + 2.0) * (-1.0 / 2.0);
                v(2, 8) = xi * 1.0 / pow(zt - 1.0, 4.0) * (xi + zt * 8.0 - xi * zt * 2.0 + (et * et) * xi + (et * et) * zt * 2.0 + xi * (zt * zt) - (zt * zt) * 1.2E+1 + (zt * zt * zt) * 8.0 - (zt * zt * zt * zt) * 2.0 - (et * et) * (zt * zt) * 2.0 - (et * et) * xi * zt * 4.0 - 2.0) * (-1.0 / 2.0);
                v(2, 9) = -1.0 / pow(zt - 1.0, 2.0) * (et + xi + zt * 4.0 - et * xi - et * zt * 2.0 - xi * zt * 2.0 + et * (zt * zt) + xi * (zt * zt) - (zt * zt) * 5.0 + (zt * zt * zt) * 2.0 - 1.0);
                v(2, 10) = -1.0 / pow(zt - 1.0, 2.0) * (et - xi + zt * 4.0 + et * xi - et * zt * 2.0 + xi * zt * 2.0 + et * (zt * zt) - xi * (zt * zt) - (zt * zt) * 5.0 + (zt * zt * zt) * 2.0 - 1.0);
                v(2, 11) = 1.0 / pow(zt - 1.0, 2.0) * (et + xi - zt * 4.0 + et * xi - et * zt * 2.0 - xi * zt * 2.0 + et * (zt * zt) + xi * (zt * zt) + (zt * zt) * 5.0 - (zt * zt * zt) * 2.0 + 1.0);
                v(2, 12) = -1.0 / pow(zt - 1.0, 2.0) * (-et + xi + zt * 4.0 + et * xi + et * zt * 2.0 - xi * zt * 2.0 - et * (zt * zt) + xi * (zt * zt) - (zt * zt) * 5.0 + (zt * zt * zt) * 2.0 - 1.0);
                v(2, 13) = 1.0 / pow(zt - 1.0, 4.0) * (zt * 1.6E+1 + (xi * xi) * (zt * zt) - (et * et) * zt * 2.0 - (xi * xi) * zt * 2.0 + et * et + xi * xi - (zt * zt) * 3.4E+1 + (zt * zt * zt) * 3.6E+1 - (zt * zt * zt * zt) * 1.9E+1 + (zt * zt * zt * zt * zt) * 4.0 + (et * et) * (xi * xi) + (et * et) * (zt * zt) - (et * et) * (xi * xi) * zt * 4.0 - 3.0);
            }
        }
        DNDS_assert(t != UnknownElem && t < ElemType_NUM);
    }

    using tNj = Eigen::RowVector<t_real, Eigen::Dynamic>;
    using tD1Nj = Eigen::Matrix<t_real, 3, Eigen::Dynamic>;
    using tD01Nj = Eigen::Matrix<t_real, 4, Eigen::Dynamic>;

    struct Element
    {
        ElemType type = UnknownElem;

        constexpr ParamSpace GetParamSpace() const
        {
            return ElemType_to_ParamSpace(type);
        }

        constexpr t_index GetDim() const
        {
            return Dim_Order_NVNNNF[type][0];
        }

        constexpr t_index GetOrder() const
        {
            return Dim_Order_NVNNNF[type][1];
        }

        constexpr t_index GetNumVertices() const
        {
            return Dim_Order_NVNNNF[type][2];
        }

        constexpr t_index GetNumNodes() const
        {
            return Dim_Order_NVNNNF[type][3];
        }

        constexpr t_index GetNumFaces() const
        {
            return Dim_Order_NVNNNF[type][4];
        }

        constexpr t_index GetNumElev_O1O2() const
        {
            return GetElemElevation_O1O2_NumNode(type);
        }

        constexpr Element ObtainFace(t_index iFace) const
        {
            DNDS_assert(iFace < this->GetNumFaces());
            return Element{GetFaceType(type, iFace)};
        }

        /**
         * @warning assuming Out has correct size
         */
        template <class TIn, class TOut>
        void ExtractFaceNodes(t_index iFace, const TIn &nodes, TOut &faceNodes)
        {
            DNDS_assert(iFace < this->GetNumFaces());
            for (t_index i = 0; i < ObtainFace(iFace).GetNumNodes(); i++)
                faceNodes[i] = nodes[FaceNodeList[type][iFace][i]];
        }

        constexpr Element ObtainElevNodeSpan(t_index iNodeElev) const
        {
            DNDS_assert(iNodeElev < this->GetNumElev_O1O2());
            return Element{GetElemElevation_O1O2_NodeSpanType(type, iNodeElev)};
        }

        constexpr Element ObtainElevatedElem() const
        {
            return Element{GetElemElevation_O1O2_ElevatedType(type)};
        }

        template <class TIn, class TOut>
        void ExtractElevNodeSpanNodes(t_index iNodeElev, const TIn &nodes, TOut &spanNodes)
        {
            DNDS_assert(iNodeElev < this->GetNumElev_O1O2());
            for (t_index i = 0; i < ObtainElevNodeSpan(iNodeElev).GetNumNodes(); i++)
                spanNodes[i] = nodes[ElemElevationSpan_O1O2List[type][iNodeElev][i]];
        }

        /**
         * @warning Nj resized within
         */
        void GetNj(const tPoint &pParam, tNj &Nj)
        {
            Nj.setZero(1, this->GetNumNodes());
            ShapeFunc_DiNj<0>(type, pParam, Nj);
        }

        /**
         * @warning D1Nj resized within
         */
        void GetD1Nj(const tPoint &pParam, tD1Nj &D1Nj)
        {
            D1Nj.setZero(3, this->GetNumNodes());
            ShapeFunc_DiNj<1>(type, pParam, D1Nj);
        }

        /**
         * @warning DiNj resized within
         */
        void GetD01Nj(const tPoint &pParam, tD01Nj &D01Nj)
        {
            tNj Nj;
            tD1Nj D1Nj;
            this->GetNj(pParam, Nj);
            this->GetD1Nj(pParam, D1Nj);

            D01Nj.resize(4, this->GetNumNodes());
            D01Nj(0, Eigen::all) = Nj;
            D01Nj({1, 2, 3}, Eigen::all) = D1Nj;
        }
    };

    Eigen::Matrix<t_real, 3, Eigen::Dynamic> GetStandardCoord(ElemType t);
}

namespace DNDS::Geom::Elem
{
    inline bool cellsAreFaceConnected(
        const std::vector<DNDS::index> &nodes_A,
        const std::vector<DNDS::index> &nodes_B,
        Element eA,
        Element eB)
    {
        DNDS_assert(nodes_A.size() >= eA.GetNumNodes());
        DNDS_assert(nodes_B.size() >= eB.GetNumVertices());
        std::vector<DNDS::index> nodes_B_Vert{nodes_B.begin(), nodes_B.begin() + eB.GetNumVertices()};
        std::sort(nodes_B_Vert.begin(), nodes_B_Vert.end());
        for (int iF = 0; iF < eA.GetNumFaces(); iF++)
        {
            auto eF = eA.ObtainFace(iF);
            std::vector<DNDS::index> fNodes(eF.GetNumNodes());
            eA.ExtractFaceNodes(iF, nodes_A, fNodes);
            std::sort(fNodes.begin(), fNodes.begin() + eF.GetNumVertices());
            if (std::includes(
                    nodes_B_Vert.begin(), nodes_B_Vert.end(),
                    fNodes.begin(), fNodes.begin() + eF.GetNumVertices()))
                return true;
        }
        return false;
    }

    inline bool cellsAreFaceConnected(
        const std::vector<DNDS::index> &verts_A,
        const std::vector<DNDS::index> &nodes_B)
    {

        return false;
    }

    template <class tCoordsIn>
    tJacobi ShapeJacobianCoordD01Nj(const tCoordsIn &cs, Eigen::Ref<const tD01Nj> DiNj)
    {
        return cs * DiNj({1, 2, 3}, Eigen::all).transpose();
    }

    template <class tCoordsIn>
    tPoint PPhysicsCoordD01Nj(const tCoordsIn &cs, Eigen::Ref<const tD01Nj> DiNj)
    {
        return cs * DiNj(0, Eigen::all).transpose();
    }

    // TODO: add a integration-based counterpart
    inline tPoint GetElemNodeMajorSpan(const tSmallCoords &coords)
    {
        tPoint c = coords.rowwise().mean();
        tSmallCoords coordsC = coords.colwise() - c;
        tPoint ve0 = coordsC(Eigen::all, 1) - coordsC(Eigen::all, 0);
        tJacobi inertia = coordsC * coordsC.transpose();
        real cond01 = HardEigen::Eigen3x3RealSymEigenDecompositionGetCond01(inertia); // ratio of 2 largest eigenvalues
        if (cond01 < 1 + 1e-6)
            inertia += ve0 * ve0.transpose() * 1e-4; // first edge gets priority
        auto decRet = HardEigen::Eigen3x3RealSymEigenDecompositionNormalized(inertia);
        coordsC = decRet.transpose() * coordsC;
        tPoint ret = coordsC.rowwise().maxCoeff() - coordsC.rowwise().minCoeff();
        std::sort(ret.begin(), ret.end(), std::greater_equal<real>());
        return ret;
    }

    template <class TIn>
    std::pair<int, std::vector<index>> ToVTKVertsAndData(Element e, const TIn &vin)
    {
        std::pair<int, std::vector<index>> ret;
        switch (e.type)
        {
        case Line2:
        {
            ret.first = 3;
            ret.second = {vin[0], vin[1]};
        }
        break;
        case Line3:
        {
            ret.first = 4;
            ret.second = {vin[0], vin[2], vin[1]};
        }
        break;
        case Tri3:
        {
            ret.first = 5;
            ret.second = {vin[0], vin[1], vin[2]};
        }
        break;
        case Tri6:
        {
            ret.first = 22;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5]};
        }
        break;
        case Quad4:
        {
            ret.first = 9;
            ret.second = {vin[0], vin[1], vin[2], vin[3]};
        }
        break;
        case Quad9:
        {
            ret.first = 23;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5], vin[6], vin[7]};
        }
        break;
        case Tet4:
        {
            ret.first = 10;
            ret.second = {vin[0], vin[1], vin[2], vin[3]};
        }
        break;
        case Tet10:
        {
            ret.first = 24;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5], vin[6], vin[7], vin[8], vin[9]};
        }
        break;
        case Hex8:
        {
            ret.first = 12;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5], vin[6], vin[7]};
        }
        break;
        case Hex27:
        {
            ret.first = 25;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5], vin[6], vin[7],
                          vin[8], vin[9], vin[10], vin[11],
                          vin[16], vin[17], vin[18], vin[19],
                          vin[12], vin[13], vin[14], vin[15]};
        }
        break;
        case Prism6:
        {
            ret.first = 13;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5]};
        }
        break;
        case Prism18:
        {
            ret.first = 26;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4], vin[5],
                          vin[6], vin[7], vin[8],
                          vin[12], vin[13], vin[14],
                          vin[9], vin[10], vin[11]};
        }
        break;
        case Pyramid5:
        {
            ret.first = 14;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4]};
        }
        break;
        case Pyramid14:
        {
            ret.first = 27;
            ret.second = {vin[0], vin[1], vin[2], vin[3], vin[4],
                          vin[5], vin[6], vin[7], vin[8],
                          vin[9], vin[10], vin[11], vin[12]};
        }
        break;
        default:
            DNDS_assert(false);
        }
        return ret;
    }
}
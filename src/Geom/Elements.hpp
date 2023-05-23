#pragma once

#include <type_traits>
#include <cstdint>
#include "Eigen/Core"
#include <array>

namespace Geom::Elem
{
    /**
     * Complying to [CGNS Element standard](https://cgns.github.io/CGNS_docs_current/sids/conv.html)
     *  !note that we use 0 based indexing (CGNS uses 1 based in the link)
     *
     */
    using t_index = int32_t;
    const t_index invalid_index = INT32_MAX;
    using t_real = double;

    static_assert(std::is_signed_v<t_index>);

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

        //                                  d  o  nv nn nf
        ret[Line2] = std::array<t_index, 5>{1, 1, 2, 2, 0};
        ret[Line3] = std::array<t_index, 5>{1, 2, 2, 3, 0};

        ret[Tri3] = std::array<t_index, 5>{2, 1, 3, 3, 3};
        ret[Tri6] = std::array<t_index, 5>{2, 2, 3, 6, 3};
        ret[Quad4] = std::array<t_index, 5>{2, 1, 4, 4, 4};
        ret[Quad9] = std::array<t_index, 5>{2, 2, 4, 9, 4};

        ret[Tet4] = std::array<t_index, 5>{3, 1, 4, 4, 4};
        ret[Tet10] = std::array<t_index, 5>{3, 2, 4, 10, 4};
        ret[Hex8] = std::array<t_index, 5>{3, 1, 8, 8, 6};
        ret[Hex27] = std::array<t_index, 5>{3, 2, 8, 27, 6};
        ret[Prism6] = std::array<t_index, 5>{3, 1, 6, 6, 5};
        ret[Prism18] = std::array<t_index, 5>{3, 2, 6, 18, 5};
        ret[Pyramid5] = std::array<t_index, 5>{3, 1, 5, 5, 5};
        ret[Pyramid14] = std::array<t_index, 5>{3, 2, 5, 14, 5};
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
        real xi = p[0];
        real et = p[1];
        real zt = p[2];

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
    }

    struct Element
    {
        ElemType type;
    };

}
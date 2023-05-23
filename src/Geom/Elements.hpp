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

    

    Eigen::Array<t_index, 3, 3> a;
}
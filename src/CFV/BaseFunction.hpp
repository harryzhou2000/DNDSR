#pragma once

#include "DNDS/Defines.hpp"

namespace DNDS::CFV
{
    /// including up to 3 orders or diffs
    static const int ndiff = 3;
    static const int ndiffSiz = 20;
    static const int ndiffSiz2D = 10;
    static const int diffOperatorOrderList[ndiffSiz][3] =
        {
            //{diffOrderX_0, diffOrderX_1, diffOrder_X2} // indexPlace, diffSeq
            {0, 0, 0}, // 00
            {1, 0, 0}, // 01
            {0, 1, 0}, // 02
            {0, 0, 1}, // 03
            {2, 0, 0}, // 04
            {0, 2, 0}, // 05
            {0, 0, 2}, // 06
            {1, 1, 0}, // 07
            {0, 1, 1}, // 08
            {1, 0, 1}, // 09
            {3, 0, 0}, // 10
            {0, 3, 0}, // 11
            {0, 0, 3}, // 12
            {2, 1, 0}, // 13
            {1, 2, 0}, // 14
            {0, 2, 1}, // 15
            {0, 1, 2}, // 16
            {1, 0, 2}, // 17
            {2, 0, 1}, // 18
            {1, 1, 1}, // 19
    };
    static const int diffOperatorOrderList2D[ndiffSiz2D][3] = {
        {0, 0, 0}, // 00 00
        {1, 0, 0}, // 01 01 0
        {0, 1, 0}, // 02 02 1
        {2, 0, 0}, // 03 04 00
        {1, 1, 0}, // 04 05 01
        {0, 2, 0}, // 05 07 11
        {3, 0, 0}, // 06 10 000
        {2, 1, 0}, // 07 11 001
        {1, 2, 0}, // 08 13 011
        {0, 3, 0}, // 09 16 111
    };
    static const int dFactorials[ndiff + 1][ndiff + 1] = {
        {1, 0, 0, 0},
        {1, 1, 0, 0},
        {1, 2, 2, 0},
        {1, 3, 6, 6}};

    static const int factorials[ndiff * 3 + 1] = {
        1,
        1,
        1 * 2,
        1 * 2 * 3,
        1 * 2 * 3 * 4,
        1 * 2 * 3 * 4 * 5,
        1 * 2 * 3 * 4 * 5 * 6,
        1 * 2 * 3 * 4 * 5 * 6 * 7,
        1 * 2 * 3 * 4 * 5 * 6 * 7 * 8,
        1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9,
    };
    static const int diffNCombs2D[ndiffSiz2D]{
        1, 1, 1, 1, 2, 1, 1, 3, 3, 1};

    inline real iPow(int p, real x)
    {
        switch (p)
        {
        case 0:
            return 1.;
        case 1:
            return x;
        case 2:
            return x * x;
        case 3:
            return x * x * x;
        default:
            return 1e300;
            break;
        }
    }

    template <int dim, int order>
    inline constexpr int PolynomialNDOF() //  2-d specific
    {
        switch (dim)
        {
        case 2:
        {
            switch (order)
            {
            case 0:
                return 1;
            case 1:
                return 3;
            case 2:
                return 6;
            case 3:
                return 10;
            default:
                return -1;
            }
        }
        break;
        case 3:
        {
            switch (order)
            {
            case 0:
                return 1;
            case 1:
                return 4;
            case 2:
                return 10;
            case 3:
                return 20;
            default:
                return -1;
            }
        }
        break;
        default:
            return -1;
        }
    }

    inline static real FPolynomial3D(int px, int py, int pz, int dx, int dy, int dz, real x, real y, real z)
    {
        int c = dFactorials[px][dx] * dFactorials[py][dy] * dFactorials[pz][dz];
        return c ? c * iPow(px - dx, x) * iPow(py - dy, y) * iPow(pz - dz, z) : 0.;
        // return c ? c * std::pow(x, px - dx) * std::pow(y, py - dy) * std::pow(z, pz - dz) : 0.;
    }
}
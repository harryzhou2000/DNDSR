#include "DNDS/Array.hpp"

int main()
{
    using namespace DNDS;
    Array<real, 3> a1;//staticFixed
    Array<real, DynamicSize> b1;//fixed
    Array<real, NonUniformSize, 3> c1; //staticMax
    Array<real, NonUniformSize, DynamicSize> d1;//Max
    Array<real, NonUniformSize> e1;

    a1.Resize(4);

    b1.Resize(4, 3);

    assert(c1.RowSizeMax() == 3);
    c1.Resize(4);
    c1.ResizeRow(0, 3);
    c1.ResizeRow(1, 3);
    c1.ResizeRow(2, 3);
    c1.ResizeRow(3, 3);

    d1.Resize(4, 3);
    assert(d1.RowSizeMax() == 3);
    d1.ResizeRow(0, 3);
    d1.ResizeRow(1, 3);
    d1.ResizeRow(2, 3);
    d1.ResizeRow(3, 3);

    e1.Resize(4);
    e1.ResizeRow(0, 3);
    e1.ResizeRow(1, 3);
    e1.ResizeRow(2, 3);
    e1.ResizeRow(3, 3);
    e1.Compress();

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            a1(i, j) = i + j;
            b1(i, j) = i + j;
            c1(i, j) = i + j;
            d1(i, j) = i + j;
            e1(i, j) = i + j;
        }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            a1(i, j) = i + j;
            b1(i, j) = i + j;
            c1(i, j) = i + j;
            d1(i, j) = i + j;
            e1(i, j) = i + j;
        }

    std::cout << a1 << std::endl;
    std::cout << b1 << std::endl;
    std::cout << c1 << std::endl;
    std::cout << d1 << std::endl;
    std::cout << e1 << std::endl;

    return 0;
}
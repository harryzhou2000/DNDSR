#include <iostream>
#include <vector>
#include <map>

#include <algorithm>
#include <execution>

std::vector<double> argD;

void testParVector()
{
    size_t size = 1024 * 1024 * 1024;
    if (argD.size() >= 1)
        size = argD[0];

    std::vector<double> vec(size), vec2(size);
    std::for_each(std::execution::par, vec.begin(), vec.end(),
                  [&](decltype(vec)::value_type &v)
                  {
                      v = 2;
                      v = v + 3;
                      v = v * double(size);
                      v = v + double((ptrdiff_t)(&vec));
                  });
    // std::sort(std::execution::par, vec.begin(), vec.end());
    double sum{0};
    std::for_each(std::execution::par_unseq, vec.begin(), vec.end(),
                  [&](decltype(vec)::value_type &v)
                  {
                      sum += v;
                  });
    std::cout << sum << std::endl;
}

int main(int argc, char *argv[])
{

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }
    testParVector();

    return 0;
}
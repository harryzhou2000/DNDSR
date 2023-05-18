#include "Eigen/Dense"
#include "DNDS/Defines.hpp"

int main()
{
    using namespace DNDS;
    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> A;
    // A.setRandom(5,5);
    A.resize(5, 5);
    A << 1, 2, 3, 4, 5,
        1, 1, 3, 4, 6,
        1, 3, 3, 4, 6,
        1, 2, 5, 4, 6,
        1, 2, 3, 1, 6;
    std::cout << A << std::endl;
    std::cout << "Ainv" << std::endl;
    Eigen::MatrixXd AI = A.colPivHouseholderQr().inverse();
    std::cout << AI << std::endl;
    std::cout << AI * A << std::endl;
    std::cout << "Test Meta:\n";
    std::cout << std::is_trivial_v<Eigen::Matrix3d> << std::endl;
    std::cout << std::is_trivial_v<Eigen::Matrix3Xd> << std::endl;
    std::cout << std::is_trivial_v<Eigen::Matrix<real, -1, -1, Eigen::ColMajor, 3, 3>> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix3d> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix3Xd> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix<real, -1, -1, Eigen::ColMajor, 3, 3>> << std::endl;
    std::cout << std::is_trivially_copyable_v<std::array<real, 5>> << std::endl;
    std::cout << "Test Size:\n";
    std::cout << sizeof(Eigen::Matrix<double, 5, 5>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, Eigen::Dynamic>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, Eigen::Dynamic, Eigen::DontAlign, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Map<Eigen::Matrix<double, 5, 5>>) << std::endl;
    std::cout << sizeof(Eigen::Map<Eigen::Matrix<double, 100, 100>>) << std::endl;

    return 0;
}
#pragma once
// #define NDEBUG
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <tuple>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <type_traits>

#include "Macros.hpp"
#include "Experimentals.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"//?It seems Mat.determinant() would be undefined rather than undeclared...

static const std::string DNDS_Defines_state =
    std::string("DNDS_Defines ") + DNDS_Macros_State + DNDS_Experimentals_State
#ifdef NDEBUG
    + " NDEBUG "
#else
    + " (no NDEBUG) "
#endif
#ifdef NINSERT
    + " NINSERT "
#else
    + " (no NINSERT) "
#endif
    ;

/***************/ // DNDS_assertS

inline void __DNDS_assert_false(const char *expr, const char *file, int line)
{
    std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]" << std::endl;
    std::abort();
}

inline void __DNDS_assert_false_info(const char *expr, const char *file, int line, const char *info)
{
    std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n"
              << info << std::endl;
    std::abort();
}

#ifdef DNDS_NDEBUG
#define DNDS_assert(expr)
#define DNDS_assert_info(expr, info)
#else
#define DNDS_assert(expr)    \
    (static_cast<bool>(expr) \
         ? void(0)           \
         : __DNDS_assert_false(#expr, __FILE__, __LINE__))
#define DNDS_assert_info(expr, info) \
    (static_cast<bool>(expr)         \
         ? void(0)                   \
         : __DNDS_assert_false_info(#expr, __FILE__, __LINE__, info))
#endif

/***************/

static_assert(sizeof(uint8_t) == 1, "bad uint8_t");

namespace DNDS
{
    typedef double real;
    typedef int64_t index;
    typedef int32_t rowsize;
    typedef int64_t real_sized_index;
    typedef int32_t real_half_sized_index;
    static_assert(sizeof(real_sized_index) == sizeof(real) && sizeof(real_half_sized_index) == sizeof(real) / 2);

#define DNDS_INDEX_MAX INT64_MAX
#define DNDS_INDEX_MIN INT64_MIN
#define DNDS_ROWSIZE_MAX INT32_MAX
#define DNDS_ROWSIZE_MIN INT32_MIN

    static const char *outputDelim = "\t";

    typedef std::vector<rowsize> t_RowsizeVec;
    typedef std::vector<index> t_IndexVec;
    typedef std::shared_ptr<t_IndexVec> t_pIndexVec;

    typedef std::tuple<index, index> t_indexerPair;

    const index indexMin = INT64_MIN;

    const real UnInitReal = std::acos(-1) * 1e299 * std::sqrt(-1.0);

    const real veryLargeReal = 3e200;
    const real largeReal = 3e10;
    const real verySmallReal = 1e-200;
    const real smallReal = 1e-10;

    const real pi = std::acos(-1);

    typedef Eigen::Matrix<real, -1, -1, Eigen::RowMajor> tDiFj;

} // namespace DNDS

namespace DNDS
{
    const rowsize DynamicSize = -1;
    const rowsize NonUniformSize = -2;
    static_assert(DynamicSize != NonUniformSize, "DynamicSize, NonUniformSize definition conflict");

    inline constexpr int RowSize_To_EigenSize(rowsize rs)
    {
        return rs >= 0 ? static_cast<int>(rs) : (rs == DynamicSize || rs == NonUniformSize ? Eigen::Dynamic : INT_MIN);
    }
}

namespace DNDS
{
    extern std::ostream *logStream;

    extern bool useCout;

    std::ostream &log();

    void setLogStream(std::ostream *nstream);
}

/*









*/

namespace DNDS
{
    // Note that TtIndexVec being accumulated could overflow
    template <class TtRowsizeVec, class TtIndexVec>
    inline void AccumulateRowSize(const TtRowsizeVec &rowsizes, TtIndexVec &rowstarts)
    {
        static_assert(std::is_signed_v<typename TtIndexVec::value_type>, "row starts should be signed");
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (typename TtIndexVec::size_type i = 1; i < rowstarts.size(); i++)
        {
            rowstarts[i] = rowstarts[i - 1] + rowsizes[i - 1];
            DNDS_assert(rowstarts[i] >= 0);
        }
    }

    template <class T>
    inline bool checkUniformVector(const std::vector<T> &dat, T &value)
    {
        if (dat.size() == 0)
            return false;
        value = dat[0];
        for (auto i = 1; i < dat.size(); i++)
            if (dat[i] != value)
                return false;
        return true;
    }

    template <class T, class TP = T>
    inline void PrintVec(const std::vector<T> &dat, std::ostream &out)
    {
        for (auto i = 0; i < dat.size(); i++)
            out << TP(dat[i]) << outputDelim;
    }

    /// \brief l must be non-negative, r must be positive. integers
    template <class TL, class TR>
    inline constexpr auto divCeil(TL l, TR r)
    {
        return l / r + (l % r) ? 1 : 0;
    }
}

/*









*/

namespace DNDS
{
    template <typename T>
    inline constexpr T sqr(const T &a)
    {
        static_assert(std::is_arithmetic_v<T>, "need arithmetic");
        return a * a;
    }

    inline constexpr real sign(real a)
    {
        return a > 0 ? 1 : (a < 0 ? -1 : 0);
    }

    inline constexpr real signP(real a)
    {
        return a >= 0 ? 1 : -1;
    }

    inline constexpr real signM(real a)
    {
        return a <= 0 ? -1 : 1;
    }

    template <typename T>
    inline constexpr T mod(T a, T b)
    {
        static_assert(std::is_signed<T>::value && std::is_integral<T>::value, "not legal mod type");
        T val = a % b;
        if (val < 0)
            val += b;
        return val;
    }

    template <typename T>
    inline constexpr T divide_ceil(T a, T b)
    {
        static_assert(std::is_integral<T>::value, "not legal mod type");
        return a / b + (a % b ? 1 : 0);
    }
    // const int a = divide_ceil(23, 11);

}

/*-----------------------------------------*/
// some meta-programming utilities
namespace DNDS
{
    namespace Meta
    {
        template <class T>
        struct is_std_array : std::false_type
        {
        };

        template <class T, size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type
        {
        };

        template <typename _Tp>
        inline constexpr bool is_std_array_v = is_std_array<_Tp>::value;

        static_assert(is_std_array_v<std::array<real, 5>> && (!is_std_array_v<std::vector<real>>)); // basic test

        /**
         * @brief see if the Actual valid data is in the struct scope (memcpy copyable)
         * @details
         * generally a fixed size Eigen::Matrix, but it seems std::is_trivially_copyable_v<> does not distinguish that,
         * see https://eigen.tuxfamily.org/dox/classEigen_1_1Matrix.html, ABI part
         * ```
         * static_assert(!is_fixed_data_real_eigen_matrix_v<std::array<real, 10>> &&
                         is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, 2, 2>> &&
                         !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, 2>> &&
                         is_fixed_data_real_eigen_matrix_v<Eigen::Vector2d> &&
                         !is_fixed_data_real_eigen_matrix_v<Eigen::Vector2f> &&
                         is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, 2, 2>> &&
                         !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, -1, 2>> &&
                         !is_fixed_data_real_eigen_matrix_v<Eigen::MatrixXd>,
                     "is_fixed_data_real_eigen_matrix_v bad");
         * ```
         *
         * @tparam T
         */
        template <class T>
        struct is_fixed_data_real_eigen_matrix
        {
            static constexpr bool value = false;
        };

        template <class T, int M, int N, int options, int max_m, int max_n>
        struct is_fixed_data_real_eigen_matrix<Eigen::Matrix<T, M, N, options, max_m, max_n>>
        {
            static constexpr bool value = std::is_same_v<real, T> &&
                                          ((M > 0 && N > 0) ||
                                           (max_m > 0 && max_n > 0));
        };

        template <typename _Tp>
        inline constexpr bool is_fixed_data_real_eigen_matrix_v = is_fixed_data_real_eigen_matrix<_Tp>::value;

        static_assert(!is_fixed_data_real_eigen_matrix_v<std::array<real, 10>> &&
                          is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, 2, 2>> &&
                          !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, 2>> &&
                          is_fixed_data_real_eigen_matrix_v<Eigen::Vector2d> &&
                          !is_fixed_data_real_eigen_matrix_v<Eigen::Vector2f> &&
                          is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, 2, 2>> &&
                          !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, -1, 2>> &&
                          !is_fixed_data_real_eigen_matrix_v<Eigen::MatrixXd>,
                      "is_fixed_data_real_eigen_matrix_v bad");
    }
}

/*









*/

/*------------------------------------------*/
// Warning disabler:

#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable \
                                                        : warningNumber))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING("-Wunused-parameter")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING("-Wunused-function")
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS DISABLE_WARNING("-Wdeprecated-declarations")
// other warnings you want to deactivate...

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate...

#endif

/*------------------------------------------*/
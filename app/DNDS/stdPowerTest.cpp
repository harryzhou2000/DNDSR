#include "DNDS/Profiling.hpp"
#include "DNDS/MPI.hpp"
#include "fmt/core.h"
#include <algorithm>
#include <random>

namespace DNDS
{
    template <typename T = double>
    class IntervalRecorder
    {
        Eigen::Vector<double, Eigen::Dynamic> intervals;
        Eigen::Vector<T, Eigen::Dynamic> data;

    public:
        void SetIntervals(const Eigen::Vector<double, Eigen::Dynamic> &newIntervals)
        {
            intervals = newIntervals;
            DNDS_assert(std::is_sorted(intervals.begin(), intervals.end()));
            ;
            data.resize(intervals.size() + 1);
            data.setZero();
        }

        const Eigen::Vector<double, Eigen::Dynamic> &getIntervals() const
        {
            return intervals;
        }

        const Eigen::Vector<T, Eigen::Dynamic> &getData() const
        {
            return data;
        }

        T &GetDataAt(double v)
        {
            auto found = std::lower_bound(intervals.begin(), intervals.end(), v, std::less_equal<double>());
            return data(found - intervals.begin());
        }

        /**
         * \warning this must sync call
         */
        void AccumulateCrossProc(const MPIInfo &mpi)
        {
            Eigen::Vector<T, Eigen::Dynamic> dataAll = data;
            MPI::Allreduce(data.data(), dataAll.data(), data.size() * BasicType_To_MPIIntType<T>().second,
                           BasicType_To_MPIIntType<T>().first,
                           MPI_SUM, mpi.comm);
            data = dataAll;
        }
    };

    double TestPow(const MPIInfo &mpi,
                   double start,
                   double end,
                   double gStart,
                   double gEnd,
                   int nRep = 1000, int nSamp = 10000, int nBin = 1000, int seed = 0)
    {
        if (mpi.rank == 0)
        {
            std::cout << fmt::format("[{},{}], gen[{},{}], nRep={}, nSamp={}, nBin={}, seed={}, nProc={}",
                                     start, end, gStart, gEnd, nRep, nSamp, nBin, seed, mpi.size)
                      << std::endl;
        }

        // std::random_device randomDevice;
        // auto gen = seed ? std::mt19937_64(seed) : std::mt19937_64(randomDevice);
        auto gen = std::mt19937_64(seed);
        std::uniform_real_distribution<double> uniformDist(gStart, gEnd);
        IntervalRecorder<uint64_t> countRecorder;
        IntervalRecorder<double> timeRecorder;
        auto intervals = Eigen::Vector<double, Eigen::Dynamic>::LinSpaced(nBin + 1, start, end);
        countRecorder.SetIntervals(intervals);
        timeRecorder.SetIntervals(intervals);

        double vAdd = 0;
        for (int iSamp = 0; iSamp < nSamp; iSamp++)
        {
            double value = uniformDist(gen);
            countRecorder.GetDataAt(value) += nRep;
            PerformanceTimer::Instance().clearTimer(PerformanceTimer::RHS);
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::RHS);
            for (int iRep = 0; iRep < nRep; iRep++)
            {
                vAdd += std::pow(value, 1.5);
            }
            PerformanceTimer::Instance().StopTimer(PerformanceTimer::RHS);
            timeRecorder.GetDataAt(value) += PerformanceTimer::Instance().getTimer(PerformanceTimer::RHS);
            if (iSamp % 100 == 0)
                if (mpi.rank == 0)
                {
                    std::cout << fmt::format(" iSamp/nSamp [{}, {}]", iSamp, nSamp) << std::endl;
                }
        }
        countRecorder.AccumulateCrossProc(mpi);
        timeRecorder.AccumulateCrossProc(mpi);
        if (mpi.rank == 0)
        {
            std::cout << "------------------- intervals ---------------------" << std::endl;
            std::cout << intervals.transpose() << std::endl;
            std::cout << "------------------- counts ---------------------" << std::endl;
            std::cout << countRecorder.getData().transpose() << std::endl;
            std::cout << "------------------- timeMean ---------------------" << std::endl;
            std::cout << (timeRecorder.getData().array() / countRecorder.getData().array().cast<double>()).transpose()
                      << std::endl;

            std::cout << std::endl;
            std::cout << std::scientific << std::setprecision(16)
                      << "data = [...\n"
                      << intervals.transpose() << "\n"
                      << countRecorder.getData().transpose()(Eigen::seq(1, Eigen::last)) << "\n"
                      << (timeRecorder.getData().array() / countRecorder.getData().array().cast<double>()).transpose()(Eigen::seq(1, Eigen::last))
                      << "\n"
                      << "];" << std::endl;
        }
        return vAdd;
    }

}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    DNDS_assert(argc >= 8);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto ret = DNDS::TestPow(mpi,
                             std::stod(argv[1]), std::stod(argv[2]),
                             std::stod(argv[3]), std::stod(argv[4]),
                             std::stoi(argv[5]), std::stoi(argv[6]),
                             std::stoi(argv[7]), argc >= 9 ? std::stoi(argv[8]) : 0);
    MPI_Finalize();
    return ret > 0 ? 0 : 1;
}

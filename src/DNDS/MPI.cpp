
#include <ctime>
#include <cstdio>
#include <cstdlib>

#include "MPI.hpp"

#ifdef NDEBUG
#define NDEBUG_DISABLED
#undef NDEBUG
#endif

namespace DNDS
{

#include <iostream>
#if defined(linux) || defined(_UNIX) || defined(__linux__)
#include <sys/ptrace.h>
#include <unistd.h>
#include <sys/stat.h>
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#include <Windows.h>
#include <process.h>
#endif
    namespace Debug
    {
        bool IsDebugged()
        {

#if defined(linux) || defined(_UNIX) || defined(__linux__)
            std::ifstream fin("/proc/self/status"); // able to detect gdb
            std::string buf;
            int tpid = 0;
            while (!fin.eof())
            {
                fin >> buf;
                if (buf == "TracerPid:")
                {
                    fin >> tpid;
                    break;
                }
            }
            fin.close();
            return tpid != 0;
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
            return IsDebuggerPresent();
#endif
        }

        void MPIDebugHold(const MPIInfo &mpi)
        {
#if defined(linux) || defined(_UNIX) || defined(__linux__)
            MPISerialDo(mpi, [&]
                        { log() << "Rank " << mpi.rank << " PID: " << getpid() << std::endl; });
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
            MPISerialDo(mpi, [&]
                        { log() << "Rank " << mpi.rank << " PID: " << _getpid() << std::endl; });
#endif
            int holdFlag = 1;
            while (holdFlag)
            {
                for (MPI_int ir = 0; ir < mpi.size; ir++)
                {
                    int newDebugFlag;
                    if (mpi.rank == ir)
                    {
                        newDebugFlag = int(IsDebugged());
                        MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);
                    }
                    else
                        MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);

                    // std::cout << "DBG " << newDebugFlag;
                    if (newDebugFlag)
                        holdFlag = 0;
                }
            }
        }
    }
}

namespace DNDS
{
    MPIBufferHandler &MPIBufferHandler::Instance()
    {
        static MPIBufferHandler instance;
        return instance;
    }
}

namespace DNDS
{
    std::string getTimeStamp(const MPIInfo &mpi)
    {
        int64_t result = static_cast<int64_t>(std::time(nullptr));
        char bufTime[512];
        char buf[512 + 32];
        int64_t pid = 0;
#if defined(linux) || defined(_UNIX) || defined(__linux__)
        // pid = Debug::getpid();
        pid = getpid();
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
        // pid = Debug::GetCurrentProcessId();
        pid = GetCurrentProcessId();
#endif
        MPI_Bcast(&result, 1, MPI_INT64_T, 0, mpi.comm);
        MPI_Bcast(&pid, 1, MPI_INT64_T, 0, mpi.comm);

        time_t time_result = static_cast<time_t>(result);

        std::strftime(bufTime, 512, "%F_%H-%M-%S", std::localtime(&time_result));

        long pidc = static_cast<long>(pid);
        std::sprintf(buf, "%s_%ld", bufTime, pidc);

        return std::string(buf);
    }
}

namespace DNDS::MPI
{
    CommStrategy::CommStrategy()
    {
        {
            auto ret = std::getenv("DNDS_ARRAY_STRATEGY_USE_IN_SITU");
            if (ret != NULL && (std::stoi(ret) != 0))
            {
                _array_strategy = InSituPack;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_ARRAY_STRATEGY_USE_IN_SITU, setting" << std::endl;
                MPI_Barrier(mpi.comm);
            }
        }

        {
            auto ret = std::getenv("DNDS_USE_STRONG_SYNC_WAIT");
            if (ret != NULL && (std::stoi(ret) != 0))
            {
                _use_strong_sync_wait = true;
                auto mpi = MPIInfo();
                mpi.setWorld();
                if (mpi.rank == 0)
                    log() << "Detected DNDS_USE_STRONG_SYNC_WAIT, setting" << std::endl;
                MPI_Barrier(mpi.comm);
            }
        }
    }

    CommStrategy &CommStrategy::Instance()
    {
        static CommStrategy strategy;
        return strategy;
    }

    CommStrategy::ArrayCommType CommStrategy::GetArrayStrategy()
    {
        return _array_strategy;
    }

    void CommStrategy::SetArrayStrategy(CommStrategy::ArrayCommType t)
    {
        _array_strategy = t;
    }

    bool CommStrategy::GetUseStrongSyncWait()
    {
        return _use_strong_sync_wait;
    }
}

#ifdef NDEBUG_DISABLED
#define NDEBUG
#undef NDEBUG_DISABLED
#endif

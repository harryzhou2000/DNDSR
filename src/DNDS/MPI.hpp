#pragma once

#include <vector>
#include <fstream>

#include <mutex>

#include "Defines.hpp"
DISABLE_WARNING_PUSH
// disable mpicxx 's many warnings in intel oneAPI mpi's header
DISABLE_WARNING_UNUSED_VALUE
#include "mpi.h"
DISABLE_WARNING_POP

#ifdef NDEBUG
#define NDEBUG_DISABLED
#undef NDEBUG
#endif

namespace DNDS
{

    using MPI_int = int;
    using MPI_index = MPI_Aint;
#define MAX_MPI_int INT32_MAX
#define MAX_MPI_Aint INT64_MAX
    static_assert(sizeof(MPI_Aint) == 8);

    using tMPI_sizeVec = std::vector<MPI_int>;
    using tMPI_intVec = tMPI_sizeVec;
    using tMPI_indexVec = std::vector<MPI_index>;
    using tMPI_AintVec = tMPI_indexVec;

    using tMPI_statVec = std::vector<MPI_Status>;
    using tMPI_reqVec = std::vector<MPI_Request>;

    /**
     * \brief maps index or other DNDS types to MPI_Datatype ids
     */
    template <class Tbasic>
    constexpr MPI_Datatype __DNDSToMPITypeInt()
    {
        static_assert(sizeof(Tbasic) == 8 || sizeof(Tbasic) == 4, "DNDS::Tbasic is not right size");
        return sizeof(Tbasic) == 8 ? MPI_INT64_T : (sizeof(Tbasic) == 4 ? MPI_INT32_T : MPI_DATATYPE_NULL);
    }

    template <class Tbasic>
    constexpr MPI_Datatype __DNDSToMPITypeFloat()
    {
        static_assert(sizeof(Tbasic) == 8 || sizeof(Tbasic) == 4, "DNDS::Tbasic is not right size");
        return sizeof(Tbasic) == 8 ? MPI_REAL8 : (sizeof(Tbasic) == 4 ? MPI_REAL4 : MPI_DATATYPE_NULL);
    }

    const MPI_Datatype DNDS_MPI_INDEX = __DNDSToMPITypeInt<index>();
    const MPI_Datatype DNDS_MPI_REAL = __DNDSToMPITypeFloat<real>();

    template <class T> // TODO: see if an array is bounded
    //! Warning, not const-expr since OpenMPI disallows it
    std::pair<MPI_Datatype, MPI_int> BasicType_To_MPIIntType()
    {
        static const auto badReturn = std::make_pair(MPI_Datatype(MPI_DATATYPE_NULL), MPI_int(-1));
        if constexpr (std::is_scalar_v<T>)
        {
            if constexpr (std::is_same_v<T, float>)
                return std::make_pair(MPI_Datatype(MPI_FLOAT), MPI_int(1));
            if constexpr (std::is_same_v<T, double>)
                return std::make_pair(MPI_Datatype(MPI_DOUBLE), MPI_int(1));
            if constexpr (std::is_same_v<T, long double>)
                return std::make_pair(MPI_Datatype(MPI_LONG_DOUBLE), MPI_int(1));

            if constexpr (std::is_same_v<T, int8_t>)
                return std::make_pair(MPI_Datatype(MPI_INT8_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int16_t>)
                return std::make_pair(MPI_Datatype(MPI_INT16_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int32_t>)
                return std::make_pair(MPI_Datatype(MPI_INT32_T), MPI_int(1));
            if constexpr (std::is_same_v<T, int64_t>)
                return std::make_pair(MPI_Datatype(MPI_INT64_T), MPI_int(1));

            if constexpr (sizeof(T) == 1)
                return std::make_pair(MPI_Datatype(MPI_UINT8_T), MPI_int(1));
            else if constexpr (sizeof(T) == 2)
                return std::make_pair(MPI_Datatype(MPI_UINT16_T), MPI_int(1));
            else if constexpr (sizeof(T) == 4)
                return std::make_pair(MPI_Datatype(MPI_UINT32_T), MPI_int(1));
            else if constexpr (sizeof(T) == 8)
                return std::make_pair(MPI_Datatype(MPI_UINT64_T), MPI_int(1));
            else
                return badReturn;
        }
        else if constexpr (std::is_array_v<T>)
        {
            std::pair<MPI_Datatype, MPI_int> SizCom = BasicType_To_MPIIntType<std::remove_extent_t<T>>();
            return std::make_pair(SizCom.first, SizCom.second * std::extent_v<T>);
        }
        else if constexpr (std::is_trivially_copyable_v<T>)
        {
            if constexpr (Meta::is_std_array_v<T>)
                return std::make_pair(
                    BasicType_To_MPIIntType<typename T::value_type>().first,
                    BasicType_To_MPIIntType<typename T::value_type>().second * T().size());
            else
                return badReturn;
        }
        else if constexpr (Meta::is_fixed_data_real_eigen_matrix_v<T>)
            return std::make_pair(DNDS_MPI_REAL, MPI_int(divide_ceil(sizeof(T), sizeof(real))));
        else
            return badReturn;
    }

    struct MPIInfo
    {
        MPI_Comm comm = MPI_COMM_NULL;
        int rank = -1;
        int size = -1;

        void setWorld()
        {
            comm = MPI_COMM_WORLD;
            int ierr;
            ierr = MPI_Comm_rank(comm, &rank), DNDS_assert(ierr == MPI_SUCCESS);
            ierr = MPI_Comm_size(comm, &size), DNDS_assert(ierr == MPI_SUCCESS);
        }

        bool operator==(const MPIInfo &r) const
        {
            return (comm == r.comm) && (rank == r.rank) && (size == r.size);
        }
    };

    inline MPI_int MPIWorldSize()
    {
        MPI_int ret{0};
        MPI_Comm_size(MPI_COMM_WORLD, &ret);
        return ret;
    }

    inline MPI_int MPIWorldRank()
    {
        MPI_int ret{0};
        MPI_Comm_rank(MPI_COMM_WORLD, &ret);
        return ret;
    }

    std::string getTimeStamp(const MPIInfo &mpi);

    inline void InsertCheck(const MPIInfo &mpi, const std::string &info = "",
                            const std::string &FUNCTION = "", const std::string &FILE = "", int LINE = -1)
    {
#if !(defined(NDEBUG) || defined(NINSERT))
        MPI::Barrier(mpi.comm);
        std::cout << "=== CHECK \"" << info << "\"  RANK " << mpi.rank << " ==="
                  << " @  FName: " << FUNCTION
                  << " @  Place: " << FILE << ":" << LINE << std::endl;
        MPI::Barrier(mpi.comm);
#endif
    }

#define DNDS_MPI_InsertCheck(mpi, info) \
    InsertCheck(mpi, info, __FUNCTION__, __FILE__, __LINE__)

    using tMPI_typePairVec = std::vector<std::pair<MPI_int, MPI_Datatype>>;
    /**
     * \brief wrapper of tMPI_typePairVec
     */
    struct MPITypePairHolder : public tMPI_typePairVec
    {
        using tMPI_typePairVec::tMPI_typePairVec;
        ~MPITypePairHolder()
        {
            for (auto &i : (*this))
                if (i.first >= 0 && i.second != 0 && i.second != MPI_DATATYPE_NULL)
                    MPI_Type_free(&i.second); //, std::cout << "Free Type" << std::endl;
        }
    };

    using tpMPITypePairHolder = ssp<MPITypePairHolder>;
    /**
     * \brief wrapper of tMPI_reqVec, so that the requests are freed automatically
     */
    struct MPIReqHolder : public tMPI_reqVec
    {
        using tMPI_reqVec::tMPI_reqVec;
        ~MPIReqHolder()
        {
            for (auto &i : (*this))
                if (i != MPI_REQUEST_NULL)
                    MPI_Request_free(&i); //, std::cout << "Free Req" << std::endl;
        }
        void clear()
        {
            for (auto &i : (*this))
                if (i != MPI_REQUEST_NULL)
                    MPI_Request_free(&i); //, std::cout << "Free Req" << std::endl;
            tMPI_reqVec::clear();
        }
    };

}

namespace DNDS::Debug
{
    bool IsDebugged();
    void MPIDebugHold(const MPIInfo &mpi);
    extern bool isDebugging;
}

// DNDS_assert_info_mpi is used to help barrier the process before exiting if DNDS::Debug::isDebugging is set
// remember to set a breakpoint here
void __DNDS_assert_false_info_mpi(const char *expr, const char *file, int line, const std::string &info, const DNDS::MPIInfo &mpi);

#ifdef DNDS_NDEBUG
#define DNDS_assert_info_mpi(expr, mpi, info) (void(0))
#else
#define DNDS_assert_info_mpi(expr, mpi, info) \
    ((static_cast<bool>(expr))                \
         ? void(0)                            \
         : __DNDS_assert_false_info_mpi(#expr, __FILE__, __LINE__, info, mpi))
#endif

namespace DNDS // TODO: get a concurrency header
{
    extern std::mutex HDF_mutex;

    namespace MPI
    {
        inline MPI_int Init_thread(int *argc, char ***argv)
        {
            int init_flag{0};
            MPI_Initialized(&init_flag);
            if (init_flag)
                return 0;
            int provided_MPI_THREAD_LEVEL{0};
            int needed_MPI_THREAD_LEVEL = MPI_THREAD_MULTIPLE;

            auto *env = std::getenv("DNDS_DISABLE_ASYNC_MPI");
            if (env != NULL && (std::stod(env) != 0))
            {
                int ienv = std::stod(env);
                if (ienv >= 1)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_SERIALIZED;
                if (ienv >= 2)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_FUNNELED;
                if (ienv >= 3)
                    needed_MPI_THREAD_LEVEL = MPI_THREAD_SINGLE;
            }
            auto ret = MPI_Init_thread(argc, argv, needed_MPI_THREAD_LEVEL, &provided_MPI_THREAD_LEVEL);
            if (provided_MPI_THREAD_LEVEL < needed_MPI_THREAD_LEVEL)
            {
                printf("ERROR: The MPI library does not have full thread support\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            return ret;
        }

        inline int GetMPIThreadLevel()
        {
            int ret;
            int ierr;
            ierr = MPI_Query_thread(&ret), DNDS_assert(ierr == MPI_SUCCESS);
            return ret;
        }
    }
}

// MPI buffer handler
#define MPIBufferHandler_REPORT_CHANGE // for monitoring
namespace DNDS
{
    class MPIBufferHandler // cxx11 + thread-safe singleton
    {
    private:
        std::vector<uint8_t> buf;

    public:
        using size_type = decltype(buf)::size_type;

    private:
        size_type claimed = 0;

    private:
        MPIBufferHandler()
        {
            uint8_t *obuf;
            int osize;
            MPI_Buffer_detach(&obuf, &osize);

            buf.resize(1024ULL * 1024ULL);
            MPI_Buffer_attach(buf.data(), int(buf.size())); //! warning, bufsize could overflow
        }
        MPIBufferHandler(const MPIBufferHandler &);
        MPIBufferHandler &operator=(const MPIBufferHandler &);

    public:
        static MPIBufferHandler &Instance();
        MPI_int size()
        {
            DNDS_assert(buf.size() <= MAX_MPI_int);
            return MPI_int(buf.size()); // could overflow!
        }
        void claim(MPI_Aint cs, int reportRank = 0)
        {
            if (buf.size() - claimed < static_cast<size_type>(cs))
            {
                // std::cout << "claim in " << std::endl;
                uint8_t *obuf;
                int osize;
                MPI_Buffer_detach(&obuf, &osize);
#ifdef MPIBufferHandler_REPORT_CHANGE
                std::cout << "MPIBufferHandler: New BUf at " << reportRank << std::endl
                          << osize << std::endl;
#endif
                DNDS_assert(static_cast<size_type>(osize) == buf.size());
                buf.resize(claimed + cs);
                MPI_Buffer_attach(buf.data(), buf.size());
#ifdef MPIBufferHandler_REPORT_CHANGE
                std::cout << " -> " << buf.size() << std::endl;
#endif
            }
            claimed += cs;
        }
        void unclaim(MPI_int cs)
        {
            DNDS_assert(claimed >= cs);
            claimed -= cs;
        }
        void *getBuf()
        {
            return (void *)(buf.data());
        }
    };

}

namespace DNDS::MPI
{
    MPI_int Bcast(void *buf, MPI_int num, MPI_Datatype type, MPI_int source_rank, MPI_Comm comm);

    MPI_int Alltoall(void *send, MPI_int sendNum, MPI_Datatype typeSend, void *recv, MPI_int recvNum, MPI_Datatype typeRecv, MPI_Comm comm);

    MPI_int Alltoallv(
        void *send, MPI_int *sendSizes, MPI_int *sendStarts, MPI_Datatype sendType,
        void *recv, MPI_int *recvSizes, MPI_int *recvStarts, MPI_Datatype recvType, MPI_Comm comm);

    MPI_int Allreduce(const void *sendbuf, void *recvbuf, MPI_int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    MPI_int Scan(const void *sendbuf, void *recvbuf, MPI_int count,
                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

    MPI_int Allgather(const void *sendbuf, MPI_int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, MPI_int recvcount,
                      MPI_Datatype recvtype, MPI_Comm comm);

    MPI_int Barrier(MPI_Comm comm);

    MPI_int BarrierLazy(MPI_Comm comm, uint64_t checkNanoSecs);

    MPI_int WaitallLazy(MPI_int count, MPI_Request *reqs, MPI_Status *statuses, uint64_t checkNanoSecs = 10000000);

    MPI_int WaitallAuto(MPI_int count, MPI_Request *reqs, MPI_Status *statuses);

    inline void AllreduceOneReal(real &v, MPI_Op op, const MPIInfo &mpi)
    {
        real vR{0};
        Allreduce(&v, &vR, 1, DNDS_MPI_REAL, op, mpi.comm);
        v = vR;
    }

    inline void AllreduceOneIndex(index &v, MPI_Op op, const MPIInfo &mpi)
    {
        index vR{0};
        Allreduce(&v, &vR, 1, DNDS_MPI_INDEX, op, mpi.comm);
        v = vR;
    }

}

namespace DNDS
{
    template <class F>
    inline void MPISerialDo(const MPIInfo &mpi, F f)
    { //! need some improvement: order could be bad?
        for (MPI_int i = 0; i < mpi.size; i++)
        {
            MPI::Barrier(mpi.comm);
            if (mpi.rank == i)
                f();
        }
    }
}

namespace DNDS::MPI
{
    /**
     * @brief // cxx11 + thread-safe singleton
     * must be constructed in MPI_COMM_WORLD
     *
     */
    class CommStrategy
    {
    public:
        enum ArrayCommType
        {
            UnknownArrayCommType = 0,
            HIndexed = 1,
            InSituPack = 2,
        };

        static const int Ntype = 10;

    private:
        ArrayCommType _array_strategy = HIndexed;
        bool _use_strong_sync_wait = false;
        bool _use_async_one_by_one = false;
        double _use_lazy_wait = 0;

        CommStrategy();
        CommStrategy(const CommStrategy &);
        CommStrategy &operator=(const CommStrategy &);

    public:
        static CommStrategy &Instance();
        ArrayCommType GetArrayStrategy();
        void SetArrayStrategy(ArrayCommType t);
        [[nodiscard]] bool GetUseStrongSyncWait() const;
        [[nodiscard]] bool GetUseAsyncOneByOne() const;
        [[nodiscard]] double GetUseLazyWait() const;
    };
}

#ifdef NDEBUG_DISABLED
#define NDEBUG
#undef NDEBUG_DISABLED
#endif

#pragma once

#include <vector>
#include <fstream>

#include "Defines.hpp"
#include "mpi.h"

#ifdef NDEBUG
#define NDEBUG_DISABLED
#undef NDEBUG
#endif

namespace DNDS
{

    typedef int MPI_int;
    typedef MPI_Aint MPI_index;
#define MAX_MPI_int INT32_MAX

    typedef std::vector<MPI_int> tMPI_sizeVec;
    typedef tMPI_sizeVec tMPI_intVec;
    typedef std::vector<MPI_index> tMPI_indexVec;
    typedef tMPI_indexVec tMPI_AintVec;

    typedef std::vector<MPI_Status> tMPI_statVec;
    typedef std::vector<MPI_Request> tMPI_reqVec;

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
            ierr = MPI_Comm_rank(comm, &rank);
            ierr = MPI_Comm_size(comm, &size);
        }

        bool operator==(const MPIInfo &r) const
        {
            return (comm == r.comm) && (rank == r.rank) && (size == r.size);
        }
    };

    std::string getTimeStamp(const MPIInfo &mpi);

    inline void InsertCheck(const MPIInfo &mpi, const std::string &info = "",
                            const std::string &FUNCTION = "", const std::string &FILE = "", int LINE = -1)
    {
#if !(defined(NDEBUG) || defined(NINSERT))
        MPI_Barrier(mpi.comm);
        std::cout << "=== CHECK \"" << info << "\"  RANK " << mpi.rank << " ==="
                  << " @  FName: " << FUNCTION
                  << " @  Place: " << FILE << ":" << LINE << std::endl;
        MPI_Barrier(mpi.comm);
#endif
    }

#define DNDS_MPI_InsertCheck(mpi, info) \
    InsertCheck(mpi, info, _FUNCTION_, __FILE__, __LINE__)

    template <class F>
    inline void MPISerialDo(const MPIInfo &mpi, F f)
    {
        for (MPI_int i = 0; i < mpi.size; i++)
        {
            MPI_Barrier(mpi.comm);
            if (mpi.rank == i)
                f();
        }
    }

    typedef std::vector<std::pair<MPI_int, MPI_Datatype>> tMPI_typePairVec;
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

    typedef std::shared_ptr<MPITypePairHolder> tpMPITypePairHolder;
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

namespace DNDS
{
    namespace Debug
    {
        bool IsDebugged();
        void MPIDebugHold(const MPIInfo &mpi);
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

            buf.resize(1024 * 1024);
            MPI_Buffer_attach(buf.data(), buf.size());
        };
        MPIBufferHandler(const MPIBufferHandler &);
        MPIBufferHandler &operator=(const MPIBufferHandler &);

    public:
        static MPIBufferHandler &Instance();
        MPI_int size()
        {
            DNDS_assert(buf.size() <= MAX_MPI_int);
            return buf.size();
        }
        void claim(MPI_int cs, int reportRank = 0)
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

namespace DNDS
{
    namespace MPI
    {
        /// @brief dumb wrapper
        template <class TData>
        MPI_int Bcast(TData *buf, MPI_int num, MPI_Datatype type, MPI_int source_rank, MPI_Comm comm)
        {
            return MPI_Bcast(buf, num, type, source_rank, comm);
        }

        template <class TData>
        MPI_int Alltoall(TData *send, MPI_int sendNum, MPI_Datatype typeSend, TData *recv, MPI_int recvNum, MPI_Datatype typeRecv, MPI_Comm comm)
        {
            return MPI_Alltoall(send, sendNum, typeSend, recv, recvNum, typeRecv, comm);
        }

        template <class TData>
        MPI_int Alltoallv(
            TData *send, MPI_int *sendSizes, MPI_int *sendStarts, MPI_Datatype sendType,
            TData *recv, MPI_int *recvSizes, MPI_int *recvStarts, MPI_Datatype recvType, MPI_Comm comm)
        {
            return MPI_Alltoallv(
                send, sendSizes, sendStarts, sendType,
                recv, recvSizes, recvStarts, recvType, comm);
        }
    }
}

#ifdef NDEBUG_DISABLED
#define NDEBUG
#undef NDEBUG_DISABLED
#endif
#pragma once

#include "Array.hpp"
#include "IndexMapping.hpp"
#include "Profiling.hpp"

namespace DNDS
{
    using t_pLGlobalMapping = std::shared_ptr<GlobalOffsetsMapping>;
    using t_pLGhostMapping = std::shared_ptr<OffsetAscendIndexMapping>;

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ParArray : public Array<T, _row_size, _row_max, _align>
    {
    public:
        using TArray = Array<T, _row_size, _row_max, _align>;
        using t_pArray = std::shared_ptr<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        using Array<T, _row_size, _row_max, _align>::Array;
        t_pLGlobalMapping pLGlobalMapping;
        MPIInfo mpi;

        void setMPI(const MPIInfo &n_mpi)
        {
            mpi = n_mpi;
            // checking if is uniform across all procs
        }

        void AssertConsistent()
        {
            if constexpr (_dataLayout == TABLE_Max ||
                          _dataLayout == TABLE_Fixed)
            {
                t_RowsizeVec uniformSizes(mpi.size);
                MPI_int rowsizeC = this->RowSizeField();
                static_assert(sizeof(MPI_int) == sizeof(rowsize));
                MPI_Allgather(&rowsizeC, 1, MPI_INT, uniformSizes.data(), 1, MPI_INT, mpi.comm);
                for (auto i : uniformSizes)
                    DNDS_assert_info(i == rowsizeC, "sizes not uniform across procs");
            }
        }

        void createGlobalMapping() // collective;
        {
            DNDS_assert_info(mpi.comm != MPI_COMM_NULL, "MPI unset");
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
            pLGlobalMapping->setMPIAlignBcast(mpi, this->Size());
        }
    };

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayTransformer
    {
    public:
        using TArray = ParArray<T, _row_size, _row_max, _align>;
        using t_pArray = std::shared_ptr<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        MPIInfo mpi;
        t_pLGhostMapping pLGhostMapping;
        t_pArray father;
        t_pArray son;

        t_pLGlobalMapping pLGlobalMapping; // reference from father

        std::shared_ptr<tMPI_typePairVec> pPushTypeVec;
        std::shared_ptr<tMPI_typePairVec> pPullTypeVec;

        // ** comm aux info: comm running structures **
        MPIReqHolder PushReqVec;
        MPIReqHolder PullReqVec;
        tMPI_statVec PushStatVec;
        tMPI_statVec PullStatVec;
        MPI_int pushSendSize;
        MPI_int pullSendSize;

        void setFatherSon(const t_pArray &n_father, const t_pArray &n_son)
        {
            father = n_father;
            son = n_son;
            mpi = father->mpi;
            DNDS_assert_info(son->mpi == father->mpi, "MPI inconsistent between father & son");
            pLGhostMapping.reset();
            pLGlobalMapping.reset();
        }

        template <class TRArrayTrans>
        void BorrowGGIndexing(const TRArrayTrans &RArrayTrans)
        {
            // DNDS_assert(father && Rarray.father); // Rarray's father is not visible...
            // DNDS_assert(father->obtainTotalSize() == Rarray.father->obtainTotalSize());
            DNDS_assert(RArrayTrans.pLGhostMapping && RArrayTrans.pLGlobalMapping);
            DNDS_assert(RArrayTrans.father->size() == father->size());
            pLGhostMapping = RArrayTrans.pLGhostMapping;
            pLGlobalMapping = RArrayTrans.pLGlobalMapping;
            father->pLGlobalMapping = RArrayTrans.pLGlobalMapping;
        }

        void createFatherGlobalMapping()
        {
            father->createGlobalMapping();
        }

        template <class TPullSet>
        void createGhostMapping(TPullSet &&pullingIndexGlobal) // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            // tMPI_intVec ghostSizes(mpi.size, 0); // == pulling sizes
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->Size(),
                std::forward<TPullSet>(pullingIndexGlobal),
                *pLGlobalMapping,
                mpi);
        }

        template <class TPushSet, class TPushStart>
        void createGhostMapping(TPushSet &&pushingIndexLocal, TPushStart &&pushStarts) // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: calculate over pushing
            // counting could overflow
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->size(),
                std::forward<TPushSet>(pushingIndexLocal),
                std::forward<TPushStart>(pushStarts),
                *pLGlobalMapping,
                mpi);
        }
#define ARRAY_COMM_USE_TYPE_HINDEXED
        /******************************************************************************************************************************/
        /**
         * \brief get real element byte info into account, with ghost indexer and comm types built, need two mappings
         * \pre has pLGlobalMapping pLGhostMapping
         * \post my indexer pPullTypeVec pPushTypeVec established
         */
        void createMPITypes() // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert(pLGlobalMapping && pLGhostMapping);
            father->Compress(); //! assure CSR is in compressed form
            // TODO: support comm for uncompressed: add working mode
            // TODO: support actual MAX arrays' size communicating: append comm types
            /*********************************************/ // starts to deal with actual byte sizes

            //*phase2.1: build push sizes and push disps
            index nSend = pLGhostMapping->pushingIndexGlobal.size();
            tMPI_intVec pushingSizes(nSend); // pushing sizes in bytes
#ifdef ARRAY_COMM_USE_TYPE_HINDEXED
            tMPI_AintVec pushingDisps(nSend); // pushing disps in bytes
#else
            tMPI_intVec pushingDisps(nSend); // pushing disps in bytes  //!hindexed is wrong
#endif
            DNDS_assert(pLGhostMapping->pushingIndexGlobal.size() < DNDS_INDEX_MAX);
            auto fatherDataStart = father->operator[](0);
            for (index i = 0; i < index(pLGhostMapping->pushingIndexGlobal.size()); i++)
            {
                MPI_int rank = -1;
                index loc = -1;
                bool found = pLGhostMapping->search(pLGhostMapping->pushingIndexGlobal[i], rank, loc);
                DNDS_assert_info(found && rank == -1, "must be at local main"); // must be at local main
                pushingDisps[i] = (father->operator[](loc) - father->operator[](0)) * sizeof(T);
                if constexpr (_dataLayout == CSR)
                    pushingSizes[i] = father->RowSizeField(loc) * sizeof(T);
                if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                    pushingSizes[i] = father->RowSize(loc) * sizeof(T);
                if constexpr (isTABLE_Fixed(_dataLayout))
                    pushingSizes[i] = father->RowSizeField() * sizeof(T);
            }
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;

            //*phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            // - legacy: indexer.buildAsGhostAlltoall(father->indexer, pushingSizes, *pLGhostMapping, mpi); // cascade from father
            auto do_son_resizing = [&]()
            {
                auto &LGhostMapping = *pLGhostMapping;
                index ghostArraySiz = LGhostMapping.ghostStart[LGhostMapping.ghostStart.size() - 1];
                DNDS_assert(mpi.size == LGhostMapping.ghostStart.size() - 1);
                if constexpr (_dataLayout == TABLE_StaticFixed)
                {
                    son->Resize(ghostArraySiz);
                    return;
                }
                if constexpr (_dataLayout == TABLE_Fixed)
                {
                    son->Resize(ghostArraySiz, father->RowSize()); // using father's row size
                    return;
                }
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    // and go on for non-uniform resizing
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    // and go on for non-uniform resizing
                }

                // obtain pulling sizes with pushing sizes
                tMPI_intVec pullingSizes(ghostArraySiz);
                MPI_Alltoallv(pushingSizes.data(), LGhostMapping.pushIndexSizes.data(), LGhostMapping.pushIndexStarts.data(), MPI_INT,
                              pullingSizes.data(), LGhostMapping.ghostSizes.data(), LGhostMapping.ghostStart.data(), MPI_INT,
                              mpi.comm);

                // std::cout << LGhostMapping.gStarts().size() << std::endl;
                if constexpr (_dataLayout == CSR)
                    son->Resize(ghostArraySiz, [&](index i)
                                { return pullingSizes[i] / sizeof(T); });
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / sizeof(T));
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / sizeof(T));
                }
                // is actually pulling disps, but is contiguous anyway

                // InsertCheck(mpi);
                // std::cout << mpi.rank << " VEC ";
                // PrintVec(pullingSizes, std::cout);
                // std::cout << std::endl;
                // InsertCheck(mpi);

                // note that Rowstart and pullingSizes are in bytes
                // pullingSizes is actual but Rowstart is before indexModder(), use indexModder[] to invert
            };
            do_son_resizing();

            // phase3: create and register MPI types of pushing and pulling
            if constexpr (isTABLE_Max(_dataLayout)) // convert back to real pushing sizes
            {
                for (auto &i : pushingSizes)
                    i = son->RowSizeField() * sizeof(T);
            }
            pPushTypeVec = std::make_shared<MPITypePairHolder>(0);
            pPullTypeVec = std::make_shared<MPITypePairHolder>(0);
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                // std::cout << "PN" << pushNumber << std::endl;
                if (pushNumber > 0)
                {

#ifdef ARRAY_COMM_USE_TYPE_HINDEXED
                    MPI_Aint
#else
                    MPI_int
#endif
                        *pPushDisps;
                    MPI_int *pPushSizes;
                    pPushDisps = pushingDisps.data() + pLGhostMapping->pushIndexStarts[r];
                    pPushSizes = pushingSizes.data() + pLGhostMapping->pushIndexStarts[r];
                    // std::cout <<mpi.rank<< " pushSlice " << pPushDisps[0] << outputDelim << pPushSizes[0] << std::endl;

                    // if (mpi.rank == 0)
                    // {
                    //     std::cout << "pushing to " << r << "  size" << pushNumber << "\n";
                    //     for (int i = 0; i < pushNumber; i++)
                    //         std::cout << "b[" << i << "] = " << pPushSizes[i] << std::endl;
                    //     for (int i = 0; i < pushNumber; i++)
                    //         std::cout << "d[" << i << "] = " << pPushDisps[i] << std::endl;
                    // }
                    // std::cout << "=== PUSH TYPE : " << mpi.rank << " from " << r << std::endl;

                    MPI_Datatype dtype;
#ifdef ARRAY_COMM_USE_TYPE_HINDEXED
                    MPI_Type_create_hindexed(pushNumber, pPushSizes, pPushDisps, MPI_UINT8_T, &dtype);
#else
                    MPI_Type_indexed(pushNumber, pPushSizes, pPushDisps, MPI_UINT8_T, &dtype);
#endif

                    MPI_Type_commit(&dtype);
                    pPushTypeVec->push_back(std::make_pair(r, dtype));
                    // OPT: could use MPI_Type_create_hindexed_block to save some space
                }
// pull
#ifdef ARRAY_COMM_USE_TYPE_HINDEXED
                MPI_Aint pullDisp[1];
#else
                MPI_int pullDisp[1];
#endif
                MPI_int pullBytes[1];
                auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                auto gStartPtr = son->operator[](index(0));
                auto ghostSpan = gRPtr - gLPtr;
                auto ghostStart = gLPtr - gStartPtr;
                DNDS_assert(ghostSpan < INT_MAX && ghostStart < INT_MAX);
                pullBytes[0] = MPI_int(ghostSpan) * sizeof(T);
                pullDisp[0] = ghostStart * sizeof(T);
                if (pullBytes[0] > 0)
                {
                    // std::cout << "=== PULL TYPE : " << mpi.rank << " from " << r << std::endl;
                    MPI_Datatype dtype;
#ifdef ARRAY_COMM_USE_TYPE_HINDEXED
                    MPI_Type_create_hindexed(1, pullBytes, pullDisp, MPI_UINT8_T, &dtype); //! hindexed is wrong
#else
                    MPI_Type_indexed(1, pullBytes, pullDisp, MPI_UINT8_T, &dtype);
#endif
                    // std::cout << mpi.rank << " pullSlice " << pullDisp[0] << outputDelim << pullBytes[0] << std::endl;
                    MPI_Type_commit(&dtype);
                    pPullTypeVec->push_back(std::make_pair(r, dtype));
                }
            }
            pPullTypeVec->shrink_to_fit();
            pPushTypeVec->shrink_to_fit();
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PushReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPush() // collective;
        {
            // DNDS_assert(pPullTypeVec && pPushTypeVec);
            DNDS_assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
            pushSendSize = 0;
            auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
            // DNDS_assert(nReqs > 0);
            PushReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PushStatVec.resize(nReqs);
            for (auto ip = 0; ip < pPullTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPullTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                MPI_Send_init
#else
                MPI_Bsend_init
#endif
                    (son->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + ip);

                // cascade from father

                // buffer calculate
                MPI_int csize;
                MPI_Pack_size(1, dtypeInfo.second, mpi.comm, &csize);
                csize += MPI_BSEND_OVERHEAD;
                DNDS_assert(MAX_MPI_int - pushSendSize >= csize && csize > 0);
                pushSendSize += csize * 2;
            }
            for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                MPI_Recv_init(father->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + pPullTypeVec->size() + ip);
                // cascade from father
            }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            // MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PullReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPull() // collective;
        {
            // DNDS_assert(pPullTypeVec && pPushTypeVec);
            DNDS_assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
            auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
            pullSendSize = 0;
            // DNDS_assert(nReqs > 0);
            PullReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PullStatVec.resize(nReqs);
            for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPullTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPullTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank; //! receives a lot of messages, this distinguishes them
                // std::cout << mpi.rank << " Recv " << rankOther << std::endl;
                MPI_Recv_init(son->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + ip);
                // std::cout << *(real *)(dataGhost.data() + 8 * 0) << std::endl;
                // cascade from father
            }
            for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                // std::cout << mpi.rank << " Send " << rankOther << std::endl;
#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                MPI_Send_init
#else
                MPI_Bsend_init
#endif
                    (father->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + pPullTypeVec->size() + ip);
                // std::cout << *(real *)(data.data() + 8 * 1) << std::endl;
                // cascade from father

                // buffer calculate
                MPI_int csize;
                MPI_Pack_size(1, dtypeInfo.second, mpi.comm, &csize);
                csize += MPI_BSEND_OVERHEAD * 8;
                DNDS_assert(MAX_MPI_int - pullSendSize >= csize && csize > 0);
                pullSendSize += csize * 2;
            }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            // MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif
        }
        /******************************************************************************************************************************/

        void startPersistentPush() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif
            if (PushReqVec.size())
                MPI_Startall(PushReqVec.size(), PushReqVec.data());
            PerformanceTimer::Instance().EndTimer(PerformanceTimer::TimerType::Comm);
        }
        void startPersistentPull() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif
            if (PullReqVec.size())
                MPI_Startall(PullReqVec.size(), PullReqVec.data());
            PerformanceTimer::Instance().EndTimer(PerformanceTimer::TimerType::Comm);
        }

        void waitPersistentPush() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            if (PushReqVec.size())
                MPI_Waitall(PushReqVec.size(), PushReqVec.data(), PushStatVec.data());
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pushSendSize);
#endif
            PerformanceTimer::Instance().EndTimer(PerformanceTimer::TimerType::Comm);
        }
        void waitPersistentPull() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            if (PullReqVec.size())
                MPI_Waitall(PullReqVec.size(), PullReqVec.data(), PullStatVec.data());
                // std::cout << "waiting DONE" << std::endl;
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pullSendSize);
#endif
            PerformanceTimer::Instance().EndTimer(PerformanceTimer::TimerType::Comm);
        }

        void clearPersistentPush() // collective;
        {
            waitPersistentPush();
            PushReqVec.clear(); // stat vec is left untouched here
        }
        void clearPersistentPull() // collective;
        {
            waitPersistentPull();
            PullReqVec.clear();
        }

        void clearMPITypes() // collective;
        {
            pPullTypeVec.reset();
            pPushTypeVec.reset();
        }

        void clearGlobalMapping() // collective;
        {
            pLGlobalMapping.reset();
        }

        void clearGhostMapping() // collective;
        {
            pLGhostMapping.reset();
        }

        void pullOnce() // collective;
        {
            initPersistentPull();
            startPersistentPull();
            waitPersistentPull();
            clearPersistentPull();
        }

        void pushOnce() // collective;
        {
            initPersistentPush();
            startPersistentPush();
            waitPersistentPush();
            clearPersistentPush();
        }
    };

    template <class TArray>
    struct ArrayTransformerType
    {
        using Type = ArrayTransformer<typename TArray::value_type, TArray::rs, TArray::rm, TArray::al>;
    };

}
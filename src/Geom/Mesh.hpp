#pragma once
#include "Elements.hpp"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "BoundaryCondition.hpp"
#include "DNDS/ArrayPair.hpp"

namespace DNDS::Geom
{
    static const t_index INTERNAL_ZONE = -1;
    struct ElemInfo
    {
        t_index type = static_cast<t_index>(Elem::UnknownElem);
        /// @brief positive for BVnum, 0 for internal Elems, Negative for ?
        t_index zone = INTERNAL_ZONE;

        Elem::ElemType getElemType()
        {
            return static_cast<Elem::ElemType>(type);
        }

        void setElemType(Elem::ElemType t)
        {
            type = static_cast<t_index>(t);
        }

        // bool ZoneIsInternal()
        // {
        //     return zone == INTERNAL_ZONE;
        // }
        // bool ZoneIsIndexed()
        // {
        //     return zone >= 0;
        // }

        static MPI_Datatype CommType() { return MPI_INT32_T; }
        static int CommMult() { return 2; }
    };

    using tAdjPair = DNDS::ArrayAdjacencyPair<DNDS::NonUniformSize>;
    using tAdj = decltype(tAdjPair::father);
    using tAdj2Pair = DNDS::ArrayAdjacencyPair<2>;
    using tAdj2 = decltype(tAdj2Pair::father);
    using tAdj1Pair = DNDS::ArrayAdjacencyPair<1>;
    using tAdj1 = decltype(tAdj1Pair::father);
    using tCoordPair = DNDS::ArrayPair<DNDS::ArrayEigenVector<3>>;
    using tCoord = decltype(tCoordPair::father);
    using tElemInfoArrayPair = DNDS::ArrayPair<DNDS::ParArray<ElemInfo>>;
    using tElemInfoArray = DNDS::ssp<DNDS::ParArray<ElemInfo>>;
    using tIndPair = DNDS::ArrayPair<DNDS::ArrayIndex>;
    using tInd = decltype(tIndPair::father);

    enum MeshAdjState
    {
        Adj_Unknown = 0,
        Adj_PointToLocal,
        Adj_PointToGlobal,
    };

    struct UnstructuredMesh
    {
        MeshAdjState adjPrimaryState{Adj_Unknown};
        /// reader
        tCoordPair coords;
        tAdjPair cell2node;
        tAdjPair bnd2node;
        tAdj2Pair bnd2cell;
        tAdjPair cell2cell;
        tElemInfoArrayPair cellElemInfo;
        tElemInfoArrayPair bndElemInfo;

        /// interpolated
        // *! currently assume all these are Adj_PointToLocal
        tAdjPair cell2face;
        tAdjPair face2node;
        tAdj2Pair face2cell;
        tElemInfoArrayPair faceElemInfo;

        // tAdj1Pair bndFaces; // no comm needed for now

        DNDS::MPIInfo mpi;
        int dim;

        UnstructuredMesh(const DNDS::MPIInfo &n_mpi, int n_dim)
            : mpi(n_mpi), dim(n_dim) {}
        /**
         * @brief building ghost (son) from primary (currently only cell2cell)
         * @details
         * the face and bnd parts are currently only local (no comm available)
         * only builds comm data of cell and node
         * cells: current-father and cell2cell neighbor (face or node neighbor)
         * nodes: needed by all cells
         * faces/bnds: needed by all father cells
         *
         */
        void BuildGhostPrimary();
        void AdjGlobal2LocalPrimary();

        void InterpolateFace();
        void AssertOnFaces();

        index NumCell() { return cell2node.father->Size(); }
        index NumCellProc() { return cell2node.Size(); }
        index NumNode() { return coords.father->Size(); }
        index NumNodeProc() { return coords.Size(); }
        index NumFace() { return face2node.father->Size(); }
        index NumFaceProc() { return face2node.Size(); }
        index NumBnd() { return bnd2node.father->Size(); }

        /// @warning must collectively call
        index NumCellGlobal() { return cell2node.father->globalSize(); }
        /// @warning must collectively call
        index NumNodeGlobal() { return coords.father->globalSize(); }
        /// @warning must collectively call
        index NumFaceGlobal() { return face2node.father->globalSize(); }
        /// @warning must collectively call
        index NumBndGlobal() { return bnd2node.father->globalSize(); }
    };

    using tFDataFieldName = std::function<std::string(int)>;
    using tFDataFieldQuery = std::function<DNDS::real(int, DNDS::index)>;

    struct UnstructuredMeshSerialRW
    {
        DNDS::ssp<UnstructuredMesh> mesh;

        enum Mode
        {
            UnknownMode,
            SerialReadAndDistribute,
            SerialOutput,
        } mode{UnknownMode};

        bool dataIsSerialOut = false;
        bool dataIsSerialIn = false;

        tCoord coordSerial;                // created through reading
        tAdj cell2nodeSerial;              // created through reading
        tAdj bnd2nodeSerial;               // created through reading
        tElemInfoArray cellElemInfoSerial; // created through reading
        tElemInfoArray bndElemInfoSerial;  // created through reading
        tAdj2 bnd2cellSerial;              // created through reading
        /***************************************************************/
        // Current Method: R/W don't manage actually used interpolation,
        // but manually get cell2cell or node2node
        // because: currently only support node based or cell based
        /***************************************************************/

        // tAdj face2nodeSerial;    // created through InterpolateTopology
        // tAdj2 face2cellSerial;   // created through InterpolateTopology
        // tAdj cell2faceSerial;    // created through InterpolateTopology
        // tElemInfoArray faceElemInfoSerial; // created through InterpolateTopology

        tAdj cell2cellSerial; // optionally created with GetCell2Cell()
        tAdj node2nodeSerial; // optionally created with GetNode2Node()

        tAdj node2cellSerial; // not used for now
        tAdj node2faceSerial; // not used for now
        tAdj node2edgeSerial; // not used for now

        tAdj cell2faceSerial; // not used for now
        tAdj cell2edgeSerial; // not used for now

        tAdj face2nodeSerial; // not used for now
        tAdj face2faceSerial; // not used for now
        tAdj face2edgeSerial; // not used for now
        tAdj face2cellSerial; // not used for now

        tAdj edge2nodeSerial; // not used for now
        tAdj edge2cellSerial; // not used for now
        tAdj edge2edgeSerial; // not used for now
        tAdj edge2faceSerial; // not used for now

        DNDS::ArrayTransformerType<tCoord::element_type>::Type coordSerialOutTrans;
        DNDS::ArrayTransformerType<tAdj::element_type>::Type cell2nodeSerialOutTrans;
        DNDS::ArrayTransformerType<tAdj::element_type>::Type bnd2nodeSerialOutTrans;
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type cellElemInfoSerialOutTrans;
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type bndElemInfoSerialOutTrans;

        std::vector<DNDS::MPI_int> cellPartition;
        std::vector<DNDS::MPI_int> nodePartition;
        std::vector<DNDS::MPI_int> bndPartition;

        DNDS::MPI_int mRank{0}, cnPart{0};

        UnstructuredMeshSerialRW(const decltype(mesh) &n_mesh, DNDS::MPI_int n_mRank)
            : mesh(n_mesh), mRank(n_mRank) {}

        /// @brief reads a cgns file as serial input
        /**
         * @details
         * the file MUST consist of one CGNS base node, and
         * multiple zones within.
         * All the zones are treated as a whole unstructured grid
         * Proofed on .cgns files generated from Pointwise
         * @warning //!Pointwise Options: with "Include Donor Information", "Treat Structured as Unstructured", "Unstructured Interfaces = Node-to-Node"
         */
        /// @todo //TODO Add some multi thread here!
        /// @param fName file name of .cgns file
        void ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID = FBC_Name_2_ID_Default);

        // void InterpolateTopology();

        /**
         * \brief build cell2cell topology, with node-neighbors included
         * \todo add support for only face-neighbors
         */
        void BuildCell2Cell(); // For cell based purpose

        void BuildNode2Node(); // For node based purpose //!not yet implemented

        void MeshPartitionCell2Cell();

        void PartitionReorderToMeshCell2Cell();

        void ClearSerial()
        {
            coordSerial.reset();
            cell2nodeSerial.reset();
            cell2cellSerial.reset();
            cellElemInfoSerial.reset();
            bnd2nodeSerial.reset();
            bnd2cellSerial.reset();
            bndElemInfoSerial.reset();
            mode = UnknownMode;
            dataIsSerialIn = dataIsSerialOut = false;
        }

        /**
         * @brief should be called to build data for serial out
         */
        void BuildSerialOut();

        // void WriteToCGNSSerial(const std::string &fName);

        // void WriteSolutionToTecBinary(const std::string &fName,
        //                               int nField, const tFDataFieldName &names, const tFDataFieldQuery &data,
        //                               int nFieldBnd, const tFDataFieldName &namesBnd, const tFDataFieldQuery &dataBnd);

        /**
         * @brief names(idata) data(idata, ivolume)
         * https://tecplot.azureedge.net/products/360/current/360_data_format_guide.pdf
         * @todo //TODO add support for bnd export
         * @todo //TODO: switch to vtk!
         */
        void PrintSerialPartPltBinaryDataArray(
            std::string fname,
            int arraySiz,
            const std::function<std::string(int)> &names,
            const std::function<DNDS::real(int, DNDS::index)> &data,
            double t, int flag = 0); //! supports 2/3d here
    };
} // namespace geom
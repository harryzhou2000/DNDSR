#pragma once
#include "Elements.hpp"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "BoundaryCondition.hpp"

namespace Geom
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

    struct UnstructuredMesh
    {

        tCoordPair coords;
        tAdjPair cell2node;
        tAdjPair cell2face;
        tAdjPair face2node;
        tAdj2Pair face2cell;

        tAdj1Pair bndFaces; // no comm needed for now

        DNDS::MPIInfo mpi;
        int dim;

        UnstructuredMesh(const DNDS::MPIInfo &n_mpi, int n_dim)
            : mpi(n_mpi), dim(n_dim) {}
    };

    using tFDataFieldName = std::function<std::string(int)>;
    using tFDataFieldQuery = std::function<DNDS::real(int, DNDS::index)>;

    struct UnstructuredMeshSerialRW
    {
        DNDS::ssp<UnstructuredMesh> mesh;

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

        DNDS::MPI_int mRank;

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

        // void WriteToCGNSSerial(const std::string &fName);

        // void WriteSolutionToTecBinary(const std::string &fName,
        //                               int nField, const tFDataFieldName &names, const tFDataFieldQuery &data,
        //                               int nFieldBnd, const tFDataFieldName &namesBnd, const tFDataFieldQuery &dataBnd);
    };
} // namespace geom

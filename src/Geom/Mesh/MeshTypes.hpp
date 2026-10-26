#pragma once

#include "Geom/Geometric.hpp"
#include "Geom/ElemEnum.hpp"
#include "DNDS/MPI.hpp"
#include "DNDS/Runtime/ElementSchema.hpp"

namespace DNDS::Geom
{
    static const t_index INTERNAL_ZONE = -1;
    struct ElemInfo
    {
        t_index type = static_cast<t_index>(Elem::UnknownElem);
        /// @brief positive for BVnum, 0 for internal Elems, Negative for ?
        t_index zone = INTERNAL_ZONE;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ElemInfo, ElemInfo)

        DNDS_DEVICE_CALLABLE [[nodiscard]] Elem::ElemType getElemType() const
        {
            return static_cast<Elem::ElemType>(type);
        }

        DNDS_DEVICE_CALLABLE void setElemType(Elem::ElemType t)
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

        static MPI_Datatype CommType()
        {
            static_assert(sizeof(ElemInfo) <= (4ULL * 2));
            return MPI_INT32_T;
        }
        static int CommMult() { return 2; }
        static std::string pybind11_name() { return "ElemInfo"; }
    };

    enum MeshAdjState
    {
        Adj_Unknown = 0,
        Adj_PointToLocal,
        Adj_PointToGlobal,
    };

    enum MeshElevationState
    {
        Elevation_Untouched = 0,
        Elevation_O1O2,
    };
}

namespace DNDS::Runtime
{
    template <>
    struct ElementSchemaProvider<Geom::ElemInfo>
    {
        static ElementSchema Get()
        {
            return ElementSchema::Record("Geom.ElemInfo.v1", sizeof(Geom::ElemInfo),
                                         {{"type", offsetof(Geom::ElemInfo, type), ElementScalar::I32},
                                          {"zone", offsetof(Geom::ElemInfo, zone), ElementScalar::I32}},
                                         "DNDS::Geom::ElemInfo", alignof(Geom::ElemInfo));
        }
    };
}

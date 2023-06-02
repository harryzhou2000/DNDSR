#pragma once
#include "Geometric.hpp"
#include "DNDS/MPI.hpp"
#include <map>
#include <functional>

namespace Geom
{
    class BCName_2_ID
    {
        std::map<std::string, t_index> name_2_id;

        void clear() { name_2_id.clear(); }

        t_index operator[](const std::string &name)
        {
            auto found = name_2_id.find(name);
            if (found == name_2_id.end())
                return name_2_id[name] = name_2_id.size();
            else
                return (*found).second;
        }
    };

    static const t_index BC_ID_INTERNAL = 0;    // do not change
    static const t_index BC_ID_PERIODIC_1 = -1; // * periodic-s do not count as external face
    static const t_index BC_ID_PERIODIC_2 = -2; // but should be identified also
    static const t_index BC_ID_PERIODIC_3 = -3;

    static const t_index BC_ID_PERIODIC_1_DONOR = -4;
    static const t_index BC_ID_PERIODIC_2_DONOR = -5;
    static const t_index BC_ID_PERIODIC_3_DONOR = -6;

    static const t_index BC_ID_NULL = 1;
    static const t_index BC_ID_DEFAULT_WALL = 2;
    static const t_index BC_ID_DEFAULT_FAR = 3;
    static const t_index BC_ID_DEFAULT_WALL_INVIS = 4;

    static const t_index BC_ID_DEFAULT_SPECIAL_DMR_FAR = 101;
    static const t_index BC_ID_DEFAULT_MAX = 200;

    using t_FBCName_2_ID = std::function<t_index(const std::string &)>;

    static const t_FBCName_2_ID FBC_Name_2_ID_Default = [](const std::string &name) -> t_index
    {
        if (name == "PERIODIC_1")
            return BC_ID_PERIODIC_1;
        if (name == "PERIODIC_2")
            return BC_ID_PERIODIC_2;
        if (name == "PERIODIC_3")
            return BC_ID_PERIODIC_3;
        if (name == "PERIODIC_1_DONOR")
            return BC_ID_PERIODIC_1_DONOR;
        if (name == "PERIODIC_2_DONOR")
            return BC_ID_PERIODIC_2_DONOR;
        if (name == "PERIODIC_3_DONOR")
            return BC_ID_PERIODIC_3_DONOR;
        if (name == "WALL" || name == "bc-4")
            return BC_ID_DEFAULT_WALL;
        if (name == "FAR" || name == "bc-2")
            return BC_ID_DEFAULT_FAR;
        if (name == "WALL_INVIS" || name == "bc-3")
            return BC_ID_DEFAULT_WALL_INVIS;
        if (name == "bc-DMRFar")
            return BC_ID_DEFAULT_SPECIAL_DMR_FAR;
        return BC_ID_NULL;
    };

    inline bool FaceIDIsExternalBC(t_index id)
    {
        return id > 0;
    }

    inline bool FaceIDIsInternal(t_index id)
    {
        return id <= 0;
    }

    inline bool FaceIDIsTrueInternal(t_index id)
    {
        return id == 0;
    }

    inline bool FaceIDIsPeriodic(t_index id)
    {
        return id == BC_ID_PERIODIC_1 ||
               id == BC_ID_PERIODIC_2 ||
               id == BC_ID_PERIODIC_3 ||
               id == BC_ID_PERIODIC_1_DONOR ||
               id == BC_ID_PERIODIC_2_DONOR ||
               id == BC_ID_PERIODIC_3_DONOR;
    }

    inline bool FaceIDIsPeriodicMain(t_index id)
    {
        return id == BC_ID_PERIODIC_1 ||
               id == BC_ID_PERIODIC_2 ||
               id == BC_ID_PERIODIC_3;
    }

    inline bool FaceIDIsPeriodicDonor(t_index id)
    {
        return id == BC_ID_PERIODIC_1_DONOR ||
               id == BC_ID_PERIODIC_2_DONOR ||
               id == BC_ID_PERIODIC_3_DONOR;
    }

    struct Periodicity
    {
        std::array<tGPoint, 4> rotation;
        std::array<tPoint, 4> translation;
        std::array<tPoint, 4> rotationCenter;
        Periodicity()
        {
            for (auto &r : rotation)
                r.setIdentity();
            for (auto &r : rotationCenter)
                r.setZero();
            translation[0].setZero();
            translation[1] = tPoint{1, 0, 0};
            translation[2] = tPoint{0, 1, 0};
            translation[3] = tPoint{0, 0, 1};
        }

        tPoint TransCoord(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodicMain(id));
            t_index i = -id;
            return rotation[i] * (c - rotationCenter[i]) + rotationCenter[i] + translation[i];
        }

        tPoint TransCoordBack(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodicDonor(id));
            t_index i = -id - 3;
            return rotation[i].transpose() * ((c + translation[i]) - rotationCenter[i]) + rotationCenter[i];
        }

        ///@todo //TODO: add support for cartesian tensor transformation
    };

    // enum BCType
    // {
    //     BcUnknown,
    //     BcWall,
    //     BcFar,
    //     BcInviscid,
    //     BcSpecial_DMR_FAR,
    // };

    // class BCManager
    // {
    //     std::vector<BCType> id_2_type;
    //     std::vector<
    // };

} // namespace Geom

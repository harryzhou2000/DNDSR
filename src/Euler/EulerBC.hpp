#pragma once
#include "Euler.hpp"
#include "Geom/BoundaryCondition.hpp"

#include <unordered_map>

#include "json.hpp"

namespace DNDS::Euler
{
    enum EulerBCType
    {
        BCUnknown = 0,
        BCFar,
        BCWall,
        BCWallInvis,
        BCSpecial,
    };

    template <EulerModel model>
    class BoundaryHandler
    {
    public:
        static const int NVars_Fixed = getNVars_Fixed(model);
        using TU = Eigen::Vector<real, NVars_Fixed>;

    private:
        std::vector<nlohmann::json> BCSettings;
        std::vector<TU> BCValues;
        std::vector<EulerBCType> BCTypes;
        std::unordered_map<std::string, Geom::t_index> name2ID;

    public:
        BoundaryHandler()
        {
            BCValues.resize(Geom::BC_ID_DEFAULT_MAX);
            for (auto &v : BCValues)
                v.setConstant(UnInitReal);
            name2ID = Geom::GetFaceName2IDDefault();
            BCTypes.resize(Geom::BC_ID_DEFAULT_MAX, BCUnknown);
            BCTypes[Geom::BC_ID_DEFAULT_FAR] = BCFar;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR] = BCSpecial;
            BCTypes[Geom::BC_ID_DEFAULT_WALL] = BCWall;
            BCTypes[Geom::BC_ID_DEFAULT_WALL_INVIS] = BCWallInvis;
            BCTypes[Geom::BC_ID_NULL] = BCUnknown;
        }

        void PushBCWithJson(const nlohmann::json &gS)
        {
            // TODO
        }

        /**
         * if periodic, return with minus
         */
        Geom::t_index GetIDFromName(const std::string &name)
        {
            if (name2ID.count(name))
                return name2ID[name];
            else
                return Geom::BC_ID_NULL;
        }

        EulerBCType GetTypeFromID(Geom::t_index id)
        {
            // std::cout << "id " << std::endl;
            return BCTypes.at(id);
        }
    };
}
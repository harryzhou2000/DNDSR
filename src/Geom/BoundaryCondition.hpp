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

    static const t_index BC_TYPE_NULL = 1;
    static const t_index BC_TYPE_DEFAULT_WALL = 2;
    static const t_index BC_TYPE_DEFAULT_FAR = 3;
    static const t_index BC_TYPE_DEFAULT_WALL_INVIS = 4;
    static const t_index BC_TYPE_DEFAULT_SPECIAL_DMR_FAR = 101;

    using t_FBCName_2_ID = std::function<t_index(const std::string &)>;

    static const t_FBCName_2_ID FBC_Name_2_ID_Default = [](const std::string &name) -> t_index
    {
        if (name == "WALL" || name == "bc-4")
            return BC_TYPE_DEFAULT_WALL;
        if (name == "FAR" || name == "bc-2")
            return BC_TYPE_DEFAULT_FAR;
        if (name == "WALL_INVIS" || name == "bc-3")
            return BC_TYPE_DEFAULT_WALL_INVIS;
        if (name == "bc-DMRFar")
            return BC_TYPE_DEFAULT_SPECIAL_DMR_FAR;
        return BC_TYPE_NULL;
    };

} // namespace Geom

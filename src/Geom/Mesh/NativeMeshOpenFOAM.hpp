#pragma once
#include "NativeMeshCGNS.hpp"

namespace DNDS::Geom
{
    NativeSerialMeshData ReadNativeOpenFOAMSerial(
        const std::string &path, const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &boundaryId = FBC_Name_2_ID_Default);
    /// Collective native import of supported 3D canonical OpenFOAM cells.
    std::shared_ptr<const NativeMeshState> ReadNativeOpenFOAM(
        const MPIInfo &mpi, const std::string &path, const std::map<std::string, std::string> &nameMapping,
        const t_FBCName_2_ID &boundaryId = FBC_Name_2_ID_Default, int readerRank = 0,
        NativeMeshBuildOptions options = {}, std::optional<NativePeriodicImportOptions> periodic = {},
        NativeSerialDistribution distribution = NativeSerialDistribution::Even);
}

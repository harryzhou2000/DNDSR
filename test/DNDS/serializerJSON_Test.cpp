#include "DNDS/SerializerJSON.hpp"
#include <iostream>

int main()
{
    using namespace DNDS;
    std::vector<real> rVec = {1, 2, 3.3, 4, 5.4};
    std::vector<rowsize> rsVec0 = {1, 0, 1, 0, 1};

    ssp<std::vector<DNDS::index>> p_iVec0;
    DNDS_MAKE_SSP(p_iVec0, std::vector<DNDS::index>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2});
    ssp<std::vector<DNDS::index>> p_iVec1 = p_iVec0;

    SerializerJSON serializer;

    // serializer.OpenFile("testOut.json", false);
    // serializer.CreatePath("main");
    // serializer.GoToPath("main");
    // serializer.WriteRealVector("aRealVec", rVec);
    // serializer.WriteRowsizeVector("aRSVec", rsVec0);
    // serializer.WriteSharedIndexVector("aIndexVec_shared", p_iVec0);
    // serializer.WriteSharedIndexVector("aIndexVec_shared_1", p_iVec1);
    // serializer.CloseFile();

    serializer.OpenFile("testOut.json", true);
    serializer.GoToPath("main");
    serializer.ReadRealVector("aRealVec", rVec);
    serializer.ReadRowsizeVector("aRSVec", rsVec0);
    serializer.ReadSharedIndexVector("aIndexVec_shared", p_iVec0);
    serializer.ReadSharedIndexVector("aIndexVec_shared_1", p_iVec1);
    serializer.CloseFile();

    return 0;
}
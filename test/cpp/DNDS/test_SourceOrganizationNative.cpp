#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
// Include native access directly, before any compatibility forwarding header.
#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include <type_traits>

#ifdef DNDS_ARRAY_HPP
#    error "Canonical native access/state headers must not define legacy DNDS::Array"
#endif

static_assert(std::is_same_v<DNDS::Geom::NativeMeshState, DNDS::Geom::UnstructuredMeshState>);
static_assert(std::is_same_v<DNDS::Geom::NativeMeshSnapshot, DNDS::Geom::UnstructuredMeshSnapshot>);
static_assert(std::is_trivially_copyable_v<DNDS::Geom::UnstructuredMeshView>);
static_assert(decltype(std::declval<DNDS::Geom::UnstructuredMeshView>().FaceCells(0))::row_width == 2);

TEST_CASE("Canonical native access requirements are independent of legacy mesh arrays")
{
    auto requirements = DNDS::Geom::MeshReadRequirements::CellFlux();
    CHECK(requirements.faces);
    CHECK_FALSE(requirements.inverse);
    CHECK_FALSE(requirements.wallDistance);
}

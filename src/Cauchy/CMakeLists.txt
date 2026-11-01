set(CPPS Core/DomainPlacement.cpp)
dnds_add_lib(cauchy "${CPPS}" "cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)

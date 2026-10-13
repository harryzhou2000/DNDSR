# Explicit configure-time values opt into per-test TIMEOUT properties.
# Otherwise leave those properties unset so ctest --timeout can control them.
if(DEFINED ENV{DNDS_TEST_TIMEOUT})
    set(DNDS_TEST_TIMEOUT "$ENV{DNDS_TEST_TIMEOUT}")
    set(DNDS_TEST_SET_TIMEOUT ON)
else()
    # Retain a fallback for the pytest command's own per-test timeout.
    set(DNDS_TEST_TIMEOUT "1800")
    set(DNDS_TEST_SET_TIMEOUT OFF)
endif()
message(STATUS "DNDS_TEST_TIMEOUT: ${DNDS_TEST_TIMEOUT} (set-timeout: ${DNDS_TEST_SET_TIMEOUT})")

math(EXPR DNDS_TEST_TIMEOUT_SOLVER "${DNDS_TEST_TIMEOUT} / 2")
math(EXPR DNDS_TEST_TIMEOUT_PIPELINE "${DNDS_TEST_TIMEOUT} * 5")

function(_dnds_maybe_set_timeout TEST_NAME VALUE)
    if(DNDS_TEST_SET_TIMEOUT)
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT ${VALUE})
    endif()
endfunction()

# A launcher failure or timeout must not satisfy an expected MPI rejection.
# Keep the command as a CMake list so launcher flags remain separate arguments.
execute_process(COMMAND ${TEST_COMMAND}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error
    TIMEOUT 20)
message("${output}${error}")
if(NOT status MATCHES "^[0-9]+$" OR status EQUAL 0)
    message(FATAL_ERROR "Expected a nonzero MPI launcher exit, got: ${status}")
endif()
if(NOT "${output}${error}" MATCHES "${EXPECTED_DIAGNOSTIC}")
    message(FATAL_ERROR "Expected MPI rejection diagnostic was absent")
endif()

set(PROGRAM_NAME skeleton)
add_executable(${PROGRAM_NAME} main.cpp)
target_link_libraries(${PROGRAM_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROGRAM_NAME} PROPERTY CXX_STANDARD 17)

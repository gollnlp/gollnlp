add_executable(ex1.exe Ex1driver.cpp)
target_link_libraries(ex1.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS} ${KNITRO_LIB})

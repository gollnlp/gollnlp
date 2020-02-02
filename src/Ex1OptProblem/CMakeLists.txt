add_executable(ex1.exe Ex1driver.cpp)
target_link_libraries(ex1.exe gollnlp ipopt coinhsl coinmetis ${LAPACK_LIBRARIES} ${GOLLNLP_LIBS_EXTRA} ${OpenMP_CXX_LIBRARIES})

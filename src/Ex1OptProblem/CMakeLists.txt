add_executable(ex1.exe Ex1driver.cpp)
target_link_libraries(ex1.exe gollnlp ${FORTRAN_LIB} ipopt coinhsl coinmetis ${LAPACK_LIBRARIES})

add_executable(test_scacopfdata.exe SCACOPFData_test.cpp)
target_link_libraries(test_scacopfdata.exe gollnlp ${LAPACK_LIBRARIES})

add_executable(Ex1.exe Ex1driver.cpp)
target_link_libraries(Ex1.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS})

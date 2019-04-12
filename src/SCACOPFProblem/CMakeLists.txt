add_executable(test_scacopfdata.exe SCACOPFData_test.cpp)
target_link_libraries(test_scacopfdata.exe gollnlp ${LAPACK_LIBRARIES})

add_library(goSCACOPF OBJECT SCACOPFData.cpp)

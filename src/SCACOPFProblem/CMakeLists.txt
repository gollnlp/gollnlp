add_executable(test_scacopfdata.exe SCACOPFData_test.cpp)
target_link_libraries(test_scacopfdata.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS} ${KNITRO_LIB})

#add_executable(recourse_sandbox.exe go_recourse_sandbox.cpp)
#target_link_libraries(recourse_sandbox.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS})

add_library(goSCACOPF OBJECT SCACOPFData.cpp go_code1.cpp go_code2.cpp OPFConstraints.cpp OPFObjectiveTerms.cpp CouplingConstraints.cpp SCACOPFProblem.cpp ContingencyProblem.cpp SCACOPFIO.cpp)
#SCMasterProblem.cpp SCRecourseProblem.cpp 


add_executable(acopf_driver.exe ACOPFDriver.cpp)
target_link_libraries(acopf_driver.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS} ${KNITRO_LIB})

add_executable(test_scacopfdata.exe SCACOPFData_test.cpp)
target_link_libraries(test_scacopfdata.exe gollnlp ipopt coinhsl coinmetis ${GOLLNLP_LIBS_EXTRA} ${OpenMP_CXX_LIBRARIES} ${LAPACK_LIBRARIES})

#add_executable(recourse_sandbox.exe go_recourse_sandbox.cpp)
#target_link_libraries(recourse_sandbox.exe gollnlp ${LAPACK_LIBRARIES} ${IPOPT_LIBS})

add_library(goSCACOPF OBJECT SCACOPFData.cpp go_code1.cpp go_code2.cpp OPFConstraints.cpp OPFObjectiveTerms.cpp CouplingConstraints.cpp SCACOPFProblem.cpp ContingencyProblem.cpp SCACOPFIO.cpp ContingencyProblemWithFixing.cpp ContingencyProblemWithFixingCode1.cpp)
#SCMasterProblem.cpp SCRecourseProblem.cpp 


#
# other utilities such as Kron reduction ACOPF
#
if(GOLLNLP_WITH_KRON_REDUCTION)
add_library(goKronRed ACOPFKronRedProblem.cpp $<TARGET_OBJECTS:goLinAlg>)

add_executable(acopf_kron_driver.exe ACOPFKronRedDriver.cpp)
target_link_libraries(acopf_kron_driver.exe gollnlp goKronRed ipopt coinhsl coinmetis ${ME57_LIB} ${LAPACK_LIBRARIES} ${GOLLNLP_LIBS_EXTRA} ${OpenMP_CXX_LIBRARIES})
endif(GOLLNLP_WITH_KRON_REDUCTION)

#add_executable(acopf_driver.exe ACOPFDriver.cpp)
#target_link_libraries(acopf_driver.exe gollnlp "-lgfortran" ipopt coinhsl coinmetis ${LAPACK_LIBRARIES})

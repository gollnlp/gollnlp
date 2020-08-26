#add_executable(test_scacopfdata.exe SCACOPFData_test.cpp)
#target_link_libraries(test_scacopfdata.exe gollnlp ipopt coinhsl coinmetis 
#                      ${HIOP_LIB} ${GOLLNLP_LIBS_EXTRA} ${OpenMP_CXX_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(goSCACOPF OBJECT 
		      SCACOPFData.cpp go_code1.cpp go_code2.cpp 
		      OPFConstraints.cpp OPFObjectiveTerms.cpp CouplingConstraints.cpp 
		      SCACOPFProblem.cpp ContingencyProblem.cpp SCACOPFIO.cpp 
		      ContingencyProblemWithFixing.cpp ContingencyProblemWithFixingCode1.cpp)

#
# other utilities such as Kron reduction ACOPF
#

if(GOLLNLP_WITH_KRON_REDUCTION)
  add_library(goKronRed ACOPFKronRedProblem.cpp OPFConstraintsKron.cpp)

  add_executable(acopf_kron_driver.exe ACOPFKronRedDriver.cpp)
  target_link_libraries(acopf_kron_driver.exe 
                        goKronRed gollnlp ipopt coinhsl coinmetis 
			hiop ${UMFPACK_LIB} ${LAPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES}
			${GOLLNLP_LIBS_EXTRA} )

  if(GOLLNLP_USE_GPU)
    target_link_libraries(acopf_kron_driver.exe hiop_cuda)
  endif(GOLLNLP_USE_GPU)
endif(GOLLNLP_WITH_KRON_REDUCTION)

#add_executable(acopf_driver.exe ACOPFDriver.cpp)
#target_link_libraries(acopf_driver.exe gollnlp "-lgfortran" ipopt coinhsl coinmetis ${LAPACK_LIBRARIES})

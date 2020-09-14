cmake_minimum_required (VERSION 2.8)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif ()


project (goSpecCpp)
# enable Fortran for Fortran name mangling
enable_language(Fortran)
set (CMAKE_CXX_STANDARD 11)

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")
add_compile_options("-std=c++11")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#on macos
#add_link_options("-w")
# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(GOLLNLP_USE_MPI "Build with MPI support" ON)
option(GOLLNLP_FAULT_HANDLING "Using handlers for select signals" ON)
#with testing drivers capable of 'selfchecking' (-selfcheck)
option(GOLLNLP_WITH_MAKETEST "Enable 'make test'" ON)

option(GOLLNLP_WITH_KRON_REDUCTION "Building Kron Reduction code (requires addtl. dependencies)" OFF)

option(GOLLNLP_USE_GPU "Use CUDA and Magma linear algebra with HiOp to enable GPU computations" OFF)

if(GOLLNLP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /usr/lib64/openmpi")
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
endif(GOLLNLP_USE_MPI)

# libdl not found on Ubuntu (?!?)
#! set(GOLLNLP_LIBS_EXTRA "-l${CMAKE_DL_LIBS} ")

find_package(OpenMP)
message("OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")

if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  set(FORTRAN_LIB "-lifcore")	
  else()
    #!set(FORTRAN_LIB "-lgfortran")
    set(FORTRAN_LIB "-L/usr/local/gfortran/lib -lgfortran")
endif(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
message("Using Fortran library flag ${FORTRAN_LIB}")
set(GOLLNLP_LIBS_EXTRA "${GOLLNLP_LIBS_EXTRA}${FORTRAN_LIB} -ldl")

if(GOLLNLP_FAULT_HANDLING)
  add_definitions(-DGOLLNLP_FAULT_HANDLING)
endif(GOLLNLP_FAULT_HANDLING)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
endif(NOT DEFINED LAPACK_LIBRARIES)
message("LAPACK/BLAS libs: ${LAPACK_LIBRARIES}")

# "/home/petra1/work/installs/Ipopt-3.12.5/"
# "/opt/Ipopt/v3.12.12/"
# "~/bin/COIN-OR/Ipopt/"
set(IPOPT_DIR $ENV{IPOPT_PATH} CACHE PATH "path to ipopt directory")
message("Using Ipopt dir ${IPOPT_DIR}")
include_directories("${IPOPT_DIR}/include/coin")
#!link_directories("${IPOPT_DIR}/lib")
link_directories("${IPOPT_DIR}/lib")
#set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--rpath -Wl,${IPOPT_DIR}/lib")
set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ldl")
#set(IPOPT_LIBS "-lgfortran -lipopt -lm  -ldl")

add_library(coinmetis STATIC IMPORTED GLOBAL)
set_target_properties(coinmetis PROPERTIES IMPORTED_LOCATION "${IPOPT_DIR}/lib/libcoinmetis.a")

add_library(coinhsl STATIC IMPORTED GLOBAL)
set_target_properties(coinhsl PROPERTIES IMPORTED_LOCATION "${IPOPT_DIR}/lib/libcoinhsl.a")

add_library(ipopt STATIC IMPORTED GLOBAL)
set_target_properties(ipopt PROPERTIES IMPORTED_LOCATION "${IPOPT_DIR}/lib/libipopt.a")

#warn if HiOp is not present
find_library(HIOP_LIB NAMES hiop HINTS "${HIOP_DIR}/lib")
if(HIOP_LIB)
  include_directories("${HIOP_DIR}/include")
  message("Using HiOp include '${HIOP_DIR}/include' and lib '${HIOP_LIB}'")
  add_library(hiop STATIC IMPORTED GLOBAL)
  set_target_properties(hiop PROPERTIES IMPORTED_LOCATION "${HIOP_LIB}")
else()
  message(FATAL_ERROR "Could not find HiOp library needed by kron reduction. Please cmake with '-DHIOP_DIR=/path/to/hiop_dist_dir'; currently HIOP_DIR='${HIOP_DIR}'")
endif(HIOP_LIB)

if(GOLLNLP_WITH_KRON_REDUCTION)
  find_library(UMFPACK_LIB NAMES umfpack HINTS ${UMFPACK_DIR} "${UMFPACK_DIR}/lib" "${UMFPACK_DIR}/Lib")
  if(UMFPACK_LIB)
    message("Using UMFPACK lib '${UMFPACK_LIB}'")

    find_path(UMFPACK_INCL NAMES "umfpack.h" HINTS ${UMFPACK_DIR} "${UMFPACK_DIR}/include" "${UMFPACK_DIR}/Include")
    if(UMFPACK_INCL)
      include_directories(${UMFPACK_INCL})
      message("Using UMFPACK include '${UMFPACK_INCL}'")
    else() 
      message(FATAL_ERROR "Could not find UMFPACK header needed by the kron reduction. Please cmake with '-DUMFPACK_DIR=/path/to/umfpack_dir'; currently, UMFPACK_DIR='${UMFPACK_DIR}'")
    endif(UMFPACK_INCL)
  else()
    message(FATAL_ERROR "Could not find UMFPACK needed by the kron reduction. Please cmake with '-DUMFPACK_DIR=/path/to/umfpack_dir'; currently, UMFPACK_DIR='${UMFPACK_DIR}'")
  endif(UMFPACK_LIB)

  add_definitions(-DGOLLNLP_WITH_KRON_REDUCTION)
  
  # metis needed (5.x needed by umfpack)
  #find_library(METIS_LIB NAMES metis HINTS ${METIS_DIR} "${METIS_DIR}/lib" "${METIS_DIR}/src/")
  #if(METIS_LIB)
  #  message("Using METIS lib '${METIS_LIB}'")
  #else()
  #  message(FATAL_ERROR "Could not find METIS lib needed by UMFPACK. Please cmake with '-DMETIS_DIR=/path/to/metis/lib'; currently, METIS_DIR='${METIS_DIR}'")
  #endif(METIS_LIB)
endif(GOLLNLP_WITH_KRON_REDUCTION)

if(GOLLNLP_USE_GPU)
  include(CheckLanguage)
  enable_language(CUDA)
  check_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  if(HIOP_NVCC_ARCH)
    set(CMAKE_CUDA_FLAGS "-arch=${HIOP_NVCC_ARCH}")
    message(STATUS "Using CUDA arch ${HIOP_NVCC_ARCH}")
  else()
    set(CMAKE_CUDA_FLAGS "-arch=sm_35")
    message(STATUS "Using CUDA arch sm_35")
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

  set(HIOP_MAGMA_DIR CACHE PATH "Path to Magma directory")

  include(FindHiopCudaLibraries)

endif(GOLLNLP_USE_GPU)

# include build directory for Fortran name mangling header
include_directories(${CMAKE_BINARY_DIR})

include_directories(src/GeneralOptProblem)
include_directories(src/SCACOPFProblem)
include_directories(src/Utils)
include_directories(src/NlpSolver)
include_directories(src/Ex1OptProblem)
add_subdirectory(src/GeneralOptProblem)

add_subdirectory(src/Utils)
add_subdirectory(src/SCACOPFProblem)
add_subdirectory(src/Ex1OptProblem)

add_library(gollnlp STATIC $<TARGET_OBJECTS:goSCACOPF> $<TARGET_OBJECTS:goUtils> $<TARGET_OBJECTS:goOptimiz>)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY})


#########################################################
# competition executables
#########################################################

#message(${GOLLNLP_LIBS_EXTRA})
add_executable(MyExe1 MyCpp1.cpp)
target_link_libraries(MyExe1 gollnlp ipopt coinhsl coinmetis 
                      hiop ${GOLLNLP_LIBS_EXTRA} 
		      ${OpenMP_CXX_LIBRARIES} ${LAPACK_LIBRARIES})
if(GOLLNLP_WITH_KRON_REDUCTION)
  target_link_libraries(MyExe1 goKronRed)
endif(GOLLNLP_WITH_KRON_REDUCTION)
		      
		   
add_executable(MyExe2 MyCpp2.cpp)
target_link_libraries(MyExe2 gollnlp ipopt coinhsl coinmetis 
                      hiop ${GOLLNLP_LIBS_EXTRA} 
		      ${OpenMP_CXX_LIBRARIES} ${LAPACK_LIBRARIES})

if(GOLLNLP_WITH_KRON_REDUCTION)
add_executable(MyExe1Kron.exe MyCpp1Kron.cpp)
target_link_libraries(MyExe1Kron.exe goKronRed gollnlp ipopt coinhsl coinmetis 
                      hiop ${UMFPACK_LIB} ${GOLLNLP_LIBS_EXTRA} 
		      ${OpenMP_CXX_LIBRARIES} ${LAPACK_LIBRARIES})
endif(GOLLNLP_WITH_KRON_REDUCTION)
##########################################################
# CMake Tests
##########################################################
#if (GOLLNLP_WITH_MAKETEST)
#  enable_testing()
#  add_test(NAME SCACOPFData COMMAND $<TARGET_FILE:test_scacopfdata.exe> ${PROJECT_SOURCE_DIR}/test_instanceslist.txt )
#endif(GOLLNLP_WITH_MAKETEST)

##########################################################
# Examples of cmake-ing
##########################################################
# export IPOPT_PATH=/home/petra1/work/projects/gocompet/Ipopt-gollnlp/build
# rm -rf *; cmake -DGOLLNLP_FAULT_HANDLING=OFF -DGOLLNLP_WITH_KRON_REDUCTION=ON .. && make -j

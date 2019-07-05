cmake_minimum_required (VERSION 2.8)
project (goSpecCpp)
# enable Fortran for Fortran name mangling
enable_language(Fortran)
set (CMAKE_CXX_STANDARD 11)

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")
add_compile_options("-std=c++11")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#on macos
#add_link_options("-w")
# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(GOLLNLP_USE_MPI "Build with MPI support" ON)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(GOLLNLP_WITH_MAKETEST "Enable 'make test'" ON)

if(GOLLNLP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /usr/lib64/openmpi")
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
endif(GOLLNLP_USE_MPI)

find_package(OpenMP)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
endif(NOT DEFINED LAPACK_LIBRARIES)
message("LAPACK/BLAS libs: ${LAPACK_LIBRARIES}")


# "/home/petra1/work/installs/Ipopt-3.12.5/"
set(IPOPT_DIR $ENV{IPOPT_3_12_13_GCC492_PATH} CACHE PATH "path to ipopt directory")
message("Using Ipopt dir ${IPOPT_DIR}")
include_directories("${IPOPT_DIR}/include/coin")
link_directories("${IPOPT_DIR}/lib")
#set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--rpath -Wl,${IPOPT_DIR}/lib")
set(IPOPT_LIBS "-lgfortran -lipopt -lm  -ldl")

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

#
# competition executables
#
add_executable(MyExe1 MyCpp1.cpp)
target_link_libraries(MyExe1 gollnlp ${LAPACK_LIBRARIES}  ${IPOPT_LIBS})

##########################################################
# CMake Tests
##########################################################
if (GOLLNLP_WITH_MAKETEST)
  enable_testing()
  add_test(NAME SCACOPFData COMMAND $<TARGET_FILE:test_scacopfdata.exe>   ${PROJECT_SOURCE_DIR}/test_instanceslist.txt )
endif(GOLLNLP_WITH_MAKETEST)

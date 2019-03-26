cmake_minimum_required (VERSION 2.8)
project (goSpecCpp)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(GOLLNLP_USE_MPI "Build with MPI support" ON)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(GOLLNLP_WITH_MAKETEST "Enable 'make test'" ON)

if(GOLLNLP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
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

# include build directory for Fortran name mangling header
include_directories(${CMAKE_BINARY_DIR})

include_directories(src/Utils)
include_directories(src/OPF)

add_subdirectory(src/Utils)
add_subdirectory(src/OPF)
add_subdirectory(src/Drivers)

add_library(gollnlp STATIC $<TARGET_OBJECTS:goSCACOPF>
                        $<TARGET_OBJECTS:goUtils>)


##########################################################
# CMake Tests
##########################################################
if (GOLLNLP_WITH_MAKETEST)
  enable_testing()
  #add_test(NAME NlpDenseCons1_5H  COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck)
endif(GOLLNLP_WITH_MAKETEST)

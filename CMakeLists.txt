cmake_minimum_required (VERSION 2.8)
project (goSpecCpp)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

#file(WRITE "src/Interface/hiop_defs.hpp" "")

option(GOLLNLP_USE_MPI "Build with MPI support" ON)
#option(HIOP_DEEPCHECKS "Extra checks and asserts in the code with a high penalty on performance" ON)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(GOLLNLP_WITH_MAKETEST "Enable 'make test'" ON)

if(GOLLNLP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
  #file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_USE_MPI\n")
else()
  #file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_USE_MPI\n#undef HIOP_USE_MPI\n#endif\n")
endif(GOLLNLP_USE_MPI)

# if(HIOP_DEEPCHECKS)
#   file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_DEEPCHECKS\n")
# else()
#   file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_DEEPCHECKS\n#undef HIOP_DEEPCHECKS\n#endif\n")
# endif(HIOP_DEEPCHECKS)

find_package(OpenMP)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
endif(NOT DEFINED LAPACK_LIBRARIES)
message("LAPACK/BLAS libs: ${LAPACK_LIBRARIES}")

# include build directory for Fortran name mangling header
include_directories(${CMAKE_BINARY_DIR})

include_directories(src/Utils)

add_subdirectory(src/Utils)

##########################################################
# CMake Tests
##########################################################
if (GOLLNLP_WITH_MAKETEST)
  enable_testing()
  #add_test(NAME NlpDenseCons1_5H  COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck)
  if(GOLLNLP_USE_MPI)
    #add_test(NAME NlpDenseCons1_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex1.exe> 50000 1.0 -selfcheck)
  endif(GOLLNLP_USE_MPI)
endif(GOLLNLP_WITH_MAKETEST)

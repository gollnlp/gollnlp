#include "ACOPFKronRedProblem.hpp"

using namespace std;
using namespace gollnlp;

#include "goTimer.hpp"

// INST_DIR=/home/petra1/work/projects/gocompet/goinstances/trial1/T1S3_Offline/Network_01O-3; 
// CON=$INST_DIR/scenario_1/case.con; INL=$INST_DIR/case.inl; RAW=$INST_DIR/scenario_1/case.raw; ROP=$INST_DIR/case.rop
// ./src/SCACOPFProblem/acopf_kron_driver.exe $CON $INL $RAW $ROP

int main(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }

  int retcode=0;
  gollnlp::goTimer ttot(gollnlp::goTimer::tStart); 

  if(argc==5) {

    SCACOPFData data;
    if(!data.readinstance(argv[3], argv[4], argv[2], argv[1])) {
      printf("error occured while reading instance\n");
      retcode=-2;
    }
    
    ACOPFKronRedProblem prob(data);
    prob.assemble();

    
    prob.use_nlp_solver("ipopt");
    bool bret = prob.optimize("ipopt");
    if(!bret) retcode=-3;

  } else {
    printf("Usage: %s con_file inl_file raw_file rop_file\n", argv[0]);
    retcode=-1;
  }

  MPI_Finalize();
  return retcode;
}

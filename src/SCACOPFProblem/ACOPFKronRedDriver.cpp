#include "ACOPFKronRedProblem.hpp"

using namespace std;
using namespace gollnlp;

#include "goTimer.hpp"

#include <complex>
#include <cmath>
#include <iostream>
#include <cassert>

/*
INST_DIR=/home/petra1/work/projects/gocompet/goinstances/trial1/T1S3_Offline/Network_01O-3; 
CON=$INST_DIR/scenario_1/case.con; INL=$INST_DIR/case.inl; RAW=$INST_DIR/scenario_1/case.raw; ROP=$INST_DIR/case.rop;
./src/SCACOPFProblem/acopf_kron_driver.exe $CON $INL $RAW $ROP

# on mac
export INST_DIR=/Users/petra1/work/projects/kron_gitlab_pnnl/hiop-framework/instances/Network_01R-3/;
CON=$INST_DIR/scenario_1/case.con; INL=$INST_DIR/case.inl; RAW=$INST_DIR/scenario_1/case.raw; ROP=$INST_DIR/case.rop;
./src/SCACOPFProblem/acopf_kron_driver.exe $CON $INL $RAW $ROP
*/


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

    prob.use_nlp_solver("hiop");
    bool bret = prob.optimize("ipopt");
    //if(!bret) retcode=-3;

  } else {
    printf("Usage: %s con_file inl_file raw_file rop_file\n", argv[0]);
    retcode=-1;
  }

  //using namespace std::complex_literals;
  int vsize = 10;
  assert(vsize>=1);

  complex<double>* va = new complex<double>[vsize];
  assert(va);
  double* vad = new double[vsize*2];
  assert(vad);

  if(true){
    complex<double> a = 1.; a.imag(1.17);
    vector<complex<double> > v(vsize, a);
    
    
    v[vsize-1] = exp(v[0]);
    
    for(auto& e : v) cout << e << " "; cout << endl;
    

    for(int i=0; i<vsize; i++)
      memcpy(va, &v[0], vsize*sizeof(complex<double>));
    memcpy(vad, va,   vsize*2*sizeof(double));
  }

  for(int i=0; i<vsize; i++) cout << va[i].real() << " " << va[i].imag() << " "; cout << endl;
  for(int i=0; i<2*vsize; i++) cout << vad[i] << " "; cout << endl;
  
  delete[] va;
  delete[] vad;

  MPI_Finalize();
  return retcode;
}

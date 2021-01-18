#include "go_code1.hpp"
#include <string>
#include <iostream>

#include "mpi.h"

#include "SCACOPFProblem.hpp"

#include "goUtils.hpp"

using namespace std;
using namespace gollnlp;

#include "unistd.h"
#include <chrono>
#include <thread>
#include "goTimer.hpp"

// usage example
//
// small problem
// ./src/SCACOPFProblem/acopf_driver.exe ../../goinstances/trial1/T1S3_Real-Time/Network_01R-3/scenario_1/case.con ../../goinstances/trial1/T1S3_Real-Time/Network_01R-3/case.inl ../../goinstances/trial1/T1S3_Real-Time/Network_01R-3/scenario_1/case.raw ../../goinstances/trial1/T1S3_Real-Time/Network_01R-3/case.rop 600 1 Network07R
//
// medium-to-small problems
// ./src/SCACOPFProblem/acopf_driver.exe ../../goinstances/trial1/T1S3_Real-Time/Network_07R-3/scenario_3/case.con ../../goinstances/trial1/T1S3_Real-Time/Network_07R-3/case.inl ../../goinstances/trial1/T1S3_Real-Time/Network_07R-3/scenario_3/case.raw ../../goinstances/trial1/T1S3_Real-Time/Network_07R-3/case.rop 600 1 Network07R
// ./src/SCACOPFProblem/acopf_driver.exe ../../goinstances/trial1/T1S3_Offline/Network_07O-3/scenario_3/case.con ../../goinstances/trial1/T1S3_Offline/Network_07O-3/case.inl ../../goinstances/trial1/T1S3_Offline/Network_07O-3/scenario_3/case.raw ../../goinstances/trial1/T1S3_Offline/Network_07O-3/case.rop 600 1 Network07
//
// medium problem
// ./src/SCACOPFProblem/acopf_driver.exe ../../goinstances/trial3/Trial_3_Offline/Network_20O-100/scenario_97/case.con ../../goinstances/trial3/Trial_3_Offline/Network_20O-100/scenario_97/case.inl ../../goinstances/trial3/Trial_3_Offline/Network_20O-100/scenario_97/case.raw ../../goinstances/trial3/Trial_3_Offline/Network_20O-100/scenario_97/case.rop 600 1 Net20
//

int main(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }

  int retcode=0;
  gollnlp::goTimer ttot; ttot.start();

  if(argc==8) {
    
    double timeLimit = atof(argv[5]);
    int scoringMethod = atoi(argv[6]);
    
    if(timeLimit <=0 ) {
      std::cout << "invalid time limit? > " << argv[5] << std::endl;
    }
    
    if(scoringMethod <1 || scoringMethod >4 ) {
      std::cout << "invalid scoring method? > " << argv[6] << std::endl;
    }
    
    SCACOPFData data;
    if(!data.readinstance(argv[3], argv[4], argv[2], argv[1])) {
      printf("error occured while reading instance\n");
      return -1;
    }
    
    
    SCACOPFProblem*    scacopf_prob = new SCACOPFProblem(data);
    //scacopf_prob->set_AGC_as_nonanticip(true);
    scacopf_prob->set_AGC_simplified(true);
    

    vector<int> K_idxs = {};
    scacopf_prob->assembly(K_idxs);

    //toogle between hiop and ipopt
    //const char solver_name[] = "hiop";
    const char solver_name[] = "ipopt";
    
    scacopf_prob->use_nlp_solver(solver_name); 
    // scacopf_prob->set_solver_option("linear_solver", "ma57"); 
    // scacopf_prob->set_solver_option("mu_init", 1e-4);
    // scacopf_prob->set_solver_option("print_frequency_iter", 1);
    // scacopf_prob->set_solver_option("mu_target", 5e-9);
    // scacopf_prob->set_solver_option("max_iter", 600);
    
    // scacopf_prob->set_solver_option("acceptable_tol", 1e-4);
    // scacopf_prob->set_solver_option("acceptable_constr_viol_tol", 1e-6);
    // scacopf_prob->set_solver_option("acceptable_iter", 7);

    // scacopf_prob->set_solver_option("bound_relax_factor", 0.);
    // scacopf_prob->set_solver_option("bound_push", 1e-16);
    // scacopf_prob->set_solver_option("slack_bound_push", 1e-16);
    // scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.4);
    // scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.4);
    
    // scacopf_prob->set_solver_option("print_level", 5);
  
    bool bret = scacopf_prob->optimize(solver_name);

    double cost = scacopf_prob->objective_value();
  
    delete scacopf_prob;
    // MyCode1 code1(argv[3], argv[4], argv[2], argv[1], 
    // 		  timeLimit, scoringMethod, argv[7]);

    // if(!(retcode=code1.initialize(argc, argv))) {
    //   printf("Error initializing code1\n");
    // } else {
    //   retcode = code1.go();
    // }
    // if(0!=retcode) {
    //   printf("Something went wrong with code1: return code %d; it took %g seconds\n",
    //        retcode, ttot.stop());
    // }
    // MPI_Finalize();
    return 0;
  } else {
    std::string root, net, scen, name;

    std::cout << argv[0] << " did not receive the correct number of parameters. Will exit.\n";

    return -1;
    
  }
}

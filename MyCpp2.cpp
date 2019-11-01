#include "go_code2.hpp"
#include <string>
#include <iostream>

#include "goTimer.hpp"

#ifdef GOLLNLP_FAULT_HANDLING
#include "goSignalHandling.hpp"
#endif

int main(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }

  int retcode=0;
  gollnlp::goTimer ttot; ttot.start();

  std::cout << "MyExe2 - v. Oct 30, 2019 - 06:36PM" << std::endl;
#ifdef DEBUG
  std::cout << "DEBUG build !!!!" << std::endl;
#endif

#ifdef GOLLNLP_FAULT_HANDLING
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank>=1) {
    std::string msg = "[warning] fault on rank=" + std::to_string(rank) + " occured!\n";
    set_fault_signal_message(msg.c_str());
    enable_fault_signal_handling(gollnlp_fault_handler);
  }
#endif

  if(argc==8) {

    double timeLimit = atof(argv[5]);
    int scoringMethod = atoi(argv[6]);

    if(timeLimit <=0 ) {
      std::cout << "invalid time limit? > " << argv[5] << std::endl;
    }

    if(scoringMethod <1 || scoringMethod >4 ) {
      std::cout << "invalid scoring method? > " << argv[6] << std::endl;
    }

    MyCode2 code2(argv[3], argv[4], argv[2], argv[1], 
		  timeLimit, scoringMethod, argv[7]);

    if(!(retcode=code2.initialize(argc, argv))) {
      printf("Error initializing code1\n");
    } else {
      retcode = code2.go();
    }
    if(0!=retcode) {
      printf("Something went wrong with code2: return code %d; it took %g seconds\n",
           retcode, ttot.stop());
    }
    MPI_Finalize();
    return retcode;
    
  } else {

    std::cout << argv[0] << " did not receive the correct number of parameters. Will exit.\n";
    MPI_Finalize();
    return -1;
  }
  

  return 0;
}


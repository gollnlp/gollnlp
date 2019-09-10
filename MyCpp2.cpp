#include "go_code2.hpp"
#include <string>
#include <iostream>

#include "goTimer.hpp"

int main(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }

  int retcode=0;
  gollnlp::goTimer ttot; ttot.start();

  std::cout << "MyExe2 - v. Sept 09, 2019 - 09:57pm" << std::endl;
#ifdef DEBUG
  std::cout << "DEBUG build !!!!" << std::endl;
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

    //MyCode1 code1(argv[1], argv[2], argv[3], argv[4], 
    MyCode2 code2(argv[3], argv[4], argv[2], argv[1], 
		  timeLimit, scoringMethod, argv[7]);

    if(!(retcode=code2.initialize(argc, argv))) {
      printf("Error initializing code1\n");
    } else {
      retcode = code2.go();
    }
    if(0!=retcode) {
      printf("Something went wrong with code1: return code %d; it took %g seconds\n",
           retcode, ttot.stop());
    }
    MPI_Finalize();
    return retcode;
    
  } else {
    MPI_Finalize();
    std::cout << argv[0] << " did not receive the correct number of parameters. Will exit.\n";
    return -1;
  }

  return 0;
}

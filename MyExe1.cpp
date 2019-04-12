#include "go_code1.hpp"

#include <iostream>

int main(int argc, char *argv[])
{

  
  return 
    myexe1_function("../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_9/case.raw",  
		    "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/case.rop", 
		    "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/case.inl",
		    "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_9/case.con", 2700.,  2, "Network_03");


  if(argc==8) {
    std::cout << "MyExe1 - v. April 12, 2019" << std::endl;
    double timeLimit = atof(argv[5]);
    int scoringMethod = atoi(argv[6]);

    if(timeLimit <=0 ) {
      std::cout << "invalid time limit? > " << argv[5] << std::endl;
    }

    if(scoringMethod <1 || scoringMethod >4 ) {
      std::cout << "invalid scoring method? > " << argv[6] << std::endl;
    }


    return myexe1_function(argv[1], argv[2], argv[3], argv[4], 
			   timeLimit, scoringMethod, argv[7]);
  } else {
    std::cout << argv[0] << " did not receive the correct number of parameters. Will exit.\n";
  }
  return 0;
}

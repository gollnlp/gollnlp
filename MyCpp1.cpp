#include "go_code1.hpp"
#include <string>
#include <iostream>

#include "goTimer.hpp"

// just for testing
// to be removed
int myexe1_function(const std::string& InFile1, const std::string& InFile2,
		    const std::string& InFile3, const std::string& InFile4,
		    double TimeLimitInSeconds, 
		    int ScoringMethod, 
		    const std::string& NetworkModel)
{
  
  MyCode1 code1(InFile1, InFile2, InFile3, InFile4,
		TimeLimitInSeconds, ScoringMethod, NetworkModel);

  if(!code1.initialize(0, NULL)) {
    printf("Error initializing code1 tmp\n");
    return -1;
  }
  return code1.go();
}

int main(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }

  int retcode=0;
  gollnlp::goTimer ttot; ttot.start();

  std::cout << "MyExe1 - v. Sept 13, 2019 - 12:17pm" << std::endl;
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
    MyCode1 code1(argv[3], argv[4], argv[2], argv[1], 
		  timeLimit, scoringMethod, argv[7]);

    if(!(retcode=code1.initialize(argc, argv))) {
      printf("Error initializing code1\n");
    } else {
      retcode = code1.go();
    }
    if(0!=retcode) {
      printf("Something went wrong with code1: return code %d; it took %g seconds\n",
           retcode, ttot.stop());
    }
    MPI_Finalize();
    return retcode;
    
  } else {
    std::string root, net, scen, name;

    std::cout << argv[0] << " did not receive the correct number of parameters. Will exit.\n";

    return -1;

    name = "Network_01O-10/";
    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/";
    net = name + "/";
    scen = "scenario_1/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop",
    //			   root+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);
    

    name = "Network_03O-10/";
    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/";
    net = name + "/";
    scen = "scenario_4/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", 
    //			   root+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    name = "Network_05O-10/";
    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/";
    net = name + "/";
    scen = "scenario_9/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", 
    //			   root+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/";
    name = "Network_07R-10";
    net = name + "/";
    scen = "scenario_9/";
    root = root+net; 
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", 
    //		   root+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    name = "Network_07O-10/";
    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/";
    net = name + "/";
    scen = "scenario_9/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", 
    //			   root+"case.inl", root+scen+"case.con", 
    //			   600.,  2, name);

    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/";
    name = "Network_10R-10/";
    net = name + "/";
    scen = "scenario_9/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", root+scen+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/";
    name = "Network_10O-10/";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", root+scen+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);
    

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_06O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_06R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+"case.rop", root+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_12O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_12R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    //return myexe1_function(root+scen+"case.raw",  root+scen+"case.rop", root+scen+"case.inl", root+scen+"case.con", 
    //			   600.,  1, name);

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_13O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_12R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    return myexe1_function(root+scen+"case.raw",  root+"case.rop", root+"case.inl", root+scen+"case.con", 
    			   600.,  1, name);

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_70O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_12R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    return myexe1_function(root+scen+"case.raw",  root+"case.rop", root+"case.inl", root+scen+"case.con", 
    			   600.,  1, name);

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_81O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_12R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    return myexe1_function(root+scen+"case.raw",  root+scen+"case.rop", root+scen+"case.inl", root+scen+"case.con", 
    			   600.,  1, name);

    root = "../../goinstances/trial1/T1S3_Offline/";
    name = "Network_84O-3";
    //root = "../../goinstances/trial1/T1S3_Real-Time/";
    //name = "Network_12R-3";
    net = name + "/";
    scen = "scenario_2/";
    root = root+net;
    return myexe1_function(root+scen+"case.raw",  root+scen+"case.rop", root+scen+"case.inl", root+scen+"case.con", 
    			   600.,  1, name);
  }
  
  return 0;
}


//"../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_9/case.con"  "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/case.inl" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_9/case.raw" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_03O-10/case.rop"  2700  2 "Network_03"

//LD_PRELOAD=/usr/lib64/libgomp.so.1:/export/home/petra1/intel_mkl2019/mkl/lib/intel64/libmkl_gnu_thread.so:/export/home/petra1/intel_mkl2019/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64/libmkl_core.so:/export/home/petra1/intel_mkl20_libraries_2019.2.187/linux/mkl/lib/intel64_lin/libmkl_avx2.so julia ../MyJulia1.jl "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_9/case.con"  "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_9/case.inl" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_9/case.raw" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/case.rop"  2700  2 "Network_10"


//LD_PRELOAD=/usr/lib64/libgomp.so.1:/export/home/petra1/intel_mkl2019/mkl/lib/intel64/libmkl_gnu_thread.so:/export/home/petra1/intel_mkl2019/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64/libmkl_core.so:/export/home/petra1/intel_mkl20_libraries_2019.2.187/linux/mkl/lib/intel64_lin/libmkl_avx2.so julia ../MyJulia1.jl "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_1/case.con"  "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_1/case.inl" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/scenario_1/case.raw" "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Offline_Edition_1/Network_10O-10/case.rop"  2700  2 "Network_10"

#ifndef GO_CODE1
#define GO_CODE1

#include "mpi.h"
#include <string>
#include <vector>

#include "SCACOPFProblem.hpp"

class MyCode1
{
public:
  MyCode1(const std::string& InFile1, const std::string& InFile2,
	  const std::string& InFile3, const std::string& InFile4,
	  double TimeLimitInSeconds, 
	  int ScoringMethod, 
	  const std::string& NetworkModelName,
	  MPI_Comm comm_world=MPI_COMM_WORLD);
  virtual ~MyCode1();

  virtual int initialize(int argc, char *argv[]);
  virtual int go();
  virtual void display_instance_info();

private: //methods
  //
  // phase 1 - solve SCACOPF with small number of scenarios, possibly 0
  //
  void phase1_ranks_allocation();
  std::vector<int> phase1_SCACOPF_contingencies();

  bool do_phase1();
  // SCACOPF problem, maintained on solver ranks, but created on others as well as
  // a template
  gollnlp::SCACOPFProblem *scacopf_prob; 
  //
  // phase 2 - evaluate contingencies scenarios corresponding to phase 1 solution
  // stop after a limited number of contingencies with high penalty are found
  //
  void phase2_ranks_allocation();

  bool do_phase2();
  //
  // phase 3 - solve SCACOPF with the (addtl) contingencies found in phase 2
  // 
  //
  void phase3_ranks_allocation();

private: //data members
  std::string InFile1, InFile2, InFile3, InFile4;
  double TimeLimitInSec;
  int ScoringMethod;
  std::string NetworkModel;

  gollnlp::SCACOPFData data;
  
  //
  // communication
  //
  bool iAmMaster;    //master rank that deals with centralizing the communication
  bool iAmSolver;    //rank(s) that solve SC-ACOPF problem
  bool iAmEvaluator; //ranks that evaluate recourse given SC-ACOPF base case solution

  int my_rank;
  
  //rank of the master, usually 0, available on all ranks
  int rank_master;

  // rank of the solver master rank (or rank 0 in 'comm_solver') within
  // the 'comm_world' communicator
  int rank_solver_rank0;
  // ranks of the evaluators, used on the master rank
  std::vector<int> ranks_evaluators; 
  
  MPI_Comm comm_world;
  MPI_Comm comm_solver;

};
#endif

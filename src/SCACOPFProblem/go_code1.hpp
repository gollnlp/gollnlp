#ifndef GO_CODE1
#define GO_CODE1

#include "mpi.h"
#include <string>
#include <vector>

#include "SCACOPFProblem.hpp"
#include "ContingencyProblem.hpp"
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
  /////////////////////////////////////////////////////////////////////////////////
  // phase 1 - solve SCACOPF with small number of scenarios, possibly 0
  /////////////////////////////////////////////////////////////////////////////////
  void phase1_ranks_allocation();
  std::vector<int> phase1_SCACOPF_contingencies();
  //K idxs considered in phase1 SCACOPF
  //holds indexes of contingencies in data.K_Contingency
  std::vector<int> K_SCACOPF_phase1;
  
  bool do_phase1();
  // SCACOPF problem, maintained on solver ranks, but created on others as well as
  // a template, which is used to get sizes (of primal and dual solutions, etc)
  //
  // on evaluators, primal and dual variables are initialized with the master's
  // respective solutions (Bcast)
  gollnlp::SCACOPFProblem *scacopf_prob;
  
  /////////////////////////////////////////////////////////////////////////////////
  // phase 2 - evaluate contingencies using phase 1 solution
  // stop after a limited number of contingencies with high penalty are found
  /////////////////////////////////////////////////////////////////////////////////
  void phase2_ranks_allocation();

  //contingencies that need to be considered
  std::vector<int> phase2_contingencies();
  //initialization of phase 2 data and initial distribution of contingencies to
  // evaluators ranks
  void phase2_initial_contingency_distribution();

  //computes the next contingency idx given the last one
  //
  //default implementation just finds the next consecutive contingency
  //from the rank's chunk that has not been evaluated yet. If no 
  //contingency left, looks for a non-evaluated one starting at the beginning
  //
  //returns -1 when no contingencies are left, otherwise the next contingency
  int get_next_contingency(int Kidx_last, int rank);
  
  //the above contingencies minus the ones in SCACOPF phase1
  //holds indexes of contingencies in data.K_Contingency
  std::vector<int> K_phase2;

  //contingencies penalty:  -1e+20 when the contingency has not been processed
  //the same order as in K_phase2
  std::vector<double> K_penalty_phase2;
  
  //primal solutions for contingencies
  //the same order as in K_phase2
  //inner vector empty for contingencies with small penalty
  std::vector<std::vector<double> > K_primals_phase2;

  //contingencies processed on each rank
  //outer size num_ranks, maintained only on master rank
  // -1 will be sent to (and pushed_back for) each rank to signal the end of evaluations
  // -2 will be pushed_back to mark that no more sends needs to be posted for the rank
  std::vector<std::vector<int> > K_on_rank;

  //contingencies processed by current rank
  std::vector<int> K_on_my_rank;

  //
  // tags
  //
  int Tag0;// = 10000;
  int MSG_TAG_SZ;//=num_K
  // send/recv Kidxs: Tag0 + Kidx_sendrecv_counter_for_rank
  // send/recv penalty obj: Tag0+MSG_TAG_SZ+sendrecv_penalty_counter_for_rank
  // send/recv solution large penalty: Tag0+2*MSG_TAG_SZ+sendrecv_solution_counter_for_rank
  // 
  
  struct ReqKidx
  {
    ReqKidx() : ReqKidx(-1) {}
    ReqKidx(const int& K_idx_)
      : K_idx(K_idx_)
    {
      buffer[0]=K_idx_;
    }
    int K_idx; //for which contingency
    MPI_Request request;
    int buffer[1];
  };
  //on master rank
  //size num_ranks; at most one request per evaluator rank
  std::vector<std::vector<ReqKidx*> > req_send_K_idx_for_rank;
  //on evaluator rank
  ReqKidx* req_recv_K_idx;
  
  struct ReqPenalty
  {
    ReqPenalty() : ReqPenalty(-1) {}
    ReqPenalty(const int& K_idx_)
      : K_idx(K_idx_)
    {
      buffer[0]=-1e+20;
    }
    int K_idx; //for which contingency
    MPI_Request request;
    double buffer[1];
  };

  //on master rank
  //size num_ranks; at most one request per evaluator rank
  std::vector<std::vector<ReqPenalty*> > req_recv_penalty_for_rank;
  //on evaluator rank
  ReqPenalty* req_send_penalty;
  
  //ranks types: master (1), solver(2), evaluator(4) and combintations of them
  // master and evaluator 5, solver and evaluator 6, ...
  std::vector<int> type_of_rank;



  void phase2_initialization();
  bool do_phase2();

  //returns true when finished: no more contingency left and send/recv
  //messages completed
  bool do_phase2_master_part();
  bool do_phase2_evaluator_part();
  //
  // phase 3 - solve SCACOPF with the (addtl) contingencies found in phase 2
  // 
  //
  void phase3_ranks_allocation();

  //
  //utilities
  //

  //K_idx is the index in data.K_Contingency that will be solved for
  //status is on return OK=0 or failure<0 or OK-ish>0
  //return penalty/objective for the contingency problem
  double solve_contingency(int K_idx, int& status);
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

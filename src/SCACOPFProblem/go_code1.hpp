#ifndef GO_CODE1
#define GO_CODE1

#include "mpi.h"
#include <string>
#include <vector>

#include "SCACOPFProblem.hpp"
#include "ContingencyProblem.hpp"

#include "goTimer.hpp"

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

  //high penalty threshold factor relative to the basecase ACOPF from phase1
  double high_pen_threshold_factor;
  double cost_basecase;
  double pen_threshold;

  // number of contingencies with penalty >= high_pen_threshold_factor*penalty_basecase
  int number_of_high_penalties(const std::vector<double>& K_penalties);

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
  // send/recv scacopf Kidxs (between master and solver): Tag0+3*MSG_TAG_SZ+rank_solver+phase3_passes
  // send/recv scacopf penalty/handshake (between master and solver): Tag0+4*MSG_TAG_SZ+rank_solver+phase3_passes
  
  //K_idx =-1 means no more contingencies to evaluate
  //K_idx =-3 is sent to solver rank to instruct him to stay on hold for scacopf solve
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

  struct ReqKidxSCACOPF
  {
    ReqKidxSCACOPF(const std::vector<int>& K_idx_scacopf)
    {
      buffer = K_idx_scacopf;
    }
    std::vector<int> buffer;
    MPI_Request request;
  private:
    ReqKidxSCACOPF() {};
  };
  ReqKidxSCACOPF* req_send_KidxSCACOPF; //on master rank
  ReqKidxSCACOPF* req_recv_KidxSCACOPF; //on solver rank

  ReqPenalty* req_recv_penalty_solver; //on master rank
  ReqPenalty* req_send_penalty_solver; //on solver rank
  //struct ReqPDBaseCaseSolution
  //{
  // ReqPDBaseCaseSolution();
  //private: 
  // ReqPDBaseCaseSolution() {};
  //};

  //ranks types: master (1), solver(2), evaluator(4) and combinations of them
  // master and evaluator 5, solver and evaluator 6, ...
  std::vector<int> type_of_rank;

  void phase2_initialization();
  bool do_phase2();

  //returns true when finished: no more contingency left and send/recv
  //messages completed
  bool do_phase2_master_part();
  bool do_phase2_evaluator_part(int& switchToSolver);
  //
  // phase 3 - solve SCACOPF with the (addtl) contingencies found in phase 2
  // 
  //
  void phase3_ranks_allocation();

  //on master rank
  bool do_phase3_master_solverpart(bool enforce_solve);
  //on solver rank
  bool do_phase3_solver_part(); 

  //on solver rank
  std::vector<int> K_SCACOPF_phase3;

  int phase3_scacopf_passes_master;
  int phase3_scacopf_passes_solver; 

  //max high penalty contingencies to wait for initially
  int phase3_max_K_evals_to_wait_for;
  //max high penalty contingencies to put in the scacopf in phase3
  int phase3_initial_num_K_in_scacopf;
  //max high penalty to stop the solver rank from evaluating
  int phase3_max_K_to_start_solver;
  //how many additional contingencies to add to SCACOPF problem after each scacopf solve pass
  int phase3_adtl_num_K_at_each_pass;

  int phase3_last_num_K_nonproximal;
  std::vector<int> K_idxs_phase3;
  std::vector<int> get_high_penalties_from(const std::vector<double>& K_penalties, 
					   const std::vector<int> K_idxs_global,
					   const int& conting_evals_done);
  std::vector<int> get_high_penalties_from2(const std::vector<double>& K_penalties, 
					   const std::vector<int> K_idxs_global);

  //
  //utilities
  //

  //K_idx is the index in data.K_Contingency that will be solved for
  //status is on return OK=0 or failure<0 or OK-ish>0
  //return penalty/objective for the contingency problem
  double solve_contingency(int K_idx, int& status);

  //phase 3 solve scacopf with newly received K_idxs (on solver rank only)
  double phase3_solve_scacopf(std::vector<int>& K_idxs);
private: //data members
  std::string InFile1, InFile2, InFile3, InFile4;
  double TimeLimitInSec;
  int ScoringMethod;
  std::string NetworkModel;

  gollnlp::SCACOPFData data;

  double TL_rate_reduction;  

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

  gollnlp::goTimer glob_timer;

};
#endif

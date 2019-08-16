#ifndef GO_CODE1
#define GO_CODE1

#include "mpi.h"
#include <string>
#include <vector>
#include <list>
#include <cmath>

#include "SCACOPFProblem.hpp"
#include "ContingencyProblem.hpp"

#include "goTimer.hpp"

#include <iostream>

using namespace std;

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

  //
  //contingencies information
  // - penalty:  -1e+20 when the contingency has not been processed
  // - power 4 doubles: value of power (of gen, line, or transf) in the basecase 
  //at which the penalty occured
  // - n_scacopf_solves
  // - n_evals: how many times the contingency was evaluated
  struct ContingInfo
  {
    ContingInfo(const int& idx_, const int& K_idx_) : ContingInfo(idx_, K_idx_, -1e+20) {}
    ContingInfo(const int& idx_, const int& K_idx_, const double& penalty_) 
      : idx(idx_), K_idx(K_idx_), penalty(penalty_) 
    { 
      p1=q1=p2=q2=0.; n_scacopf_solves=n_evals=0; 
      scacopfed_at_pass=0; evaled_with_sol_at_pass=0;
      rank_eval=-1;
      max_K_evals = MyCode1::MAX_K_EVALS;
    }

    int idx; //in K_phase2, which is formed of indexes in data.K_contingencies
    int K_idx; 

    int scacopfed_at_pass;
    int evaled_with_sol_at_pass;
    double penalty;
    double p1,q1,p2,q2;

    int n_scacopf_solves; //scacopf solves with this contingecny
    int n_evals; //how many times the conting was evaluated
    int rank_eval; //on which rank the evaluation was done last time
    vector<int> scacopf_actions; //-102, -101, or positive x (contingency was combined with conting idx x)
    int max_K_evals;

    inline bool operator==(const ContingInfo& other) const
    {
      if(idx != other.idx) return false;
      if(n_scacopf_solves != other.n_scacopf_solves) return false;
      if(scacopf_actions != other.scacopf_actions) return false;
      if(std::fabs(p1 - other.p1) > 1e-10) return false;
      if(std::fabs(q1 - other.q1) > 1e-10) return false;
      if(std::fabs(p2 - other.p2) > 1e-10) return false;
      if(std::fabs(q2 - other.q2) > 1e-10) return false;
      if(std::fabs(penalty - other.penalty) > 1e-10) return false;
      return true;
    }
  };
  friend ostream& operator<<(ostream& os, const ContingInfo& o);

  //std::vector<double> K_penalty_phase2;
  //the same order as in K_phase2
  std::vector<ContingInfo> K_info_phase2;  
  std::vector<ContingInfo> K_info_last_scacopf_solve;

  //computes the next contingency idx given the last one
  //
  //default implementation just finds the next consecutive contingency
  //from the rank's chunk that has not been evaluated yet. If no 
  //contingency left, looks for a non-evaluated one starting at the beginning
  //
  //returns -1 when no contingencies are left, otherwise the next contingency
  int get_next_contingency(int Kidx_last, int rank);
  int get_next_conting_foreval(int Kidx_last, int rank, vector<ContingInfo>& K_info_all);
  //the above contingencies minus the ones in SCACOPF phase1
  //holds indexes of contingencies in data.K_Contingency
  std::vector<int> K_phase2;
  
  


  //high penalty threshold factor relative to the basecase ACOPF from phase1
  double high_pen_threshold_factor;
  double cost_basecase;
  double pen_threshold;

  // number of contingencies with penalty >= high_pen_threshold_factor*penalty_basecase
  int number_of_high_penalties(const vector<ContingInfo>& K_info);

  void determine_solver_actions(const vector<ContingInfo>& K_info_all, 
				const vector<int>& K_idxs_all, 
				const bool& master_evalpart_done,
				const vector<ContingInfo>& K_info_last_solved,
				vector<ContingInfo>& K_info_next_solve, 
				bool& changed_since_last);

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
  //int Tag0;// = 10000;
  //int MSG_TAG_SZ;//=num_K

  // Tag1 : send/recv Kidxs 
  //!: Tag0 + Kidx_sendrecv_counter_for_rank
  
  // Tag2 : send/recv penalty obj
  //! Tag0+MSG_TAG_SZ+sendrecv_penalty_counter_for_rank
 
  // Tag3 : send/recv solution large penalty
  //! Tag0+2*MSG_TAG_SZ+sendrecv_solution_counter_for_rank

  // Tag4 : send/recv scacopf Kidxs (between master and solver)
  //! Tag0+3*MSG_TAG_SZ+rank_solver+phase3_passes
  
  // Tag5 :send/recv scacopf penalty/handshake (between master and solver)
  //!  Tag0+4*MSG_TAG_SZ+rank_solver+phase3_passes
  enum Tags{Tag1=1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7};
  
  //K_idx =-1 means no more contingencies to evaluate
  //K_idx =-3 is sent to solver rank to instruct him to stay on hold for scacopf solve
  struct ReqKidx
  {
    ReqKidx() : ReqKidx(-1, -1) {}
    ReqKidx(const int& K_idx_, const int& scacopf_solve_pass)
    {
      buffer[0]=K_idx_;
      buffer[1]=scacopf_solve_pass;
    }

    int test() {
      int mpi_test_flag; MPI_Status mpi_status;
      int ierr = MPI_Test(&request, &mpi_test_flag, &mpi_status);
      assert(MPI_SUCCESS == ierr);
      return mpi_test_flag;
    }

    void post_recv(int tag, int rank_from, MPI_Comm comm) 
    {
      int ierr = MPI_Irecv(buffer, 2, MPI_INT, rank_from, tag, comm, &request);
      assert(MPI_SUCCESS == ierr);
    }
    void post_send(int tag, int rank_to, MPI_Comm comm)
    {
      int ierr = MPI_Isend(buffer, 2, MPI_INT, rank_to, tag, comm, &request);
      assert(MPI_SUCCESS == ierr);
    }

    int K_idx(){ return buffer[0]; } //for which contingency
    int scacopf_pass() { return buffer[1]; }
  private:
    MPI_Request request;
    int buffer[2];
  };
  //on master rank
  //size num_ranks; at most one request per evaluator rank
  std::vector<std::vector<ReqKidx*> > req_send_K_idx_for_rank;
  //on evaluator rank
  ReqKidx* req_recv_K_idx;
  
  struct ReqPenalty
  {
    ReqPenalty(const int& idx_, const double& penalty, const int& scacopf_pass) 
      : ReqPenalty(idx_, penalty) 
    { 
      buffer[5] = (double) scacopf_pass;
    }
    void post_send(int tag, int rank_to, MPI_Comm comm)
    {
      int ierr = MPI_Isend(buffer, 6, MPI_DOUBLE, rank_to, tag, comm, &request);
      assert(MPI_SUCCESS == ierr);
    }

    void post_recv(int tag, int rank_from, MPI_Comm comm)
    {
      int ierr = MPI_Irecv(buffer, 6, MPI_DOUBLE, rank_from, tag, comm, &request);
      assert(MPI_SUCCESS == ierr);     
    }

    int  get_generation_level() { return buffer[1]; }
    void get_transmission_levels(double& p1, double& q1, double& p2, double& q2)
    { p1=buffer[1]; q1=buffer[2]; p2=buffer[3];  q2=buffer[3];}

    int get_scacopf_pass() { return (int) buffer[5]; }

    int idxK; //for which contingency; index in K_info_phase2 and K_phase2
    MPI_Request request;
    double buffer[6];
  private:
    ReqPenalty() : ReqPenalty(-1, -1e+20) {}
    ReqPenalty(const int& idx_) : ReqPenalty(idx_, -1e+20) {} 
    ReqPenalty(const int& idx_, const double& penalty)
      : idxK(idx_)
    {
      buffer[0]=penalty; 
      buffer[1]=-1e+20; buffer[2]=-1e+20; buffer[3]=-1e+20; buffer[4]=-1e+20;
      buffer[5]=-1.;
    }

  };

  //on master rank
  //size num_ranks; at most one request per evaluator rank
  std::vector<std::vector<ReqPenalty*> > req_recv_penalty_for_rank;
  //on evaluator rank
  ReqPenalty* req_send_penalty;

  struct ReqKidxSCACOPF
  {
    ReqKidxSCACOPF()
    {
      buffer = vector<double>(7*MAX_NUM_Kidxs_SCACOPF, -1.);
    }
    ReqKidxSCACOPF(const vector<ContingInfo>& K_info_scacopf_solve) 
    {
      assert(K_info_scacopf_solve.size()<=MAX_NUM_Kidxs_SCACOPF);

      int nn = MAX_NUM_Kidxs_SCACOPF;
      buffer = std::vector<double>();
      
      for(auto& k : K_info_scacopf_solve) buffer.push_back((double)k.idx);
      while(buffer.size()<nn) buffer.push_back(-1.);
      
      for(auto& k : K_info_scacopf_solve) buffer.push_back(k.penalty);
      while(buffer.size()<2*nn) buffer.push_back(-1.);
      
      for(auto& k : K_info_scacopf_solve) {
	if(k.scacopf_actions.size()>=1)
	  buffer.push_back((double)k.scacopf_actions.back());
	else 
	  buffer.push_back(-10000.); //this is when the "finish" signal is sent
      }
      while(buffer.size()<3*nn) buffer.push_back(-1.);
      
      for(auto& k : K_info_scacopf_solve) buffer.push_back((double)k.p1);
      while(buffer.size()<4*nn) buffer.push_back(-1.);

      for(auto& k : K_info_scacopf_solve) buffer.push_back((double)k.q1);
      while(buffer.size()<5*nn) buffer.push_back(-1.);

      for(auto& k : K_info_scacopf_solve) buffer.push_back((double)k.p2);
      while(buffer.size()<6*nn) buffer.push_back(-1.);

      for(auto& k : K_info_scacopf_solve) buffer.push_back((double)k.q2);
      while(buffer.size()<7*nn) buffer.push_back(-1.);

    }
    void post_recv(int tag, int rank_from, MPI_Comm commwrld)
    {
      int ierr = MPI_Irecv(buffer.data(), buffer.size(), MPI_DOUBLE, rank_from, 
			   tag, commwrld, &request);
      assert(MPI_SUCCESS == ierr);
    }
    void post_send(int tag, int rank_to, MPI_Comm commwrld)
    {
      int ierr = MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, rank_to, 
			   tag, commwrld, &request);
      assert(MPI_SUCCESS == ierr);
    }
    std::vector<int> K_idxs()
    {
      std::vector<int> idxs;
      for(int i=0; i<buffer.size()/7; i++) {
	if(buffer[i]!=-1.) {
	  assert(buffer[i]>=0 || buffer[i]==-2);
	  idxs.push_back((int)buffer[i]);
	}
      }
      return idxs;
    }
    std::vector<int> K_actions()
    {
      std::vector<int> v; int n=buffer.size()/7; assert(n*7 == buffer.size());
      for(int i=2*n; i<3*n; i++) {
	if(buffer[i]!=-1.) {
	  //-10000. - is when the "finish" signal occurs
	  assert(buffer[i]>=0 || buffer[i]==-101 || buffer[i]==-102 || buffer[i]==-10000.);
	  v.push_back((int)buffer[i]);
	}
      }
      return v;
    }
    std::vector<double> K_penalties()
    {
      std::vector<double> v; int n=buffer.size()/7; assert(n*7 == buffer.size());
      for(int i=n; i<2*n; i++) {
	if(buffer[i]!=-1.) {
	  assert(buffer[i]>=-1e-8 || buffer[i]==-1e+20);
	  v.push_back(buffer[i]);
	}
      }
      return v;
    }
    std::vector<std::vector<double> > K_powers()
    {
      std::vector<std::vector<double> > v; 
      int n=buffer.size()/7; assert(n*7 == buffer.size());
      for(int i=3*n; i<4*n; i++) {
	if(buffer[i]!=-1.) {
	  v.push_back(vector<double>());
	  v.back().push_back(buffer[i]);
	}
      }

      for(int i=4*n; i<5*n; i++) {
	if(buffer[i]!=-1.) {
	  assert(i-4*n < v.size());
	  v[i-4*n].push_back(buffer[i]);
	}
      }
      for(int i=5*n; i<6*n; i++) {
	if(buffer[i]!=-1.) {
	  assert(i-5*n < v.size());
	  v[i-5*n].push_back(buffer[i]);
	}
      }
      for(int i=6*n; i<7*n; i++) {
	if(buffer[i]!=-1.) {
	  assert(i-6*n < v.size());
	  v[i-6*n].push_back(buffer[i]);
	}
      }
      
      return v;
    }

    std::vector<double> buffer;
    MPI_Request request;
  };
  ReqKidxSCACOPF* req_send_KidxSCACOPF; //on master rank
  ReqKidxSCACOPF* req_recv_KidxSCACOPF; //on solver rank

  ReqPenalty* req_recv_penalty_solver; //on master rank
  ReqPenalty* req_send_penalty_solver; //on evaluator ranks
  
  //used by solver rank to send to all other ranks
  struct ReqPDBaseCaseSolutionSend
  {
    ReqPDBaseCaseSolutionSend(gollnlp::SCACOPFProblem* prob, int num_scacopf_solve)
    {
      //prob->primal_variables()->copy_to(buffer);
      prob->copy_basecase_primal_variables_to(buffer);
      buffer.push_back((double)num_scacopf_solve);
    }

    void post(int tag, int from_rank, MPI_Comm comm_all) 
    {
      int err, r, comm_size;
#ifdef DEBUG
      int my_rank;
      err = MPI_Comm_rank(comm_all, &my_rank); assert(err==MPI_SUCCESS);
      assert(my_rank==from_rank);
#endif
      err = MPI_Comm_size(comm_all, &comm_size); assert(err==MPI_SUCCESS);

      for(int r=0; r<comm_size; r++) {
	if(r==from_rank) continue;
	requests.push_back(MPI_Request());
	err = MPI_Isend(buffer.data(), buffer.size(), MPI_DOUBLE, r,
			 tag, comm_all, &requests.back());
	assert(err==MPI_SUCCESS);
      }
      assert(requests.size()==comm_size-1);
    }
    bool all_are_done() {
      int ierr, mpi_test_flag; bool done=true;
      for(MPI_Request& req: requests) {
	MPI_Status mpi_status;
	ierr = MPI_Test(&req, &mpi_test_flag, &mpi_status); assert(ierr == MPI_SUCCESS);
	if(mpi_test_flag != 0) { 
	  //completed
	} else {
	  done = false;
	}
      }
      return done;
    }

    std::vector<double> buffer;
    std::vector<MPI_Request> requests;
  private: 
  };

  struct ReqPDBaseCaseSolutionSendList
  {
    virtual ~ReqPDBaseCaseSolutionSendList()
    {
      for(auto& s: sends_list) delete s;
    }
    void post_new_sol(gollnlp::SCACOPFProblem* prob, int tag, int from_rank, MPI_Comm comm_all, int num_scacopf_solve)
    {
      sends_list.push_back(new ReqPDBaseCaseSolutionSend(prob, num_scacopf_solve));
      sends_list.back()->post(tag, from_rank, comm_all);
    }
    //returns number of send requests that completed
    int attempt_cleanup() 
    {
      int ndone=0; bool do_it=true;
      while(do_it) {
	do_it=false;
	for(std::list<ReqPDBaseCaseSolutionSend*>::iterator it=sends_list.begin(); it!=sends_list.end(); ++it) {
	  if((*it)->all_are_done()) {
	    delete (*it);
	    sends_list.erase(it);
	    do_it = true; ndone++;
	    break;
	  }
	}
	return ndone;
      }
    }
    //private:
  public:
    std::list<ReqPDBaseCaseSolutionSend*> sends_list;
  };
  //on solver rank
  ReqPDBaseCaseSolutionSendList req_send_base_sols;


  //used by master and workers to receive solution from solver rank
  struct ReqPDBaseCaseSolutionRecv
  {
    void post(gollnlp::SCACOPFProblem* prob, int tag, int from_rank, MPI_Comm comm_all)
    {
      buffer = std::vector<double>(prob->primal_variables()->n()+1, -1e+10);
      int ierr, r, my_rank, comm_size;
      ierr = MPI_Irecv(buffer.data(), buffer.size(), MPI_DOUBLE, from_rank,
		       tag, comm_all, &request);
      assert(ierr == MPI_SUCCESS);
    }
    bool is_done()
    {
      int ierr, mpi_test_flag; MPI_Status mpi_status;
      ierr = MPI_Test(&request, &mpi_test_flag, &mpi_status); assert(ierr == MPI_SUCCESS);
      return (mpi_test_flag != 0);
    }
    //returns the number of the SCACOPF solve this solution belongs to
    int update_prob_variables(gollnlp::SCACOPFProblem* prob)
    {
      prob->primal_variables()->copy_from(buffer);
      return (int)buffer.back();
    }
  private:
    std::vector<double> buffer;
    MPI_Request request;
  };
  //on master and workers ranks
  ReqPDBaseCaseSolutionRecv req_recv_base_sol;

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

  int phase3_scacopf_passes_master;  //on master
  int phase3_scacopf_passes_solver;  //on solver
  // which scacopf solve the basecase solution is from
  int phase3_scacopf_pass_solution;  //on evaluators

  //max high penalty contingencies to wait for initially
  int phase3_max_K_evals_to_wait_for;
  //max high penalty contingencies to put in the scacopf in phase3
  int phase3_initial_num_K_in_scacopf;
  //max high penalty to stop the solver rank from evaluating
  int phase3_max_K_to_start_solver;
  //how many additional contingencies to add to SCACOPF problem after each scacopf solve pass
  int phase3_adtl_num_K_at_each_pass;

  int phase3_last_num_K_nonproximal;
  std::vector<int> get_high_penalties_from(const std::vector<double>& K_penalties, 
					   const std::vector<int> K_idxs_global,
					   const int& conting_evals_done);
  std::vector<int> get_high_penalties_from2(const std::vector<double>& K_penalties, 
					   const std::vector<int> K_idxs_global);

  //
  //utilities
  //

  void process_contingency(const int& K_idx, int& status_out,
			   double& penalty_out, double* info_out);
			   

  //K_idx is the index in data.K_Contingency that will be solved for
  //status is on return OK=0 or failure<0 or OK-ish>0
  //return penalty/objective for the contingency problem
  double solve_contingency(int K_idx, int& status);
  double solve_contingency_with_basecase(int K_idx, int& status);

  //phase 3 solve scacopf with newly received K_idxs (on solver rank only)
  double phase3_solve_scacopf(std::vector<int>& K_idxs, 
			      const std::vector<double>& K_penalties,
			      const std::vector<std::vector<double> >& K_powers,
			      const std::vector<int>& K_actions);
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

  static int MAX_NUM_Kidxs_SCACOPF, MAX_K_EVALS;
};


#endif

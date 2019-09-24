#include "go_code1.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"

#include "goUtils.hpp"

//using namespace std;
using namespace gollnlp;

#include <algorithm>
#include <unordered_map>

#include "unistd.h"
#include <chrono>
#include <thread>

//#define DEBUG_COMM 1
#define DEBUG_SCHED 1
//#define MAX_NUM_Kidxs_SCACOPF 512

ostream& operator<<(ostream& os, const MyCode1::ContingInfo& o)
{
  os << "idx=" << o.idx << " K_idx=" << o.K_idx << " restr=" << o.restrictions
     << " scacopfed_at_pass " << o.scacopfed_at_pass 
     << " evaled_with_sol_at_pass " << o.evaled_with_sol_at_pass
     << " -> penalty=" << o.penalty 
     << " scacopf_solves=" << o.n_scacopf_solves << " n_evals=" << o.n_evals 
     << " max_K_evals=" << o.max_K_evals <<  " force_reev=" << o.force_reeval
     << " rank=" << o.rank_eval 
    //<< " P| " << o.p1 << ' ' << o.q1 << ' ' << o.p2 << ' ' << o.q2 << " | "
     << " scacopf actions: [";
  for(auto a : o.scacopf_actions) os << a << ' ';
  
  os << ']' << std::endl;
  return os;
};

int MyCode1::MAX_NUM_Kidxs_SCACOPF=512;
int MyCode1::MAX_K_EVALS=3;
MyCode1::MyCode1(const std::string& InFile1_, const std::string& InFile2_,
		 const std::string& InFile3_, const std::string& InFile4_,
		 double TimeLimitInSeconds, 
		 int ScoringMethod_, 
		 const std::string& NetworkModelName,
		 MPI_Comm comm_world_)
  : InFile1(InFile1_), InFile2(InFile2_), InFile3(InFile3_), InFile4(InFile4_),
    TimeLimitInSec(TimeLimitInSeconds), ScoringMethod(ScoringMethod_),
    NetworkModel(NetworkModelName),
    rank_master(-1), rank_solver_rank0(-1),
    comm_world(comm_world_), comm_solver(MPI_COMM_NULL)
{
  glob_timer.start();

  iAmMaster=iAmSolver=iAmEvaluator=false;
  scacopf_prob = NULL;
  my_rank = -1; 

  req_recv_K_idx = NULL;
  req_send_penalty = NULL;

  phase3_max_K_evals_to_wait_for = -1;
  phase3_initial_num_K_in_scacopf = -1;
  phase3_max_K_to_start_solver = -1;
  phase3_last_num_K_nonproximal=0;

  phase3_scacopf_passes_master=0;
  phase3_scacopf_passes_solver=0;
  phase3_scacopf_pass_solution=-1;
  
  pen_threshold = 1; //dolars -> updated later
  high_pen_threshold_factor = 0.25; //25% relative to basecase cost
  cost_basecase=0.;

  req_send_KidxSCACOPF=NULL;
  req_recv_KidxSCACOPF=NULL;

  req_recv_penalty_solver=NULL;
  req_send_penalty_solver=NULL;

  TL_rate_reduction=1.;
}

MyCode1::~MyCode1()
{

}

int MyCode1::initialize(int argc, char *argv[])
{
  int ret = MPI_Comm_rank(comm_world, &my_rank); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    return false;
  }

  rank_master = 0;
  rank_solver_rank0 = 1;
  if(my_rank == rank_master) iAmMaster=true;

  if(my_rank!=rank_solver_rank0 && my_rank!=rank_master)  
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

  data.my_rank = my_rank;

  //load data
  if(!data.readinstance(InFile1, InFile2, InFile3, InFile4)) {
    printf("error occured while reading instance\n");
    return false;
  }


  for(int i=0; i<data.N_Vlb.size(); i++) {
    data.N_Vlb[i] *= 1.02;
    data.N_Vub[i] *= 0.98;
    //printf("%5d %8.3f %8.3f %8.3f %8.3f\n", i, data.N_Vlb[i], data.N_Vub[i], data.N_EVlb[i], data.N_EVub[i]);
  }






  phase3_initial_num_K_in_scacopf = (ScoringMethod==1 || ScoringMethod==3) ? 2 : 2;
  phase3_max_K_evals_to_wait_for = 1;//2*phase3_initial_num_K_in_scacopf;
  phase3_max_K_to_start_solver = 1;//phase3_initial_num_K_in_scacopf;
  phase3_adtl_num_K_at_each_pass = 4;

  if(data.L_Line.size()>1999 || data.N_Bus.size()>1999) {
    phase3_initial_num_K_in_scacopf = (ScoringMethod==1 || ScoringMethod==3) ? 1 : 2;
    phase3_max_K_evals_to_wait_for = 1;//2*phase3_initial_num_K_in_scacopf;
    phase3_max_K_to_start_solver = 1;//phase3_initial_num_K_in_scacopf;
    phase3_adtl_num_K_at_each_pass = 2;
  }
  if(data.L_Line.size()>4999 || data.N_Bus.size()>4999) {
    phase3_initial_num_K_in_scacopf = (ScoringMethod==1 || ScoringMethod==3) ? 1 : 1;
    phase3_max_K_evals_to_wait_for = 1;//2*phase3_initial_num_K_in_scacopf;
    phase3_max_K_to_start_solver = 1;//phase3_initial_num_K_in_scacopf;
    phase3_adtl_num_K_at_each_pass = 1;
  }
  if(data.L_Line.size()>9999 || data.N_Bus.size()>9999) {
    phase3_initial_num_K_in_scacopf = (ScoringMethod==1 || ScoringMethod==3) ? 1 : 1;
    phase3_max_K_evals_to_wait_for = 1;//2*phase3_initial_num_K_in_scacopf;
    phase3_max_K_to_start_solver = 1;//phase3_initial_num_K_in_scacopf;
    phase3_adtl_num_K_at_each_pass = 1;
  }
  if(data.L_Line.size()>14999 || data.N_Bus.size()>14999) {
    phase3_initial_num_K_in_scacopf = (ScoringMethod==1 || ScoringMethod==3) ? 1 : 1;
    phase3_max_K_evals_to_wait_for = 1;//phase3_initial_num_K_in_scacopf;
    phase3_max_K_to_start_solver = 1;//phase3_initial_num_K_in_scacopf;
    phase3_adtl_num_K_at_each_pass = 1;
  }
  phase3_scacopf_passes_solver = 0;
  phase3_scacopf_passes_master = 0;
  phase3_scacopf_pass_solution=-1;

  pen_threshold = 1.*data.K_Contingency.size(); //dolars; violations of O(1) or less allowed per contingency
  if(data.N_Bus.size()<7000) pen_threshold = 0.5*data.K_Contingency.size();
  if(data.N_Bus.size()<4000) pen_threshold = 0.25*data.K_Contingency.size();



  return true;
}
void MyCode1::phase1_ranks_allocation()
{
  iAmMaster=iAmSolver=iAmEvaluator=false;
  assert(comm_world != MPI_COMM_NULL);
  int ret, comm_size;
  ret = MPI_Comm_size(comm_world, &comm_size); assert(ret==MPI_SUCCESS);

  rank_master = 0;
  rank_solver_rank0 = 1;

  if(my_rank == rank_master) iAmMaster=true;
  if(comm_size==1) {
    iAmSolver=true; iAmEvaluator=true;
    rank_solver_rank0 = rank_master;
  } else {
    if(my_rank==rank_solver_rank0) {
      iAmSolver=true;
      iAmEvaluator=false;
    } else {
      //ranks 0, 2, 3, 4, ...
      iAmEvaluator=true;
      //no need to have master as an evaluator since evaluators do not do much
      if(my_rank==0) iAmEvaluator=false;
    }
  }
  //#ifdef DEBUG_SCHED
  //printf("[comm] [Phase 1] Rank %d ismaster %d issolver %d isevaluator %d\n",
  //	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
  //#endif
}

void MyCode1::phase2_ranks_allocation()   
{
  assert(comm_world != MPI_COMM_NULL);
  int ret, comm_size;
  ret = MPI_Comm_size(comm_world, &comm_size); assert(ret==MPI_SUCCESS);

  //solver is rank_solver_rank0
  iAmSolver = my_rank==rank_solver_rank0;

  if(my_rank == rank_master) {assert(iAmMaster); iAmMaster = true;}
  if(comm_size==1) {
    iAmEvaluator=true;
  } else {

    //ranks 0, 1, 2, 3, 4
    iAmEvaluator=true; //rank 0 is also an evaluator as long as comm_size<4
    if(my_rank==rank_master && comm_size>=4) iAmEvaluator=false;
    if(my_rank==rank_solver_rank0 && comm_size>=4) iAmEvaluator=false;
  }

  //on master
  if(iAmMaster) {
    type_of_rank.clear();

    //rank 0
    int type = 1; //master
    //rank 0 is also an evaluator as long as comm_size<4
    if(comm_size<4) type = 5;
    type_of_rank.push_back(type);

    //solver
    if(comm_size>1)
      type_of_rank.push_back(2);

    for(int r=2; r<comm_size; r++) 
      type_of_rank.push_back(4);
  }
  //#ifdef DEBUG_SCHED 
  //printf("[comm] [Phase 2] Rank %d ismaster %d issolver %d isevaluator %d\n",
  //	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
  //#endif
}

void MyCode1::phase3_ranks_allocation()
{
  //to do: grow the pool of solvers with the contingencies considered in
  //phase 3

  //for now use same strategy as in phase 1
  phase1_ranks_allocation();
}

bool MyCode1::do_phase1()
{
  printf("[ph1] rank %d  starts phase 1 global time %g\n", 
	   my_rank, glob_timer.measureElapsedTime());

  K_SCACOPF_phase1 = phase1_SCACOPF_contingencies();
  assert(NULL == scacopf_prob);

  phase1_ranks_allocation();

  //contingencies to be considered in phase 2
  K_phase2 = phase2_contingencies();//set_diff(phase2_contingencies(), K_SCACOPF_phase1);

  if(iAmMaster) {
    K_info_phase2.clear();
    for(int it=0; it<K_phase2.size(); it++)
      K_info_phase2.push_back(ContingInfo(it, K_phase2[it]));
  }


  //
  // solver scacopf problem on solver rank(s) xxxbase1
  //
  scacopf_prob = new SCACOPFProblem(data);

  scacopf_prob->my_rank = my_rank;

  scacopf_prob->update_AGC_smoothing_param(1e-4);
  scacopf_prob->update_PVPQ_smoothing_param(1e-2);
  //scacopf_prob->set_AGC_as_nonanticip(true);
  //scacopf_prob->set_AGC_simplified(true);
  //scacopf_prob->set_PVPQ_as_nonanticip(true);

  //reduce T and L rates to min(RateBase, TL_rate_reduction*RateEmer)
  TL_rate_reduction = 0.85;
  //if((ScoringMethod==1 || ScoringMethod==3))
  //  TL_rate_reduction = 0.85;

  scacopf_prob->set_basecase_L_rate_reduction(TL_rate_reduction);
  scacopf_prob->set_basecase_T_rate_reduction(TL_rate_reduction);

  //scacopf_prob->set_quadr_penalty_qg0(true);

  scacopf_prob->assembly(K_SCACOPF_phase1);

  if(iAmMaster) {
    vector<int> Kgen_infeas_gen_idxs, Kgen_infeas_K_idxs;
    scacopf_prob->find_AGC_infeasible_Kgens(Kgen_infeas_gen_idxs, 
					    Kgen_infeas_K_idxs);
    for(int& K_idx : Kgen_infeas_K_idxs) {
      for(int k=0; k<K_info_phase2.size(); k++) {
	if(K_info_phase2[k].K_idx == K_idx)
	  K_info_phase2[k].restrictions = 1; //do not include in SCACOPF
      }
    }

    //build info for contingencies priorities
  }
  
  bool blarge_prob = false;//data.N_Bus.size() > 20000;

  scacopf_prob->use_nlp_solver("ipopt"); 
  scacopf_prob->set_solver_option("sb","yes");
  scacopf_prob->set_solver_option("linear_solver", "ma57"); 

  scacopf_prob->set_solver_option("print_frequency_iter", 5);
  scacopf_prob->set_solver_option("max_iter", 2000);    
  scacopf_prob->set_solver_option("acceptable_tol", 1e-3);
  scacopf_prob->set_solver_option("acceptable_constr_viol_tol", 1e-6);
  scacopf_prob->set_solver_option("acceptable_iter", 7);

  if(!blarge_prob) {
    scacopf_prob->set_solver_option("mu_init", 1.);
    scacopf_prob->set_solver_option("tol", 1e-9);
    scacopf_prob->set_solver_option("mu_target", 1e-9);

    scacopf_prob->set_solver_option("bound_relax_factor", 0.);
    scacopf_prob->set_solver_option("bound_push", 1e-16);
    scacopf_prob->set_solver_option("slack_bound_push", 1e-16);
    scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.4);
    scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.4);

  } else {
    scacopf_prob->set_solver_option("mu_init", 0.1);
    scacopf_prob->set_solver_option("tol", 1e-8);
    scacopf_prob->set_solver_option("mu_target", 1e-9);
    
    //scacopf_prob->set_solver_option("bound_relax_factor", 0.);
    //scacopf_prob->set_solver_option("bound_push", 1e-16);
    //scacopf_prob->set_solver_option("slack_bound_push", 1e-16);
    scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.5);
    scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.2);
  }

  if(iAmSolver) {    assert(my_rank==rank_solver_rank0);
    //if(true) {
    scacopf_prob->set_solver_option("print_level", 5);

  } else {
    //master and evaluators do not solve, but we call optimize to force an
    //allocation of the internals, such as the dual variables
    scacopf_prob->set_solver_option("print_level", 1);
    scacopf_prob->set_solver_option("max_iter", 1);
  }

  printf("[ph1] rank %d  starts scacopf solve phase 1 global time %g\n", 
	   my_rank, glob_timer.measureElapsedTime());

  
  bool bret = scacopf_prob->optimize("ipopt");

  if(!iAmSolver)
    printf("[ph1] rank %d  scacopf solve phase 1 done at global time %g\n", 
	   my_rank, glob_timer.measureElapsedTime());


  //if(scacopf_prob->data_K.size()>0)
  //  scacopf_prob->print_reactive_power_balance_info(*scacopf_prob->data_K[0]);
  //scacopf_prob->print_reactive_power_balance_info(data);
  //scacopf_prob->print_line_limits_info(data);
  

  //
  //communication -> solver rank0 bcasts basecase solutions
  //

  //scacopf_prob->primal_variables()->
  //  MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);

  //scacopf_prob->duals_bounds_lower()->
  //  MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
  //scacopf_prob->duals_bounds_upper()->
  //  MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
  //scacopf_prob->duals_constraints()->
  //  MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);

  MPI_Bcast(&cost_basecase, 1, MPI_DOUBLE, rank_solver_rank0, comm_world);
  //printf("[ph1] rank %d  phase 1 basecase bcasts done at global time %g\n", 
  //	   my_rank, glob_timer.measureElapsedTime());


  //force a have_start set
  if(!iAmSolver) {
    scacopf_prob->set_have_start();
  } else {

    //scacopf_prob->print_summary();

    K_SCACOPF_phase3 = K_SCACOPF_phase1;
    attempt_write_solutions(scacopf_prob, bret);
  }

  if(my_rank<=3)
    printf("[ph1] rank %d  basecase obj %g global time %g\n", 
	   my_rank, cost_basecase, glob_timer.measureElapsedTime());
  fflush(stdout);

  // all non-solver ranks posts recv for basecase solution from solver rank
  if(!iAmSolver) {
    req_recv_base_sol.post(scacopf_prob, Tag7, rank_solver_rank0, comm_world);

    while(!req_recv_base_sol.is_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    int scacopf_pass = req_recv_base_sol.update_prob_variables(scacopf_prob);
    //printf("rank %d  basecase solution pass %d received\n", my_rank, scacopf_pass);
  } else {
    req_send_base_sols.post_new_sol(scacopf_prob, Tag7, my_rank, comm_world, phase3_scacopf_passes_solver);
    
    if(blarge_prob) {
      scacopf_prob->set_solver_option("tol", 1e-9);
      scacopf_prob->set_solver_option("bound_relax_factor", 1e-8);
      scacopf_prob->set_solver_option("bound_push", 1e-8);
      scacopf_prob->set_solver_option("slack_bound_push", 1e-8);

      //scacopf_prob->set_solver_option("bound_push", 1e-12);
      //scacopf_prob->set_solver_option("slack_bound_push", 1e-12);
      //scacopf_prob->set_solver_option("warm_start_init_point", "yes");
      //scacopf_prob->set_solver_option("warm_start_bound_push", 1e-12);
      //scacopf_prob->set_solver_option("warm_start_slack_bound_push", 1e-12);
      //scacopf_prob->set_solver_option("warm_start_mult_bound_push", 1e-12);
      //scacopf_prob->set_solver_option("warm_start_bound_frac", 1e-12);
      //scacopf_prob->set_solver_option("warm_start_slack_bound_frac", 1e-12);
      
      scacopf_prob->set_solver_option("mu_target", 1e-8);
      scacopf_prob->set_solver_option("mu_init", 1e-8);
      
      bool bret = scacopf_prob->reoptimize(OptProblem::primalDualRestart);
      attempt_write_solutions(scacopf_prob, bret);
    }      
    cost_basecase = scacopf_prob->objective_value();
    scacopf_prob->print_objterms_evals();
    printf("[ph1] rank %d  scacopf solve phase 1 done at global time %g\n", 
	   my_rank, glob_timer.measureElapsedTime());

      
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    while(!req_send_base_sols.sends_list.back()->all_are_done()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    //printf("rank %d  basecase solution pass %d send to all\n", my_rank, phase3_scacopf_passes_solver);
  }

  if(iAmMaster) {
    gollnlp::goTimer tm; tm.start();
    string str; char msg[1024];
    vector<int> Kgens_highp, Ktransm_highp;
    get_high_priority_Kgens(1024, Kgens_highp, scacopf_prob);
    get_high_priority_Ktransm(2048, Ktransm_highp, scacopf_prob);
    sprintf(msg, "Kscreen: %lu gens %lu transm ", Kgens_highp.size(), Ktransm_highp.size()); str += msg;

    auto itg = Kgens_highp.begin(), itt=Ktransm_highp.begin();
    while(itg!=Kgens_highp.end() || itt!=Ktransm_highp.end()) {
      if(itg!=Kgens_highp.end()) { K_high_prio_phase2.push_back(*itg); ++itg; }
      if(itt!=Ktransm_highp.end()) { K_high_prio_phase2.push_back(*itt); ++itt; }
    }
    sprintf(msg, " took %.4f sec\n", tm.getElapsedTime()); str += msg;
    printf("[rank 0] %s", str.c_str());
  }

  return true;
}

void MyCode1::attempt_write_solutions(gollnlp::SCACOPFProblem* prob, bool opt_success)
{
  printf("[ph1] rank %d  phase 1 writes solution1.txt at global time %g\n", 
	 my_rank, glob_timer.measureElapsedTime());
  
  //write solution
  prob->write_solution_basecase();
  prob->write_pridua_solution_basecase();
  if(!opt_success) {
    printf("[warning] Solver rank %d: initial basecase solve failed; solution1 was written though at global time=%g\n",
	   my_rank, glob_timer.measureElapsedTime());
  }
  
#ifdef DEBUG
  //write solution extras
  prob->write_solution_extras_basecase();
  if(!opt_success) {
    printf("[warning] Solver rank %d: initial basecase solve failed; solution1 extras were written though\n",
	   my_rank);
  }
#endif  
}

vector<int> MyCode1::phase1_SCACOPF_contingencies()
{
  bool testing = true;
  if(true) {
 
    //vector<int> cont_list = {0, 141, 102}; //net 01
    //vector<int> cont_list = {52}; //net 03
    //return  {101,  102,  106,  110,  249,  344,  394,  816,  817};
    vector<int> cont_list = {};//102,  817};

    return  cont_list;
  } else {
    return vector<int>();
  }
}
std::vector<int> MyCode1::phase2_contingencies()
{
  return data.K_Contingency;
  //return {0, 1, 48, 106};

  //return {101,  102,  103,  110,  249,  344,  394,  816,  817, 55, 2, 2, 3, 4, 5, 6, 7, 8, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};// 101,  102,  106,  110,  249,  344,  394,  816,  817, 55}; //net 7
  return {55, 101,  102,  106,  110,  249,  344,  394,  816,  817, 55, 497}; //net 7
  //return {106};
  return {101, 106, 344};
  return {101,  102, 106, 110,  249,  344,  394,  816,  817, 55, 497, 0, 1, 2};// 101,  102,  106,  110,  249,  344,  394,  816,  817, 55}; //net 7

}

void MyCode1::phase2_initial_contingency_distribution()
{
  hardclear(K_on_rank); hardclear(K_on_my_rank); 
  
  if(!iAmMaster) {
    //--non-master phase 2 initialization
    if(iAmSolver)
      K_on_my_rank.push_back(-2);
    //nothing at this point
    return;
  }

  //num ranks
  int R = type_of_rank.size(); assert(R>0); assert(type_of_rank[0]>=1);
  int num_ranks = R;
  for(int r=0; r<num_ranks; r++) {
    if(type_of_rank[r]==4 || type_of_rank[r]==5 || type_of_rank[r]==6) {
      //evaluator rank
    } else {
      R--;
    }
  }

  //initialize K_on_rank (vector of K idxs on each rank)
  for(int it=0; it<num_ranks; it++)
    K_on_rank.push_back(vector<int>());
  
  //num contingencies; K_phase2 contains contingencies ids in increasing order
  int S = K_phase2.size();
  //assert(S >= R);
  
  int perRank = S/R; int remainder = S-R*(S/R);
  //printf("evaluators=%d contingencies=%d perRank=%d remainder=%d\n",
  //	 R, S, perRank, remainder);

  //each rank gets one contingency idx = r*perRank
  int nEvaluators=-1;
  for(int r=0; r<num_ranks; r++) {
    if(type_of_rank[r]==4 || type_of_rank[r]==5 || type_of_rank[r]==6) {
      //evaluator rank
      nEvaluators++;
      
      int K_idx_phase2=-55;
      if(1.0*r/num_ranks <= 110.0/144 && !K_high_prio_phase2.empty()) {
	K_idx_phase2 = K_high_prio_phase2.front();
	K_high_prio_phase2.pop_front();
	K_on_rank[r].push_back(K_idx_phase2);

      } else {
	K_idx_phase2 = nEvaluators*perRank;
	if(nEvaluators<remainder)
	  K_idx_phase2 += nEvaluators;
	else
	  K_idx_phase2 += remainder;
	
	if(nEvaluators<S) {
	  //printf("r=%d K_idx=%d  K_value=%d\n", r, K_idx, K_phase2[K_idx]);
	  assert(K_idx_phase2 < K_phase2.size());
	  erase_elem_from(K_high_prio_phase2, K_idx_phase2);
	  K_on_rank[r].push_back(K_idx_phase2);
	}
      }
    }
  }
  //printvecvec(K_on_rank);

  if(K_on_rank[1].size()==0) K_on_rank[1].push_back(-2);

  assert(nEvaluators+1==R);
}

int MyCode1::get_next_conting_foreval(int Kidx_last, int rank, vector<ContingInfo>& K_info_all)
{
  assert(iAmMaster);
  assert(Kidx_last>=0);

  bool only_first_timers_last_iters = true;
  int nmax_K_prev_firsttimers_evals=3; //contingencies that were evaluated only once

  int nK_on_rank = K_on_rank[rank].size();
  if(nK_on_rank>=nmax_K_prev_firsttimers_evals) {
    //were the last few new evals? if yes, try to schedule a reevaluation of a high penalty contingency
    for(int i=0; i<nmax_K_prev_firsttimers_evals; i++) {
      assert(nK_on_rank-i-1>=0);
      int idx = K_on_rank[rank][nK_on_rank-i-1];

      if(idx ==-3) break; //rank is on hold, ok to let only_first_timers_last_iters==true
      assert(idx>=0 && idx < K_info_phase2.size());
      assert(K_info_phase2[idx].n_evals>0);
      if(K_info_phase2[idx].n_evals>1 || 
	 (K_info_phase2[idx].n_evals==1 && K_info_phase2[idx].scacopfed_at_pass>0)) 
	only_first_timers_last_iters=false;
	
    }
  }

  bool have_new_firsttimer=false;
  //basic strategy to pick the next contingency
  // get one from the front of K_high_prio_phase2
  // or 
  // try Kidx_last+1, Kidx_last+2, ..., numK, 1, 2, ..., Kidx_last-1 (and remove it from K_high_prio_phase2)
  bool done=false, is_high_prior=false; int comm_size = K_on_rank.size(); 
  int Kidx_next = Kidx_last;
  while(!done) {

    if(1.0*rank/comm_size <= 110.0/144 && !K_high_prio_phase2.empty()) {
      Kidx_next = K_high_prio_phase2.front();
      is_high_prior=true;
    } else {
      Kidx_next += 1;
      if(Kidx_next>=K_phase2.size()) Kidx_next=0;
    }

    bool previously_scheduled = false;
    for(auto& Kr : K_on_rank) {
      for(auto Kidx : Kr) {
	if(Kidx_next == Kidx) {
	  previously_scheduled=true;
	  break;
	}
      }
      if(previously_scheduled) break;
    }
    
    if(!previously_scheduled) {
      if(false == only_first_timers_last_iters) {
#ifdef DEBUG_SCHED
	printf("[sched] Master: next contingency for K_idx=%d (globidx=%d) to have "
	       "idx=%d (globidx=%d) [1] [%s] for rank %d\n",
	       Kidx_last, K_phase2[Kidx_last], Kidx_next, K_phase2[Kidx_next],
	       is_high_prior ? "high prior" : "", rank);
#endif      
	erase_elem_from(K_high_prio_phase2, Kidx_next);
	return Kidx_next;
      } else {
	//we found a new one but we don't return it since the rank has only evaluated first timers
	//recently and we may want to use the rank to reevaluate a high penalty contingency
	have_new_firsttimer=true;
	done = true;
      }
    } else { assert(is_high_prior==false); }
    
    if(Kidx_next == Kidx_last) {
      //we looped over all elements of K_phase2 and couldn't find a contingency
      //that were not already assigned to a rank
      done = true;
    }
  }

  assert(done==true);
  if(have_new_firsttimer) { assert( only_first_timers_last_iters ); assert(Kidx_next != Kidx_last); }

  bool hold=false;

  for(ContingInfo& kinfo : K_info_all) {
    if(kinfo.rank_eval==rank) {

      //try to find one that was already evaluated and is in need of a new evaluation because
      //it still has a high penalty after the evaluation or was rescheduled after an scacopf solve
      if(kinfo.n_evals==kinfo.n_scacopf_solves && 
	 kinfo.n_evals < kinfo.max_K_evals && 
	 kinfo.scacopfed_at_pass <= phase3_scacopf_passes_master) {
#ifdef DEBUG_SCHED
	printf("[sched] Master: next contingency for K_idx=%d (globidx=%d) to have "
	       "idx=%d (globidx=%d) [4] [%s] for rank %d\n",
	       Kidx_last, K_phase2[Kidx_last], kinfo.idx, K_phase2[kinfo.idx],
	       is_high_prior ? "high prior" : "", rank);
#endif   
	  return kinfo.idx;
      } 
    }
  }
  //
  // second pass - find one that has been scheduled for reval
  //
  for(ContingInfo& kinfo : K_info_all) {
    if(kinfo.rank_eval==rank) {
      if(kinfo.force_reeval) {
#ifdef DEBUG_SCHED
	printf("[sched] Master: next contingency for K_idx=%d (globidx=%d) to have "
	       "idx=%d (globidx=%d) [5] [%s] for rank %d\n",
	       Kidx_last, K_phase2[Kidx_last], kinfo.idx, K_phase2[kinfo.idx],
	       is_high_prior ? "high prior" : "", rank);
#endif 
	return kinfo.idx;
      }
    }
  }
  
  //
  // third pass -> find first one evaluated with an old basecase solution
  //
  for(ContingInfo& kinfo : K_info_all) {
    if(kinfo.rank_eval==rank) {
      //if !have_new_first_timer -> check that evaled_with_sol_at_pass == phase3_scacopf_passes_master 
      // if not, return it (evaled_with_sol_at_pass < phase3_scacopf_passes_master)
      if(!have_new_firsttimer) {
	if(kinfo.evaled_with_sol_at_pass<phase3_scacopf_passes_master) {
#ifdef DEBUG_SCHED
	  printf("[sched] Master: next contingency for K_idx=%d (globidx=%d) to have "
		 "idx=%d (globidx=%d) [3] [%s] for rank %d\n",
		 Kidx_last, K_phase2[Kidx_last], kinfo.idx, K_phase2[kinfo.idx],
		 is_high_prior ? "high prior" : "", rank);
#endif   
	  return kinfo.idx;
	}
      } //end of if have_new_firsttimer
      
      //if there are NO firsttimer contingencies and no Kidx is to be found in this pass, 
      //we may want to put the rank on hold
      if(!have_new_firsttimer) {
	if(kinfo.penalty>=pen_threshold && kinfo.n_evals<kinfo.max_K_evals) 
	  hold=true; //need to eval or eval is in progress
	if(kinfo.penalty>=pen_threshold && kinfo.scacopf_actions.size()>kinfo.n_scacopf_solves)
	  hold=true; //high penalty that is in progress of being scacopfed
      }
    }
  }
  
  if(hold) {
    assert(false==have_new_firsttimer);
    return -3;
  } else {
    if(have_new_firsttimer) {
      assert(Kidx_last!=Kidx_next);
      assert(Kidx_next>=0 && Kidx_next<K_phase2.size());
#ifdef DEBUG_SCHED
      printf("[comm] Master: found next  contingency for K_idx=%d (globidx=%d) to have "
	     "idx=%d (globidx=%d) (no reevals were needed) [%s]  for rank %d\n",
	     Kidx_last, K_phase2[Kidx_last], Kidx_next, K_phase2[Kidx_next],
	     is_high_prior ? "high prior" : "", rank);
#endif
      erase_elem_from(K_high_prio_phase2, Kidx_next);
      return Kidx_next;
      
    } else { //!have_new_first_timer

      

      for(ContingInfo& kinfo: K_info_phase2) {
	//on hold if high pens (without restriction==1)
	if((kinfo.penalty>=pen_threshold && kinfo.restrictions!=1) || kinfo.penalty<=-1e20) return -3;
	if(kinfo.force_reeval!=0) return -3;
      }



#ifdef DEBUG_SCHED  
    printf("[comm] Master: did NOT find a next contingency to follow Kidx=%d on rank %d, will return K_idx=-1\n",
      Kidx_last, rank);
    //printvec(K_info_phase2);
#endif
    }
    return -1;
  }
}

bool MyCode1::do_phase2_master_part()
{
  int mpi_test_flag, ierr; MPI_Status mpi_status;
  int num_ranks = K_on_rank.size();

  bool finished = true;

  int num_high_penalties = number_of_high_penalties(K_info_phase2);

  //check recv for basecase solution (from solver rank)
  //post a new one if the previous was completed
  while(req_recv_base_sol.is_done()) {
    int scacopf_pass = req_recv_base_sol.update_prob_variables(scacopf_prob);

    assert(scacopf_pass >= phase3_scacopf_pass_solution);
    phase3_scacopf_pass_solution = scacopf_pass;

#ifdef DEBUG_COMM
    printf("[comm] Master: recv basecase solution from scacopf_pass %d completed [current master scacopf_pass=%d] "
      "at global time %g\n", 
      scacopf_pass, phase3_scacopf_passes_master, glob_timer.measureElapsedTime());
#endif 
    assert(phase3_scacopf_passes_master >= scacopf_pass-1);
    req_recv_base_sol.post(scacopf_prob, Tag7, rank_solver_rank0, comm_world);
  }


  for(int r=0; r<num_ranks; r++) {
    if(K_on_rank[r].size()==0) {
      if(r==rank_master)
	continue;
    }

    //done with communication for this rank since it was marked as not having any
    //contingencies left
    if(K_on_rank[r].size()>0 && K_on_rank[r].back()==-2) {
      //printf("[comm] Master : no more comm for rank=%d. it was marked with -2\n", r);
      continue;
    }
    
    bool send_new_K_idx = false;

    //check for recv of penalty
    if(req_recv_penalty_for_rank[r].size()>0) {

      ReqPenalty* req_pen = req_recv_penalty_for_rank[r].back();
      //test penalty receive
      ierr = MPI_Test(&req_pen->request, &mpi_test_flag, &mpi_status);
      assert(ierr == MPI_SUCCESS);
      
      if(mpi_test_flag != 0) {
	//completed
	double penalty = req_pen->buffer[0];
	int solution_pass_of_eval = req_pen->get_scacopf_pass();
#ifdef DEBUG_COMM
	printf("[comm] Master: recv penalty=%g from rank=%d K_idx=%d "
	       "eval_with_sol_from_pass=%d completed (current master scacopf_pass=%d)\n", 
	       penalty, r, K_phase2[req_pen->idxK], solution_pass_of_eval, phase3_scacopf_passes_master);
#endif

	assert(req_pen->idxK>=0); assert(req_pen->idxK<K_info_phase2.size()); assert(penalty>-1e+20);

	int idx = req_pen->idxK; assert(K_info_phase2[idx].idx == idx);

	assert(K_info_phase2[idx].evaled_with_sol_at_pass<=solution_pass_of_eval);

	K_info_phase2[idx].penalty = penalty; K_info_phase2[idx].n_evals++;
	K_info_phase2[idx].p1 = req_pen->buffer[1]; K_info_phase2[idx].q1 = req_pen->buffer[2];
	K_info_phase2[idx].p2 = req_pen->buffer[3]; K_info_phase2[idx].q2 = req_pen->buffer[4];
	K_info_phase2[idx].rank_eval = r;
	K_info_phase2[idx].evaled_with_sol_at_pass = solution_pass_of_eval;
	K_info_phase2[idx].force_reeval=0;

	assert(K_info_phase2[idx].scacopf_actions.size() < K_info_phase2[idx].n_evals);
	assert(K_info_phase2[idx].n_scacopf_solves       < K_info_phase2[idx].n_evals);
#ifdef DEBUG_COMM
	cout << "[comm] Master recv penalty\n   ContingInfo: " << K_info_phase2[idx];
#endif
	//remove the request irecv for this rank
	delete req_pen;
	req_recv_penalty_for_rank[r].pop_back(); assert(req_recv_penalty_for_rank[r].size() == 0);

	// the next K_idx/index in K_phase2 to be sent or to put solver rank on hold
	assert(K_on_rank[r].back()>=0); 

	int K_idx_next;
	if(r==rank_solver_rank0 && num_high_penalties>=phase3_max_K_to_start_solver) {
          K_idx_next=-1;
#ifdef DEBUG_COMM
	printf("[comm] Master: will send solver_start to rank %d (num_high_pen %d   its_thresh %d)\n", 
	       r, num_high_penalties, phase3_max_K_to_start_solver);
#endif
	} else {
          K_idx_next = get_next_conting_foreval(K_on_rank[r].back(), r, K_info_phase2);
	  assert(K_idx_next>=-1 || K_idx_next==-3);
	}
	
	if(K_idx_next==-3) {
          //signal for putting the rank on hold
          send_new_K_idx = false;
	  K_on_rank[r].push_back(-3);
	} else {
	  K_on_rank[r].push_back(K_idx_next);
	  send_new_K_idx = true;
        }
      } else {
	//request did not complete
      }
    } else {
      //post send <- this is for the first call (a K_idx has been put in K_on_rank[r])
      send_new_K_idx = true;

      if(K_on_rank[r].size()>0) {

	if(K_on_rank[r].back() == -3) {
	  assert(K_on_rank[r].size()>=2);
	  int K_idx_next = get_next_conting_foreval(K_on_rank[r][K_on_rank[r].size()-2], r, K_info_phase2);
	  if(K_idx_next==-3) send_new_K_idx = false;
	  else K_on_rank[r].back() = K_idx_next;
	  //was on hold
	}
      }  
    }

    //check if the send for the last idx is complete 
    if( req_send_K_idx_for_rank[r].size()>0 ) {
      assert(req_send_K_idx_for_rank[r].size()==1);

      ReqKidx* req_K_idx = req_send_K_idx_for_rank[r].back();

      mpi_test_flag = req_K_idx->test();
      if(mpi_test_flag != 0) {
	//completed
#ifdef DEBUG_COMM	
	printf("[comm] Master: send K_idx=%d to rank=%d completed\n", req_K_idx->K_idx(), r);
#endif
	//was this one the last one (K_idx==-1) ?
	if(req_K_idx->K_idx()==-1) {
	  //we're done with this rank -> do not send a new one
	  send_new_K_idx = false;
	  K_on_rank[r].push_back(-2);
	} 
	
	delete req_K_idx;
	req_send_K_idx_for_rank[r].pop_back();
      } else {
	//last one didn't complete
	send_new_K_idx = false;
      }
    }
    
    // i. post a new send for K idx
    // ii.post the recv for corresponding penalty objective
    if(send_new_K_idx) {
      if(K_on_rank[r].size()>0) { assert(-3 != K_on_rank[r].back()); }
      assert(req_recv_penalty_for_rank[r].size() == 0);
      assert(req_send_K_idx_for_rank[r].size() == 0);
      
      int K_idx_next = -1; // in case K_on_rank[r] is empty (no contingencies for this rank)
      if(K_on_rank[r].size()>0) {
	K_idx_next = K_on_rank[r].back();

#ifdef DEBUG_COMM   
	char msg[100]; sprintf(msg, "K_on_rank[%d] (on master before send new Kidx)", r);
	printvec(K_on_rank[r], msg);
#endif
      }
      
      //
      {
	int pass = K_idx_next>=0 ? K_info_phase2[K_idx_next].scacopfed_at_pass : -1;
	if(pass!=-1 && pass<phase3_scacopf_passes_master) pass = phase3_scacopf_passes_master;
	//isend K_idx_next
	ReqKidx* req_K_idx = new ReqKidx( K_idx_next, pass ); 
#ifdef DEBUG_COMM
	int id = K_idx_next>=0 ? K_phase2[K_idx_next] : K_idx_next;
	int sol_pass = K_idx_next>=0 ? K_info_phase2[K_idx_next].evaled_with_sol_at_pass : -1;
	printf("[comm] Master posted send for K_idx=%d to rank=%d "
	       "should_use_sol_at_pass=%d  evaled_with_sol_at_pass=%d (current master_scacopf_pass=%d)  tag=%d\n", 
	       id, r, pass, sol_pass, phase3_scacopf_passes_master, Tag1);
#endif
	req_K_idx->post_send(Tag1, r, comm_world);
	req_send_K_idx_for_rank[r].push_back(req_K_idx);
	
	if( K_idx_next>=0 ) {

	  //post the irecv for penalty
	  ReqPenalty* req_pen = new ReqPenalty( K_idx_next, -1e+20, -1 );  
	  req_pen->post_recv(Tag2, r, comm_world);

	  req_recv_penalty_for_rank[r].push_back(req_pen);

#ifdef DEBUG_COMM
	  printf("[comm] Master posted recv for penalty for K_idx=%d rank=%d tag=%d\n", 
		 K_phase2[K_idx_next], r, Tag2);
#endif
	}
      }
    } // end of the new send

    if(req_recv_penalty_for_rank[r].size()==0 &&
       req_send_K_idx_for_rank[r].size() == 0) {
      //done on rank r, will return finished==true or false depending on
      //whether other ranks are done
      if(K_on_rank[r].size()>0) if(K_on_rank[r].back()==-3) finished=false;
    } else {
      finished=false;
    }
  } // end of 'for' loop over ranks


  return finished;
}


int MyCode1::number_of_high_penalties(const std::vector<ContingInfo>& K_info)
{
  int count=0;
  for(auto o: K_info)
    if(o.penalty >= pen_threshold) count++;

  return count;
}

void remove_close_idxs(vector<int>& vidxs, int proximity, 
		       const vector<double>& K_penalties, 
		       const std::vector<int> K_idxs_global)
{
  //remove close idxs
  vector<int> to_remove;
  int it=0, itf;
  while(it<vidxs.size()) {
    itf = it+1;
    while(itf<vidxs.size() && K_idxs_global[vidxs[itf]]<K_idxs_global[vidxs[it]]+proximity) {
      to_remove.push_back(vidxs[itf]);
      itf++;
    }
    it = itf;
  }
  for(auto idx: to_remove) {
    bool ret = erase_elem_from(vidxs, idx);
    assert(ret==true);
  }
}

std::vector<int> sort_high_penalties_w_remove_close(const std::vector<double>& K_penalties,
				     const std::vector<int> K_idxs_global,
				     double thresh, int proximity)
{
  vector<int> K_idxs_all;

  for(int it=0; it<K_penalties.size(); it++) {
    if(K_penalties[it]>=thresh)
      K_idxs_all.push_back(it);  
  }

  //printvec(K_idxs_all, "K_idxs_all 11111");

  if(proximity>0)  {
    sort(K_idxs_all.begin(),  K_idxs_all.end());
    remove_close_idxs(K_idxs_all, proximity, K_penalties, K_idxs_global);
  }
  sort(K_idxs_all.begin(), 
       K_idxs_all.end(), 
       [&](const int& a, const int& b) { return (K_penalties[a] > K_penalties[b]); });

  //printvec(K_idxs_all, "K_idxs_all 3333");
  
  return K_idxs_all;
}

void MyCode1::determine_solver_actions(const vector<ContingInfo>& K_info_all, 
				const vector<int>& K_idxs_all, 
				const bool& master_evalpart_done,
				const vector<ContingInfo>& K_info_last_solved,
				vector<ContingInfo>& K_info_next_solve, 
				bool& changed_since_last)
{
  

  assert(K_info_all.size() == K_idxs_all.size());

  vector<ContingInfo> Ks_to_include;

  K_info_next_solve.clear();

  for(int it=0; it<K_info_all.size(); it++) {
    ContingInfo kinfo = K_info_all[it];

    assert(kinfo.K_idx == K_idxs_all[it]);
    assert(kinfo.idx   == it);

    if(kinfo.penalty>pen_threshold) {
      //attempt to "include" if "penalized" was done MAX_K_EVALS and did not reduce penalty
      // this is disabled for medium and large networks
      if(kinfo.n_evals>=kinfo.max_K_evals  && kinfo.n_evals-1==kinfo.n_scacopf_solves && data.N_Bus.size()<3200) {

	bool include_it=true;
       
	//do not include if it is restricted from being included (infeasible Kgen)
	if(kinfo.restrictions==1) include_it=false;

	//do not include if it was already included
	if(kinfo.scacopf_actions.end() != find(kinfo.scacopf_actions.begin(), 
					       kinfo.scacopf_actions.end(),
					       -102))  {
	  include_it = false;
	}

	if(include_it) {
	  Ks_to_include.push_back(kinfo);
	}
	
      } else {
	if(kinfo.n_evals>kinfo.n_scacopf_solves) {

	  if((data.N_Bus.size()>=3200 && kinfo.n_scacopf_solves<=3) || data.N_Bus.size()<3200) {
	    kinfo.scacopf_actions.push_back(-101);
	    assert(kinfo.n_scacopf_solves == kinfo.scacopf_actions.size()-1);
	    K_info_next_solve.push_back(kinfo);
	  }
	}
      }
    }
  }


  //decide which one to include -> the first 'two'
  size_t two=1;
  for(int it=0; it<std::min(two, Ks_to_include.size()); it++) {
    ContingInfo& kinfo = Ks_to_include[it];
    
    K_info_next_solve.push_back(kinfo);
    K_info_next_solve.back().scacopf_actions.push_back(-102);

    assert(K_info_next_solve.back().n_scacopf_solves == K_info_next_solve.back().scacopf_actions.size()-1);
  }

  //no more than MAX_NUM_Kidxs_SCACOPF entries in K_info_next_solve 
  if(K_info_next_solve.size() > MAX_NUM_Kidxs_SCACOPF) {
    vector<ContingInfo> K_info_next_solve2;
    for(int i=0; i<MAX_NUM_Kidxs_SCACOPF; i++) 
      K_info_next_solve2.push_back(K_info_next_solve[i]);
    K_info_next_solve = K_info_next_solve2;
      
    //K_info_next_solve.resize(MAX_NUM_Kidxs_SCACOPF);
  }
  assert(K_info_next_solve.size() <= MAX_NUM_Kidxs_SCACOPF);


  changed_since_last = !(K_info_next_solve == K_info_last_solved);

  //if(!changed_since_last) {
  //  printvec(K_info_last_solved);
  //  printvec(K_info_next_solve);
  //}
}

std::vector<int> MyCode1::get_high_penalties_from(const std::vector<double>& K_penalties, 
						  const std::vector<int> K_idxs_global,
						  const int& conting_evals_done)
{
  bool is_late = glob_timer.measureElapsedTime() > 0.6*TimeLimitInSec;
  double thresh = high_pen_threshold_factor * cost_basecase;
  if(thresh > 20000.) thresh=20000;
  else if(thresh<1000.) thresh=1000.;

  int proximity = 5;
  vector<int> K_idxs_all, K_idxs_new; 
  vector<double> K_pens;

  int max_num_K = phase3_last_num_K_nonproximal+ phase3_adtl_num_K_at_each_pass;

  K_idxs_all = sort_high_penalties_w_remove_close(K_penalties, K_idxs_global,
						  thresh, proximity);

  if(K_idxs_all.size()<max_num_K) {
    thresh *= 0.25;
    K_idxs_all = sort_high_penalties_w_remove_close(K_penalties, K_idxs_global,
						    thresh, proximity);
  }

  K_idxs_new.clear();
  int num_to_keep = max_num_K;
  if(num_to_keep>K_idxs_all.size()) num_to_keep = K_idxs_all.size();
  K_idxs_new = vector<int>(K_idxs_all.begin(), K_idxs_all.begin()+num_to_keep);

  assert(K_idxs_new.size() >= phase3_last_num_K_nonproximal);
  phase3_last_num_K_nonproximal = K_idxs_new.size();

  if(K_idxs_new.size() < max_num_K  && (is_late || conting_evals_done)) {

    //printvec(K_idxs_new, "K_idxs - is late or evals done, non-proximal, from get_high_pen");
    //if we don't have enough non-proximal, append the highest-penalty proximal ones

    //first increase num K from last pass; disregard num_K_nonproximal

    if(max_num_K<K_info_last_scacopf_solve.size()+phase3_adtl_num_K_at_each_pass)
      max_num_K = K_info_last_scacopf_solve.size()+phase3_adtl_num_K_at_each_pass;

    //disable proximity removal
    proximity = 0;
    K_idxs_all = sort_high_penalties_w_remove_close(K_penalties, K_idxs_global,
						    thresh, proximity);

    vector<int> K_idxs_diff = set_diff(K_idxs_all, K_idxs_new);
    //sort in !increasing! order to be able to use pop_back()
    sort(K_idxs_diff.begin(), K_idxs_diff.end(),
       [&](const int& a, const int& b) { return (K_penalties[a] < K_penalties[b]); });

    while( !K_idxs_diff.empty() && K_idxs_new.size() < max_num_K) {
      K_idxs_new.push_back(K_idxs_diff.back());
      K_idxs_diff.pop_back();
    }
    
  } else {
  }

  //printvec(K_idxs_new, "K_idxs - final, from get_high_pen");

  return K_idxs_new;
}


bool MyCode1::do_phase3_master_solverpart(bool master_evalpart_done)
{
  int r = rank_solver_rank0; //shortcut

  //check recv for basecase solution (from solver rank)
  //post a new one if the previous was completed
  while(req_recv_base_sol.is_done()) {
    int scacopf_pass = req_recv_base_sol.update_prob_variables(scacopf_prob);
    assert(scacopf_pass > phase3_scacopf_pass_solution);
    phase3_scacopf_pass_solution = scacopf_pass;

#ifdef DEBUG_COMM
    printf("[comm] Master: recv basecase solution from scacopf_pass %d completed at global time %g\n", 
	   scacopf_pass, glob_timer.measureElapsedTime());
#endif 
    assert(phase3_scacopf_passes_master >= scacopf_pass-1);
    req_recv_base_sol.post(scacopf_prob, Tag7, rank_solver_rank0, comm_world);
  }

  int num_high_penalties = number_of_high_penalties(K_info_phase2);
  //printf("numhigh penalties = %d\n", num_high_penalties);
  //printvec(K_info_phase2, "K_penalty_phase2");

  int mpi_test_flag, ierr; MPI_Status mpi_status;

  if(K_on_rank[r].back()==-2) {
    if(req_send_KidxSCACOPF==NULL) {

      if(master_evalpart_done )
	printf("[ph3] Master - starting solverpart with master_evalpart_done = true\n");

      bool changed_since_last;
      vector<ContingInfo> K_info_next_scacopf_solve;
      determine_solver_actions(K_info_phase2, K_phase2, master_evalpart_done, K_info_last_scacopf_solve,
			       K_info_next_scacopf_solve, changed_since_last);

      if(master_evalpart_done && K_info_next_scacopf_solve.size()==0) {
	//this is the "finish" signal for the solver rank
	K_info_next_scacopf_solve.push_back(ContingInfo(-2, 0.));
	K_info_last_scacopf_solve = K_info_next_scacopf_solve;

      } else {

	if(!changed_since_last) return false;
	if(K_info_next_scacopf_solve.size()==0) return false;

	K_info_last_scacopf_solve = K_info_next_scacopf_solve;

	//update scacopf pass
	for(ContingInfo& kinfo : K_info_next_scacopf_solve) {
	  kinfo.scacopfed_at_pass = K_info_phase2[kinfo.idx].scacopfed_at_pass = phase3_scacopf_passes_master+1;
	  K_info_phase2[kinfo.idx].scacopf_actions = kinfo.scacopf_actions;
	}
      }

      //
      //post idxs send msg
      //
      req_send_KidxSCACOPF = new ReqKidxSCACOPF(K_info_next_scacopf_solve);
       
#ifdef DEBUG_COMM
      printf("[comm] Master posted send for scacopf K_idxs to (solver) rank %d   tag %d scacopf_pass=%d\n", 
	     r, Tag4, phase3_scacopf_passes_master);
#endif
#ifdef DEBUG_SCHED
      if(data.K_Contingency.size()<1250)
	printvec(K_info_next_scacopf_solve, "[comm] Ks on master to be solved next");
#endif

      req_send_KidxSCACOPF->post_send(Tag4, r, comm_world);

      //
      // post recv penalty from solver rank 
      //
      assert(req_recv_penalty_solver==NULL);
      req_recv_penalty_solver = new ReqPenalty(-1, 1e6, -1);
      req_recv_penalty_solver->post_recv(Tag5, rank_solver_rank0, comm_world);

#ifdef DEBUG_COMM
      printf("[comm] Master rank %d posted recv for scacopf penalty to (solver) rank %d  pass=%d tag %d\n",
	     my_rank, r, phase3_scacopf_passes_master, Tag5);
#endif

    } // end of if(req_send_KidxSCACOPF==NULL) 
    else {

      //Kidxs send posted -> check the solution/penalty recv
      ierr = MPI_Test(&req_recv_penalty_solver->request, &mpi_test_flag, &mpi_status);
      assert(ierr == MPI_SUCCESS);
      if(mpi_test_flag != 0) {
	//completed
	phase3_scacopf_passes_master++;

	bool last_pass = false;
	int scacopf_pass_solver = req_recv_penalty_solver->get_scacopf_pass();

#ifdef DEBUG_COMM
	printf("[comm] Master rank %d recv-ed scacopf penalty from (solver) "
	       "rank %d  penalty %g master_pass=%d fromsolver_pass=%d\n",
	       my_rank, r, req_recv_penalty_solver->buffer[0], 
	       phase3_scacopf_passes_master, scacopf_pass_solver);
#endif
	assert(scacopf_pass_solver == phase3_scacopf_passes_master);
	assert(req_send_KidxSCACOPF != NULL);
	if(req_send_KidxSCACOPF->buffer.size()>0) {
	  if(req_send_KidxSCACOPF->buffer[0] == -2) {
	    printf("[comm] Master - pass: the last one !!!\n");
	    last_pass = true;
	  }
	}

	////////////////////////////////////////////////////////////////////////////////
	// actions after the SCACOPF solve
	////////////////////////////////////////////////////////////////////////////////
	bool scacopf_includes = false, scacopf_penalize=false, force_reevals=false; 
	//update scacopf solves counter for elems of K_info_phase2
	for(int itk=0; itk<K_info_phase2.size(); itk++) {
	  ContingInfo& kinfo=K_info_phase2[itk];
	  if(kinfo.scacopfed_at_pass == phase3_scacopf_passes_master) {
	    kinfo.n_scacopf_solves++;
	    assert(kinfo.n_scacopf_solves <= kinfo.n_evals);
	    assert(kinfo.n_scacopf_solves == kinfo.scacopf_actions.size());

	    if(kinfo.scacopf_actions.back() == -102) {
	      scacopf_includes = true;
	      //kinfo.max_K_evals++;
	      //force_reevals=true;
	      kinfo.force_reeval = 1;
	    }
	    if(kinfo.scacopf_actions.back() == -101) {
	      scacopf_penalize = true;
	      kinfo.force_reeval = 1;
	    }
	  }
	}
	if(scacopf_includes) {
	  MAX_K_EVALS = MAX_K_EVALS + 1;
	}
	//if(scacopf_penalize || scacopf_includes) {
	if(scacopf_includes) {
	  bool all_in=true;
	  for(auto& ci : K_info_phase2) { if(ci.rank_eval<0) { all_in = false; break; } }
	  if(all_in) force_reevals=true;
	}
	

	if(force_reevals) {
	  for(int itk=0; itk<K_info_phase2.size(); itk++) {
	    ContingInfo& kinfo=K_info_phase2[itk];
	    if(kinfo.n_evals>=kinfo.max_K_evals) {
	      kinfo.max_K_evals = kinfo.n_evals+1;
	    } else { 
	      //assert(kinfo.n_evals<kinfo.max_K_evals); 
	      if(kinfo.n_evals>kinfo.max_K_evals) {
		printf("[warning] kinfo.n_evals<kinfo.max_K_evals (%d < %d) for Kidx=%d\n",
		       kinfo.n_evals, kinfo.max_K_evals, kinfo.idx);
		//printvec(K_info_phase2);
	      }
	    }
	    kinfo.force_reeval = 1;
	  }
	}


#ifdef DEBUG_SCHED
	if(scacopf_includes) printf("[sched] MAX_K_EVALS=%d (was increased)\n", MAX_K_EVALS);
	if(data.K_Contingency.size()<1250)
	  printvec(K_info_phase2, "[sched] K_info_phase2 on master after recv pen");
#endif
	ierr = MPI_Test(&req_send_KidxSCACOPF->request, &mpi_test_flag, &mpi_status); 
	assert(ierr == MPI_SUCCESS);
	if(mpi_test_flag != 0) {
	  printf("[comm] Master  completed send for scacopf idxs from (solver) rank %d\n", r);
	} else {
	  assert(false && "[ph3] Master did not complete send for scacopf idxs!?!");
	}
	
	delete req_recv_penalty_solver; 
	req_recv_penalty_solver=NULL;
	delete req_send_KidxSCACOPF;
	req_send_KidxSCACOPF=NULL;

	return last_pass;
      }
    } // end of else if(req_send_KidxSCACOPF==NULL) 


  } else { //else for if(K_on_rank[rank_solver]==-2) 
    //nothing -> phase3 has not started yet
  }
  
  return false;
}


bool MyCode1::do_phase2_evaluator_part(int& switchToSolver)
{
  int mpi_test_flag, ierr; MPI_Status mpi_status;

  //test for recv of K_idx
  if(req_recv_K_idx!=NULL) {

    mpi_test_flag = req_recv_K_idx->test();
    if(mpi_test_flag != 0) {
      //completed
      int K_idx = req_recv_K_idx->K_idx();
      int scacopf_pass_from_Kidxreq = req_recv_K_idx->scacopf_pass();
      if(K_idx<0) {
	//no more contingencies coming from master or switching to solver mode

	K_on_my_rank.push_back(K_idx);

	delete req_recv_K_idx;
	req_recv_K_idx = NULL;

	if(req_send_penalty==NULL) {
	  if(iAmSolver) {
	    assert(my_rank==rank_solver_rank0);
	    switchToSolver = true;
	    K_on_my_rank.push_back(-2);

	    printf("Evaluator Rank %d recv K_idx %d - switching "
		   "to solver (it may have finished evals) global time %g\n", 
		   my_rank, K_idx, glob_timer.measureElapsedTime());
	    return false;
	  } else {
	    printf("Evaluator Rank %d recv K_idx %d - finished evaluations  global time %g\n", 
		   my_rank, K_idx, glob_timer.measureElapsedTime());
	  return true;
	  }
	}
	
      } else {
#ifdef DEBUG_COMM
	printf("[comm] Evaluator Rank %d recv K_idx=%d completed\n", my_rank, K_phase2[K_idx]);
#endif
	K_on_my_rank.push_back(K_idx);

	//
	// solve recourse problem
	//

	//first check if recv for basecase solution (from solver rank) completed
	//post a new one if the previous was completed
	int scacopf_pass_of_solution = phase3_scacopf_pass_solution;
	if(!iAmSolver) {
	  if(req_recv_base_sol.is_done()) {
	    scacopf_pass_of_solution = req_recv_base_sol.update_prob_variables(scacopf_prob);

	    //assert(scacopf_pass > phase3_scacopf_pass_solution);
	    //phase3_scacopf_pass_solution = scacopf_pass;
#ifdef DEBUG_COMM
	    printf("[comm] Evaluator Rank %d recv basecase solution from scacopf_pass %d completed at global time %g \n", 
		   my_rank, scacopf_pass_of_solution, glob_timer.measureElapsedTime());
#endif 
	    phase3_scacopf_pass_solution = scacopf_pass_of_solution;
	    req_recv_base_sol.post(scacopf_prob, Tag7, rank_solver_rank0, comm_world);
	  }
	}

	while(scacopf_pass_from_Kidxreq > phase3_scacopf_pass_solution) {
	  printf("[comm] [warning] Evaluator Rank %d basecase solution is from scacopf_pass %d "
		 "while the request K_idx=%d requires sol from scacopf_pass %d at global time %g. will idle\n", 
		 my_rank, scacopf_pass_of_solution, K_phase2[K_idx], scacopf_pass_from_Kidxreq, 
		 glob_timer.measureElapsedTime());
	  
	  
	  if(req_recv_base_sol.is_done()) {
	    scacopf_pass_of_solution = req_recv_base_sol.update_prob_variables(scacopf_prob);
#ifdef DEBUG_COMM
	    printf("[comm] Evaluator Rank %d recv basecase solution "
		   "from scacopf_pass %d completed at global time %g [K_idx=%d][2]\n", 
		   my_rank, scacopf_pass_of_solution, K_phase2[K_idx],
		   glob_timer.measureElapsedTime());
#endif 
	    phase3_scacopf_pass_solution = scacopf_pass_of_solution;
	    req_recv_base_sol.post(scacopf_prob, Tag7, rank_solver_rank0, comm_world);
	  } else {
	    std::this_thread::sleep_for(std::chrono::milliseconds(500));
	  }
	}	    
	
	assert(phase3_scacopf_pass_solution>=0);
	assert(scacopf_pass_from_Kidxreq <= phase3_scacopf_pass_solution);

	assert(req_send_penalty == NULL);
	req_send_penalty = new ReqPenalty(K_idx, -1e+20, phase3_scacopf_pass_solution);
	//
	// solve contingency
	//
	int solve_status; double penalty;
	process_contingency(K_phase2[K_idx], solve_status, penalty, req_send_penalty->buffer);

	//send the penalty to master
	req_send_penalty->post_send(Tag2, rank_master, comm_world); //Tag0 + MSG_TAG_SZ + K_on_my_rank.size();

#ifdef DEBUG_COMM	
	printf("[comm] Evaluator Rank %d posted penalty send value=%g for Kidx=%d "
	       "to master rank %d solutionfrom_pass=%d tag=%d\n", 
	       my_rank, penalty, K_phase2[K_idx], rank_master, phase3_scacopf_pass_solution, Tag2);
#endif
	//delete recv request
	delete req_recv_K_idx;
	req_recv_K_idx = NULL;
      }
    } // end of if(mpi_test_flag != 0) 
  }

  //test the send of the penalty send
  if(req_send_penalty!=NULL) {
    ierr = MPI_Test(&req_send_penalty->request, &mpi_test_flag, &mpi_status);
    assert(MPI_SUCCESS == ierr);

    if(mpi_test_flag != 0) {
      //completed
#ifdef DEBUG_COMM
      int idx = req_send_penalty->idxK >=0 ? K_phase2[req_send_penalty->idxK] : req_send_penalty->idxK;
      int pass = req_send_penalty->get_scacopf_pass();
      printf("[comm] Evaluator Rank %d send penalty for K_idx=%d completed. penalty=%g used solutionfrom_pass=%d\n",
	     my_rank, idx, req_send_penalty->buffer[0], pass);
#endif
      delete req_send_penalty; req_send_penalty=NULL;
    }
  }

  //post the recv for K_idx
  if(req_recv_K_idx==NULL) {
    req_recv_K_idx = new ReqKidx();
    
    //int tag = Tag1; //Tag0 + K_on_my_rank.size()+1;
    //ierr = MPI_Irecv(&req_recv_K_idx->buffer, 1, MPI_INT, rank_master, tag,
    //		     comm_world, &req_recv_K_idx->request);
    req_recv_K_idx->post_recv(Tag1, rank_master, comm_world);
#ifdef DEBUG_COMM
    printf("[comm] Evaluator Rank %d posted recv for K_idx tag=%d\n", my_rank, Tag1);
#endif
  }
  
  return false;
}

void MyCode1::phase2_initialization()
{
  if(iAmMaster) {
    int num_ranks = K_on_rank.size();
    for(int r=0; r<num_ranks; r++) {
      req_send_K_idx_for_rank.push_back(vector<ReqKidx*>());
      req_recv_penalty_for_rank.push_back(vector<ReqPenalty*>());
    }   
  }

  if(iAmEvaluator) { }
}

double counter=0.;

bool MyCode1::do_phase2()
{
  phase2_ranks_allocation();

  //assert( K_info_phase2.size() ==  data.K_Contingency.size());

  phase2_initial_contingency_distribution();

  phase2_initialization();
  
  bool finished=false; 
  bool master_evalpart_done=!iAmMaster;
  bool master_solvepart_done=!iAmMaster;
  bool evaluator_part_done=false;
  while(!finished) {

    if(iAmMaster) {

      counter++;

      if(!master_evalpart_done) {
	finished = do_phase2_master_part();
	if(finished)
	  printf("master_evalpart finished=%d counter=%g master_evalpart_done=%d\n", 
		 finished, counter, master_evalpart_done);
	if(finished) master_evalpart_done = true;
      }

      //if(!master_solvepart_done) 
      {
	//solver related communication

	bool finished_solverpart = do_phase3_master_solverpart(master_evalpart_done);
	if(finished_solverpart)
	  printf("master solverpart finished=%d counter=%g\n", finished_solverpart, counter);

	if(finished_solverpart) {
	  master_solvepart_done = true;
	}
      }
      finished = master_solvepart_done && master_evalpart_done;
    }

    int switchToSolver=false;
    if(iAmEvaluator && !evaluator_part_done) {
      finished = do_phase2_evaluator_part(switchToSolver);

      //if(finished)
      //printf("evaluator part finished on rank %d\n", my_rank);

      if(finished) {
	evaluator_part_done=true;
      } else {
	//need to finish the send of penalty before switching
	switchToSolver=false;
      }
      finished = finished && master_evalpart_done; 
    }
    
    if(switchToSolver) {
      assert(my_rank==rank_solver_rank0);
#ifdef DEBUG_COMM
      printf("[comm] Evaluator Rank %d switched to solver\n", my_rank);
#endif
    }

    if(iAmSolver) {
      finished = do_phase3_solver_part();
    }

    if(iAmMaster && !iAmEvaluator) {
      fflush(stdout);
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
  }

  if(iAmMaster) {
    printvec(K_info_phase2, "penalties on master");
  }  
  return true;
}

bool MyCode1::do_phase3_solver_part()
{
  assert(my_rank==rank_solver_rank0);
  int mpi_test_flag, ierr; MPI_Status mpi_status;

  if(K_on_my_rank.size()==0) {
    return false;
  }


  if(K_on_my_rank.back() ==-2) {
    if(req_recv_KidxSCACOPF==NULL) {
      //post the recv
      req_recv_KidxSCACOPF = new ReqKidxSCACOPF();
      req_recv_KidxSCACOPF->post_recv(Tag4, rank_master, comm_world);

#ifdef DEBUG_COMM
      printf("[ph3] Solver rank %3d recv posted for scacopf contingencies"
	     "to (master) rank %3d  tag %d current pass %d\n",
	     my_rank, rank_master, Tag4, phase3_scacopf_passes_solver);
#endif

      return false;

    } else {
      //test req_recv_KidxSCACOPF
      ierr = MPI_Test(&req_recv_KidxSCACOPF->request, &mpi_test_flag, &mpi_status);
      if(mpi_test_flag != 0) {
	//completed
	vector<int> K_idxs = req_recv_KidxSCACOPF->K_idxs();
	vector<double> K_penalties = req_recv_KidxSCACOPF->K_penalties();
	vector<int> K_actions = req_recv_KidxSCACOPF->K_actions();
	vector<vector<double> > K_powers = req_recv_KidxSCACOPF->K_powers();
	assert(K_idxs.size()==K_penalties.size());
	assert(K_idxs.size()==K_actions.size());


	printf("[ph3] Solver rank %3d request n=%3lu conting recv-ed scacopf solve current pass %d "
	       "at global time: %g K_idxs=[ ", 
	       my_rank, K_idxs.size(), phase3_scacopf_passes_solver, glob_timer.measureElapsedTime());
	for(auto c: K_idxs) { 
	  if(c>=0) printf("%d  ", K_phase2[c]); 
	  else    printf("%d  ", c); 
	}; 
	printf("]\n"); fflush(stdout);

	bool last_pass = false;
	assert(K_idxs.size()>0);
	if(K_idxs.size()>0) {
	  double objective;
	  if(K_idxs[0]>=0) {

	    //solve SCACOPF
	    K_idxs = selectfrom(K_phase2, K_idxs);

	    //////////////////////////////////////////////////////////////////////////
	    // solve SCACOPF
	    //////////////////////////////////////////////////////////////////////////
	    objective = phase3_solve_scacopf(K_idxs, K_penalties, K_powers, K_actions);

	  } else {
	    assert(K_idxs[0]==-2);
	    assert(K_idxs.size()==1);
	    // this is the finish / last evaluation signal
	    objective = -2.;
	    last_pass = true;
	    phase3_scacopf_passes_solver++;
	  }

	  //send penalty/handshake 
	  req_send_penalty_solver = new ReqPenalty(-1, objective, phase3_scacopf_passes_solver);

	  //int tag = Tag5; //Tag0 + 4*MSG_TAG_SZ + rank_solver_rank0 + phase3_scacopf_passes_solver;
	  req_send_penalty_solver->post_send(Tag5, rank_master, comm_world);

#ifdef DEBUG_COMM
	  printf("[ph3] Solver rank %d sent scacopf objective=%g to rank %d  scacopf_pass=%d  tag %d\n\n",
		 my_rank, objective, rank_master, phase3_scacopf_passes_solver, Tag5);
#endif
	  
	}
	//cleanup requests to make myself available for another scacopf evaluation
	delete req_recv_KidxSCACOPF;
	req_recv_KidxSCACOPF=NULL;
	delete req_send_penalty_solver;
	req_send_penalty_solver=NULL;

	return last_pass;
      } else {
	//request for SCACOPF idxs did not complete
      }
    }
  }
  return false;
}

int MyCode1::go()
{
  goTimer ttot; ttot.start();

  if(iAmMaster)
    display_instance_info();

  //
  //phase 1
  //
  if(!do_phase1()) {
    printf("Error occured in phase 1\n");
    return -1;
  }
  printf("Phase 1 completed on rank %d after %g sec\n", my_rank, glob_timer.measureElapsedTime());
  fflush(stdout);

  //
  // phase 2
  //
  if(!do_phase2()) {
    printf("Error occured in phase 2\n");
    return -1;
  }
  printf("Phase 2 completed/finished on rank %d after %g sec\n", 
	 my_rank, glob_timer.measureElapsedTime());
  fflush(stdout);

  //
  //cleanup
  //
  delete scacopf_prob;

  if(my_rank==rank_master)
    printf("--finished in %g sec  global time %g sec\n", ttot.stop(), glob_timer.measureElapsedTime());



  return 0;
}

void MyCode1::process_contingency(const int& K_idx, int& status, double& penalty, double* info_out)
{
  //
  // the actual solve 
  //
  penalty = solve_contingency(K_idx, status);

  //
  // prepare info for master rank
  //
  info_out[0]=penalty;

  if(data.K_ConType[K_idx] == SCACOPFData::kGenerator) {

    auto pg0 = scacopf_prob->variable("p_g", data);
    assert(pg0->n == data.G_Generator.size());
    assert(K_idx>=0 && K_idx<data.K_outidx.size());
    assert(data.K_outidx.size() == data.K_Contingency.size());

    int idx_gen = data.K_outidx[K_idx];
    assert(idx_gen>=0 && idx_gen<pg0->n);
    
    info_out[1]=pg0->x[idx_gen];
    info_out[2]=info_out[3]=info_out[4]=0.;

  } else if(data.K_ConType[K_idx] == SCACOPFData::kLine) {

    auto qli1 = scacopf_prob->variable("q_li1", data);
    auto qli2 = scacopf_prob->variable("q_li2", data); assert(qli2->n==qli1->n);
    auto pli1 = scacopf_prob->variable("p_li1", data); assert(pli1->n==qli1->n);
    auto pli2 = scacopf_prob->variable("p_li2", data); assert(pli2->n==qli1->n);
    assert(data.L_Line.size() == qli1->n);
    assert(K_idx>=0 && K_idx<data.K_outidx.size());

    int idx = data.K_outidx[K_idx];
    assert(idx>=0 && idx<qli1->n);
    
    info_out[1]=pli1->x[idx];
    info_out[2]=qli1->x[idx];
    info_out[3]=pli2->x[idx];
    info_out[4]=qli2->x[idx];

  } else if(data.K_ConType[K_idx] == SCACOPFData::kTransformer) {
    auto qti1 = scacopf_prob->variable("q_ti1", data);
    auto qti2 = scacopf_prob->variable("q_ti2", data);
    auto pti1 = scacopf_prob->variable("p_ti1", data);
    auto pti2 = scacopf_prob->variable("p_ti2", data);

    assert(data.T_Transformer.size() == qti1->n);
    assert(K_idx>=0 && K_idx<data.K_outidx.size());
    
    int idx = data.K_outidx[K_idx];
    assert(idx>=0 && idx<qti1->n);
    
    info_out[1]=pti1->x[idx];
    info_out[2]=qti1->x[idx];
    info_out[3]=pti2->x[idx];
    info_out[4]=qti2->x[idx];

  }

  //status is OK=0 or failure<0 or OK-ish>0
  //return penalty/objective for the contingency problem

}
double MyCode1::solve_contingency(int K_idx, int& status)
{
  assert(iAmEvaluator);
  assert(scacopf_prob != NULL);
  
  status = 0; //be positive
  auto p_g0 = scacopf_prob->variable("p_g", data); 
  auto v_n0 = scacopf_prob->variable("v_n", data);

  goTimer t; t.start();
  
  ContingencyProblem prob(data, K_idx, my_rank);

  if(data.N_Bus.size()>8999) {
    //prob.monitor.is_active = true;
    //prob.monitor.pen_threshold = pen_threshold;
  }

  prob.update_AGC_smoothing_param(1e-2);
  prob.update_PVPQ_smoothing_param(1e-2);

  //xxxcont

  if(!prob.default_assembly(p_g0, v_n0)) {
    printf("Evaluator Rank %d failed in default_assembly for contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -1;
    return 1e+20;
  }

  if(!prob.set_warm_start_from_base_of(*scacopf_prob)) {
    status = -2;
    return 1e+20;
  }

  prob.use_nlp_solver("ipopt");
  prob.set_solver_option("sb","yes");
  prob.set_solver_option("print_frequency_iter", 1);
  prob.set_solver_option("linear_solver", "ma57"); 
  prob.set_solver_option("print_level", 2);
  prob.set_solver_option("mu_init", 1e-4);
  prob.set_solver_option("mu_target", 1e-9);//!

  //return if it takes too long in phase2
  prob.set_solver_option("max_iter", 1700);
  prob.set_solver_option("acceptable_tol", 1e-3);
  prob.set_solver_option("acceptable_constr_viol_tol", 1e-5);
  prob.set_solver_option("acceptable_iter", 5);

  //if(data.N_Bus.size()<10000) 
  {
    prob.set_solver_option("bound_relax_factor", 0.);
    prob.set_solver_option("bound_push", 1e-16);
    prob.set_solver_option("slack_bound_push", 1e-16);
  }
  prob.set_solver_option("mu_linear_decrease_factor", 0.4);
  prob.set_solver_option("mu_superlinear_decrease_power", 1.25);

  //scacopf_prob->duals_bounds_lower()->print_summary("duals bounds lower");
  //scacopf_prob->duals_bounds_upper()->print_summary("duals bounds upper");
  //scacopf_prob->duals_constraints()->print_summary("duals constraints");

  double penalty; 
  if(!prob.eval_obj(p_g0, v_n0, penalty)) {
    printf("Evaluator Rank %d failed in the eval_obj of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    penalty=1e+6;
  }
  int num_iter = prob.number_of_iterations();

  // if(false) {
  //   vector<double> smoothing_params = {1e-3, 1e-4, 1e-5, 1e-6};  

  //   for(auto smo: smoothing_params) {
  //     prob.update_AGC_smoothing_param(smo);
  //     prob.update_PVPQ_smoothing_param(smo);
  //     prob.set_solver_option("mu_init", 1e-8);
  //     prob.reoptimize(OptProblem::primalDualRestart);
  //     num_iter += prob.number_of_iterations();
  //     printf("Evaluator Rank %3d contingency reoptimize done K_idx=%4d - smoothing=%8.2e\n", 
  // 	     my_rank, K_idx, smo);
  //   }
  //   //prob.print_p_g_with_coupling_info(*prob.data_K[0], p_g0);
  // }
  //if(true)   {
    //prob.print_objterms_evals();
    
    //scacopf_prob->print_p_g_with_coupling_info(*scacopf_prob->data_K[0]);
    //if(scacopf_prob->data_K.size()>0)
    //prob.print_PVPQ_info(*prob.data_K[0]);

    //auto v_n0 = scacopf_prob->variable("v_n", data);
    //v_n0->print();
    //auto v_nk = scacopf_prob->variable("v_n", *scacopf_prob->data_K[0]);
    //v_nk->print();

    //prob.print_active_power_balance_info(*prob.data_K[0]);
    //prob.print_line_limits_info(*prob.data_K[0]);
    //if(scacopf_prob->data_K.size()>0)

    //prob.print_p_g_with_coupling_info(*prob.data_K[0], p_g0);


    //prob.print_active_power_balance_info(*prob.data_K[0]);
    //prob.print_reactive_power_balance_info(*prob.data_K[0]);
    //prob.print_line_limits_info(*prob.data_K[0]);
  // }
  printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	 "in %5.3f sec and %3d iterations  sol_from_scacopf_pass %d  global time %g\n",
	 my_rank, K_idx, penalty, t.stop(), 
	 num_iter, phase3_scacopf_pass_solution, glob_timer.measureElapsedTime());
  
  return penalty;
}


double MyCode1::phase3_solve_scacopf(std::vector<int>& K_idxs,
				     const std::vector<double>& K_penalties,
				     const std::vector<std::vector<double> >& K_powers,
				     const std::vector<int>& K_actions)
{
  assert(scacopf_prob!=NULL);
  assert(iAmSolver);
  assert(K_idxs.size()==K_penalties.size());
  assert(K_idxs.size()==K_actions.size());


  goTimer t; t.start();

  {
    auto K_idxs_temp = K_idxs;
    sort(K_idxs_temp.begin(), K_idxs_temp.end());
    sort(K_SCACOPF_phase3.begin(), K_SCACOPF_phase3.end());
    if(K_idxs_temp == K_SCACOPF_phase3) {
      printf("Solver Rank %d - phase3_solve_scacopf - no new contingencies detected\n",
	     my_rank);
      return -11111;
    }
  }

  vector<int> Ks_to_add_as_blocks;

  string smsg="Solver Rank 1 actions: "; char msg[1024];
  for(int it=0; it<K_idxs.size(); it++) {

    sprintf(msg, "K_idx=%d >> %d ", K_idxs[it], K_actions[it]); smsg += msg;

    double penalty_weight = 10./data.K_Contingency.size();

    if(K_actions[it] == -101) {
      //penalize
      int idx_elem = data.K_outidx[K_idxs[it]];
      double penalty = K_penalties[it];
      assert(4 == K_powers[it].size());


      if(data.K_ConType[K_idxs[it]]==SCACOPFData::kGenerator) {
	scacopf_prob->remove_quadr_conting_penalty_pg0(idx_elem);
	scacopf_prob->add_quadr_conting_penalty_pg0(idx_elem, 
						    K_powers[it][0], 
						    penalty_weight*penalty);
	sprintf(msg, "[pen gen] |");
      } else if(data.K_ConType[K_idxs[it]]==SCACOPFData::kLine) {
	scacopf_prob->remove_conting_penalty_line0(idx_elem);
	scacopf_prob->add_conting_penalty_line0(idx_elem, 
						K_powers[it][0], K_powers[it][1],
						K_powers[it][2], K_powers[it][3],
						penalty_weight*penalty);
	sprintf(msg, "[pen line] |");
      } else if(data.K_ConType[K_idxs[it]]==SCACOPFData::kTransformer) {
	scacopf_prob->remove_conting_penalty_transf0(idx_elem);
	scacopf_prob->add_conting_penalty_transf0(idx_elem, 
						K_powers[it][0], K_powers[it][1],
						K_powers[it][2], K_powers[it][3],
						penalty_weight*penalty);
	sprintf(msg, "[pen transf] |");
      } else { assert(false); }

      smsg += msg;

      scacopf_prob->primal_problem_changed();

    } else if(K_actions[it] == -102) {
      //include in scacopf
      if(!scacopf_prob->has_contigency(K_idxs[it])) {
	smsg += " [new]";
	Ks_to_add_as_blocks.push_back(K_idxs[it]);
      }
      smsg += " |";

    } else if(K_actions[it] >=0) {
      //combine with K_idx = K_actions[i
      assert(false);
    }
  }
  printf("%s at glob_time %g\n", smsg.c_str(), glob_timer.measureElapsedTime());

  
  if(Ks_to_add_as_blocks.size()>0) {
    for(int K_idx: Ks_to_add_as_blocks)
      scacopf_prob->add_contingency_block(K_idx);

    scacopf_prob->dual_problem_changed();
    

    for(int K_idx: Ks_to_add_as_blocks)
      scacopf_prob->set_warm_start_for_cont_from_base_of(K_idx, *scacopf_prob);


  } else {

  }

  //scacopf_prob->set_AGC_as_nonanticip(true);
  //scacopf_prob->set_AGC_simplified(true);
  //scacopf_prob->set_PVPQ_as_nonanticip(true);

  //TL_rate_reduction was computed in phase1
  //reduce T and L rates to min(RateBase, TL_rate_reduction*RateEmer)
  scacopf_prob->set_basecase_L_rate_reduction(TL_rate_reduction);
  scacopf_prob->set_basecase_T_rate_reduction(TL_rate_reduction);

  //!  scacopf_prob->set_quadr_penalty_qg0(true);

  bool blarge_prob = data.N_Bus.size() > 20000;

  scacopf_prob->use_nlp_solver("ipopt"); 
  scacopf_prob->set_solver_option("linear_solver", "ma57"); 
  scacopf_prob->set_solver_option("mu_init", 1e-4);
  scacopf_prob->set_solver_option("print_frequency_iter", 5);
  scacopf_prob->set_solver_option("print_level", 5);

  scacopf_prob->set_solver_option("acceptable_tol", 1e-4);
  scacopf_prob->set_solver_option("acceptable_constr_viol_tol", 1e-6);
  scacopf_prob->set_solver_option("acceptable_iter", 7);
    
  if(!blarge_prob) {
    scacopf_prob->set_solver_option("mu_target", 1e-9);
    scacopf_prob->set_solver_option("max_iter", 600);
    scacopf_prob->set_solver_option("bound_relax_factor", 0.);
    scacopf_prob->set_solver_option("bound_push", 1e-18);
    scacopf_prob->set_solver_option("slack_bound_push", 1e-18);
    scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.4);
    scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.4); 
  } else {
    scacopf_prob->set_solver_option("mu_target", 1e-8);
    scacopf_prob->set_solver_option("max_iter", 400);
    scacopf_prob->set_solver_option("bound_relax_factor", 1e-8);
    scacopf_prob->set_solver_option("bound_push", 1e-8);
    scacopf_prob->set_solver_option("slack_bound_push", 1e-8);
    scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.5);
    scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.2);     
  }

  
  //bool bret = scacopf_prob->optimize("ipopt");
  bool bret = scacopf_prob->reoptimize(OptProblem::primalDualRestart);
  //bret = scacopf_prob->reoptimize(OptProblem::primalDualRestart);

  //if(scacopf_prob->data_K.size()>0)
  //  scacopf_prob->print_p_g_with_coupling_info(*scacopf_prob->data_K[0]);

  phase3_scacopf_passes_solver++;
  req_send_base_sols.post_new_sol(scacopf_prob, Tag7, my_rank, comm_world, phase3_scacopf_passes_solver);

#ifdef DEBUG_COMM
  printf("Solver Rank %d send posted basecase solution from pass %d at global time %g\n",
	 my_rank, phase3_scacopf_passes_solver, glob_timer.measureElapsedTime());
#endif

  double cost = scacopf_prob->objective_value();
  printf("Solver Rank %d - finished scacopf solve in %g seconds "
	 "%d iters -- cost=%g  TL_rate=%g  glob time %g K_idxs=\n",
	 my_rank, t.stop(), scacopf_prob->number_of_iterations(), 
	 cost, TL_rate_reduction, glob_timer.measureElapsedTime());
  printvec(scacopf_prob->get_contingencies());

  //write solution
  if(!bret) {
    printf("[warning] Solver rank %d: scacopf solve failed; solution1 NOT written\n",
	   my_rank);
  } else {
    scacopf_prob->write_solution_basecase();
    scacopf_prob->write_pridua_solution_basecase();
#ifdef DEBUG
    scacopf_prob->write_solution_extras_basecase();
#endif
  }


  for(int i=0; i<100; i++)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

  //this will free memory associated with all basecase solution send requests that completed
  int n_cleanups = req_send_base_sols.attempt_cleanup();
#ifdef DEBUG_COMM
  printf("Solver Rank %d -> cleanups: %d  [at pass %d]\n",
  	 my_rank, n_cleanups, phase3_scacopf_passes_solver);
#endif

  return cost;
}

void MyCode1::display_instance_info()
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSec);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}

struct KGenInfo {
  KGenInfo() : Kidx(-1), genidx(-1), loss(0.) {};
  KGenInfo(const int& Kidx_, const int& genidx_, const double& loss_) 
    : Kidx(Kidx_), genidx(genidx_), loss(loss_) {};
  int Kidx, genidx;
  double loss;
};

void MyCode1::get_high_priority_Kgens(int Kmax, std::vector<int>& K_high_prio, gollnlp::SCACOPFProblem* prob)
{
  //gollnlp::goTimer timer; timer.start();
  auto p_g0 = prob->variable("p_g", data), q_g0 = prob->variable("p_g", data);

  std::unordered_map<int, KGenInfo> N_gens_loss;
  std::unordered_map<int, KGenInfo> N_gens_gain;

  int nKgens=0;

  //find the largest generator loss and gain at each Bus with Kgenerators 

  for(int i=0; i<data.K_Contingency.size(); i++) {
    if(data.K_ConType[i] != gollnlp::SCACOPFData::kGenerator) continue;
    nKgens++;
    const int Kidx = data.K_Contingency[i];
    const int genidx = data.K_outidx[i]; 
    assert(genidx>=0); assert(genidx<p_g0->n); assert(genidx<q_g0->n); 

    if(p_g0->x[i] > 1e-4) {
      double loss = p_g0->x[genidx];
      auto it = N_gens_loss.find(data.G_Bus[genidx]);
      if(it != N_gens_loss.end()) {
	if(it->second.loss < loss) {
	  it->second.loss = loss; 
	  it->second.Kidx = Kidx;
	  it->second.genidx = genidx;
	}
	//printf(" ++++ bus %d  loss %.4f genidx %d lb=%g ub=%g\n", 
	//it->first, loss, genidx, data.G_Plb[genidx],  data.G_Pub[genidx]);
      } else {

	N_gens_loss[data.G_Bus[i]] = KGenInfo(Kidx, genidx, loss);

	auto it = N_gens_loss.find(data.G_Bus[i]);
	//printf(" !!!! bus %d  loss %.4f genidx %d lb=%g ub=%g\n", 
	//it->first, loss, genidx, data.G_Plb[genidx],  data.G_Pub[genidx]);
      }

    } else if(p_g0->x[i] < -1e-4) {
      double gain = p_g0->x[genidx];
      auto it = N_gens_gain.find(data.G_Bus[genidx]);
      if(it != N_gens_gain.end()) {
	if(it->second.loss > gain) {
	  it->second.loss = gain; 
	  it->second.Kidx = Kidx;
	  it->second.genidx = genidx;
	}
      } else {
	N_gens_gain[data.G_Bus[i]] = KGenInfo(Kidx, genidx, gain);
      }
    }
  }

#ifdef DEBUG
  int nbuseswithgens = 0;
  {
    std::unordered_map<int, int> N_with_gens;
    for(int i=0; i<data.G_Bus.size(); i++) {
      auto it = N_with_gens.find(data.G_Bus[i]);
      if(it != N_with_gens.end()) {
	N_with_gens[data.G_Bus[i]]++;
      } else {
	N_with_gens[data.G_Bus[i]]=1;
      }
    }
    nbuseswithgens = N_with_gens.size();
  }

  printf("found %lu buses with Kgens having losses (total buses with gens %d); total gens %lu out of which %d are Kgens\n", 
	 N_gens_loss.size(), nbuseswithgens, data.G_Bus.size(), nKgens);
  printf("found %lu buses with Kgens having gains; total gens %lu out of which %d are Kgens\n", 
	 N_gens_gain.size(), data.G_Bus.size(), nKgens);
  
#endif

  vector<KGenInfo> v;
  for(auto it=N_gens_loss.begin(); it != N_gens_loss.end(); ++it) {
    //printf(" -------- bus %d  loss %.4f genidx %d\n", it->first, it->second.loss, it->second.genidx);
    v.push_back(it->second);
  }
  std::sort(v.begin(), v.end(), [&](const KGenInfo& a, const KGenInfo& b) { return a.loss > b.loss; });
  int Kmax1 = std::min(Kmax, (int)v.size());
  for(int i=0; i<Kmax1; i++) K_high_prio.push_back(v[i].Kidx);


#ifdef DEBUG
  //printvec(K_high_prio, "Kidx largest losses");
  //for(int i=0; i<K_high_prio.size(); i++) {
  //  printf(" [%d] Kidx %d  loss %.3f genidx %d\n", i, K_high_prio[i], p_g0->x[data.K_outidx[K_high_prio[i]]], data.K_outidx[K_high_prio[i]]);
  //}
#endif

  v.clear();
  for(auto it=N_gens_gain.begin(); it != N_gens_gain.end(); ++it)
    v.push_back(it->second);
  std::sort(v.begin(), v.end(), [&](const KGenInfo& a, const KGenInfo& b) { return a.loss < b.loss; });
  Kmax1 = std::min(Kmax, (int)N_gens_gain.size());
  for(int i=0; i<Kmax1; i++) K_high_prio.push_back(v[i].Kidx);
  //printvec(K_high_prio, "Kidx largest gains");

  //printf("append_high_priority_Kgens took %.3f sec\n", timer.getElapsedTime());
}

struct KTransmInfo {
  KTransmInfo() : Kidx(-1), TL_idx(-1), priority(-1.17) {};
  KTransmInfo(const int& Kidx_, const int& TL_idx_, const double& priority_) 
    : Kidx(Kidx_), TL_idx(TL_idx_), priority(priority_) {};
  int Kidx, TL_idx;
  double priority;
};

ostream& operator<<(ostream& os, const KTransmInfo& o)
{
  os<< "Kidx=" << o.Kidx << " TL_idx=" << o.TL_idx << " priori=" << o.priority << endl;
  return os;
}

void MyCode1::get_high_priority_Ktransm(int Kmax, std::vector<int>& K_high_prio, gollnlp::SCACOPFProblem* prob)
{
  gollnlp::goTimer timer; timer.start();
  auto p_g = prob->variable("p_g", data), q_g = prob->variable("p_g", data);
  auto v_n = prob->variable("v_n", data); 
  auto p_li1 = prob->variable("p_li1", data), p_li2 = prob->variable("p_li2", data);
  auto q_li1 = prob->variable("q_li1", data), q_li2 = prob->variable("q_li2", data);
  auto p_ti1 = prob->variable("p_ti1", data), p_ti2 = prob->variable("p_ti2", data);
  auto q_ti1 = prob->variable("q_ti1", data), q_ti2 = prob->variable("q_ti2", data);

  //compute max capacity
  const int L = data.L_Line.size(), T=data.T_Transformer.size();
  vector<double> pli1_capac(L), pli2_capac(L), pti1_capac(T), pti2_capac(T);
  double aux1, aux2;
  for(int i=0; i<L; i++) {
    aux1 = 0.85*data.L_RateEmer[i]; aux1 = aux1*aux1;
    aux2=aux1 - q_li1->x[i]*q_li1->x[i];
    pli1_capac[i] = aux2>0 ? sqrt(aux2) : 0.;
    aux2=aux1 - q_li2->x[i]*q_li2->x[i];
    pli2_capac[i] = aux2>0 ? sqrt(aux2) : 0.;
  }
  for(int i=0; i<T; i++) {
    aux1 = 0.85*data.T_RateEmer[i]; aux1 = aux1*aux1;
    aux2=aux1 - q_ti1->x[i]*q_ti1->x[i];
    pti1_capac[i] = aux2>0 ? sqrt(aux2) : 0.;

    aux2=aux1 - q_ti2->x[i]*q_ti2->x[i];
    pti2_capac[i] = aux2>0 ? sqrt(aux2) : 0.;
  }
  //printvec(pli1_capac);   printvec(pli2_capac);   printvec(pti1_capac);   printvec(pti2_capac);

  double pli_max=0., pti_max=0.;
  for(int i=0; i<L; i++) {
    aux1 = abs(p_li1->x[i]); if(aux1>pli_max) pli_max = aux1;
    aux1 = abs(p_li2->x[i]); if(aux1>pli_max) pli_max = aux1;
  }
  for(int i=0; i<T; i++) {
    aux1 = abs(p_ti1->x[i]); if(aux1>pti_max) pti_max = aux1;
    aux1 = abs(p_ti2->x[i]); if(aux1>pti_max) pti_max = aux1;
  }
  pti_max = std::max(pli_max, pti_max); pli_max=pti_max;

  //priorities range from 0 to +1e+10
  vector<KTransmInfo> kline_info, ktransf_info;

  int B = data.N_Bus.size();
  //maximum ramping for inflow/positive and outflow/negative powers as the sum of the capacities of 
  //inflow and outflow transmission and generators
  vector<double> bus_inflow_ramping(B, -1e20),  bus_outflow_ramping(B, +1e20);
  for(int i=0; i<data.K_Contingency.size(); i++) {
    if(data.K_ConType[i] == gollnlp::SCACOPFData::kGenerator) continue;
    const int TL_idx = data.K_outidx[i];
    const int Kidx = data.K_Contingency[i];
    int N_from_idx = -1, N_to_idx = -1;
    if(data.K_ConType[i] == gollnlp::SCACOPFData::kLine) { 
      N_from_idx = data.L_Nidx[0][TL_idx];//data.L_From[TL_idx]; 
      N_to_idx   = data.L_Nidx[1][TL_idx];//data.L_To[TL_idx]; 
      //printf("KLineidx %d N_from_idx %d  N_to_idx %d\n", TL_idx, N_from_idx, N_to_idx);
    } else {
      assert(data.K_ConType[i] == gollnlp::SCACOPFData::kTransformer);
      N_from_idx = data.T_Nidx[0][TL_idx];//data.T_From[TL_idx]; 
      N_to_idx   = data.T_Nidx[1][TL_idx];//data.T_To[TL_idx]; 
      //printf("KKTransfidx %d N_from_idx %d  N_to_idx %d\n", TL_idx, N_from_idx, N_to_idx);
    }

    if(bus_inflow_ramping[N_from_idx] == -1e20 || bus_outflow_ramping[N_from_idx] == +1e20) { 
      double in, out;
      compute_bus_inflow_outflow_rampings(N_from_idx, p_g, v_n, 
					  p_li1, p_li2, p_ti1, p_ti2, 
					  pli1_capac, pli2_capac, pti1_capac, pti2_capac,
					  in, out);
      assert(in>=0 && out<=0);
      bus_inflow_ramping[N_from_idx] = in;
      bus_outflow_ramping[N_from_idx] = out;
    }

    if(bus_outflow_ramping[N_to_idx] == -1e20 || bus_outflow_ramping[N_to_idx] == +1e20) {
      double in, out;
      compute_bus_inflow_outflow_rampings(N_to_idx, p_g, v_n, 
					  p_li1, p_li2, p_ti1, p_ti2, 
					  pli1_capac, pli2_capac, pti1_capac, pti2_capac,
					  in, out); assert(in>=0 && out<=0);
      bus_inflow_ramping[N_to_idx] = in;
      bus_outflow_ramping[N_to_idx] = out;
    }
    
    if(data.K_ConType[i] == gollnlp::SCACOPFData::kLine) {

      bool special=false;

      double r1=0.;
      if(p_li1->x[TL_idx] < -1e-8) {
	double remaining_inflow_ramping = bus_inflow_ramping[N_from_idx] - std::max(0., pli1_capac[TL_idx] + p_li1->x[TL_idx]);

	//printf(" --- lidx=%d, pli=%g remaining_inflow_ramping=%g\n", 
	//       TL_idx, p_li1->x[TL_idx], remaining_inflow_ramping);
	assert(remaining_inflow_ramping >=0);

	r1 = 1. + (-p_li1->x[TL_idx])/pli_max;
	if(remaining_inflow_ramping>1e-8) {
	  r1 = - p_li1->x[TL_idx] / remaining_inflow_ramping;
	  if(r1>1) r1 = 1. + (-p_li1->x[TL_idx])/pli_max;
	} else {
	  special=true;
	}

      } else if(p_li1->x[TL_idx] > 1e-8) {
	double remaining_outflow_ramping = bus_outflow_ramping[N_from_idx] + std::max(0., pli1_capac[TL_idx]-p_li1->x[TL_idx]);
	assert(remaining_outflow_ramping<=0);

	r1 = 1. + (p_li1->x[TL_idx])/pli_max;
	if(remaining_outflow_ramping<-1e-8) {
	  r1 = p_li1->x[TL_idx] / (0-remaining_outflow_ramping);
	  if(r1>1.) r1 = 1. + (p_li1->x[TL_idx])/pli_max;
	} else {
	  special=true;
	}

      } else {

      }
      
      
      double r2 = 0.;
      if(p_li2->x[TL_idx] < -1e-8) {

	double remaining_inflow_ramping = bus_inflow_ramping[N_to_idx] - std::max(0., p_li2->x[TL_idx] + pli2_capac[TL_idx]);

	assert(remaining_inflow_ramping >=0);
	r2 = 1. + (-p_li2->x[TL_idx])/pli_max;;//-100*p_li2->x[TL_idx];

     	if(remaining_inflow_ramping>1e-8) {
     	  r2 = - p_li2->x[TL_idx] / remaining_inflow_ramping; assert(r2>=0);
	  if(r2>1.) r2 = 1. + (-p_li2->x[TL_idx])/pli_max;
	} else {
	  special=true;
	}

      } else if(p_li2->x[TL_idx] > 1e-8) {
	double remaining_outflow_ramping = bus_outflow_ramping[N_to_idx] +  std::max(0., pli2_capac[TL_idx]-p_li2->x[TL_idx]);
	assert(remaining_outflow_ramping<=0);
	r2 = 1. + (p_li2->x[TL_idx])/pli_max;
     	if(remaining_outflow_ramping<-1e-8) {
     	  r2 = p_li2->x[TL_idx] / (0-remaining_outflow_ramping); assert(r2>=0);
	  if(r2>1.) r2 = 1. + (p_li2->x[TL_idx])/pli_max;
	} else {
	  special=true;
	}

      } else {
	
      }
      
      kline_info.push_back(KTransmInfo(Kidx, TL_idx, std::max(r1,r2))); 

      //printf("Kidx=%d Kline_idx %d N_from_idx %d  N_to_idx %d r1,r2=%g %g %s\n", 
      //     Kidx, TL_idx, N_from_idx, N_to_idx, r1, r2, special ? "!S!" : "");
    } // end of if ConType==kLine
    else     if(data.K_ConType[i] == gollnlp::SCACOPFData::kTransformer) {

      bool special=false;

      double r1=0.;
      if(p_ti1->x[TL_idx] < -1e-8) {
	double remaining_inflow_ramping = bus_inflow_ramping[N_from_idx] - std::max(0., pti1_capac[TL_idx] + p_ti1->x[TL_idx]);

	//printf(" --- lidx=%d, pli=%g remaining_inflow_ramping=%g\n", 
	//       TL_idx, p_li1->x[TL_idx], remaining_inflow_ramping);
	assert(remaining_inflow_ramping >=0);

	r1 = 1. + (-p_ti1->x[TL_idx])/pti_max;
	if(remaining_inflow_ramping>1e-8) {
	  r1 = - p_ti1->x[TL_idx] / remaining_inflow_ramping;
	  if(r1>1) r1 = 1. + (-p_ti1->x[TL_idx])/pti_max;
	} else {
	  special=true;
	}

      } else if(p_ti1->x[TL_idx] > 1e-8) {
	double remaining_outflow_ramping = bus_outflow_ramping[N_from_idx] + std::max(0., pti1_capac[TL_idx]-p_ti1->x[TL_idx]);
	assert(remaining_outflow_ramping<=0);

	r1 = 1. + (p_ti1->x[TL_idx])/pti_max;
	if(remaining_outflow_ramping<-1e-8) {
	  r1 = p_ti1->x[TL_idx] / (0-remaining_outflow_ramping);
	  if(r1>1.) r1 = 1. + (p_ti1->x[TL_idx])/pti_max;
	} else {
	  special=true;
	}

      } else {

      }
      
      
      double r2 = 0.;
      if(p_ti2->x[TL_idx] < -1e-8) {

	double remaining_inflow_ramping = bus_inflow_ramping[N_to_idx] - std::max(0., p_ti2->x[TL_idx] + pti2_capac[TL_idx]);

	//printf("bus_inflow_ramping[N_to_idx]=%12.6e remaining_inflow_ramping=%12.6e pli2_capac[TL_idx]=%12.6e\n", 
	//       bus_inflow_ramping[N_to_idx], remaining_inflow_ramping, pli2_capac[TL_idx]);

	assert(remaining_inflow_ramping >=0);
	r2 = 1. + (-p_ti2->x[TL_idx])/pti_max;;//-100*p_li2->x[TL_idx];

     	if(remaining_inflow_ramping>1e-8) {
     	  r2 = - p_ti2->x[TL_idx] / remaining_inflow_ramping; assert(r2>=0);
	  if(r2>1.) r2 = 1. + (-p_ti2->x[TL_idx])/pti_max;
	} else {
	  special=true;
	}

      } else if(p_ti2->x[TL_idx] > 1e-8) {
	double remaining_outflow_ramping = bus_outflow_ramping[N_to_idx] +  std::max(0., pti2_capac[TL_idx]-p_ti2->x[TL_idx]);
	assert(remaining_outflow_ramping<=0);
	r2 = 1. + (p_ti2->x[TL_idx])/pti_max;
     	if(remaining_outflow_ramping<-1e-8) {
     	  r2 = p_ti2->x[TL_idx] / (0-remaining_outflow_ramping); assert(r2>=0);
	  if(r2>1.) r2 = 1. + (p_ti2->x[TL_idx])/pti_max;
	} else {
	  special=true;
	}

      } else {
	
      }
      
      ktransf_info.push_back(KTransmInfo(Kidx, TL_idx, std::max(r1,r2))); 

      //printf("Kidx=%d KTran_idx %d N_from_idx %d  N_to_idx %d r1,r2=%g %g %s\n", 
      //     Kidx, TL_idx, N_from_idx, N_to_idx, r1, r2, special ? "!S!" : "");
    } // end of if ConType==kTransformer
      
  } 

#ifdef DEBUG
  std::sort(kline_info.begin(), kline_info.end(), 
	    [&](const KTransmInfo& a, const KTransmInfo& b) { return a.priority > b.priority; });
  std::sort(ktransf_info.begin(), ktransf_info.end(), 
	    [&](const KTransmInfo& a, const KTransmInfo& b) { return a.priority > b.priority; });


  //printvec(kline_info, "Klines in order");
  //printvec(ktransf_info, "Ktransf in order");
#endif
  kline_info.insert(kline_info.end(), ktransf_info.begin(), ktransf_info.end());
  std::sort(kline_info.begin(), kline_info.end(), 
	    [&](const KTransmInfo& a, const KTransmInfo& b) { return a.priority > b.priority; });
  
  int Kmax1 = std::min(Kmax, (int)kline_info.size());

  for(int i=0; i<Kmax1; i++)  {
    K_high_prio.push_back(kline_info[i].Kidx);
    if(kline_info[i].priority <= 0.1) break;
  }
  
}


double MyCode1::
compute_bus_inflow_outflow_rampings(const int Nidx, 
				    gollnlp::OptVariablesBlock* p_g, 
				    gollnlp::OptVariablesBlock* v_n,
				    gollnlp::OptVariablesBlock* p_li1,
				    gollnlp::OptVariablesBlock* p_li2,
				    gollnlp::OptVariablesBlock* p_ti1,
				    gollnlp::OptVariablesBlock* p_ti2,
				    std::vector<double>& pli1_capac,
				    std::vector<double>& pli2_capac,
				    std::vector<double>& pti1_capac,
				    std::vector<double>& pti2_capac,
				    double& inflow_ramp, double& outflow_ramp)
{
  inflow_ramp = outflow_ramp = 0.;
#ifdef DEBUG
  double inflow2 = 0,  outflow2 = 0;

  if(data.N_Pd[Nidx] > 0) outflow2  -= data.N_Pd[Nidx];
  else                    inflow2   -= data.N_Pd[Nidx];

  if(data.N_Gsh[Nidx] > 0) outflow2 -= data.N_Gsh[Nidx] * v_n->xref[Nidx] * v_n->xref[Nidx];
  else                     inflow2  -= data.N_Gsh[Nidx] * v_n->xref[Nidx] * v_n->xref[Nidx];
  string str; char msg[1024];
  sprintf(msg, "Bus[%d] ", Nidx); str += msg;
#endif

  for(int ig=0; ig<data.Gn[Nidx].size(); ig++) {
    const int genidx=data.Gn[Nidx][ig]; assert(genidx>=0 && genidx<p_g->n);
    const double pg = p_g->x[genidx];
    if(pg>0) {
      inflow_ramp += p_g->ub[genidx]-pg;
      outflow_ramp -=  pg - p_g->lb[genidx];
#ifdef DEBUG
      inflow2 += pg;
#endif
    } else {
      inflow_ramp += p_g->ub[genidx]-pg;
      outflow_ramp -=  pg - p_g->lb[genidx];
#ifdef DEBUG
      outflow2 += pg;
#endif   
    }
#ifdef DEBUG
    sprintf(msg, "g[%d]->%g [%g,%g]  ", genidx, pg, p_g->lb[genidx], p_g->ub[genidx]); str += msg;
#endif
  }

  for(int ilix=0; ilix<data.Lidxn1[Nidx].size(); ilix++) {
    const int lidx = data.Lidxn1[Nidx][ilix];
    const double pli = p_li1->x[lidx];
    //printf("---- pli1=%15.8f capac=%15.8f \n", pli, pli1_capac[lidx]);
    if(pli>0) { 
      outflow_ramp -= std::max(0., pli1_capac[lidx]-pli);
#ifdef DEBUG
      outflow2 -= pli;
#endif
    } else { //assert(pli<=pli1_capac[lidx]);
      inflow_ramp +=  std::max(0., pli+pli1_capac[lidx]);
#ifdef DEBUG
      inflow2 -= pli;
#endif
    }
#ifdef DEBUG
    sprintf(msg, "pli1[%d]->%g  ", lidx, pli); str += msg;
#endif
  }

  for(int ilix=0; ilix<data.Lidxn2[Nidx].size(); ilix++) {
    const int lidx = data.Lidxn2[Nidx][ilix];
    const double pli = p_li2->x[lidx];
    //printf("---- pli2=%15.8f capac=%15.8f idx=%d\n", pli, pli1_capac[lidx], lidx);
    if(pli>0) {
      outflow_ramp -= std::max(0., pli2_capac[lidx]-pli);
#ifdef DEBUG
      outflow2 -= pli;
#endif
    } else {
      inflow_ramp += std::max(0., pli + pli2_capac[lidx]);
#ifdef DEBUG
      inflow2 -= pli;
#endif
    }
#ifdef DEBUG
    sprintf(msg, "pli2[%d]->%g ", lidx, pli); str += msg;
#endif
  }

  for(int itix=0; itix<data.Tidxn1[Nidx].size(); itix++) {
    const int tidx = data.Tidxn1[Nidx][itix];
    const double pti = p_ti1->x[tidx];
    //printf("---- pti1=%15.8f capac=%15.8f idx=%d\n", pti, pti1_capac[tidx], tidx);
    if(pti>0) {
      outflow_ramp -= std::max(0., pti1_capac[tidx] - pti);
#ifdef DEBUG
      outflow2 -= pti;
#endif
    } else {
      inflow_ramp += std::max(0., pti + pti1_capac[tidx]);
#ifdef DEBUG
      inflow2 -= pti;
#endif
    }
#ifdef DEBUG
    sprintf(msg, "pti1[%d]->%g ", tidx, pti); str += msg;
#endif
  }

  for(int itix=0; itix<data.Tidxn2[Nidx].size(); itix++) {
    const int tidx = data.Tidxn2[Nidx][itix];
    const double pti = p_ti2->x[data.Tidxn2[Nidx][itix]];
    //printf("---- pti2=%15.8f capac=%15.8f idx=%d\n", pti, pti2_capac[tidx], tidx);
    if(pti>0) {
      outflow_ramp -= std::max(0., pti2_capac[tidx] - pti);
#ifdef DEBUG
      outflow2 -= pti;
#endif
    } else {
      inflow_ramp += std::max(0., pti + pti2_capac[tidx]);
#ifdef DEBUG
      inflow2 -= pti;
#endif
    }
#ifdef DEBUG
    sprintf(msg, "pti2[%d]->%g ", tidx, pti); str += msg;
#endif
  }

#ifdef DEBUG
  sprintf(msg, "[inflow %15.8e outflow %15.8e] [inflow_ramp %6.2f  outflow_ramp %6.2f] ", 
  	  inflow2, outflow2, inflow_ramp, outflow_ramp);


  //sprintf(msg, "[inflow/outflow_ramps %6.2f %6.2f][flowoneside %6.3f] ", 
  //	  inflow_ramp, outflow_ramp, inflow2);
  str = msg+str;
  //printf("%s\n", str.c_str());
#endif

  return 0;
}

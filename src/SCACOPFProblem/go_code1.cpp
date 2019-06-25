#include "go_code1.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"

#include "goTimer.hpp"
#include "goUtils.hpp"

using namespace std;
using namespace gollnlp;

#include "unistd.h"
#include <chrono>
#include <thread>

//#define DEBUG_COMM 1


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
  iAmMaster=iAmSolver=iAmEvaluator=false;
  scacopf_prob = NULL;
  my_rank = -1;

  req_recv_K_idx = NULL;
  req_send_penalty = NULL;
}

MyCode1::~MyCode1()
{

}

int MyCode1::initialize(int argc, char *argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    return false;
  }

  ret = MPI_Comm_rank(comm_world, &my_rank); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    return false;
  }

  rank_master = 0;
  rank_solver_rank0 = 1;
  if(my_rank == rank_master) iAmMaster=true;
  
  //load data
  if(!data.readinstance(InFile1, InFile2, InFile3, InFile4)) {
    printf("error occured while reading instance\n");
    return false;
  }


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
    if(my_rank==1) {
      iAmSolver=true;
    } else {
      //ranks 0, 2, 3, 4, ...
      iAmEvaluator=true;
      //no need to have master as an evaluator since evaluators do not do much
      if(my_rank==0) iAmEvaluator=false;
    }
  }
#ifdef DEBUG_COMM
  printf("[Phase 1] Rank %d ismaster %d issolver %d isevaluator %d\n",
  	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
#endif
}

void MyCode1::phase2_ranks_allocation()   
{
  assert(comm_world != MPI_COMM_NULL);
  int ret, comm_size;
  ret = MPI_Comm_size(comm_world, &comm_size); assert(ret==MPI_SUCCESS);

  //no solver work needed in this phase
  //all solver ranks become evaluators
  iAmSolver=false; 
  if(my_rank == rank_master) {assert(iAmMaster); iAmMaster = true;}
  if(comm_size==1) {
    iAmEvaluator=true;
  } else {

    //ranks 0, 1, 2, 3, 4
    iAmEvaluator=true; //rank 0 is also an evaluator as long as comm_size<4
    if(my_rank==rank_master && comm_size>=4) iAmEvaluator=false;
    
  }

  //on master
  if(iAmMaster) {
    type_of_rank.clear();

    //rank 0
    int type = 1; //master
    //rank 0 is also an evaluator as long as comm_size<4
    if(comm_size<4) type = 5;
    type_of_rank.push_back(type);
    
    for(int r=1; r<comm_size; r++) 
      type_of_rank.push_back(4);
  }
#ifdef DEBUG_COMM  
  printf("[Phase 2] Rank %d ismaster %d issolver %d isevaluator %d\n",
  	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
#endif
}

void MyCode1::phase3_ranks_allocation()
{
  //to do: grow the pool of solvers with the contingencies considered in
  //phase 3

  //for now use same strategy as in phase 1
  phase1_ranks_allocation();
}


vector<int> MyCode1::phase1_SCACOPF_contingencies()
{
  bool testing = true;
  if(true) {
 
    vector<int> cont_list = {};//1512, 696};//1512,650//10,58,53,1};
 
    return  cont_list;
  } else {
    return vector<int>();
  }
}
bool MyCode1::do_phase1()
{
  K_SCACOPF_phase1 = phase1_SCACOPF_contingencies();
  assert(NULL == scacopf_prob);

  phase1_ranks_allocation();


  //
  // solver scacopf problem on solver rank(s)
  //
  scacopf_prob = new SCACOPFProblem(data);

  scacopf_prob->set_AGC_as_nonanticip(true);

  scacopf_prob->assembly(K_SCACOPF_phase1);

  scacopf_prob->use_nlp_solver("ipopt"); 
  scacopf_prob->set_solver_option("linear_solver", "ma57"); 
  scacopf_prob->set_solver_option("mu_init", 1.);
  scacopf_prob->set_solver_option("print_frequency_iter", 1);
  scacopf_prob->set_solver_option("mu_target", 1e-8);

  scacopf_prob->set_solver_option("acceptable_tol", 1e-3);
  scacopf_prob->set_solver_option("acceptable_constr_viol_tol", 1e-5);
  scacopf_prob->set_solver_option("acceptable_iter", 7);

  if(iAmSolver) {    assert(my_rank==rank_solver_rank0);
    //if(true) {
    scacopf_prob->set_solver_option("print_level", 5);

  } else {
    //master and evaluators do not solve, but we call optimize to force an
    //allocation of the internals, such as the dual variables
    scacopf_prob->set_solver_option("print_level", 5);
    scacopf_prob->set_solver_option("max_iter", 1);
  }
  
  bool bret = scacopf_prob->optimize("ipopt");

  //scacopf_prob->print_p_g_with_coupling_info(*scacopf_prob->data_K[0]);

  //
  //communication -> solver rank0 bcasts basecase solutions
  //
  if(true) {

  scacopf_prob->primal_variables()->
    MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);

  //usleep(1e6*(1+my_rank));
  //char msg[1024];
  //sprintf(msg, "primal vars on rank %d", my_rank);
  //scacopf_prob->primal_variables()->print(msg);

  scacopf_prob->duals_bounds_lower()->
    MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
  scacopf_prob->duals_bounds_upper()->
    MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
  scacopf_prob->duals_constraints()->
    MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
  }
  //force a have_start set
  if(!iAmSolver) {
    scacopf_prob->set_have_start();
    //delete scacopf_prob; scacopf_problem=NULL;
  }

  return true;
}

std::vector<int> MyCode1::phase2_contingencies()
{
  //assert(false);
  //return data.K_Contingency;
  
  //or, for testing purposes
  //return {0,1,2,3,4,5};
  //return {818, 1523, 275};
  return {650,1391,1512, 1514, 1515, 1111, 1112, 696, 1525, 1526, 1652, 1653, 378, 1539, 275};
  //return {0,10,20,30,40,50,60,70,80,90};
  //return {204,1,2,3,4,5,6,7,8,9};
  //return {0,1,2,3};
  //return {17, 426, 960, 961}; //network 7
}

void MyCode1::phase2_initial_contingency_distribution()
{
  hardclear(K_on_rank); hardclear(K_on_my_rank);
  
  if(!iAmMaster) {
    //--non-master phase 2 initialization
    
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
      
      int K_idx_phase2 = nEvaluators*perRank;
      if(nEvaluators<remainder)
	K_idx_phase2 += nEvaluators;
      else
	K_idx_phase2 += remainder;

      if(nEvaluators<S) {
	//printf("r=%d K_idx=%d  K_value=%d\n", r, K_idx, K_phase2[K_idx]);
	assert(K_idx_phase2 < K_phase2.size());
	//!K_on_rank[r].push_back(K_phase2[K_idx]);
	K_on_rank[r].push_back(K_idx_phase2);
      }
    }
  }
  assert(nEvaluators+1==R);
}

int MyCode1::get_next_contingency(int Kidx_last, int rank)
{
  //basic strategy to pick the next contingency
  // try Kidx_last+1, Kidx_last+2, ..., numK, 1, 2, ..., Kidx_last-1

  assert(iAmMaster);
  assert(Kidx_last>=0);
  
  bool done=false;
  int Kidx_next = Kidx_last;
  while(!done) {
    Kidx_next += 1;
    if(Kidx_next>=K_phase2.size())
      Kidx_next=0;

    bool found = false;
    for(auto& Kr : K_on_rank) {
      for(auto Kidx : Kr) {
	if(Kidx_next == Kidx) {
	  found=true;
	  break;
	}
      }
      if(found) break;
    }
			    
    if(!found) {
#ifdef DEBUG_COMM
      printf("Master: found next  contingency for K_idx=%d to have "
	     "idx=%d (global conting index=%d)\n",
	     Kidx_last, Kidx_next, K_phase2[Kidx_next]);
#endif      
      return Kidx_next;
    }

    if(Kidx_next == Kidx_last) {
      //we looped over all elements of K_phase2 and couldn't find a contingency
      //that were not already assigned to a rank
      done=true;
    }
  }
#ifdef DEBUG_COMM  
  printf("Master: did NOT find a next contingency for K_idx=%d, will return K_idx=-1\n",
	 Kidx_last);
#endif
  return -1;
}

bool MyCode1::do_phase2_master_part()
{
  int mpi_test_flag, ierr; MPI_Status mpi_status;
  int num_ranks = K_on_rank.size();

  bool finished = true;
  
  for(int r=0; r<num_ranks; r++) {
    if(K_on_rank[r].size()==0) {
      assert(r==rank_master);
      continue;
    }

    //done with communication for this rank since it was marked as not having any
    //contingencies left
    if(K_on_rank[r].back()==-2) {
#ifdef DEBUG_COMM
      //printf("Master : no more comm for rank=%d. it was marked with -2\n", r);
#endif
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
#ifdef DEBUG_COMM
	printf("Master: recv penalty=%g from rank=%d completed\n", penalty, r);
#endif
	
	// if completed and penalty large, irecv the solution
	// to do
	
	//remove the request irecv for this rank
	delete req_pen;
	req_recv_penalty_for_rank[r].pop_back();
	assert(req_recv_penalty_for_rank[r].size() == 0);


	// the next K_idx/index in K_phase2 to be sent
	assert(K_on_rank[r].back()>=0); 
	int K_idx_next = get_next_contingency(K_on_rank[r].back(), r);
	assert(K_idx_next>=-1);
	K_on_rank[r].push_back(K_idx_next);
	
	send_new_K_idx = true;
      }
    } else {
      //post send <- we're at the first call (a K_idx has been put in K_on_rank[r])
      send_new_K_idx = true;
    }
    
    //check if the send for the last idx is complete 
    if( req_send_K_idx_for_rank[r].size()>0 ) {
      assert(req_send_K_idx_for_rank[r].size()==1);
      ReqKidx* req_K_idx = req_send_K_idx_for_rank[r].back();
      ierr = MPI_Test(&req_K_idx->request, &mpi_test_flag, &mpi_status);
      assert(ierr == MPI_SUCCESS);
      if(mpi_test_flag != 0) {
	//completed
#ifdef DEBUG_COMM	
	printf("Master: send K_idx=%d to rank=%d completed\n", req_K_idx->K_idx, r);
#endif
	//was this one the last one (K_idx==-1) ?
	if(req_K_idx->K_idx==-1) {
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
      assert(req_recv_penalty_for_rank[r].size() == 0);
      assert(req_send_K_idx_for_rank[r].size() == 0);
      
      int K_idx_next = K_on_rank[r].back();

#ifdef DEBUG_COMM   
      char msg[100];
      sprintf(msg, "K_on_rank[%d]", r);
      printvec(K_on_rank[r], msg);
#endif
      
      //
      {
	//isend K_idx_next
	ReqKidx* req_K_idx = new ReqKidx( K_idx_next );
	int tag = Tag0 + K_on_rank[r].size();
	ierr = MPI_Isend(req_K_idx->buffer, 1, MPI_INT, r,
			 tag, comm_world, &req_K_idx->request);
#ifdef DEBUG_COMM
	printf("Master posted send for K_idx=%d to rank=%d tag=%d\n",
	       K_idx_next, r, tag);
#endif
	assert(MPI_SUCCESS == ierr);
	req_send_K_idx_for_rank[r].push_back(req_K_idx);
	
	if( K_idx_next>=0 ) {

	  //post the irecv for penalty
	  ReqPenalty* req_pen = new ReqPenalty( K_idx_next );
	  tag = Tag0 + MSG_TAG_SZ + K_on_rank[r].size();
	  ierr = MPI_Irecv(req_pen->buffer, 1, MPI_DOUBLE, r,
			   tag, comm_world, &req_pen->request);
	  assert(MPI_SUCCESS == ierr);
	  req_recv_penalty_for_rank[r].push_back(req_pen);
#ifdef DEBUG_COMM
	  printf("Master posted recv for penalty for K_idx=%d rank=%d tag=%d\n",
		 K_idx_next, r, tag);
#endif
	}
      }
    } // end of the new send
    
    if(req_recv_penalty_for_rank[r].size()==0 &&
       req_send_K_idx_for_rank[r].size() == 0) {
      //done on rank r, will return finished==true or false depending on
      //whether other ranks are done
    } else {
      finished=false;
    }
  } // end of 'for' loop over ranks
  return finished;
}

bool MyCode1::do_phase2_evaluator_part()
{
  int mpi_test_flag, ierr; MPI_Status mpi_status;
  

  //test for recv of K_idx
  if(req_recv_K_idx!=NULL) {
    //test for completion
    ierr = MPI_Test(&req_recv_K_idx->request, &mpi_test_flag, &mpi_status);
    assert(MPI_SUCCESS == ierr);

    if(mpi_test_flag != 0) {
      //completed
      int K_idx = req_recv_K_idx->buffer[0];
      if(K_idx<0) {
	//no more contingencies coming from master
#ifdef DEBUG_COMM	
	printf("Evaluator Rank %d recv K_idx=-1 finished evaluations\n", my_rank);
#endif
	if(req_send_penalty==NULL)
	  return true;
	
      } else {
#ifdef DEBUG_COMM
	printf("Evaluator Rank %d recv K_idx=%d completed\n", my_rank, K_idx);
#endif
	K_on_my_rank.push_back(K_idx);

	//
	// solve recourse problem
	//

	// this will reuse the solution of the basecase from scacopf_prob
	// or from the previous recourse problem to warm start contingency
	// K_phase2[K_idx]
	int status;
	double penalty = solve_contingency(K_phase2[K_idx], status);
	
	//double penalty = 100*my_rank+K_on_my_rank.size();

	//send penalty
	assert(req_send_penalty == NULL);
	req_send_penalty = new ReqPenalty(K_idx);
	req_send_penalty->buffer[0] = penalty;

	
	int tag = Tag0 + MSG_TAG_SZ + K_on_my_rank.size();
	ierr = MPI_Isend(req_send_penalty->buffer, 1, MPI_DOUBLE,
			 rank_master, tag, comm_world, &req_send_penalty->request);
	assert(MPI_SUCCESS == ierr);

#ifdef DEBUG_COMM	
	printf("Evaluator Rank %d posted penalty send value=%g for Kidx=%d "
	       "to master rank %d  tag=%d\n", my_rank, penalty, K_idx, rank_master, tag);
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
      printf("Evaluator Rank %d send penalty for K_idx=%d completed with penalty=%g\n",
	     my_rank, req_send_penalty->K_idx, req_send_penalty->buffer[0]);
#endif
      delete req_send_penalty; req_send_penalty=NULL;
    }
  }

  //post the recv for K_idx
  if(req_recv_K_idx==NULL) {
    req_recv_K_idx = new ReqKidx();
    
    int tag = Tag0 + K_on_my_rank.size()+1;
    ierr = MPI_Irecv(&req_recv_K_idx->buffer, 1, MPI_INT, rank_master, tag,
		     comm_world, &req_recv_K_idx->request);
#ifdef DEBUG_COMM
    printf("Evaluator Rank %d posted recv for K_idx tag=%d\n", my_rank, tag);
#endif
    assert(MPI_SUCCESS == ierr);
  }
  
  return false;
}

void MyCode1::phase2_initialization()
{
  Tag0 = 10000; MSG_TAG_SZ=data.K_Contingency.size();
  
  if(iAmMaster) {
    int num_ranks = K_on_rank.size();
    for(int r=0; r<num_ranks; r++) {
      req_send_K_idx_for_rank.push_back(vector<ReqKidx*>());
      req_recv_penalty_for_rank.push_back(vector<ReqPenalty*>());
    }
      
  }

  if(iAmEvaluator) {

  }
}



bool MyCode1::do_phase2()
{
  phase2_ranks_allocation();

  //contingencies to be considered in phase 2
  K_phase2 = phase2_contingencies();//set_diff(phase2_contingencies(), K_SCACOPF_phase1);

  phase2_initial_contingency_distribution();

  phase2_initialization();
  
  bool finished=false; 
  bool master_part_done=!iAmMaster;
  bool evaluator_part_done=false;
  while(!finished) {
    if(iAmMaster) {
      finished = do_phase2_master_part();

      if(finished) master_part_done = true;
    }
    
    if(iAmEvaluator && !evaluator_part_done) {
      finished = do_phase2_evaluator_part();

      if(finished) evaluator_part_done=true;
      finished = finished && master_part_done; 
    }

    if(iAmMaster && !iAmEvaluator) {
      //usleep(200); //microseconds
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      //std::this_thread::sleep_for(std::chrono::milliseconds(100));      
    }
  }
  
  return true;
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
  printf("Phase 1 completed on rank %d\n", my_rank);

  //
  // phase 2
  //
  if(!do_phase2()) {
    printf("Error occured in phase 2\n");
    return -1;
  }
  printf("Phase 2 completed on rank %d\n", my_rank);

 
  //if(!do_phase3()) {
  //  printf("Error occured in phase 3\n");
  //  return -1;
  //}

  //
  //cleanup
  //
  delete scacopf_prob;

  if(my_rank==rank_master)
    printf("--finished in %g seconds\n", ttot.stop());

  MPI_Finalize();
  return 0;
}

//K_idx is the index in data.K_Contingency
//status is OK=0 or failure<0 or OK-ish>0
//return penalty/objective for the contingency problem
double MyCode1::solve_contingency(int K_idx, int& status)
{
  assert(iAmEvaluator);
  assert(scacopf_prob != NULL);
  
  status = 0; //be positive
  auto p_g0 = scacopf_prob->variable("p_g", data); 
  auto v_n0 = scacopf_prob->variable("v_n", data);

  //usleep(1e6*my_rank);
  //p_g0->print();
  //v_n0->print();

  goTimer t; t.start();
  
  ContingencyProblem prob(data, K_idx, my_rank);
  
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
  prob.set_solver_option("print_frequency_iter", 1);
  prob.set_solver_option("linear_solver", "ma57"); 
  prob.set_solver_option("print_level", 2);
  prob.set_solver_option("mu_init", 1e-4);
  prob.set_solver_option("mu_target", 1e-8);

  //return if it takes too long in phase2
  prob.set_solver_option("max_iter", 120);
  prob.set_solver_option("acceptable_tol", 1e-3);
  prob.set_solver_option("acceptable_constr_viol_tol", 1e-5);
  prob.set_solver_option("acceptable_iter", 5);


  //scacopf_prob->duals_bounds_lower()->print_summary("duals bounds lower");
  //scacopf_prob->duals_bounds_upper()->print_summary("duals bounds upper");
  //scacopf_prob->duals_constraints()->print_summary("duals constraints");

  double penalty;
  if(!prob.eval_obj(p_g0, v_n0, penalty)) {
    printf("Evaluator Rank %d failed in the evaluation of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return 1e+20;
  }

  //prob.print_p_g_with_coupling_info(*prob.data_K[0], p_g0);

  printf("Evaluator Rank %3d K_idx %5d finished with penalty %12.3f "
	 "in %5.3f sec and %3d iterations\n",
	 my_rank, K_idx, penalty, t.stop(), prob.number_of_iterations());
  
  return penalty;
}

void MyCode1::display_instance_info()
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSec);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}

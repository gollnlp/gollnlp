#include "go_code2.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"

#include "SCACOPFIO.hpp"
#include "goUtils.hpp"

using namespace std;
using namespace gollnlp;

#include "unistd.h"
#include <chrono>
#include <thread>

//#define DEBUG_COMM 1
//#define MAX_NUM_Kidxs_SCACOPF 512

MyCode2::MyCode2(const std::string& InFile1_, const std::string& InFile2_,
		 const std::string& InFile3_, const std::string& InFile4_,
		 double TimeLimitInSeconds, 
		 int ScoringMethod_, 
		 const std::string& NetworkModelName,
		 MPI_Comm comm_world_)
  : InFile1(InFile1_), InFile2(InFile2_), InFile3(InFile3_), InFile4(InFile4_),
    TimeLimitInSec(TimeLimitInSeconds), ScoringMethod(ScoringMethod_),
    NetworkModel(NetworkModelName),
    rank_master(-1), my_rank(-1), comm_size(-1), iAmMaster(false),
    comm_world(comm_world_)
{
  glob_timer.start();

  v_n0 = theta_n0 = b_s0 = p_g0 = q_g0 = NULL;
  last_Kidx_written=-1;
  size_sol_block=-1;
}

MyCode2::~MyCode2()
{
  delete v_n0;
  delete theta_n0;
  delete b_s0;
  delete p_g0;
  delete q_g0;
}

int MyCode2::initialize(int argc, char *argv[])
{
  int ret = MPI_Comm_rank(comm_world, &my_rank); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    return false;
  }

  int ierr = MPI_Comm_size(comm_world, &comm_size); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    return false;
  }

  if(comm_size<=1) {
    printf("[error] MyExe2 needs at least 2 ranks to run\n"); 
    exit(-1);
  }

  rank_master = 0;
  if(my_rank == rank_master) iAmMaster=true;

  if(!iAmMaster)  
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

  data.my_rank = my_rank;

  num_req4Kidx = 1;
  num_req4Kidx_for_ranks = vector<int>(comm_size, 1);
  comm_contidxs_done = vector<bool>(comm_size, false);

  //load data
  if(!data.readinstance(InFile1, InFile2, InFile3, InFile4)) {
    printf("error occured while reading instance\n");
    return false;
  }
  
  SCACOPFIO::read_solution1(&v_n0, &theta_n0, &b_s0, &p_g0, &q_g0, data, "solution1.txt");
  size_sol_block = v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n + 1;


  if(iAmMaster) {
    for(int it=0; it<comm_size; it++)
      K_on_slave_ranks.push_back(vector<int>());
    
    for(int it=0; it<comm_size; it++)
      req_recv_Ksln.push_back(std::vector<ReqKSln*>());

    for(int it=0; it<comm_size; it++)
      req_recv_req4Kidx.push_back(std::vector<ReqKidx*>());
    
    for(int it=0; it<comm_size; it++) 
      req_send_Kidx.push_back(std::vector<ReqKidx*>());
  }

  //K_Contingency = data.K_Contingency;
  //!
  K_Contingency = {106, 344 };
  //K_Contingency = {106, 101,  102,  110,  249,  344,  394,  816,  817, 55, 497, 0, 1, 2, 3, 4, 5, 6,7,8,9,10, 15,16,17,18,19};
//{1,2, 101, 106, 497, 816, 817};


  K_left = vector<int>(K_Contingency.size());
  iota(K_left.begin(), K_left.end(), 0);

  //!!!
  //for(double& p : data.G_Pub) p*=1.05;

  //for(double& p : data.G_Qub) p*=2.05;
  //for(double& p : data.G_Qlb) p*=0.5;


  //for(double& p : data.N_EVub) p*=1.01;
  //for(double& p : data.N_EVlb) p*=0.98;

  return true;
}

int MyCode2::go()
{
  goTimer ttot; ttot.start();

  if(iAmMaster)
    display_instance_info();

  //assign one contingency per rank
  initial_K_distribution();

  //master posts recv for the initial Ks
  if(iAmMaster) {
    for(int r=0; r<comm_size; r++) {
      for(int Kidx: K_on_slave_ranks[r]) {
	if(Kidx<0) continue;
	int tag2 = 222;     
	ReqKSln* req_recv_sln = new ReqKSln(Kidx, vector<double>(size_sol_block+1,-12.17));
	int ierr = MPI_Irecv(req_recv_sln->buffer.data(), req_recv_sln->get_msg_size(), 
			     MPI_DOUBLE, r, tag2, comm_world, &req_recv_sln->request);
	assert(ierr == MPI_SUCCESS);
	req_recv_Ksln[r].push_back(req_recv_sln);
#ifdef DEBUG_COMM
	printf("[rank 0] contsol recv created for conting %d on rank %d\n", Kidx, r);
#endif
      }
    }
  }

  //this is only on the master rank
  // 1==size() means solution has not been received for contingency k;
  // 0==size() means solution was written for contingency k (and vvslns[k] was "deallocated")
  vector<vector<double> > vvslns(data.K_Contingency.size(), vector<double>(1));

  bool ask_for_conting=true;
  //main loop
  while(true) {

    if(!K_local.empty()) {

      int k = K_local.front(); 
      K_local.pop_front();

      vector<double> sol;

      bool bret = solve_contingency(K_Contingency[k], sol);
      assert(sol.size() == size_sol_block+1);

      //send the solution
      int tag2 = 222;      
      ReqKSln* req_send_sol = new ReqKSln(k, sol);
      int ierr = MPI_Isend(req_send_sol->buffer.data(), req_send_sol->get_msg_size(), 
			   MPI_DOUBLE, rank_master, tag2, comm_world, &req_send_sol->request);

#ifdef DEBUG_COMM
      printf("[rank %d] posted send solution for K_idx=%d to master rank %d globtime %g\n", 
	     my_rank, k, rank_master, glob_timer.measureElapsedTime());
#endif
	assert(MPI_SUCCESS == ierr);
	req_send_Ksln.push_back(req_send_sol);
    }

    //master checks receive of contingency solution
    if(my_rank == rank_master) {

      int mpi_test_flag, ierr; MPI_Status mpi_status;
      for(int r=0; r<comm_size; r++) {
	for(int i=0; i<K_on_slave_ranks[r].size(); i++) {
	  int k = K_on_slave_ranks[r][i];
	  if(k<0) continue;
	  
	  int ierr = MPI_Test(&(req_recv_Ksln[r][i]->request), &mpi_test_flag, &mpi_status);
	  if(mpi_test_flag != 0) {
	    //completed
	    assert(req_recv_Ksln[r][i]->buffer.back() == k);
#ifdef DEBUG_COMM
	    printf("[rank 0] contsol recv from rank=%d conting=%d done\n", r, k); 
#endif	    
	    vvslns[k] = req_recv_Ksln[r][i]->buffer;
	    vvslns[k].pop_back(); //remove the last entry, which is the Kidx (used for checking)
	    
	    delete req_recv_Ksln[r][i];
	    req_recv_Ksln[r][i] = NULL;
	    K_on_slave_ranks[r][i] = -1;

	    attempt_write_solution2(vvslns);
	  }
	}
      }
    } //end of master checks receive of contingency solution

    // workers part -> request additional contingencies 
    if(my_rank != rank_master) {
      if(K_local.size()<=2 && ask_for_conting) {
	//there may be data in the pipe - only if the send and recv completed
	if(req_recv_Kidx.empty() && req_send_req4Kidx.empty()) {
	  ReqKidx* req_send_4idx = new ReqKidx(num_req4Kidx);
	  
	  int tag3 = 333;                       
	  int ierr = MPI_Isend(req_send_4idx->buffer.data(), req_send_4idx->buffer.size(),
			       MPI_INT, rank_master, tag3, comm_world, &req_send_4idx->request);
	  
#ifdef DEBUG_COMM	  
	  printf("[rank %d] created send request %d for contingencies globtime %g\n", 
		 my_rank, num_req4Kidx, glob_timer.measureElapsedTime());
#endif
	  req_send_req4Kidx.push_back(req_send_4idx);
	}
      }
    }


    // workers part -> test completion of  requests for contidxs 
    if(my_rank != rank_master) {
      if(ask_for_conting && !req_send_req4Kidx.empty()) {

	int mpi_test_flag, ierr; MPI_Status mpi_status;
	ReqKidx* req_send_4idx = req_send_req4Kidx.back();


	ierr = MPI_Test(&(req_send_4idx->request), &mpi_test_flag, &mpi_status);
	if(mpi_test_flag != 0) {
	  //completed
#ifdef DEBUG_COMM	  
	  printf("[rank %d] completed send request %d for contingencies globtime %g\n", 
		 my_rank, num_req4Kidx, glob_timer.measureElapsedTime());
#endif      
	  // post the recv for the actual indexes
	  int tag4 = 444;
	  ReqKidx* req_recv_idx = new ReqKidx(num_req4Kidx);
	  ierr = MPI_Irecv(req_recv_idx->buffer.data(), req_recv_idx->buffer.size(), 
			   MPI_INT, rank_master, tag4, comm_world, &req_recv_idx->request);
	  assert(MPI_SUCCESS == ierr);
	  req_recv_Kidx.push_back(req_recv_idx);
#ifdef DEBUG_COMM	  
	  printf("[rank %d] created irecv request %d for indexes globtime %g\n", 
		 my_rank, num_req4Kidx, glob_timer.measureElapsedTime());
#endif 


	  assert(req_recv_Kidx.size() == req_send_req4Kidx.size());
	  assert(req_recv_Kidx.size() == 1);

	  delete req_send_4idx;
	  req_send_req4Kidx.pop_back();
	}
      } // end of -- if ask_for_conting && !isempty(req_contidxs_send)
    } // end of workers part -> check on requests for contidxs 

    //workers part - process receive of Kidxs if completed
    if(!req_recv_Kidx.empty()) {

      int mpi_test_flag, ierr; MPI_Status mpi_status;
      ReqKidx* req_recv_idx = req_recv_Kidx.back();


      ierr = MPI_Test(&(req_recv_idx->request), &mpi_test_flag, &mpi_status);
      if(mpi_test_flag != 0) {
	//completed
#ifdef DEBUG_COMM
	printf("[rank %d] irecv request %d for indexes completed globtime %g\n", 
	  my_rank, num_req4Kidx, glob_timer.measureElapsedTime());
#endif
	for(auto k: req_recv_idx->buffer) {
	  if(k>=0) {
	    K_local.push_back(k);
	  } else if(k==-1) {
	    ask_for_conting=false;
	  } else {
	    assert(false && "check this");
	  }
	  delete req_recv_idx;
	  req_recv_Kidx.pop_back();
	  
	  num_req4Kidx += 1;
	}
      }
    } // end of - workers part - process receive of Kidxs if completed

    //master part - 
    // i. check for requests for additional contingencies; 
    // ii. (I)send these contingencies
    // iii. create contsol recv requests for the contingencies that were just sent 
    if(my_rank==rank_master) {
      bool all_comm_contidxs_done=true;
      
      for(int r=comm_size-1; r>=1; r--) {
	if(comm_contidxs_done[r]) continue;
	all_comm_contidxs_done=false;
      

	//initiate receive for Kidx if need to
	if(req_recv_req4Kidx[r].empty() && req_send_Kidx[r].empty()) {

	  // initiate Irecv 
	  ReqKidx* req_recv_idx = new ReqKidx(num_req4Kidx_for_ranks[r]);
	  int tag3 = 333;
	  int ierr = MPI_Irecv(req_recv_idx->buffer.data(), req_recv_idx->buffer.size(), 
			       MPI_INT, r, tag3, comm_world, &req_recv_idx->request);
	  req_recv_req4Kidx[r].push_back(req_recv_idx);
#ifdef DEBUG_COMM
	  printf("[rank 0] created recv request %d for contingencies for rank %d\n", 
		 num_req4Kidx_for_ranks[r], r);
#endif
	}
	
	//test recv of Kidx request -> send Kidx and also initiate recvs for the solution of these Kidxs
	if(!req_recv_req4Kidx[r].empty()) {

	  int mpi_test_flag, ierr; MPI_Status mpi_status;
	  ReqKidx* req_recv_idx = req_recv_req4Kidx[r].back();
	  ierr = MPI_Test(&(req_recv_idx->request), &mpi_test_flag, &mpi_status);
	  if(mpi_test_flag != 0) {
	    //completed
#ifdef DEBUG_COMM
	    printf("[rank 0] completed recv request %d for contingencies for rank %d\n", num_req4Kidx_for_ranks[r], r);
#endif
	    //pick new contingency 
	    int perRank = K_Contingency.size()/(comm_size-1);
	    int Kstart = perRank*(r-1);
	    auto Kidxs = findall(K_left, [Kstart](int val) {return val>Kstart;});
	    if(Kidxs.empty()) 
	      Kidxs = findall(K_left, [](int val) {return val>=0;});
	    
	    int Kidx = -1;
	    if(!Kidxs.empty()) {
	      Kidx = K_left[Kidxs[0]];
	      erase_idx_from(K_left, Kidxs[0]);
	    } 

	    //create send
	    ReqKidx* req_send_idx = new ReqKidx(Kidx);
	    int tag4 = 444;
	    ierr = MPI_Isend(req_send_idx->buffer.data(), req_send_idx->buffer.size(),
			     MPI_INT, r, tag4, comm_world, &req_send_idx->request);
	    assert(MPI_SUCCESS==ierr);
#ifdef DEBUG_COMM
	    printf("[rank 0] created send indexes request %d to rank %d (444)\n", num_req4Kidx_for_ranks[r], r);
#endif
	    req_send_Kidx[r].push_back(req_send_idx);

	    assert(req_send_Kidx[r].size() == req_recv_req4Kidx[r].size());

	    delete req_recv_idx;
	    req_recv_req4Kidx[r].pop_back();
	    
	  
	    // create contsol recvs for the contingency that was just sent 
	    if(Kidx>=0) {
	      K_on_slave_ranks[r].push_back(Kidx);
	      
	      int tag2 = 222;     
	      ReqKSln* req_recv_sln = new ReqKSln(Kidx, vector<double>(size_sol_block+1,-12.17));
	      ierr = MPI_Irecv(req_recv_sln->buffer.data(), req_recv_sln->get_msg_size(), 
			       MPI_DOUBLE, r, tag2, comm_world, &req_recv_sln->request);
#ifdef DEBUG_COMM
	      printf("[rank 0] contsol recv created for conting %d on rank %d\n", Kidx, r);
#endif
	      req_recv_Ksln[r].push_back(req_recv_sln);
	    } else {
	      // apparently we're out of contingencies
	      assert(0==K_left.size());
	      comm_contidxs_done[r]=true;
	    } // end recv done
	  }
	} // end of  if(!req_recv_req4Kidx[r].empty()) 

	// test completion of indexes send 'req_send_Kidx'
	if(!req_send_Kidx[r].empty()) {
	  ReqKidx* req_send_idx = req_send_Kidx[r].back();
	  int mpi_test_flag, ierr; MPI_Status mpi_status;
	  ierr = MPI_Test(&(req_send_idx->request), &mpi_test_flag, &mpi_status);
	  if(mpi_test_flag != 0) {
	    //completed
#ifdef DEBUG_COMM
	    printf("[rank 0] completed send indexes request %d to rank %d (444)\n", num_req4Kidx_for_ranks[r], r);
#endif
	    num_req4Kidx_for_ranks[r]++;
	    delete req_send_idx;
	    req_send_Kidx[r].pop_back();
	  }
	} // end of - if(!req_send_Kidx[r].empty()) {

      }

      // we're still under myRank==0
      
      //test for termination on master rank
      if(all_comm_contidxs_done) {
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
        fflush(stdout);
	break;
      }
      fflush(stdout);
    }  else {// end of - master part
      //on worker ranks
      
      //termination tests
      if(!ask_for_conting && K_local.empty() && req_recv_Kidx.empty() && req_send_req4Kidx.empty()) {
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	fflush(stdout);
	break;
      }
      fflush(stdout);
    }
  } // end of while(true)

  printf("[rank %d] All contingencies solved in %6.2f sec\n", my_rank, glob_timer.measureElapsedTime());
  fflush(stdout);

 
  //master waits for all solutions to come in
  if(iAmMaster) {
    bool recv_sln_done=false;
    while(!recv_sln_done) {
      recv_sln_done = true;
      for(int r=0; r<comm_size; r++) {
	for(int i=0; i<K_on_slave_ranks[r].size(); i++) {
	  int k = K_on_slave_ranks[r][i];
	  if(k<0) continue;
	  
	  int mpi_test_flag, ierr; MPI_Status mpi_status;
	  ierr = MPI_Test(&(req_recv_Ksln[r][i]->request), &mpi_test_flag, &mpi_status);
	  if(mpi_test_flag != 0) {
	    //completed

	    assert(req_recv_Ksln[r][i]->buffer.back()==k);
	    vvslns[k] = req_recv_Ksln[r][i]->buffer;
	    vvslns[k].pop_back(); //remove the last element, which is Kidx

	    delete req_recv_Ksln[r][i];
	    req_recv_Ksln[r][i]=NULL;
	    K_on_slave_ranks[r][i] = -1;

	    attempt_write_solution2(vvslns);
	  } else {
	    recv_sln_done = false;
	  }
	}
      }
    }
  }

  //write solution2.txt
  if(iAmMaster) {
    bool writing_done = attempt_write_solution2(vvslns);
    if(!writing_done) 
      printf("[warning] couldn't write all contingency solutions; some are missing from vvslns\n");
    else 
      printf("writing of 'solution2.txt' completed\n");
  }

  // final message
  if(iAmMaster)
    printf("--finished in %g sec  global time %g sec\n", ttot.stop(), glob_timer.measureElapsedTime());

  return 0;
}


void MyCode2::initial_K_distribution()
{

  int num_ranks = comm_size;

  //debuging code
  //vector<vector<int> > Ks_to_ranks(num_ranks, vector<int>());
  //Ks_to_ranks[1].push_back(48);

  //num contingencies; 
  int S = K_Contingency.size(), R = num_ranks-1;
  if(R<=0) R=1;

  int perRank = S/R; int remainder = S-R*perRank;


  if(0==perRank) {
    perRank=1; remainder=0;
  }

  if(iAmMaster) {
    printf("ranks=%d contingencies=%d perRank=%d remainder=%d\n", num_ranks, S, perRank, remainder);

    //each slave rank gets one contingency idx = r*perRank
    for(int r=1; r<num_ranks; r++) {
      int K_idx = (r-1) * perRank;
      if(r < remainder)
	K_idx += r;
      else
	K_idx += remainder;
       
      if(K_idx < S) {
	K_on_slave_ranks[r].push_back(K_idx);
	bool bret = erase_elem_from(K_left, K_idx);
	assert(bret);
      }
    }
    //"initial Ks on each rank: "
    printvecvec(K_on_slave_ranks, "initial Ks on each rank: ");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //printvec(K_left, "contingencies left");
  } else {

    int r = my_rank;
    int K_idx = (r-1) * perRank;
    if(r < remainder)
      K_idx += r;
    else
      K_idx += remainder;
    
    if(K_idx < S) {
      K_local.push_back(K_idx);
      char s[1000]; sprintf(s, "K_local on rank %d", my_rank);
      printlist(K_local, string(s));
    } else {
      printf("K_local on rank %d  is empty\n", my_rank);
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

bool MyCode2::solve_contingency(int K_idx, std::vector<double>& sln)
{

  goTimer t; t.start();
  int status;
  ContingencyProblem prob(data, K_idx, my_rank);
  
  prob.update_AGC_smoothing_param(1e-4);
  prob.update_PVPQ_smoothing_param(1e-4);



  //prob.reg_vn = true;
  //prob.reg_thetan = true;
  //prob.reg_bs = true;
  //prob.reg_pg = true;
  //prob.reg_qg = true;


  if(!prob.default_assembly(v_n0, theta_n0, b_s0, p_g0, q_g0)) {
    printf("Evaluator Rank %d failed in default_assembly for contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -1;
    return false;
  }

  //  if(!prob.set_warm_start_from_base_of(*scacopf_prob)) {
  //status = -2;
  //return 1e+20;
  //}

  prob.use_nlp_solver("ipopt");
  prob.set_solver_option("print_frequency_iter", 25);
  prob.set_solver_option("linear_solver", "ma57"); 
  prob.set_solver_option("print_level", 5);
  prob.set_solver_option("mu_init", 1e-1);
  prob.set_solver_option("mu_target", 5e-9);

  //return if it takes too long in phase2
  prob.set_solver_option("max_iter", 1000);
  prob.set_solver_option("acceptable_tol", 1e-3);
  prob.set_solver_option("acceptable_constr_viol_tol", 1e-6);
  prob.set_solver_option("acceptable_iter", 5);

  prob.set_solver_option("tol", 1e-9);

  prob.set_solver_option("bound_relax_factor", 0.);
  prob.set_solver_option("bound_push", 1e-16);
  prob.set_solver_option("slack_bound_push", 1e-16);
  prob.set_solver_option("mu_linear_decrease_factor", 0.4);
  prob.set_solver_option("mu_superlinear_decrease_power", 1.25);


  double penalty;
  if(!prob.optimize(p_g0, v_n0, penalty)) {
    printf("Evaluator Rank %d failed in the evaluation of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return false;
  }
  if(penalty>100)
    prob.print_objterms_evals();
  int numiter1 =  prob.number_of_iterations();

  /////////////////////////////////////////
if(true)
{

  prob.set_solver_option("print_frequency_iter", 11);
  prob.set_solver_option("mu_init", 1e-1);

  prob.update_AGC_smoothing_param(1e-6);
  prob.update_PVPQ_smoothing_param(1e-6);

  if(!prob.reoptimize(OptProblem::primalDualRestart)) {
    printf("Evaluator Rank %d failed in the evaluation 2 of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return false;
  }

  penalty = prob.objective_value();
  if(penalty>100)
    prob.print_objterms_evals();
  
}

  if(false) {

  prob.set_solver_option("mu_init", 1e-4);

  prob.update_AGC_smoothing_param(1e-8);
  prob.update_PVPQ_smoothing_param(1e-8);

  if(!prob.reoptimize(OptProblem::primalDualRestart)) {
    printf("Evaluator Rank %d failed in the evaluation 2 of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return false;
  }

  penalty = prob.objective_value();
  if(penalty>100)
    prob.print_objterms_evals();

  }

if(false) {
  prob.set_solver_option("mu_init", 1e-5);

  prob.update_AGC_smoothing_param(1e-6);
  prob.update_PVPQ_smoothing_param(1e-6);

  if(!prob.reoptimize(OptProblem::primalDualRestart)) {
    printf("Evaluator Rank %d failed in the evaluation 2 of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return false;
  
  penalty = prob.objective_value();
  if(penalty>100)
    prob.print_objterms_evals();
  }

  prob.set_solver_option("mu_init", 1e-5);

  prob.update_AGC_smoothing_param(1e-7);
  prob.update_PVPQ_smoothing_param(1e-7);

  if(!prob.reoptimize(OptProblem::primalDualRestart)) {
    printf("Evaluator Rank %d failed in the evaluation 2 of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    return false;
  }
  penalty = prob.objective_value();
  if(penalty>100)
    prob.print_objterms_evals();


    }


  prob.get_solution_simplicial_vectorized(sln);
  assert(size_sol_block == sln.size());

  prob.print_p_g_with_coupling_info(*prob.data_K[0], p_g0);
  //prob.print_PVPQ_info(*prob.data_K[0], v_n0);


  sln.push_back((double)K_idx);
  printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	 "in %5.3f sec and %3d/%3d iterations  global time %g \n",
	 my_rank, K_idx, penalty, t.stop(), 
	 numiter1, prob.number_of_iterations(), glob_timer.measureElapsedTime());

  return true;
}

void MyCode2::display_instance_info()
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSec);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}


// 1. sweeps 'vvsols' and writes to 'solution2.txt' the solutions 
// received on master; this is done in order of the Kidx;
// 2. once the solution for contingcency k is written, vvsols[k] is deallocated
//
// return true when all solutions are written; false otherwise
bool MyCode2::attempt_write_solution2(std::vector<std::vector<double> >& vvsols)
{
  while(true) {
    int Kidx = last_Kidx_written+1; assert(Kidx>=0);

    if(Kidx==K_Contingency.size()) return true;
    assert(Kidx<K_Contingency.size());

    if(vvsols[Kidx].size()==1) return false; //solution has not arrived

    assert(size_sol_block>=2);
    assert(vvsols[Kidx].size()==size_sol_block+1);

#ifdef DEBUG
    if(Kidx>=1) assert(vvsols[Kidx-1].size() == 0); //should have been written already
    assert(size_sol_block == v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n + 1);
#endif
    
    const double 
      *v_n     = vvsols[Kidx].data(), 
      *theta_n = vvsols[Kidx].data() + v_n0->n, 
      *b_s     = vvsols[Kidx].data() + v_n0->n + theta_n0->n, 
      *p_g     = vvsols[Kidx].data() + v_n0->n + theta_n0->n + b_s0->n, 
      *q_g     = vvsols[Kidx].data() + v_n0->n + theta_n0->n + b_s0->n + p_g0->n,
      delta    = vvsols[Kidx][v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n];

    SCACOPFIO::write_solution2_block(Kidx, v_n, theta_n, b_s, p_g, q_g, delta,
				     data, "solution2.txt");
    printf("[rank 0] wrote solution2 block for contingency %d [%d] label '%s'\n", 
	   Kidx, K_Contingency[Kidx], data.K_Label[Kidx].c_str());

    hardclear(vvsols[Kidx]);
    last_Kidx_written++;
  }

  return false;
}

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

std::ostream& operator<<(std::ostream& os, const MyCode2::Kinfo_worker& o)
{
  os << "K_idx=" << o.id;
  if(o.safe_mode_solve) os << "(safemode)";
  return os;
};

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

  //v_n0 = theta_n0 = b_s0 = p_g0 = q_g0 = NULL;
  last_Kidx_written=-1;
  size_sol_block=-1;

  num_K_done=0; 
}

MyCode2::~MyCode2()
{
  for(auto& p : dict_basecase_vars) 
    delete p.second;
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

  //will populate 'dict_basecase_vars'
  read_solution1();

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

  vector<int> K_Cont = data.K_Contingency;
  //!
  //K_Cont = {0,1,1936};//{1936, 913, 792};
  //344
  //K_Cont={3222};//, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223, 3223}; 
  
  //K_Cont = {913, 4286}; for(int i=0; i<4900; i++) K_Cont.push_back(3180+i);
  //K_Cont = {11971};//, 776}; //for(int i=0; i<4900; i++) K_Cont.push_back(3180+i);
  //K_Cont = {2223, 136, 10112, 482,0 };
  for(auto& id : K_Cont) 
    K_Contingency.push_back(Kinfo(id));

  K_left = vector<int>(K_Contingency.size());
  iota(K_left.begin(), K_left.end(), 0);

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

      int k = K_local.front().id; 
      int safe_mode = K_local.front().safe_mode_solve;
      K_local.pop_front();

      vector<double> sol;

      bool bret = solve_contingency(K_Contingency[k].id, safe_mode, sol);
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

    //delete ReqKSln* req_send_sol that completed (relevant on workers)
    attempt_cleanup_req_send_Ksln();

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
	    num_K_done++;
#ifdef DEBUG_COMM
	    printf("[rank 0] contsol recv 1 from rank=%d conting=%d done; so far K_done=%d\n", 
		   r, k, num_K_done); 
#endif	    
	    K_Contingency[k].solve_done = true;
	    if(vvslns[k].size()==1) {
	      vvslns[k] = req_recv_Ksln[r][i]->buffer;
	      vvslns[k].pop_back(); //remove the last entry, which is the Kidx (used for checking)
	    } else {
	      //this was a timedout solve; we already have the solution
	      assert(vvslns[k].size()==0 || vvslns[k].size()==req_recv_Ksln[r][i]->buffer.size()-1);
	    }
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
	  ReqKidx* req_send_4idx = new ReqKidx(num_req4Kidx,-1);
	  
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
	  ReqKidx* req_recv_idx = new ReqKidx(num_req4Kidx, -1);
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
	assert(req_recv_idx->buffer.size()>=2);
	if(req_recv_idx->buffer.size()>=2) {
	  num_K_done = req_recv_idx->buffer.back();
	  req_recv_idx->buffer.pop_back();
	}

	for(auto k: req_recv_idx->buffer) {
	  if(k>=0) {
	    K_local.push_back(Kinfo_worker(k));
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
	  ReqKidx* req_recv_idx = new ReqKidx(num_req4Kidx_for_ranks[r],-1);
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
	    //
	    // picking
	    //
	    bool late_Kidx=false;
	    int Kidx = pick_late_contingency(r);
	    if(Kidx>=0) late_Kidx=true;
	    if(Kidx<0) Kidx = pick_new_contingency(r);

	    bool all_contingencies_are_solved = all_contingencies_solved();

	    if(Kidx>=0 || all_contingencies_are_solved) {
	      int Kidx2 = late_Kidx ? 1000000+Kidx : Kidx;
	      //create send
	      ReqKidx* req_send_idx = new ReqKidx(Kidx2, num_K_done);
	      int tag4 = 444;
	      ierr = MPI_Isend(req_send_idx->buffer.data(), req_send_idx->buffer.size(),
			       MPI_INT, r, tag4, comm_world, &req_send_idx->request);
	      assert(MPI_SUCCESS==ierr);
#ifdef DEBUG_COMM
	      printf("[rank 0] created send indexes %d  request %d to rank %d (444)\n", Kidx, num_req4Kidx_for_ranks[r], r);
#endif
	      req_send_Kidx[r].push_back(req_send_idx);
	      
	      assert(req_send_Kidx[r].size() == req_recv_req4Kidx[r].size());
	    
	      delete req_recv_idx;
	      req_recv_req4Kidx[r].pop_back();
	    }
	    
	  
	    // create contsol recvs for the contingency that was just sent 
	    if(Kidx>=0) {
	      K_on_slave_ranks[r].push_back(Kidx);
	      K_Contingency[Kidx].tmSent.push_back(MPI_Wtime());

	      int tag2 = 222;     
	      ReqKSln* req_recv_sln = new ReqKSln(Kidx, vector<double>(size_sol_block+1,-12.17));
	      ierr = MPI_Irecv(req_recv_sln->buffer.data(), req_recv_sln->get_msg_size(), 
			       MPI_DOUBLE, r, tag2, comm_world, &req_recv_sln->request);
#ifdef DEBUG_COMM
	      printf("[rank 0] contsol recv created for conting %d on rank %d\n", Kidx, r);
#endif
	      req_recv_Ksln[r].push_back(req_recv_sln);
	    } else {
	      // apparently we're out of new contingencies
	      assert(0==K_left.size());
	      if(all_contingencies_are_solved)
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
	//if(all_contingencies_are_solved) {
	  fflush(stdout);
	  break;
	  //} else {
	  //std::this_thread::sleep_for(std::chrono::milliseconds(20));
	  //}
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

	    K_Contingency[k].solve_done=true;
	    assert(req_recv_Ksln[r][i]->buffer.back()==k);
	    if(vvslns[k].size() == 1) {
	      vvslns[k] = req_recv_Ksln[r][i]->buffer;
	      vvslns[k].pop_back(); //remove the last element, which is Kidx
	    } else {
	      assert(vvslns[k].size()==0 || vvslns[k].size()==req_recv_Ksln[r][i]->buffer.size()-1);
	    }

	    delete req_recv_Ksln[r][i];
	    req_recv_Ksln[r][i]=NULL;
	    K_on_slave_ranks[r][i] = -1;

	    num_K_done++;

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
      printf("[warning] couldn't write all contingency solutions; some are missing from vvslns; num_K_done=%d\n",
	     num_K_done);
    else 
      printf("writing of 'solution2.txt' completed at glob time %g; num_K_done=%d\n", 
	     glob_timer.measureElapsedTime(), num_K_done);
  } else {
    attempt_cleanup_req_send_Ksln();
  }

  // final message
  if(iAmMaster)
    printf("--finished in %g sec  glob time %g sec\n", ttot.stop(), glob_timer.measureElapsedTime());

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
	K_Contingency[K_idx].tmSent.push_back(MPI_Wtime());
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
      K_local.push_back(Kinfo_worker(K_idx));
      char s[1000]; sprintf(s, "K_local on rank %d", my_rank);
      printlist(K_local, string(s));
    } else {
      printf("K_local on rank %d  is empty\n", my_rank);
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

#include "goSignalHandling.hpp"

bool MyCode2::_guts_of_solve_contingency(ContingencyProblemWithFixing& prob, int K_idx)
{
  double pen_accept=1., pen_accept_inipt=1., pen_accept_solve1=1.,
    pen_accept_emer=1000., pen_accept_safemode=10000.;
  double timeout = 1200; // contingency solves will switch to emergency mode after this limit is reached
  
  double tm_percentage = glob_timer.measureElapsedTime()/(2.0*data.K_Contingency.size());
  double avgtm = 1.5; //average time per contingency in seconds
  if(tm_percentage<0.075) {
    if(data.N_Bus.size()<31000) { //tm_percentage in [0%,7.5%]
      pen_accept = pen_accept_inipt = pen_accept_solve1 = 1.;
      pen_accept_emer = 1000.;
      pen_accept_safemode=10000.;
    } else {
      pen_accept = pen_accept_inipt = pen_accept_solve1 = 1000.;
      pen_accept_emer = 10000.;
      pen_accept_safemode=100000.;
    }
  } else if(tm_percentage<0.20) {
    if(avgtm<2.25) {
      if(data.N_Bus.size()<31000) {
	pen_accept = pen_accept_inipt = pen_accept_solve1 = 1.;
	pen_accept_emer = 1000.;
	pen_accept_safemode=10000.;
      } else {
	pen_accept = pen_accept_inipt = pen_accept_solve1 = 1000.;
	pen_accept_emer = 10000.;
	pen_accept_safemode=100000.;
      }
    } else if(avgtm<3.0) { //avgtm in [2.25, 3]
      pen_accept = 500.; pen_accept_inipt = 2000.; pen_accept_solve1 = 1000.;
      pen_accept_emer = 10000.;
      pen_accept_safemode=50000.;
    } else { //avgtm > 3
      pen_accept = 2000.; pen_accept_inipt = 10000.; pen_accept_solve1 = 5000.;
      pen_accept_emer = 25000.;
      pen_accept_safemode=50000.;
    }
  } else if(tm_percentage<0.70) { //tm_percentage in [20%,70%]
    if(avgtm<2.0) {
      pen_accept = pen_accept_inipt = pen_accept_solve1 = 1.;
      pen_accept_emer = 1000.;
      pen_accept_safemode=10000.;
    } else if(avgtm<2.5) { //avgtm in [2, 2.5]
      pen_accept = 1000.; pen_accept_inipt = 2000.; pen_accept_solve1 = 2000.;
      pen_accept_emer = 10000.;
      pen_accept_safemode=50000.;
    } else if(avgtm<5) { //avgtm in [2.5, 5]
      pen_accept = 2000.; pen_accept_inipt = 10000.; pen_accept_solve1 = 5000.;
      pen_accept_emer = 25000.;
      pen_accept_safemode=100000.;
    } else { //avgtm>5
      pen_accept = 20000.; pen_accept_inipt = 50000.; pen_accept_solve1 = 30000.;
      pen_accept_emer = 100000.;
      pen_accept_safemode=500000.;
    }
  } else if(tm_percentage<0.85) {//tm_percentage in [70%,85%]
    if(avgtm<1.9) {
      pen_accept = pen_accept_inipt = pen_accept_solve1 = 1.;
      pen_accept_emer = 1000.;
      pen_accept_safemode=50000.;
    } else if(avgtm<2.25) { //avgtm in [1.9, 2.25]
      pen_accept = 1000.; pen_accept_inipt = 4000.; pen_accept_solve1 = 2000.;
      pen_accept_emer = 20000.;
      pen_accept_safemode=200000.;
    } else { //avgtm >2.25
      pen_accept = 20000.; pen_accept_inipt = 50000.; pen_accept_solve1 = 30000.;
      pen_accept_emer = 100000.;
      pen_accept_safemode=1000000.;//1M
    } 
  } else if(tm_percentage<0.95) { //tm_percentage in [85%,95%]
    if(avgtm<1.90) {
      pen_accept = 5000.; pen_accept_inipt = 10000.; pen_accept_solve1 = 10000.;
      pen_accept_emer = 50000.;
      pen_accept_safemode=2000000.;//2M
      timeout = 400;
    } else  { //avgtm > 1.9
      pen_accept = 50000.; pen_accept_inipt = 100000.; pen_accept_solve1 = 100000.;
      pen_accept_emer = 1000000.; //2M
      pen_accept_safemode=1e+20;//infinity
      timeout = 200;
    } 
  } else {//tm_percentage > 95%
    timeout = 0. - glob_timer.measureElapsedTime() + 2.0*data.K_Contingency.size();
    timeout *= 0.6;
    pen_accept = pen_accept_inipt = pen_accept_solve1 = pen_accept_emer = pen_accept_safemode=1e+20;
  }

  prob.pen_accept = pen_accept;
  prob.pen_accept_initpt=pen_accept_inipt;
  prob.pen_accept_solve1=pen_accept_solve1;
  prob.pen_accept_emer=pen_accept_emer;
  prob.pen_accept_safemode;

  //if(data.N_Bus.size()>8999)
  {
    ContingencyProblemWithFixing::g_bounds_abuse = 0.00009999;
    prob.monitor.is_active = true;
  }

  if(!prob.default_assembly(v_n0(), theta_n0(), b_s0(), p_g0(), q_g0())) {
    printf("rank=%d failed in default_assembly for contingency K_idx=%d\n",
	   my_rank, K_idx);
    //status = -1;
    return false;
  }

  prob.use_nlp_solver("ipopt");
  return true;
}

extern int g_max_memory_ma57;

bool MyCode2::solve_contingency(int K_idx, bool safe_mode, std::vector<double>& sln)
{
  if(false) {
  g_max_memory_ma57=900;//Mbytes

  ///////////////////////////////////////
  //
  // solver scacopf problem on solver rank(s) xxxbase1
  //
  SCACOPFProblem* scacopf_prob = new SCACOPFProblem(data);

  scacopf_prob->my_rank = my_rank;

  scacopf_prob->update_AGC_smoothing_param(1e-4);
  scacopf_prob->update_PVPQ_smoothing_param(1e-2);
  //scacopf_prob->set_AGC_as_nonanticip(true);
  //scacopf_prob->set_AGC_simplified(true);
  //scacopf_prob->set_PVPQ_as_nonanticip(true);

  //reduce T and L rates to min(RateBase, TL_rate_reduction*RateEmer)
  double TL_rate_reduction = 0.85;
  //if((ScoringMethod==1 || ScoringMethod==3))
  //  TL_rate_reduction = 0.85;

  scacopf_prob->set_basecase_L_rate_reduction(TL_rate_reduction);
  scacopf_prob->set_basecase_T_rate_reduction(TL_rate_reduction);

  //scacopf_prob->set_quadr_penalty_qg0(true);

  scacopf_prob->assembly({});

  
  scacopf_prob->use_nlp_solver("ipopt"); 
  scacopf_prob->set_solver_option("sb","yes");
  scacopf_prob->set_solver_option("linear_solver", "ma57"); 

  scacopf_prob->set_solver_option("print_frequency_iter", 5);
  scacopf_prob->set_solver_option("max_iter", 2000);    
  scacopf_prob->set_solver_option("acceptable_tol", 1e-3);
  scacopf_prob->set_solver_option("acceptable_constr_viol_tol", 1e-6);
  scacopf_prob->set_solver_option("acceptable_iter", 7);

  scacopf_prob->set_solver_option("mu_init", 0.1);
  scacopf_prob->set_solver_option("tol", 1e-8);
  scacopf_prob->set_solver_option("mu_target", 1e-9);
  
  //scacopf_prob->set_solver_option("bound_relax_factor", 0.);
  //scacopf_prob->set_solver_option("bound_push", 1e-16);
  //scacopf_prob->set_solver_option("slack_bound_push", 1e-16);
  scacopf_prob->set_solver_option("mu_linear_decrease_factor", 0.5);
  scacopf_prob->set_solver_option("mu_superlinear_decrease_power", 1.2);


  scacopf_prob->set_solver_option("print_level", 5);

  printf("[ph1] rank %d  starts scacopf solve phase 1 global time %g\n", 
	   my_rank, glob_timer.measureElapsedTime());

  
  bool bret = scacopf_prob->optimize("ipopt");

  auto p_g0_ = scacopf_prob->variable("p_g", data); 
  auto v_n0_ = scacopf_prob->variable("v_n", data);

  ///////////////////////////////////////////////////////////////////////////////
  {
  int status = 0; //be positive


  
  
  ContingencyProblem prob(data, K_idx, my_rank);

  if(data.N_Bus.size()>8999) {
    //prob.monitor.is_active = true;
    //prob.monitor.pen_threshold = pen_threshold;
  }

  prob.update_AGC_smoothing_param(1e-4);
  prob.update_PVPQ_smoothing_param(1e-4);

  //xxxcont

  if(!prob.default_assembly(p_g0_, v_n0_)) {
    printf("Evaluator Rank %d failed in default_assembly for contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -1;
    return false;
  }

  if(!prob.set_warm_start_from_base_of(*scacopf_prob)) {
    status = -2;
    return false;
  }

  prob.use_nlp_solver("ipopt");
  // prob.set_solver_option("sb","yes");
  // prob.set_solver_option("print_frequency_iter", 10);
  // prob.set_solver_option("linear_solver", "ma57"); 
  // prob.set_solver_option("print_level", 5);
  // prob.set_solver_option("mu_init", 1e-4);
  // prob.set_solver_option("mu_target", 5e-9);

  // //return if it takes too long in phase2
  // prob.set_solver_option("max_iter", 1700);
  // prob.set_solver_option("acceptable_tol", 1e-3);
  // prob.set_solver_option("acceptable_constr_viol_tol", 1e-6);
  // prob.set_solver_option("acceptable_iter", 5);

  // //if(data.N_Bus.size()<10000) 
  // {
  //   prob.set_solver_option("bound_relax_factor", 0.);
  //   prob.set_solver_option("bound_push", 1e-16);
  //   prob.set_solver_option("slack_bound_push", 1e-16);
  // }
  // prob.set_solver_option("mu_linear_decrease_factor", 0.4);
  // prob.set_solver_option("mu_superlinear_decrease_power", 1.25);

  double penalty; 
  if(!prob.eval_obj(p_g0_, v_n0_, penalty)) {
    printf("Evaluator Rank %d failed in the eval_obj of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    penalty=1e+6;
  }
  int num_iter = prob.number_of_iterations();

  printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	 "in %5.3f sec and %3d iterations  sol_from_scacopf_pass %d  global time %g\n",
	 my_rank, K_idx, penalty, -117., 
	 num_iter, -117, glob_timer.measureElapsedTime());
  }

  ////////////////////////////////////////////////////////////////////////////////////
  }
  goTimer t; t.start();

  int status; double penalty;
  ContingencyProblemWithFixing* prob = 
    new ContingencyProblemWithFixing(data, K_idx, 
				     my_rank, comm_size, 
				     dict_basecase_vars, 
				     num_K_done, 
				     glob_timer.measureElapsedTime(), 
				     safe_mode);
  
  _guts_of_solve_contingency(*prob, K_idx);

  if(!prob->optimize(p_g0(), v_n0(), penalty, sln)) {
    printf("rank=%d failed in the evaluation of contingency K_idx=%d\n",
	   my_rank, K_idx);
    status = -3;
    delete prob;
    return false;
  }

  //prob.get_solution_simplicial_vectorized(sln);
#ifdef DEBUG
  if(size_sol_block != sln.size()) {
    printf("Evaluator Rank %d size mismatch in the evaluation of contingency K_idx=%d\n",
	   my_rank, K_idx);
  }
  assert(size_sol_block == sln.size());
#endif
  sln.push_back((double)K_idx);

  //prob.print_p_g_with_coupling_info(*prob.data_K[0], p_g0);
  //prob.print_PVPQ_info(*prob.data_K[0], v_n0);
  //prob->print_reactive_power_balance_info(*prob->data_K[0]);

  printf("rank=%d K_idx=%d finished with penalty %12.3f "
  	 "in %5.3f sec time %g \n",
  	 my_rank, K_idx, penalty, t.stop(), glob_timer.measureElapsedTime());
  delete prob;
  //delete scacopf_prob;
  return true;
}

void MyCode2::read_solution1()
{
  gollnlp::SCACOPFIO::read_variables_blocks(data, dict_basecase_vars);
  //for(auto& p: dict_basecase_vars) cout << "   - [" << p.first << "] size->" << p.second->n << endl;

  size_sol_block = v_n0()->n + theta_n0()->n + b_s0()->n + p_g0()->n + q_g0()->n + 1;

#ifdef DEBUG  
  OptVariablesBlock *v_n00, *theta_n00, *b_s00, *p_g00, *q_g00;
  SCACOPFIO::read_solution1(&v_n00, &theta_n00, &b_s00, &p_g00, &q_g00, data, "solution1.txt");
  

  assert(v_n00->n == v_n0()->n);
#ifdef DEBUG
  if(diff_two_norm(v_n00->n, v_n00->x, v_n0()->x)>1e-12)
     printf("[warning] difference between read_variables read_solution1 in v\n");
#endif
  delete v_n00;

  assert(theta_n00->n == theta_n0()->n);
#ifdef DEBUG
  if(diff_two_norm(theta_n00->n, theta_n0()->x, theta_n00->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in theta\n");
#endif
  delete theta_n00;

  assert(b_s00->n == b_s0()->n);
#ifdef DEBUG
  if(diff_two_norm(b_s00->n, b_s00->x, b_s0()->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in b_s\n");
#endif
  delete b_s00;

  assert(p_g00->n == p_g0()->n);
#ifdef DEBUG
  if(diff_two_norm(p_g00->n, p_g00->x, p_g0()->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in p_g\n");
#endif
  delete p_g00;

  assert(q_g00->n == q_g0()->n);
#ifdef DEBUG
  if(diff_two_norm(q_g00->n, q_g00->x, q_g0()->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in q_g\n");
#endif
  delete q_g00;

#endif
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
    assert(size_sol_block == v_n0()->n + theta_n0()->n + b_s0()->n + p_g0()->n + q_g0()->n + 1);
#endif
    
    const double 
      *v_n     = vvsols[Kidx].data(), 
      *theta_n = vvsols[Kidx].data() + v_n0()->n, 
      *b_s     = vvsols[Kidx].data() + v_n0()->n + theta_n0()->n, 
      *p_g     = vvsols[Kidx].data() + v_n0()->n + theta_n0()->n + b_s0()->n, 
      *q_g     = vvsols[Kidx].data() + v_n0()->n + theta_n0()->n + b_s0()->n + p_g0()->n,
      delta    = vvsols[Kidx][v_n0()->n + theta_n0()->n + b_s0()->n + p_g0()->n + q_g0()->n];

    SCACOPFIO::write_solution2_block(Kidx, v_n, theta_n, b_s, p_g, q_g, delta,
				     data, "solution2.txt");
    printf("[rank 0] wrote solution2 block for contingency %d [%d] label '%s'\n", 
	   Kidx, K_Contingency[Kidx].id, data.K_Label[Kidx].c_str());

    hardclear(vvsols[Kidx]);
    last_Kidx_written++;
  }

  return false;
}

int MyCode2::pick_new_contingency(int rank) 
{
  //pick new contingency 
  int perRank = K_Contingency.size()/(comm_size-1);
  int Kstart = perRank*(rank-1);
  if(rank<=20) Kstart=0;
  auto Kidxs = findall(K_left, [Kstart](int val) {return val>=Kstart;});
  if(Kidxs.empty()) 
    Kidxs = findall(K_left, [](int val) {return val>=0;});
  
  int Kidx = -1;
  if(!Kidxs.empty()) {
    Kidx = K_left[Kidxs[0]];
    erase_idx_from(K_left, Kidxs[0]);
  } 
  return Kidx;
}
int MyCode2::pick_late_contingency(int rank)
{
  double tm = MPI_Wtime();
  for(auto & kinfo : K_Contingency) {
    if(kinfo.tmSent.size()>0 && !kinfo.solve_done) {
      if(tm-kinfo.tmSent.back()>1400 && kinfo.tmSent.size()==1) {
	printf("[warning] rank=%d detected that K_idx=%d is late by %.2f sec\n", rank, kinfo.id, tm-kinfo.tmSent.back());
	return kinfo.id;
      }
    }
  }
  return -1;
}

bool MyCode2::all_contingencies_solved()
{
  for(auto & kinfo : K_Contingency) {
    if(!kinfo.solve_done) 
      return false;
  }
  return true;
}

void MyCode2::attempt_cleanup_req_send_Ksln()
{
  bool done=false; int mpi_test_flag, ierr; MPI_Status mpi_status;
  while(!done) {
    vector<ReqKSln*>::iterator it = req_send_Ksln.begin();
    for(;it!=req_send_Ksln.end(); ++it) {
      ReqKSln* req_sln = (*it);
      const int ierr = MPI_Test(&(req_sln->request), &mpi_test_flag, &mpi_status);
      if(mpi_test_flag != 0) {
	//completed
	delete req_sln;
	req_send_Ksln.erase(it);
	break;
      }
    }
    if(it==req_send_Ksln.end()) done=true;
  }
}

 

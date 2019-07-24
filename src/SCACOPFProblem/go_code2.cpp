#include "go_code2.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"

#include "goUtils.hpp"

using namespace std;
using namespace gollnlp;

#include "unistd.h"
#include <chrono>
#include <thread>

#define DEBUG_COMM 1
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
}

MyCode2::~MyCode2()
{
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

  rank_master = 0;
  if(my_rank == rank_master) iAmMaster=true;

  if(!iAmMaster)  
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

  data.my_rank = my_rank;

  num_req4Kidx = 1;
  num_req4Kidx_for_ranks = vector<int>(comm_size, 1);
  comm_contidxs_done = vector<bool>(comm_size, false);

  //load data
  if(!data.readinstance(InFile1, InFile2, InFile3, InFile4)) {
    printf("error occured while reading instance\n");
    return false;
  }

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
  for(int r=0; r<comm_size; r++) {
    for(int Kidx: K_on_slave_ranks[r]) {
      if(Kidx<0) continue;
      int tag2 = 222;     
      ReqKSln* req_recv_sln = new ReqKSln(Kidx, vector<double>(200,-12.17));
      int ierr = MPI_Irecv(req_recv_sln->buffer.data(), req_recv_sln->get_msg_size(), 
			   MPI_DOUBLE, r, tag2, comm_world, &req_recv_sln->request);
      assert(ierr == MPI_SUCCESS);
      req_recv_Ksln[r].push_back(req_recv_sln);
      printf("[rank 0] contsol recv created for conting %d on rank %d\n", Kidx, r);
    }
  }

  bool ask_for_conting=true;

  //main loop
  while(true) {

    if(!K_local.empty()) {
      printf("2222\n");

      int k = K_local.front(); 
      K_local.pop_front();

      vector<double> sol;

      printf("Solving contingency on rank %d\n", my_rank);

      bool bret = solve_contingency(k, sol);

      //send the solution
      int tag2 = 222;      
      ReqKSln* req_send_sol = new ReqKSln(k, sol);
      int ierr = MPI_Isend(req_send_sol->buffer.data(), req_send_sol->get_msg_size(), 
			   MPI_DOUBLE, rank_master, tag2, comm_world, &req_send_sol->request);

#ifdef DEBUG_COMM
      printf("on rank %d: posted send solution for K_idx=%d to master rank %d globtime %g\n", 
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
	    printf("[on rank 0] contsol recv from rank=%d conting=%d done\n", r, k); 
#endif	    
	    K_on_slave_ranks[r][i] = -1;
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
	  printf("on rank %d: created send request %d for contingencies globtime %g\n", 
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
	  printf("on rank %d: completed send request %d for contingencies globtime %g\n", 
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
	  printf("on rank %d: created irecv request %d for indexes globtime %g\n", 
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
	printf("on rank %d: irecv request %d for indexes completed globtime %g\n", 
	  my_rank, num_req4Kidx, glob_timer.measureElapsedTime());
#endif
	for(auto k: req_recv_idx->buffer) {
	  if(k>0) {
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
    if(my_rank==0) {
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
	    int perRank = data.K_Contingency.size()/comm_size;
	    int Kstart = perRank*r;
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
	    printf("[rank 0] created send indexes request %d to rank %d (444)\n", num_req4Kidx_for_ranks[r], r);

	    req_send_Kidx[r].push_back(req_send_idx);

	    assert(req_send_Kidx[r].size() == req_recv_req4Kidx[r].size());

	    delete req_recv_idx;
	    req_recv_req4Kidx[r].pop_back();
	    
	  
	    // create contsol recvs for the contingency that was just sent 
	    if(Kidx>0) {
	      K_on_slave_ranks[r].push_back(Kidx);
	      
	      int tag2 = 222;     
	      ReqKSln* req_recv_sln = new ReqKSln(Kidx, vector<double>(200,-12.17));
	      ierr = MPI_Irecv(req_recv_sln->buffer.data(), req_recv_sln->get_msg_size(), 
			       MPI_DOUBLE, r, tag2, comm_world, &req_recv_sln->request);

	      printf("[rank 0] contsol recv created for conting %d on rank %d\n", Kidx, r);
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
    } // end of - master part
    
  } // end of while(true)

  // final message
  if(my_rank==rank_master)
    printf("--finished in %g sec  global time %g sec\n", ttot.stop(), glob_timer.measureElapsedTime());

  return 0;
}


void MyCode2::initial_K_distribution()
{

  int num_ranks = comm_size;
  //initialize K_on_rank (vector of K idxs on each rank)
  for(int it=0; it<num_ranks; it++)
    K_on_slave_ranks.push_back(vector<int>());
  
  //num contingencies; 
  int S = data.K_Contingency.size(), R = num_ranks-1;
  if(R<=0) R=1;

  K_left = vector<int>(S);
  iota(K_left.begin(), K_left.end(), 0);

  int perRank = S/R; int remainder = S-R*perRank;
  printf("ranks=%d contingencies=%d perRank=%d remainder=%d\n",
  	 num_ranks, S, perRank, remainder);

  if(iAmMaster) {

    //each slave rank gets one contingency idx = r*perRank
    for(int r=1; r<num_ranks; r++) {
      int K_idx = (r-1) * perRank;
      if(r < remainder)
	K_idx += r;
      else
	K_idx += remainder;
      
      assert(K_idx < S);
      K_on_slave_ranks[r].push_back(K_idx);
      bool bret = erase_elem_from(K_left, K_idx);
      assert(bret);
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
    K_local.push_back(K_idx);
    
    char s[1000]; sprintf(s, "K_local on rank %d", my_rank);
    printlist(K_local, string(s));
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

bool MyCode2::solve_contingency(int K_idx, std::vector<double>& sln)
{
  sln = vector<double>(200, 17);

  printf("solve_contingency on rank %d\n", my_rank);

  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  return true;
}

void MyCode2::display_instance_info()
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSec);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}

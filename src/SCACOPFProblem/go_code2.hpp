#ifndef GO_CODE2
#define GO_CODE2

#include "mpi.h"
#include <string>
#include <vector>
#include <list>

#include "SCACOPFProblem.hpp"
#include "ContingencyProblem.hpp"

#include "goTimer.hpp"

class MyCode2
{
public:
  MyCode2(const std::string& InFile1, const std::string& InFile2,
	  const std::string& InFile3, const std::string& InFile4,
	  double TimeLimitInSeconds, 
	  int ScoringMethod, 
	  const std::string& NetworkModelName,
	  MPI_Comm comm_world=MPI_COMM_WORLD);
  virtual ~MyCode2();

  virtual int initialize(int argc, char *argv[]);
  virtual int go();
  virtual void display_instance_info();

private:
  struct ReqKSln
  {
    ReqKSln(int K_idx, const std::vector<double>& sln)
    {
      buffer = sln;
      buffer.push_back( (double) K_idx);
    }
    std::vector<double> buffer;
    MPI_Request request;
    inline int get_msg_size() { return buffer.size(); }
  private:
    ReqKSln() {};
  };
  //on master rank
  std::vector<std::vector<ReqKSln*> > req_recv_Ksln;
  //on slave ranks
  std::vector<ReqKSln*> req_send_Ksln;

  struct ReqKidx
  {
    ReqKidx(int K_idx)
    {
      buffer.push_back(K_idx);
    }
    std::vector<int> buffer;
    MPI_Request request;
  private:
    ReqKidx() {};
  };
  //on master rank
  std::vector<std::vector<ReqKidx*> > req_recv_req4Kidx;
  std::vector<std::vector<ReqKidx*> > req_send_Kidx;
  //size of the comm_size
  std::vector<bool> comm_contidxs_done;
  //on slave ranks
  std::vector<ReqKidx*> req_send_req4Kidx;
  std::vector<ReqKidx*> req_recv_Kidx;
  int num_req4Kidx;
  std::vector<int> num_req4Kidx_for_ranks;

  void initial_K_distribution();
  bool solve_contingency(int K_idx, std::vector<double>& sln);

  // 1. sweeps 'vvsols' and writes to 'solution2.txt' the solutions 
  // received on master; this is done in order of the Kidx;
  // 2. once the solution for contingcency k is written, vvsols[k] is deallocated
  //
  // return true when all solutions are written; false otherwise
  bool attempt_write_solution2(std::vector<std::vector<double> >& vvsols);
  int last_Kidx_written;
  int size_sol_block;
private:
  std::string InFile1, InFile2, InFile3, InFile4;
  double TimeLimitInSec;
  int ScoringMethod;
  std::string NetworkModel;

  gollnlp::SCACOPFData data;

  gollnlp::OptVariablesBlock *v_n0, *theta_n0, *b_s0, *p_g0, *q_g0;

  //
  // communication
  //
  MPI_Comm comm_world;
  int my_rank, rank_master;
  int comm_size;
  bool iAmMaster;
  gollnlp::goTimer glob_timer;

  //contingencies processed on each rank
  //outer size num_ranks, maintained only on master rank
  std::vector<std::vector<int> > K_on_slave_ranks;
  std::vector<int> K_left;

  //contingencies on current rank
  std::list<int> K_local;

};
#endif
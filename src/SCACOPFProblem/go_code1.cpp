#include "go_code1.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"
#include "goTimer.hpp"

using namespace std;
using namespace gollnlp;


MyCode1::MyCode1(const std::string& InFile1_, const std::string& InFile2_,
		 const std::string& InFile3_, const std::string& InFile4_,
		 double TimeLimitInSeconds, 
		 int ScoringMethod_, 
		 const std::string& NetworkModelName,
		 MPI_Comm comm_world_)
  : InFile1(InFile1_), InFile2(InFile2_), InFile3(InFile3_), InFile4(InFile4_),
    TimeLimitInSec(TimeLimitInSeconds), ScoringMethod(ScoringMethod_),
    NetworkModel(NetworkModelName),
    rank_master(-1), rank_solver_master(-1),
    comm_world(comm_world_), comm_solver(MPI_COMM_NULL)
{
  iAmMaster=iAmSolver=iAmEvaluator=false;
  scacopf_prob = NULL;
  my_rank = -1;
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
  if(my_rank == 0) iAmMaster=true;
  if(comm_size==1) {
    iAmSolver=true; iAmEvaluator=true;
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
  printf("[Phase 1] Rank %d ismaster %d issolver %d isevaluator %d\n",
  	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
}

void MyCode1::phase2_ranks_allocation()   
{
  assert(comm_world != MPI_COMM_NULL);
  int ret, comm_size;
  ret = MPI_Comm_size(comm_world, &comm_size); assert(ret==MPI_SUCCESS);

  //no solver work needed in this phase
  //all solver ranks become evaluators
  iAmSolver=false; 
  if(my_rank == 0) {assert(iAmMaster); iAmMaster = true;}
  if(comm_size==1) {
    iAmEvaluator=true;
  } else {

    //ranks 0, 1, 2, 3, 4
    iAmEvaluator=true; //rank 0 is also an evaluator as long as comm_size<4
    if(my_rank==0 && comm_size>=4) iAmEvaluator=false;
    
  }
  printf("[Phase 2] Rank %d ismaster %d issolver %d isevaluator %d\n",
  	 my_rank, iAmMaster, iAmSolver, iAmEvaluator);
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
  vector<int> K_idxs = phase1_SCACOPF_contingencies();
  assert(NULL == scacopf_prob);

  phase1_ranks_allocation();
  
  if(iAmSolver) {
    scacopf_prob = new SCACOPFProblem(data);
    scacopf_prob->assembly(K_idxs);
    scacopf_prob->use_nlp_solver("ipopt"); 
    scacopf_prob->set_solver_option("linear_solver", "ma57"); 
    scacopf_prob->set_solver_option("mu_init", 1.);
    scacopf_prob->set_solver_option("print_frequency_iter", 5);
    scacopf_prob->set_solver_option("print_level", 5);
    bool bret = scacopf_prob->optimize("ipopt");
    
    
  } else {
    //master and evaluators do nothing
  }

  //communication -> solver rank0 bcasts basecase solution
  MPI_Barrier(comm_world);
  return true;
}
vector<int> MyCode1::phase1_SCACOPF_contingencies()
{
  bool testing = true;
  if(true) {
    //net 07R scenario 9
    vector<int> cont_list = {426//, //line/trans conting, penalty $417
				  //960, // gen conting, penalty $81,xxx
				  //961
    };//963};// gen conting, penalty $52,xxx
    return  cont_list;
  } else {
    return vector<int>();
  }
}
int MyCode1::go()
{
  goTimer ttot; ttot.start();

  if(iAmMaster)
    display_instance_info();

  if(!do_phase1()) {
    printf("Error occured in phase 1\n");
    return -1;
  }

  //if(!do_phase2()) {
  //  printf("Error occured in phase 2\n");
  //  return -1;
  //}

  //if(!do_phase3()) {
  //  printf("Error occured in phase 3\n");
  //  return -1;
  //}

  //
  //cleanup
  //
  delete scacopf_prob;
  MPI_Finalize();
  return 0;
}
  //use a small list of contingencies for testing/serial
  //  std::vector<int> cont_list = {0, 88, 89, 92};//, 407};

  //net 07R scenario 9
  // std::vector<int> cont_list = {426//, //line/trans conting, penalty $417
  // 				//960, // gen conting, penalty $81,xxx
  // 				//961
  // 				};//963};// gen conting, penalty $52,xxx
  
  
  // SCMasterProblem master_prob(data, cont_list);
  // master_prob.default_assembly();

  // //
  // //phase 1
  // //
  // master_prob.use_nlp_solver("ipopt"); 
  // master_prob.set_solver_option("linear_solver", "ma57"); 
  // master_prob.set_solver_option("mu_init", 1.);
  // bool bret = master_prob.optimize("ipopt");

  // printf("*** PHASE 1 finished - master problem solved: obj_value %g\n\n", master_prob.objective_value());

  // //
  // //phase 2
  // //
  // SCRecourseObjTerm* rec;
  // master_prob.append_objterm(rec=new SCRecourseObjTerm(d, master_prob, 
  // 						       master_prob.p_g0_vars(), master_prob.v_n0_vars(), 
  // 						       cont_list));
  // //master_prob.append_objterm(new SCRecourseObjTerm(d, master_prob.p_g0_vars(), master_prob.v_n0_vars()));

  // //bret = master_prob.optimize("ipopt");
  // master_prob.set_solver_option("mu_init", 1e-8);
  // master_prob.set_solver_option("bound_push", 1e-16);
  // master_prob.set_solver_option("slack_bound_push", 1e-16);
  // bret = master_prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  // printf("*** PHASE 2 finished - master problem solved: obj_value %g\n\n", master_prob.objective_value());


  // master_prob.set_solver_option("mu_init", 1e-2);
  // master_prob.set_solver_option("bound_push", 1e-16);
  // master_prob.set_solver_option("slack_bound_push", 1e-16);
  // rec->stop_evals=false;
  // bret = master_prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  // ttot.stop(); printf("MyExe1 took %g sec.\n", ttot.getElapsedTime());


void MyCode1::display_instance_info()
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSec);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}

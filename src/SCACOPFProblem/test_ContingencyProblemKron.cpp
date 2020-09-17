#include "ContingencyProblemWithFixingCode1.hpp"
#include "ContingencyProblemKronRedWithFixingCode1.hpp"
#include "ContingencyProblemKronRed.hpp"

#include "SCACOPFData.hpp"
#include "SCACOPFIO.hpp"
#include "goTimer.hpp"

using namespace gollnlp;

#include <unordered_map>
using namespace std;

int main(int argc, char *argv[])
{

  int ret;
  ret = MPI_Init(&argc, &argv); assert(ret==MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    std::cerr << "MPI_Init failed\n";
  }
  goTimer glob_timer, t;
  glob_timer.start();
  t.start();
  
  int my_rank = 0;
  
  if(argc!=8) return -1;
  
  
  double timeLimit = atof(argv[5]);
  int scoringMethod = atoi(argv[6]);
  
  if(timeLimit <=0 ) {
    std::cout << "invalid time limit? > " << argv[5] << std::endl;
  }
  
  if(scoringMethod <1 || scoringMethod >4 ) {
    std::cout << "invalid scoring method? > " << argv[6] << std::endl;
  }
  
  std::unordered_map<std::string, gollnlp::OptVariablesBlock*> dict_basecase_vars;
  SCACOPFData data;
  data.my_rank = my_rank;
  
  if(!data.readinstance(argv[3], argv[4], argv[2], argv[1])) {
    printf("error occured while reading instance\n");
    return false;
  }

  
  gollnlp::SCACOPFIO::read_variables_blocks(data, dict_basecase_vars);
  //for(auto& p: dict_basecase_vars) cout << "   - [" << p.first << "] size->" << p.second->n << endl;

  auto v_n0 = dict_basecase_vars["v_n_0"];
  auto theta_n0 = dict_basecase_vars["theta_n_0"];
  auto b_s0 = dict_basecase_vars["b_s_0"];
  auto p_g0 = dict_basecase_vars["p_g_0"];
  auto q_g0 = dict_basecase_vars["q_g_0"];
  
  size_t size_sol_block = v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n + 1;

  OptVariablesBlock *v_n00, *theta_n00, *b_s00, *p_g00, *q_g00;
  SCACOPFIO::read_solution1(&v_n00, &theta_n00, &b_s00, &p_g00, &q_g00, data, "solution1.txt");
  

  assert(v_n00->n == v_n0->n);
#ifdef DEBUG
  if(diff_two_norm(v_n00->n, v_n00->x, v_n0->x)>1e-12)
     printf("[warning] difference between read_variables read_solution1 in v\n");
#endif
  delete v_n00;

  assert(theta_n00->n == theta_n0->n);
#ifdef DEBUG
  double diff=0.;
  if( (diff=diff_two_norm(theta_n00->n, theta_n0->x, theta_n00->x))>1e-12) {
    //for(int i=0; i<theta_n00->n; i++)
    //  if(fabs(theta_n0()->x[i] - theta_n00->x[i])>1e-18) printf("[%d %12.5e %12.5e]",
    //								i, theta_n0()->x[i],theta_n00->x[i]);	 
    printf("[warning] difference [%12.5e] between read_variables read_solution1 in theta\n", diff);

  }
#endif
  delete theta_n00;

  assert(b_s00->n == b_s0->n);
#ifdef DEBUG
  if(diff_two_norm(b_s00->n, b_s00->x, b_s0->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in b_s\n");
#endif
  delete b_s00;

  assert(p_g00->n == p_g0->n);
#ifdef DEBUG
  if(diff_two_norm(p_g00->n, p_g00->x, p_g0->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in p_g\n");
#endif
  delete p_g00;

  assert(q_g00->n == q_g0->n);
#ifdef DEBUG
  if(diff_two_norm(q_g00->n, q_g00->x, q_g0->x)>1e-12)
    printf("[warning] difference between read_variables read_solution1 in q_g\n");
#endif
  delete q_g00;

  auto p_li1 = dict_basecase_vars["p_li1_0"];
  auto q_li1 = dict_basecase_vars["q_li1_0"];
  auto p_li2 = dict_basecase_vars["p_li2_0"];
  auto q_li2 = dict_basecase_vars["q_li2_0"];
  auto p_ti1 = dict_basecase_vars["p_ti1_0"];
  auto q_ti1 = dict_basecase_vars["q_ti1_0"];
  auto p_ti2 = dict_basecase_vars["p_ti2_0"];
  auto q_ti2 = dict_basecase_vars["q_ti2_0"];
  
  //
  // ========================================================
  //
  //100 works after it enters restauration phase
  int K_idx = 100;
  int comm_size=1;
  double data_for_master[111];
  double pen_threshold = 1e+5;
  bool run_full_prob = true;
  bool run_kron_prob = true;

  if(run_full_prob) {
    ContingencyProblemWithFixingCode1 prob(data, K_idx, 
					   my_rank, comm_size, 
					   dict_basecase_vars, 
					   -1, -1., false);
  
    ContingencyProblemWithFixing::g_bounds_abuse = 0.000095;
    prob.monitor.is_active = true;


  
    prob.pen_accept = 0.99*pen_threshold;
    prob.pen_accept_initpt=0.99*pen_threshold;
    prob.pen_accept_solve1= 0.99*pen_threshold;
    prob.pen_accept_emer= 0.99*pen_threshold;
    prob.pen_accept_safemode= 0.99*pen_threshold;

    prob.use_nlp_solver("ipopt");


    if(!prob.default_assembly(v_n0, theta_n0, b_s0, p_g0, q_g0,
			      p_li1, q_li1, p_li2, q_li2, p_ti1, q_ti1, p_ti2, q_ti2)) {

      printf("rank=%d failed in default_assembly for contingency K_idx=%d\n",
	     my_rank, K_idx);
      //status = -1;
      data_for_master[0]=-0.117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
      return -1170;
    }

    double penalty; int num_iter = -117;
    if(!prob.eval_obj(p_g0, v_n0, penalty, data_for_master)) {
      printf("Evaluator Rank %d failed in the eval_obj of contingency K_idx=%d  global time %g\n",
	     my_rank, K_idx, glob_timer.measureElapsedTime());
      //status = -3;
      data_for_master[0]=penalty=-0.0117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
    } else {
      num_iter = prob.number_of_iterations();
    }  

    printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	   "in %5.3f sec and %3d iterations  sol_from_scacopf_pass XXX  global time %g\n",
	   my_rank, K_idx, penalty, t.stop(), 
	   num_iter, glob_timer.measureElapsedTime());
  }

  if(run_kron_prob) {
    gollnlp::ContingencyProblemKronRedWithFixingCode1 prob(data, K_idx, 
							   my_rank, comm_size, 
							   dict_basecase_vars, 
							   -1, -1., false);

    gollnlp::ContingencyProblemWithFixing::g_bounds_abuse = 0.000095;
    prob.monitor.is_active = true;
    
    prob.pen_accept = 0.99*pen_threshold;
    prob.pen_accept_initpt=0.99*pen_threshold;
    prob.pen_accept_solve1= 0.99*pen_threshold;
    prob.pen_accept_emer= 0.99*pen_threshold;
    prob.pen_accept_safemode= 0.99*pen_threshold;
    
    prob.use_nlp_solver("hiop");
    
    //prob.use_nlp_solver("ipopt");

    if(!prob.default_assembly(v_n0, theta_n0, b_s0, p_g0, q_g0,
			      p_li1, q_li1, p_li2, q_li2, p_ti1, q_ti1, p_ti2, q_ti2)) {
      
      printf("rank=%d failed in default_assembly for contingency K_idx=%d\n",
	     my_rank, K_idx);
      //status = -1;
      data_for_master[0]=-0.117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
      return -1170;
    }
    
    double penalty; int num_iter = -117;
    if(!prob.eval_obj(p_g0, v_n0, penalty, data_for_master)) {
      printf("Evaluator Rank %d failed in the eval_obj of contingency K_idx=%d  global time %g\n",
	     my_rank, K_idx, glob_timer.measureElapsedTime());
      //status = -3;
      data_for_master[0]=penalty=-0.0117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
    } else {
      num_iter = prob.number_of_iterations();
    }  
    
    printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	   "in %5.3f sec and %3d iterations  sol_from_scacopf_pass xxx  global time %g\n",
	   my_rank, K_idx, penalty, t.stop(), 
	   num_iter, glob_timer.measureElapsedTime());
  }
  
  
  MPI_Finalize();
  return 0;
}

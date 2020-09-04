#ifndef GO_CODE1_KRON
#define GO_CODE1_KRON

#include "go_code1.hpp"

using namespace std;

#include "ContingencyProblemKronRedWithFixingCode1.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"

class MyCode1Kron : public MyCode1
{
public:
  MyCode1Kron(const std::string& InFile1, const std::string& InFile2,
	      const std::string& InFile3, const std::string& InFile4,
	      double TimeLimitInSeconds, 
	      int ScoringMethod, 
	      const std::string& NetworkModelName,
	      MPI_Comm comm_world=MPI_COMM_WORLD)
    : MyCode1(InFile1, InFile2, InFile3, InFile4,
	      TimeLimitInSeconds, ScoringMethod, NetworkModelName, comm_world)
  {

  }
  virtual ~MyCode1Kron()
  {

  }

protected: //methods

  double solve_contingency_use_fixing(int K_idx, int& status, double* data_for_master)
  {
    gollnlp::goTimer t; t.start();
    assert(iAmEvaluator);
    assert(scacopf_prob != NULL);
  
    status = 0; //be positive
    auto p_g0 = scacopf_prob->variable("p_g", data); 
    auto q_g0 = scacopf_prob->variable("q_g", data); 
    auto v_n0 = scacopf_prob->variable("v_n", data);
    auto theta_n0 = scacopf_prob->variable("theta_n", data);
    auto b_s0 = scacopf_prob->variable("b_s", data); 
    auto p_li1 = scacopf_prob->variable("p_li1", data); 
    auto q_li1 = scacopf_prob->variable("q_li1", data); 
    auto p_li2 = scacopf_prob->variable("p_li2", data); 
    auto q_li2 = scacopf_prob->variable("q_li2", data);
    auto p_ti1 = scacopf_prob->variable("p_ti1", data); 
    auto q_ti1 = scacopf_prob->variable("q_ti1", data); 
    auto p_ti2 = scacopf_prob->variable("p_ti2", data); 
    auto q_ti2 = scacopf_prob->variable("q_ti2", data); 
    

    assert(p_g0 == dict_basecase_vars["p_g_0"]);
    assert(q_g0 == dict_basecase_vars["q_g_0"]);
    assert(v_n0 == dict_basecase_vars["v_n_0"]);
    assert(theta_n0 == dict_basecase_vars["theta_n_0"]);
    assert(b_s0 == dict_basecase_vars["b_s_0"]);
    assert(p_li1 == dict_basecase_vars["p_li1_0"]);
    assert(q_li1 == dict_basecase_vars["q_li1_0"]);
    assert(p_li2 == dict_basecase_vars["p_li2_0"]);
    assert(q_li2 == dict_basecase_vars["q_li2_0"]);
    
#ifdef GOLLNLP_WITH_KRON_REDUCTION          
    
    gollnlp::ContingencyProblemKronRedWithFixingCode1 prob(data, K_idx, 
							   my_rank, comm_size, 
							   dict_basecase_vars, 
							   -1, -1., false);
#else
    assert(false);
#endif
    
    gollnlp::ContingencyProblemWithFixing::g_bounds_abuse = 0.000095;
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
      status = -1;
      data_for_master[0]=-0.117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
      return -1170;
    }
    
    double penalty; int num_iter = -117;
    if(!prob.eval_obj(p_g0, v_n0, penalty, data_for_master)) {
      printf("Evaluator Rank %d failed in the eval_obj of contingency K_idx=%d  global time %g\n",
	     my_rank, K_idx, glob_timer.measureElapsedTime());
      status = -3;
      data_for_master[0]=penalty=-0.0117;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
    } else {
      num_iter = prob.number_of_iterations();
    }  
    
    printf("Evaluator Rank %3d K_idx=%d finished with penalty %12.3f "
	   "in %5.3f sec and %3d iterations  sol_from_scacopf_pass %d  global time %g\n",
	   my_rank, K_idx, penalty, t.stop(), 
	   num_iter, phase3_scacopf_pass_solution, glob_timer.measureElapsedTime());
    
    return penalty; 
  }
  
};


#endif

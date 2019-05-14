#include <string>
#include <iostream>

#include "SCACOPFData.hpp"
#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"
using namespace gollnlp;
using namespace std;

int main(int argc, char *argv[])
{
  std::string root, net, scen, name;

  root = "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/";
  name = "Network_07R-10";
  net = name + "/";
  scen = "scenario_9/";
  root = root+net; 

  SCACOPFData d;
  d.readinstance(root+scen+"case.raw",
		 root+"case.rop",
		 root+"case.inl",
		 root+scen+"case.con");
  //net 07R scenario 9
  std::vector<int> cont_list = {
    //426,//, //line/trans conting, penalty $417
    //960, // gen conting, penalty $81,xxx
    //961,
    963
  };// gen conting, penalty $52,xxx
  

  SCMasterProblem master_prob(d, cont_list);
  //SCACOPFProblem master_prob(d);
  master_prob.default_assembly();

  //
  //phase 1
  //
  master_prob.use_nlp_solver("ipopt"); 
  master_prob.set_solver_option("linear_solver", "ma57"); 
  master_prob.set_solver_option("mu_init", 1.);
  //master_prob.set_solver_option("hessian_approximation", "limited-memory");
  bool bret = master_prob.optimize("ipopt");

  printf("*** PHASE 1 finished - master problem solved: obj_value %g\n\n",
	 master_prob.objective_value());


  //
  //phase 2
  //


  if(false) {
    SCMasterProblem p(d, cont_list);    
    auto p_g0 = new OptVariablesBlock(d.G_Generator.size(), "p_g_0", 
				      d.G_Plb.data(), d.G_Pub.data());
    p.append_variables(p_g0);
    p_g0->set_start_to(master_prob.vars_block("p_g_0")->x);
    p_g0->xref = p_g0->x; 
    
    printf("creating recourse objective\n");
    SCRecourseObjTerm* rec;
    p.append_recourse_obj(rec=new SCRecourseObjTerm(d,
						    p_g0,
						    NULL, //master_prob.v_n0_vars(),
						    cont_list));
    p.use_nlp_solver("ipopt");
    p.set_solver_option("linear_solver", "ma57"); 
    
    //p.set_solver_option("derivative_test", "first-order");
    //p.set_solver_option("derivative_test_perturbation",  1e-2);
    //p.set_solver_option("derivative_test_tol", 1e-3);
    
    p.set_solver_option("mu_init", 1e-6);
    p.set_solver_option("mu_target", 1e-6);
    p.set_solver_option("tol", 1e-4);
    
    if(false) {
      p.set_solver_option("hessian_approximation", "limited-memory");
      //https://arxiv.org/pdf/1901.05682.pdf
      p.set_solver_option("limited_memory_initialization", "scalar1");
      //p.set_solver_option("limited_memory_max_history", 0);
      p.set_solver_option("limited_memory_init_val", 100.);
      p.set_solver_option("print_level", 5);
    }
    p.optimize("ipopt");
  } else {
    SCRecourseObjTerm* rec;
    master_prob.append_recourse_obj(rec=new SCRecourseObjTerm(d, 
    							 master_prob.vars_block("p_g_0"), 
    							 NULL, 
    							 cont_list));
    master_prob.set_solver_option("mu_init", 1e-9);
    master_prob.set_solver_option("bound_push", 1e-18);
    master_prob.set_solver_option("slack_bound_push", 1e-18);
    master_prob.problem_changed();
    //master_prob.reoptimize("ipopt");
    bret = master_prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu
  }
  
  printf("Test finished\n");
  
  return 0;
}

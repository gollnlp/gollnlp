#include "go_code1.hpp"

#include "SCMasterProblem.hpp"
#include "SCRecourseProblem.hpp"
#include "goTimer.hpp"

using namespace std;
using namespace gollnlp;

static void display_instance_info(const std::string& InFile1, const std::string& InFile2,
		    const std::string& InFile3, const std::string& InFile4,
		    double TimeLimitInSeconds, 
		    int ScoringMethod, 
				  const std::string& NetworkModel);

int myexe1_function(const std::string& InFile1, const std::string& InFile2,
		    const std::string& InFile3, const std::string& InFile4,
		    double TimeLimitInSeconds, 
		    int ScoringMethod, 
		    const std::string& NetworkModel)
{
  goTimer ttot; ttot.start();

  display_instance_info(InFile1, InFile2, InFile3, InFile4, TimeLimitInSeconds, ScoringMethod, NetworkModel);

  //use a small list of contingencies for testing/serial
  //  std::vector<int> cont_list = {0, 88, 89, 92};//, 407};
  //net 07R scenario 9
  std::vector<int> cont_list = {426, //line conting, penalty $417
				960, // gen conting, penalty $81,xxx
				961,
				963};// gen conting, penalty $52,xxx
  
  
  SCACOPFData d;
  d.readinstance(InFile1, InFile2, InFile3, InFile4);

  SCMasterProblem master_prob(d, cont_list);
  master_prob.default_assembly();

  //
  //phase 1
  //
  master_prob.use_nlp_solver("ipopt"); 
  master_prob.set_solver_option("linear_solver", "ma57"); 
  master_prob.set_solver_option("mu_init", 1.);
  bool bret = master_prob.optimize("ipopt");

  printf("*** PHASE 1 finished - master problem solved: obj_value %g\n\n", master_prob.objective_value());

  //
  //phase 2
  //
  master_prob.append_objterm(new SCRecourseObjTerm(d, master_prob.p_g0_vars(), master_prob.v_n0_vars(), cont_list));
  //master_prob.append_objterm(new SCRecourseObjTerm(d, master_prob.p_g0_vars(), master_prob.v_n0_vars()));

  //bret = master_prob.optimize("ipopt");
  master_prob.set_solver_option("mu_init", 1e-8);
  master_prob.set_solver_option("bound_push", 1e-16);
  master_prob.set_solver_option("slack_bound_push", 1e-16);
  bret = master_prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  printf("*** PHASE 2 finished - master problem solved: obj_value %g\n\n", master_prob.objective_value());

  ttot.stop(); printf("MyExe1 took %g sec.\n", ttot.getElapsedTime());
  return 0;
}

void 
display_instance_info(const std::string& InFile1, const std::string& InFile2,
		      const std::string& InFile3, const std::string& InFile4,
		      double TimeLimitInSeconds, 
		      int ScoringMethod, 
		      const std::string& NetworkModel)
{
  printf("Model %s ScoringMethod %d TimeLimit %g\n", NetworkModel.c_str(), ScoringMethod, TimeLimitInSeconds);
  printf("Paths to data files:\n");
  printf("[%s]\n[%s]\n[%s]\n[%s]\n\n", InFile1.c_str(), InFile2.c_str(), InFile3.c_str(), InFile4.c_str());
}

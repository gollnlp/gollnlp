#include "SCMasterProblem.hpp"

namespace gollnlp {

bool SCMasterProblem::default_assembly()
{
  useQPen = true;
  slacks_scale = 1.;
  
  SCACOPFData& d = data_sc; //shortcut
  
  //
  // base case
  //
  if(false) {
  add_variables(d);
  add_cons_lines_pf(d);
  add_cons_transformers_pf(d);
  add_cons_active_powbal(d);
  add_cons_reactive_powbal(d);
  add_cons_thermal_li_lims(d);
  add_cons_thermal_ti_lims(d);
  add_obj_prod_cost(d);
  } else {
    SCACOPFProblem::default_assembly();
  }

  return true;
}
  bool SCMasterProblem::iterate_callback(int iter, const double& obj_value, const double* primals,
					 const double& inf_pr, const double& inf_du, 
					 const double& mu, 
					 const double& alpha_du, const double& alpha_pr,
					 int ls_trials)
  {
    if(NULL != recou_objterm) {

      auto p_g0 = p_g0_vars(); 
      recou_objterm->end_of_iteration(iter); 
      //printf("SCMasterProblem end of iter %d\n", iter, obj_value);
    }
    return true;
  }
} //end namespace

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
  add_variables(d);
  add_cons_lines_pf(d);
  add_cons_transformers_pf(d);
  add_cons_active_powbal(d);
  add_cons_reactive_powbal(d);
  add_cons_thermal_li_lims(d);
  add_cons_thermal_ti_lims(d);
  add_obj_prod_cost(d);



  return true;
}

} //end namespace

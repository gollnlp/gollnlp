#include "Ex1OptObjects.hpp"
#include "IpoptSolver.hpp"


int main()
{
  OptProblem prob;
  int nx=2000000, nyz=nx/5;
  OptVariablesBlock* x = new OptVariablesBlock(nx, "x");
  OptVariablesBlock* y = new OptVariablesBlock(nyz, "y", 0.5, 10);
  OptVariablesBlock* z = new OptVariablesBlock(nyz, "z", -1e+20, 5);

  //by default variables blocks have no starting values attached
  //example
  y->set_start_to(0.5);
  z->set_start_to(*y); // can also have z->set_start_to( some_array_double* )
  //x still does not have start values and OptProblem will pass zeros for x to the NlpSolver

  prob.append_variables(x);
  prob.append_variables(y);
  prob.append_variables(z);
  
  prob.append_objterm(new Ex1SingleVarQuadrObjTerm("xsquare", x));
  prob.append_objterm(new Ex1TwoVarsQuadrObjTerm("yzsquare", y, z));

  prob.append_constraints(new Ex1SumOfSquaresConstraints("sumxsquare", x));
  prob.append_constraints(new Ex1Constraint2("constraint2", x, y, z));

  prob.use_nlp_solver("ipopt");
  //set options
  prob.set_solver_option("linear_solver", "ma57");
  prob.set_solver_option("print_timing_statistics", "yes");
  //prob.set_solver_option("print_level", 6);

  bool bret = prob.optimize("ipopt");

  //set initial mu to something low when restarting
  prob.set_solver_option("mu_init", 1e-9);

  bret = prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  //
  //modify rhs of constraints2 from 0 to 0.1 and resolve
  OptConstraintsBlock* con2 = prob.get_constraints_block("constraint2");
  assert(con2);
  for(int i=0; i<con2->n; i++)
    con2->lb[i] = con2->ub[i] = 0.01;

  bret = prob.reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  return 0;
}

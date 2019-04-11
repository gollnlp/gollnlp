#include "Ex1OptObjects.hpp"
#include "IpoptSolver.hpp"


int main()
{
  OptProblem prob;
  int nx=100000, nyz=nx/5;
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

  bool bret = prob.optimize("ipopt");

  bret = prob.reoptimize(OptProblem::primalDualRestart);

  return 0;
}

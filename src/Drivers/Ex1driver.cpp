#include "Ex1OptObjects.hpp"
#include "IpoptSolver.hpp"


int main()
{
  OptProblem prob;
  OptVariablesBlock* x = new OptVariablesBlock(3, "x");
  OptVariablesBlock* y = new OptVariablesBlock(5, "y");
  OptVariablesBlock* z = new OptVariablesBlock(5, "z");
  prob.append_variables(x);
  prob.append_variables(y);
  prob.append_variables(z);
  
  prob.append_objterm(new Ex1SingleVarQuadrObjTerm("xsquare", x));
  //prob.append_objterm(new Ex1TwoVarsQuadrObjTerm("yzsquare", y, z));

  prob.append_constraints(new Ex1SumOfSquaresConstraints("sumxsquare", x));
  //prob.append_constraints(new Ex1Constraint2("constraint2", x, y, z));

  bool bret = prob.optimize("ipopt");

  return 0;
}

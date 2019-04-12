#include "go_code1.hpp"

#include "OptProblem.hpp"

#include "SCACOPFData.hpp"
#include "OPFConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

using namespace std;
using namespace gollnlp;

int myexe1_function(const std::string& InFile1, const std::string& InFile2,
		    const std::string& InFile3, const std::string& InFile4,
		    double TimeLimitInSeconds, 
		    int ScoringMethod, 
		    const std::string& NetworkModel)
{
  SCACOPFData d;
  d.readinstance(InFile1, InFile2, InFile3, InFile4);
  d.buildindexsets();

  OptProblem prob;
  
  //!starting point
  auto v_n = new OptVariablesBlock(d.N_Bus.size(), "v_n", d.N_Vlb.data(), d.N_Vub.data()); 
  prob.append_variables(v_n);
  prob.append_objterm(new DummySingleVarQuadrObjTerm("v_n_sq", v_n));

  auto theta_n = new OptVariablesBlock(d.N_Bus.size(), "theta_n", d.N_Vlb.data(), d.N_Vub.data());
  prob.append_variables(theta_n);
  prob.append_objterm(new DummySingleVarQuadrObjTerm("theta_n_sq", theta_n));

  auto p_li = new OptVariablesBlock(2*d.L_Line.size(), "p_li");
  auto q_li = new OptVariablesBlock(2*d.L_Line.size(), "q_li");
  prob.append_variables(p_li); prob.append_variables(q_li);
  prob.append_objterm(new DummySingleVarQuadrObjTerm("p_li_sq", p_li));
  prob.append_objterm(new DummySingleVarQuadrObjTerm("q_li_sq", q_li));

  auto p_ti = new OptVariablesBlock(2*d.T_Transformer.size(), "p_ti");
  auto q_ti = new OptVariablesBlock(2*d.T_Transformer.size(), "q_ti");
  prob.append_variables(p_ti); prob.append_variables(q_ti);
  prob.append_objterm(new DummySingleVarQuadrObjTerm("p_ti_sq", p_ti));
  prob.append_objterm(new DummySingleVarQuadrObjTerm("q_ti_sq", q_ti));

  auto SSh = new OptVariablesBlock(d.SSh_SShunt.size(), "SSh", d.SSh_Blb.data(), d.SSh_Bub.data());
  prob.append_variables(SSh);
  prob.append_objterm(new DummySingleVarQuadrObjTerm("SSh_sq", SSh));

  auto p_g = new OptVariablesBlock(d.G_Generator.size(), "p_g", d.G_Plb.data(), d.G_Pub.data());
  auto q_g = new OptVariablesBlock(d.G_Generator.size(), "q_g", d.G_Qlb.data(), d.G_Qub.data());
  prob.append_variables(p_g); prob.append_variables(q_g); 
  prob.append_objterm(new DummySingleVarQuadrObjTerm("p_g_sq", p_g));
  prob.append_objterm(new DummySingleVarQuadrObjTerm("q_g_sq", q_g));


  //
  //constraints
  //
  prob.append_constraints(new PFConRectangular("p_li_powerflow", p_li, v_n, theta_n, d));

  prob.use_nlp_solver("ipopt");
  //set options
  prob.set_solver_option("linear_solver", "ma57");
  prob.set_solver_option("print_timing_statistics", "yes");
  //prob.set_solver_option("print_level", 6);
  bool bret = prob.optimize("ipopt");
  
  cout << "end" << endl;


  return 0;
}

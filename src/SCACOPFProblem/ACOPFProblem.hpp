#ifndef ACOPF_PROBLEM
#define ACOPF_PROBLEM

#include "OptProblem.hpp"

#include "SCACOPFData.hpp"
#include "OPFConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

#include <cstring>
#include "blasdefs.hpp"
//this class is for ACOPF base case and is inherited by ACOPFContingencyProblem

namespace gollnlp {
  
  class ACOPFProblem : public OptProblem
  {
  public:
    ACOPFProblem(SCACOPFData& d_in) : d(d_in) {}
    OptProblem opt_prob;
    
    virtual bool default_assembly()
    {
      //!starting point
      auto v_n = new OptVariablesBlock(d.N_Bus.size(), "v_n", d.N_Vlb.data(), d.N_Vub.data()); 
      append_variables(v_n);
      v_n->set_start_to(d.N_v0.data());
      append_objterm(new DummySingleVarQuadrObjTerm("v_n_sq", v_n));
      
      auto theta_n = new OptVariablesBlock(d.N_Bus.size(), "theta_n");
      append_variables(theta_n);
      theta_n->set_start_to(d.N_theta0.data());
      append_objterm(new DummySingleVarQuadrObjTerm("theta_n_sq", theta_n));
      
      auto p_li1 = new OptVariablesBlock(d.L_Line.size(), "p_li1", -3, 4);
      append_variables(p_li1);

      auto p_li2 = new OptVariablesBlock(d.L_Line.size(), "p_li2");
       append_variables(p_li2);
      auto q_li1 = new OptVariablesBlock(d.L_Line.size(), "q_li1");
      auto q_li2 = new OptVariablesBlock(d.L_Line.size(), "q_li2");

      append_variables(q_li1); append_variables(q_li2);
      append_objterm(new DummySingleVarQuadrObjTerm("p_li1_sq", p_li1));
      append_objterm(new DummySingleVarQuadrObjTerm("p_li2_sq", p_li2));
      append_objterm(new DummySingleVarQuadrObjTerm("q_li1_sq", q_li1));
      append_objterm(new DummySingleVarQuadrObjTerm("q_li2_sq", q_li2));

      auto p_ti1 = new OptVariablesBlock(d.T_Transformer.size(), "p_t1i");
      auto p_ti2 = new OptVariablesBlock(d.T_Transformer.size(), "p_ti2");
      auto q_ti1 = new OptVariablesBlock(d.T_Transformer.size(), "q_ti1");
      auto q_ti2 = new OptVariablesBlock(d.T_Transformer.size(), "q_ti2");
      append_variables(p_ti1); append_variables(p_ti2); 
      append_variables(q_ti1); append_variables(q_ti2); 
      append_objterm(new DummySingleVarQuadrObjTerm("p_ti1_sq", p_ti1));
      append_objterm(new DummySingleVarQuadrObjTerm("p_ti2_sq", p_ti1));
      append_objterm(new DummySingleVarQuadrObjTerm("q_ti1_sq", q_ti1));
      append_objterm(new DummySingleVarQuadrObjTerm("q_ti2_sq", q_ti2));
      
      auto SSh = new OptVariablesBlock(d.SSh_SShunt.size(), "SSh", d.SSh_Blb.data(), d.SSh_Bub.data());
      SSh->set_start_to(d.SSh_B0.data());
      append_variables(SSh);
      append_objterm(new DummySingleVarQuadrObjTerm("SSh_sq", SSh));
      
      auto p_g = new OptVariablesBlock(d.G_Generator.size(), "p_g", d.G_Plb.data(), d.G_Pub.data());
      auto q_g = new OptVariablesBlock(d.G_Generator.size(), "q_g", d.G_Qlb.data(), d.G_Qub.data());
      append_variables(p_g); append_variables(q_g); 
      p_g->set_start_to(d.G_p0.data());
      q_g->set_start_to(d.G_q0.data());
      append_objterm(new DummySingleVarQuadrObjTerm("p_g_sq", p_g));
      append_objterm(new DummySingleVarQuadrObjTerm("q_g_sq", q_g));
      
      //
      //constraints
      //
      int one=1; double zero=1.;

      auto pf_cons = new PFConRectangular("p_li1_powerflow", d.L_Line.size(), 
					  p_li1, v_n, theta_n,
					  d.L_Nidx[0], d.L_Nidx[1]);
      //set the coefficients directly
      DCOPY(&(pf_cons->n), d.L_G.data(), &one, pf_cons->get_A(), &one);

      double *B=pf_cons->get_B(), *LG=d.L_G.data();
      for(int i=0; i<pf_cons->n; i++) B[i]=-LG[i];

      double *C=pf_cons->get_C(), *LB=d.L_B.data();
      for(int i=0; i<pf_cons->n; i++) C[i]=-LB[i];

      double* T=pf_cons->get_T();
      for(int i=0; i<pf_cons->n; i++) T[i]=0.;
      
      append_constraints(pf_cons);

      use_nlp_solver("ipopt");
      //set options
      set_solver_option("linear_solver", "ma57");
      set_solver_option("print_timing_statistics", "yes");
      set_solver_option("max_iter", 20);
      //prob.set_solver_option("print_level", 6);
      bool bret = optimize("ipopt");

      return true;
    }

  protected: 
    SCACOPFData& d;
  };

}

#endif

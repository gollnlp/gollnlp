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
      append_objterm(new DummySingleVarQuadrObjTerm("p_li1_sq", p_li1));
      append_objterm(new DummySingleVarQuadrObjTerm("p_li2_sq", p_li2));

      auto q_li1 = new OptVariablesBlock(d.L_Line.size(), "q_li1");
      auto q_li2 = new OptVariablesBlock(d.L_Line.size(), "q_li2");
      append_variables(q_li1); append_variables(q_li2);
      append_objterm(new DummySingleVarQuadrObjTerm("q_li1_sq", q_li1));
      append_objterm(new DummySingleVarQuadrObjTerm("q_li2_sq", q_li2));
 
      auto p_ti1 = new OptVariablesBlock(d.T_Transformer.size(), "p_t1i");
      auto p_ti2 = new OptVariablesBlock(d.T_Transformer.size(), "p_ti2");
      append_variables(p_ti1); 
      append_objterm(new DummySingleVarQuadrObjTerm("p_ti1_sq", p_ti1));
      append_variables(p_ti2); 
      append_objterm(new DummySingleVarQuadrObjTerm("p_ti2_sq", p_ti2));

      auto q_ti1 = new OptVariablesBlock(d.T_Transformer.size(), "q_ti1");
      auto q_ti2 = new OptVariablesBlock(d.T_Transformer.size(), "q_ti2");
      append_variables(q_ti1); append_variables(q_ti2); 
      append_objterm(new DummySingleVarQuadrObjTerm("q_ti1_sq", q_ti1));
      append_objterm(new DummySingleVarQuadrObjTerm("q_ti2_sq", q_ti2));
      
      auto b_s = new OptVariablesBlock(d.SSh_SShunt.size(), "b_s", d.SSh_Blb.data(), d.SSh_Bub.data());
      b_s->set_start_to(d.SSh_B0.data());
      append_variables(b_s);
      append_objterm(new DummySingleVarQuadrObjTerm("b_s_sq", b_s));
      
      auto p_g = new OptVariablesBlock(d.G_Generator.size(), "p_g", d.G_Plb.data(), d.G_Pub.data());
      append_variables(p_g); 
      p_g->set_start_to(d.G_p0.data());
      append_objterm(new DummySingleVarQuadrObjTerm("p_g_sq", p_g));

      auto q_g = new OptVariablesBlock(d.G_Generator.size(), "q_g", d.G_Qlb.data(), d.G_Qub.data());
      q_g->set_start_to(d.G_q0.data());
      append_variables(q_g); 
      append_objterm(new DummySingleVarQuadrObjTerm("q_g_sq", q_g));
    
      //
      //constraints
      //
      int one=1; double zero=1., neghalf=-0.5;;
      // lines power flows
      {
	// i=1 addpowerflowcon!(m, p_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], L[:G][l], -L[:G][l], -L[:B][l])
	auto pf_cons1 = new PFConRectangular("p_li1_powerflow", d.L_Line.size(), 
					     p_li1, v_n, theta_n,
					     d.L_Nidx[0], d.L_Nidx[1]);
	// i=2 addpowerflowcon!(m, p_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], L[:G][l], -L[:G][l], -L[:B][l])
	auto pf_cons2 = new PFConRectangular("p_li2_powerflow", d.L_Line.size(), 
					     p_li2, v_n, theta_n,
					     d.L_Nidx[1], d.L_Nidx[0]);
	//set the coefficients directly
	DCOPY(&(pf_cons1->n), d.L_G.data(), &one, pf_cons1->get_A(), &one);
	DCOPY(&(pf_cons2->n), d.L_G.data(), &one, pf_cons2->get_A(), &one);
	
	double *B=pf_cons1->get_B(), *LG=d.L_G.data();
	for(int i=0; i<pf_cons1->n; i++) B[i]=-LG[i];
	DCOPY(&(pf_cons2->n), B, &one, pf_cons2->get_B(), &one);

	double *C=pf_cons1->get_C(), *LB=d.L_B.data();
	for(int i=0; i<pf_cons1->n; i++) C[i]=-LB[i];
	DCOPY(&(pf_cons2->n), C, &one, pf_cons2->get_C(), &one);
	
	double* T=pf_cons1->get_T();
	for(int i=0; i<pf_cons1->n; i++) T[i]=0.;
	DCOPY(&(pf_cons2->n), T, &one, pf_cons2->get_T(), &one);
	
	append_constraints(pf_cons1);
	append_constraints(pf_cons2);
      }
      {
	// i=1 addpowerflowcon!(m, q_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], 
	//                         theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], 
	//                          -L[:B][l]-L[:Bch][l]/2, L[:B][l], -L[:G][l])
	auto pf_cons1 = new PFConRectangular("q_li1_powerflow", d.L_Line.size(), 
					     q_li1, v_n, theta_n,
					     d.L_Nidx[0], d.L_Nidx[1]);
	// i=2 
	auto pf_cons2 = new PFConRectangular("q_li2_powerflow", d.L_Line.size(), 
					     q_li2, v_n, theta_n,
					     d.L_Nidx[1], d.L_Nidx[0]);
	
	//set the coefficients directly
	double *A=pf_cons1->get_A(), *LB=d.L_B.data();
	for(int i=0; i<pf_cons1->n; i++) A[i]=-LB[i];
	// A += -0.5*L_Bch
	DAXPY(&(pf_cons1->n), &neghalf, d.L_Bch.data(), &one, A, &one); 
	DCOPY(&(pf_cons2->n), A, &one, pf_cons2->get_A(), &one);

	
	DCOPY(&(pf_cons1->n), d.L_B.data(), &one, pf_cons1->get_B(), &one);
	DCOPY(&(pf_cons2->n), d.L_B.data(), &one, pf_cons2->get_B(), &one);

	double *C=pf_cons1->get_C(), *LG=d.L_G.data();
	for(int i=0; i<pf_cons1->n; i++) C[i]=-LG[i];
	DCOPY(&(pf_cons2->n), C, &one, pf_cons2->get_C(), &one);
	
	double* T=pf_cons1->get_T();
	for(int i=0; i<pf_cons1->n; i++) T[i]=0.;
	DCOPY(&(pf_cons2->n), T, &one, pf_cons2->get_T(), &one);
	
	append_constraints(pf_cons1);
	append_constraints(pf_cons2);
      }
      // transformers power flows
      {
	// i=1 addpowerflowcon!(m, p_ti[t,1], v_n[T_Nidx[t,1]], v_n[T_Nidx[t,2]],
	//		theta_n[T_Nidx[t,1]], theta_n[T_Nidx[t,2]],
	//		T[:G][t]/T[:Tau][t]^2+T[:Gm][t], -T[:G][t]/T[:Tau][t], -T[:B][t]/T[:Tau][t], -T[:Theta][t])
	auto pf_cons1 = new PFConRectangular("p_ti1_powerflow", d.T_Transformer.size(), 
					     p_ti1, v_n, theta_n,
					     d.T_Nidx[0], d.T_Nidx[1]);
	//set the coefficients directly
	double *A = pf_cons1->get_A(), *TG=d.T_G.data(), *TTau=d.T_Tau.data();
	DCOPY(&(pf_cons1->n), d.T_Gm.data(), &one, A, &one);
	for(int t=0; t<pf_cons1->n; t++) 
	  A[t] += TG[t] / (TTau[t]*TTau[t]);

	double *B=pf_cons1->get_B();
	for(int t=0; t<pf_cons1->n; t++) 
	  B[t]=-TG[t]/TTau[t];
	
	double *C=pf_cons1->get_C(), *TB=d.T_B.data();
	for(int t=0; t<pf_cons1->n; t++) 
	  C[t]=-TB[t]/TTau[t];

	double *T=pf_cons1->get_T(), *TTheta=d.T_Theta.data();
	for(int t=0; t<pf_cons1->n; t++) 
	  T[t] = -TTheta[t];

	// i=2 addpowerflowcon!(m, p_ti[t,2], v_n[T_Nidx[t,2]], v_n[T_Nidx[t,1]],
	//		theta_n[T_Nidx[t,2]], theta_n[T_Nidx[t,1]],
	//		T[:G][t], -T[:G][t]/T[:Tau][t], -T[:B][t]/T[:Tau][t], T[:Theta][t])
	auto pf_cons2 = new PFConRectangular("p_ti2_powerflow", d.T_Transformer.size(), 
					     p_ti2, v_n, theta_n,
					     d.T_Nidx[1], d.T_Nidx[0]);
	//set the coefficients directly
	DCOPY(&(pf_cons2->n), d.T_G.data(), &one, pf_cons2->get_A(), &one);
	DCOPY(&(pf_cons2->n), pf_cons1->get_B(), &one, pf_cons2->get_B(), &one);
	DCOPY(&(pf_cons2->n), pf_cons1->get_C(), &one, pf_cons2->get_C(), &one);	
	DCOPY(&(pf_cons2->n), TTheta, &one, pf_cons2->get_T(), &one);
	
	append_constraints(pf_cons1);
	append_constraints(pf_cons2);
      }
      {
      	// i=1 addpowerflowcon!(m, q_ti[t,1], v_n[T_Nidx[t,1]], v_n[T_Nidx[t,2]],
	//		theta_n[T_Nidx[t,1]], theta_n[T_Nidx[t,2]],
	//		-T[:B][t]/T[:Tau][t]^2-T[:Bm][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], -T[:Theta][t])
      	auto pf_cons1 = new PFConRectangular("q_ti1_powerflow", d.T_Transformer.size(), 
      					     q_ti1, v_n, theta_n,
      					     d.T_Nidx[0], d.T_Nidx[1]);
      	//set the coefficients directly
	double *A=pf_cons1->get_A(), *TB=d.T_B.data(), *TTau=d.T_Tau.data(), *TBM=d.T_Bm.data();
	for(int t=0; t<pf_cons1->n; t++) 
	  A[t] = -TBM[t]-TB[t]/(TTau[t]*TTau[t]);

	double *B=pf_cons1->get_B();
	for(int t=0; t<pf_cons1->n; t++) 
	  B[t] = TB[t]/TTau[t];

	double *C=pf_cons1->get_C(), *TG=d.T_G.data();
	for(int t=0; t<pf_cons1->n; t++) 
	  C[t] = -TG[t]/TTau[t];

	double* T=pf_cons1->get_T(), *TTheta=d.T_Theta.data();
	for(int t=0; t<pf_cons1->n; t++) T[t] = -TTheta[t];

      	// i=2 addpowerflowcon!(m, q_ti[t,2], v_n[T_Nidx[t,2]], v_n[T_Nidx[t,1]],
	//		theta_n[T_Nidx[t,2]], theta_n[T_Nidx[t,1]],
	//		-T[:B][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], T[:Theta][t])
      	auto pf_cons2 = new PFConRectangular("q_ti2_powerflow", d.T_Transformer.size(), 
      					     q_ti2, v_n, theta_n,
      					     d.T_Nidx[1], d.T_Nidx[0]);
	
	A=pf_cons2->get_A();
      	for(int i=0; i<pf_cons1->n; i++) A[i]=-TB[i];
      	DCOPY(&(pf_cons2->n), pf_cons1->get_B(), &one, pf_cons2->get_B(), &one);
      	DCOPY(&(pf_cons2->n), pf_cons1->get_C(), &one, pf_cons2->get_C(), &one);
	DCOPY(&(pf_cons2->n), TTheta, &one, pf_cons2->get_T(), &one);
	
      	append_constraints(pf_cons1);
      	append_constraints(pf_cons2);
      }
      {
	//active power balance
	auto pf_p_bal = new PFActiveBalance("p_balance", d.N_Bus.size(), p_g, v_n, p_li1, p_li2, p_ti1, p_ti2, d);
	append_constraints(pf_p_bal);
	//active power balance
	auto pf_q_bal = new PFReactiveBalance("q_balance", d.N_Bus.size(), q_g, v_n, q_li1, q_li2, q_ti1, q_ti2, b_s, d);
	append_constraints(pf_q_bal);

      }

      {
	//thermal line limits
	auto pf_line_lim1 = new PFLineLimits("line_limits1", d.L_Line.size(),
					     p_li1, q_li1, v_n, 
					     d.L_Nidx[0], d.L_RateBase, d);
	auto pf_line_lim2 = new PFLineLimits("line_limits2", d.L_Line.size(),
					     p_li2, q_li2, v_n, 
					     d.L_Nidx[1], d.L_RateBase, d);
	append_constraints(pf_line_lim1);

	vars_block("sslack_li_line_limits1")->set_start_to(0.1);
	append_constraints(pf_line_lim2);
      }

      use_nlp_solver("ipopt");
      //set options
      set_solver_option("linear_solver", "ma57");
      //set_solver_option("print_timing_statistics", "yes");
      set_solver_option("max_iter", 150);
      //prob.set_solver_option("print_level", 6);
      bool bret = optimize("ipopt");
      
      //this->problem_changed();
      //set_solver_option("max_iter", 200);
      //bret = optimize("ipopt");

      return true;
    }

  protected: 
    SCACOPFData& d;
  };

}

#endif

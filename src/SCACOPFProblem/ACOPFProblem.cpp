#include "ACOPFProblem.hpp"

#include "OPFConstraints.hpp"
#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

#include <numeric>
#include <typeinfo> 

using namespace std;

namespace gollnlp {

ACOPFProblem::~ACOPFProblem()
{
  for(auto d: data_K)
    delete d;
}

bool ACOPFProblem::default_assembly()
{
  useQPen = true;
  slacks_scale = 128.;

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

  //
  // contingencies
  //

  //vector<int> K_Cont = {8, 83, 366}; //net 01; gen, line, transf, id out=[27,61,126]
  //vector<int> K_Cont = {0, 71, 85, 97, 98}; //net 03 scen 9
  //vector<int> K_Cont ={0, 386, 428, 435}; //net 10 scen 9; first two are gen, then a line and a trans
  vector<int> K_Cont ={435};
  int nK = K_Cont.size();
  //for(auto K : data_sc.K_Contingency) {
  for(auto K : K_Cont) {
    bool SysCond_BaseCase = false;
    data_K.push_back(new SCACOPFData(data_sc));
    SCACOPFData& dK = *(data_K).back(); //shortcut
    dK.rebuild_for_conting(K,nK);

    printf("adding blocks for contingency K=%d IDOut=%d outidx=%d Type=%s\n", 
	   K, d.K_IDout[K], d.K_outidx[K], d.cont_type_string(K).c_str());

    add_variables(dK);
    add_cons_lines_pf(dK);
    add_cons_transformers_pf(dK);
    add_cons_active_powbal(dK);
    add_cons_reactive_powbal(dK);
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    //coupling AGC and PVPQ; also creates delta_k
    add_cons_coupling(dK);
  }

  //print_summary();

  use_nlp_solver("ipopt");
  //set options
  set_solver_option("linear_solver", "ma57");
  set_solver_option("mu_init", 1.);
  //set_solver_option("print_timing_statistics", "yes");
  set_solver_option("max_iter", 1000);
  //prob.set_solver_option("print_level", 6);
  bool bret = optimize("ipopt");

  
  //print_p_g(data_sc);
  //for(auto d: data_K)
  //  print_p_g_with_coupling_info(*d);
  
      
  //this->problem_changed();
  //t_solver_option("max_iter", 50);
  //set_solver_option("mu_init", 1e-2);
  //bret = optimize("ipopt");

  //bret = reoptimize(OptProblem::primalDualRestart); //warm_start_target_mu

  return true;
}
void ACOPFProblem::add_cons_coupling(SCACOPFData& dB)
{
  int K_id = dB.K_Contingency[0];

  //indexes in data_sc.G_Generator
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  assert(Gk.size() == dB.G_Generator.size());

  add_cons_nonanticip(dB, Gknop);
  add_cons_AGC(dB, Gkp);

  //voltages
  add_cons_PVPQ(dB, Gk);
}

// Gk are the indexes of all gens other than the outgen (for generator contingencies) 
// in data_sc.G_Generator
void ACOPFProblem::add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk)
{
  printvec(Gk);
  auto G_Nidx_Gk = selectfrom(dB.G_Nidx, Gk);
  assert(G_Nidx_Gk == dB.G_Nidx);
  printvec(G_Nidx_Gk);
  sort(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  printvec(G_Nidx_Gk);
  auto last = unique(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  G_Nidx_Gk.erase(last, G_Nidx_Gk.end());
  printvec(G_Nidx_Gk);
  auto &N_PVPQ = G_Nidx_Gk; //nodes with PVPQ generators;

  //generators at each node with PVPQ generators
  vector<vector<int> > gen_agg;
  vector<double> Qlb, Qub;
  int nPVPQGens=0, nPVPQCons=0;

  for(auto n: N_PVPQ) {
    assert(dB.Gn[n].size()>0);
    double Qagglb=0., Qaggub=0.;

    int numfixed = 0;
    gen_agg.push_back( vector<int>() );
    for(auto g: dB.Gn[n]) {
#ifdef DEBUG
      assert(dB.K_Contingency.size()==1);
      assert(dB.K_outidx.size()==1);
      if(dB.K_ConType[0]==SCACOPFData::kGenerator) 
	assert(data_sc.G_Generator[dB.K_outidx[0]]!=dB.G_Generator[g]);
#endif
      if(abs(dB.G_Qub[g]-dB.G_Qlb[g])<=1e-8) {
	numfixed++;
	printf("PVPQ: gen ID=%d p_q is fixed; will not add PVPQ constraint\n");
	continue;
      }
      gen_agg.back().push_back(g);
      Qagglb += dB.G_Qlb[g];
      Qaggub += dB.G_Qub[g];
    }
    assert(gen_agg.back().size()+numfixed == dB.Gn[n].size());
    nPVPQGens += gen_agg.back().size()+numfixed;
    Qlb.push_back(Qagglb);
    Qub.push_back(Qaggub);
  }
  assert(gen_agg.size()==Qlb.size());
  assert(gen_agg.size()==Qub.size());
  assert(N_PVPQ.size()==gen_agg.size());

  


  for(int i=0; i<Qlb.size(); i++) {
    printf("%d %g %g \n", N_PVPQ[i], Qlb[i], Qub[i]);
    printvec(gen_agg[i]);
  }
  //printvecvec(gen_agg, "active generators");
  //printvec(Qlb, "Qlb");
  //printvec(Qub, "Qub");
}
void ACOPFProblem::add_cons_nonanticip(SCACOPFData& dB, const std::vector<int>& G_idxs_no_AGC)
{
  if(G_idxs_no_AGC.size()>0) {

    OptVariablesBlock* pg0 = variable("p_g", data_sc);
    if(NULL==pg0) {
      printf("Contingency %d: p_g var not found in the base case; will add NO nonanticip coupling constraints.\n", dB.id);
      return;
    }
    OptVariablesBlock* pgK = variable("p_g", dB);
    if(NULL==pgK) {
      printf("Contingency %d: p_g var not found in conting problem; will add NO nonanticip coupling constraints.\n", dB.id);
      return;
    }
    
    auto ids_no_AGC = selectfrom(data_sc.G_Generator, G_idxs_no_AGC);
    auto conting_matching_idxs = indexin(dB.G_Generator, ids_no_AGC);
    conting_matching_idxs = findall(conting_matching_idxs, [](int val) {return val!=-1;});
#ifdef DEBUG
    assert(G_idxs_no_AGC.size() == conting_matching_idxs.size());
    for(int i0=0, iK=0; i0<G_idxs_no_AGC.size(); i0++, iK++) {
      //all dB.G_Generator should be in data_sc.G_Generator
      assert(conting_matching_idxs[iK]>=0); 
      //all ids should match in order
      assert(dB.G_Generator[conting_matching_idxs[iK]] == data_sc.G_Generator[G_idxs_no_AGC[i0]]);
    }
#endif
    
    auto cons = new NonAnticipCons(con_name("non_anticip",dB), G_idxs_no_AGC.size(),
				   pg0, pgK, G_idxs_no_AGC, conting_matching_idxs);
    append_constraints(cons);
  }
  printf("AGC: %d gens NOT participating: added one nonanticip constraint for each\n", G_idxs_no_AGC.size());
}

void ACOPFProblem::add_cons_AGC(SCACOPFData& dB, const std::vector<int>& G_idxs_AGC)
{
  if(G_idxs_AGC.size()==0) {
    printf("AGC: NO gens participating !?! in contingency %d\n", dB.id);
    return;
  }
  auto ids_agc = selectfrom(data_sc.G_Generator, G_idxs_AGC);
  auto conting_matching_idxs = indexin(dB.G_Generator, ids_agc);
  conting_matching_idxs = findall(conting_matching_idxs, [](int val) {return val!=-1;});
#ifdef DEBUG
  assert(G_idxs_AGC.size() == conting_matching_idxs.size());
  for(int i0=0, iK=0; i0<G_idxs_AGC.size(); i0++, iK++) {
    //all dB.G_Generator should be in data_sc.G_Generator
    assert(conting_matching_idxs[iK]>=0); 
    //all ids should match in order
    assert(dB.G_Generator[conting_matching_idxs[iK]] == data_sc.G_Generator[G_idxs_AGC[i0]]);
  }
  //printvec(conting_matching_idxs, "conting gen idxs");
  //printvec(G_idxs_AGC, "base case gen idxs");
#endif
  OptVariablesBlock* pg0 = variable("p_g", data_sc);
  if(NULL==pg0) {
    printf("Contingency %d: p_g var not found in the base case; will NOT add AGC coupling constraints.\n", dB.id);
    return;
  }
  OptVariablesBlock* pgK = variable("p_g", dB);
  if(NULL==pgK) {
    printf("Contingency %d: p_g var not found in conting problem; will NOT add AGC coupling constraints.\n", dB.id);
    return;
  }

  OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dB));
  append_variables(deltaK);
  deltaK->set_start_to(0.);

  double AGCSmoothing = 1e-5;
  auto cons = new AGCComplementarityCons(con_name("AGC", dB), 3*G_idxs_AGC.size(),
					 pg0, pgK, deltaK, 
					 G_idxs_AGC, conting_matching_idxs,
					 selectfrom(data_sc.G_Plb, G_idxs_AGC), selectfrom(data_sc.G_Pub, G_idxs_AGC),
					 data_sc.G_alpha,
					 AGCSmoothing); 
  append_constraints(cons);

  //starting point for rhop and rhom that were added by AGCComplementarityCons
  auto rhop=cons->get_rhop(), rhom=cons->get_rhom();
  cons->compute_rhos(rhop, rhom);
  rhop->providesStartingPoint=true; rhom->providesStartingPoint=true;

  //for(int i=0; i<rhop->n; i++) 
  //  printf("%g %g   %g\n", rhop->x[i], rhom->x[i], cons->gb[i]);
  //printf("\n");

  printf("AGC: %d gens participating: added %d constraints\n", G_idxs_AGC.size(), cons->n);
}
  
void ACOPFProblem::add_variables(SCACOPFData& d)
{
  auto v_n = new OptVariablesBlock(data_sc.N_Bus.size(), var_name("v_n",d), 
				   data_sc.N_Vlb.data(), data_sc.N_Vub.data()); 
  append_variables(v_n);
  v_n->set_start_to(data_sc.N_v0.data());
  //v_n->print();

  auto theta_n = new OptVariablesBlock(data_sc.N_Bus.size(), var_name("theta_n",d));
  append_variables(theta_n);
  theta_n->set_start_to(data_sc.N_theta0.data());
  int RefBus = data_sc.bus_with_largest_gen();

  theta_n->lb[RefBus] = theta_n->ub[RefBus] = data_sc.N_theta0[RefBus];
  if(data_sc.N_theta0[RefBus]!=0) {
    printf("We should set theta at RefBus to 0");
  }

  auto p_li1 = new OptVariablesBlock(d.L_Line.size(), var_name("p_li1",d));
  append_variables(p_li1);
  auto p_li2 = new OptVariablesBlock(d.L_Line.size(), var_name("p_li2",d));
  append_variables(p_li2);

  auto q_li1 = new OptVariablesBlock(d.L_Line.size(), var_name("q_li1",d));
  auto q_li2 = new OptVariablesBlock(d.L_Line.size(), var_name("q_li2",d));
  append_variables(q_li1); 
  append_variables(q_li2);
 
  auto p_ti1 = new OptVariablesBlock(d.T_Transformer.size(), var_name("p_ti1",d));
  auto p_ti2 = new OptVariablesBlock(d.T_Transformer.size(), var_name("p_ti2",d));
  append_variables(p_ti1); 
  append_variables(p_ti2); 

  auto q_ti1 = new OptVariablesBlock(d.T_Transformer.size(), var_name("q_ti1",d));
  auto q_ti2 = new OptVariablesBlock(d.T_Transformer.size(), var_name("q_ti2",d));
  append_variables(q_ti1); append_variables(q_ti2); 
      
  auto b_s = new OptVariablesBlock(data_sc.SSh_SShunt.size(), var_name("b_s",d), 
  				   data_sc.SSh_Blb.data(), data_sc.SSh_Bub.data());
  b_s->set_start_to(data_sc.SSh_B0.data());
  append_variables(b_s);

  auto p_g = new OptVariablesBlock(d.G_Generator.size(), var_name("p_g",d), 
				   d.G_Plb.data(), d.G_Pub.data());
  append_variables(p_g); 
  p_g->set_start_to(d.G_p0.data());
  //append_objterm(new DummySingleVarQuadrObjTerm("p_g_sq", p_g));

  auto q_g = new OptVariablesBlock(d.G_Generator.size(), var_name("q_g",d), 
  				   d.G_Qlb.data(), d.G_Qub.data());
  q_g->set_start_to(d.G_q0.data());
  append_variables(q_g); 
}

void ACOPFProblem::add_cons_lines_pf(SCACOPFData& d)
{
  auto p_li1 = variable("p_li1",d), p_li2 = variable("p_li2",d);
  auto v_n = variable("v_n",d), theta_n = variable("theta_n",d);
  //
  // active power power flow constraints
  //
  // i=1 addpowerflowcon!(m, p_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], L[:G][l], -L[:G][l], -L[:B][l])
  auto pf_cons1 = new PFConRectangular(con_name("p_li1_powerflow",d), 
				       d.L_Line.size(), 
				       p_li1, v_n, theta_n,
				       d.L_Nidx[0], d.L_Nidx[1]);
  // i=2 addpowerflowcon!(m, p_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], L[:G][l], -L[:G][l], -L[:B][l])
  auto pf_cons2 = new PFConRectangular(con_name("p_li2_powerflow",d),
				       d.L_Line.size(), 
				       p_li2, v_n, theta_n,
				       d.L_Nidx[1], d.L_Nidx[0]);
  //set the coefficients directly
  DCOPY(&(pf_cons1->n), d.L_G.data(), &ione, pf_cons1->get_A(), &ione);
  DCOPY(&(pf_cons2->n), d.L_G.data(), &ione, pf_cons2->get_A(), &ione);
  
  double *B=pf_cons1->get_B(), *LG=d.L_G.data();
  for(int i=0; i<pf_cons1->n; i++) B[i]=-LG[i];
  DCOPY(&(pf_cons2->n), B, &ione, pf_cons2->get_B(), &ione);
  
  double *C=pf_cons1->get_C(), *LB=d.L_B.data();
  for(int i=0; i<pf_cons1->n; i++) C[i]=-LB[i];
  DCOPY(&(pf_cons2->n), C, &ione, pf_cons2->get_C(), &ione);
  
  double* T=pf_cons1->get_T();
  for(int i=0; i<pf_cons1->n; i++) T[i]=0.;
  DCOPY(&(pf_cons2->n), T, &ione, pf_cons2->get_T(), &ione);
  
  append_constraints(pf_cons1);
  append_constraints(pf_cons2);
  
  //compute starting points
  pf_cons1->compute_power(p_li1); p_li1->providesStartingPoint=true;
  pf_cons2->compute_power(p_li2); p_li2->providesStartingPoint=true;


  auto q_li1 = variable("q_li1",d), q_li2 = variable("q_li2",d);
  //
  // reactive power power flow constraints
  //
  // i=1 addpowerflowcon!(m, q_li[l,i], v_n[L_Nidx[l,i]], v_n[L_Nidx[l,3-i]], 
  //                         theta_n[L_Nidx[l,i]], theta_n[L_Nidx[l,3-i]], 
  //                          -L[:B][l]-L[:Bch][l]/2, L[:B][l], -L[:G][l])
  pf_cons1 = new PFConRectangular(con_name("q_li1_powerflow",d), d.L_Line.size(), 
				  q_li1, v_n, theta_n,
				  d.L_Nidx[0], d.L_Nidx[1]);
  // i=2 
  pf_cons2 = new PFConRectangular(con_name("q_li2_powerflow",d), d.L_Line.size(), 
				  q_li2, v_n, theta_n,
				  d.L_Nidx[1], d.L_Nidx[0]);
  
  //set the coefficients directly
  double neghalf=-0.5;
  double *A=pf_cons1->get_A(); LB=d.L_B.data();
  for(int i=0; i<pf_cons1->n; i++) A[i]=-LB[i];
  // A += -0.5*L_Bch
  DAXPY(&(pf_cons1->n), &neghalf, d.L_Bch.data(), &ione, A, &ione); 
  DCOPY(&(pf_cons2->n), A, &ione, pf_cons2->get_A(), &ione);
  
  
  DCOPY(&(pf_cons1->n), d.L_B.data(), &ione, pf_cons1->get_B(), &ione);
  DCOPY(&(pf_cons2->n), d.L_B.data(), &ione, pf_cons2->get_B(), &ione);
  
  C=pf_cons1->get_C(); LG=d.L_G.data();
  for(int i=0; i<pf_cons1->n; i++) C[i]=-LG[i];
  DCOPY(&(pf_cons2->n), C, &ione, pf_cons2->get_C(), &ione);
  
  T=pf_cons1->get_T();
  for(int i=0; i<pf_cons1->n; i++) T[i]=0.;
  DCOPY(&(pf_cons2->n), T, &ione, pf_cons2->get_T(), &ione);
  
  append_constraints(pf_cons1);
  append_constraints(pf_cons2);
  pf_cons1->compute_power(q_li1); q_li1->providesStartingPoint=true;
  pf_cons2->compute_power(q_li2); q_li2->providesStartingPoint=true;
}
void ACOPFProblem::add_cons_transformers_pf(SCACOPFData& d)
{
  //
  // transformers active power flows
  //
  auto p_ti1 = variable("p_ti1",d), p_ti2 = variable("p_ti2",d);
  auto v_n = variable("v_n",d), theta_n = variable("theta_n",d);

  // i=1 addpowerflowcon!(m, p_ti[t,1], v_n[T_Nidx[t,1]], v_n[T_Nidx[t,2]],
  //		theta_n[T_Nidx[t,1]], theta_n[T_Nidx[t,2]],
  //		T[:G][t]/T[:Tau][t]^2+T[:Gm][t], -T[:G][t]/T[:Tau][t], -T[:B][t]/T[:Tau][t], -T[:Theta][t])
  auto pf_cons1 = new PFConRectangular(con_name("p_ti1_powerflow",d), d.T_Transformer.size(), 
				       p_ti1, v_n, theta_n,
				       d.T_Nidx[0], d.T_Nidx[1]);
  //set the coefficients directly
  double *A = pf_cons1->get_A(), *TG=d.T_G.data(), *TTau=d.T_Tau.data();
  DCOPY(&(pf_cons1->n), d.T_Gm.data(), &ione, A, &ione);
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
  auto pf_cons2 = new PFConRectangular(con_name("p_ti2_powerflow",d), d.T_Transformer.size(), 
				       p_ti2, v_n, theta_n,
				       d.T_Nidx[1], d.T_Nidx[0]);
  //set the coefficients directly
  DCOPY(&(pf_cons2->n), d.T_G.data(), &ione, pf_cons2->get_A(), &ione);
  DCOPY(&(pf_cons2->n), pf_cons1->get_B(), &ione, pf_cons2->get_B(), &ione);
  DCOPY(&(pf_cons2->n), pf_cons1->get_C(), &ione, pf_cons2->get_C(), &ione);	
  DCOPY(&(pf_cons2->n), TTheta, &ione, pf_cons2->get_T(), &ione);
  
  append_constraints(pf_cons1);
  append_constraints(pf_cons2);
  
  pf_cons1->compute_power(p_ti1); p_ti1->providesStartingPoint=true;
  pf_cons2->compute_power(p_ti2); p_ti2->providesStartingPoint=true;


  //
  // transformers reactive power flows
  //
  auto q_ti1 = variable("q_ti1",d), q_ti2 = variable("q_ti2",d);
  // i=1 addpowerflowcon!(m, q_ti[t,1], v_n[T_Nidx[t,1]], v_n[T_Nidx[t,2]],
  //		theta_n[T_Nidx[t,1]], theta_n[T_Nidx[t,2]],
  //		-T[:B][t]/T[:Tau][t]^2-T[:Bm][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], -T[:Theta][t])
  pf_cons1 = new PFConRectangular(con_name("q_ti1_powerflow",d), d.T_Transformer.size(), 
				  q_ti1, v_n, theta_n,
				  d.T_Nidx[0], d.T_Nidx[1]);
  //set the coefficients directly
  A=pf_cons1->get_A(); TB=d.T_B.data(); TTau=d.T_Tau.data(); double* TBM=d.T_Bm.data();
  for(int t=0; t<pf_cons1->n; t++) 
    A[t] = -TBM[t]-TB[t]/(TTau[t]*TTau[t]);
  
  B=pf_cons1->get_B();
  for(int t=0; t<pf_cons1->n; t++) 
    B[t] = TB[t]/TTau[t];
  
  C=pf_cons1->get_C(); TG=d.T_G.data();
  for(int t=0; t<pf_cons1->n; t++) 
    C[t] = -TG[t]/TTau[t];
  
  T=pf_cons1->get_T(); TTheta=d.T_Theta.data();
  for(int t=0; t<pf_cons1->n; t++) T[t] = -TTheta[t];
  
  // i=2 addpowerflowcon!(m, q_ti[t,2], v_n[T_Nidx[t,2]], v_n[T_Nidx[t,1]],
  //		theta_n[T_Nidx[t,2]], theta_n[T_Nidx[t,1]],
  //		-T[:B][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], T[:Theta][t])
  pf_cons2 = new PFConRectangular(con_name("q_ti2_powerflow",d), d.T_Transformer.size(), 
				  q_ti2, v_n, theta_n,
				  d.T_Nidx[1], d.T_Nidx[0]);
  
  A=pf_cons2->get_A();
  for(int i=0; i<pf_cons1->n; i++) A[i]=-TB[i];
  DCOPY(&(pf_cons2->n), pf_cons1->get_B(), &ione, pf_cons2->get_B(), &ione);
  DCOPY(&(pf_cons2->n), pf_cons1->get_C(), &ione, pf_cons2->get_C(), &ione);
  DCOPY(&(pf_cons2->n), TTheta, &ione, pf_cons2->get_T(), &ione);
  
  append_constraints(pf_cons1);
  append_constraints(pf_cons2);
  pf_cons1->compute_power(q_ti1); q_ti1->providesStartingPoint=true;
  pf_cons2->compute_power(q_ti2); q_ti2->providesStartingPoint=true;
}

void ACOPFProblem::add_cons_active_powbal(SCACOPFData& d)
{
  //!temp 
  bool useQPenActiveBalance = useQPen; //double slacks_scale=1.;

  //active power balance
  auto p_li1 = variable("p_li1",d), p_li2 = variable("p_li2",d), 
    p_ti1 = variable("p_ti1",d), p_ti2 = variable("p_ti2",d),
    p_g = variable("p_g",d), v_n = variable("v_n",d);

  auto pf_p_bal = new PFActiveBalance(con_name("p_balance",d), data_sc.N_Bus.size(), 
				      p_g, v_n, p_li1, p_li2, p_ti1, p_ti2, 
				      data_sc.N_Gsh, data_sc.N_Pd, 
				      d.Gn, d.Lidxn1, d.Lidxn2, d.Tidxn1, d.Tidxn2,
				      slacks_scale);
  append_constraints(pf_p_bal);

  //pslackm_n and pslackp_n
  OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
  pf_p_bal->compute_slacks(pslacks_n); pslacks_n->providesStartingPoint=true;
      
  if(useQPenActiveBalance) {
    append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + pslacks_n->id, pslacks_n, 
						    d.P_Penalties[SCACOPFData::pP], d.P_Quantities[SCACOPFData::pP], 
						    d.PenaltyWeight, slacks_scale) );
    
  } else {
    PFPenaltyAffineCons* cons_apb_pen = 
      new PFPenaltyAffineConsTwoSlacks(string("pcwslin_cons_") + pslacks_n->id, 
				       pf_p_bal->n, pslacks_n, 
				       d.P_Penalties[SCACOPFData::pP], 
				       d.P_Quantities[SCACOPFData::pP], 
				       d.PenaltyWeight, slacks_scale);
    append_constraints(cons_apb_pen);
    
    //sigmas for this block
    OptVariablesBlock* sigma = cons_apb_pen->get_sigma();
    cons_apb_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
  }
}

void ACOPFProblem::add_cons_reactive_powbal(SCACOPFData& d)
{
  //!temp 
  bool useQPenReactiveBalance = useQPen; //double slacks_scale=1.;

  auto q_li1 = variable("q_li1",d), q_li2 = variable("q_li2",d), 
    q_ti1 = variable("q_ti1",d), q_ti2 = variable("q_ti2",d),
    q_g = variable("q_g",d), v_n = variable("v_n",d),
    b_s = variable("b_s",d);
  

  //reactive power balance
  auto pf_q_bal = new PFReactiveBalance(con_name("q_balance",d), data_sc.N_Bus.size(), 
					q_g, v_n, q_li1, q_li2, q_ti1, q_ti2, b_s, 					
					data_sc.N_Bsh, data_sc.N_Qd,  
					d.Gn, d.SShn,
					d.Lidxn1, d.Lidxn2, d.Tidxn1, d.Tidxn2,
					slacks_scale);
  append_constraints(pf_q_bal);
  OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
  pf_q_bal->compute_slacks(qslacks_n); qslacks_n->providesStartingPoint=true;

  if(useQPenReactiveBalance) {
    append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + qslacks_n->id,
						    qslacks_n,
						    d.P_Penalties[SCACOPFData::pQ], 
						    d.P_Quantities[SCACOPFData::pQ], 
						    d.PenaltyWeight, slacks_scale) );
  } else {
    PFPenaltyAffineCons* cons_rpb_pen = 
      new PFPenaltyAffineConsTwoSlacks(string("pcwslin_cons_") + qslacks_n->id, 
				       pf_q_bal->n, qslacks_n, 
				       d.P_Penalties[SCACOPFData::pQ], 
				       d.P_Quantities[SCACOPFData::pQ], 
				       d.PenaltyWeight, slacks_scale);
    append_constraints(cons_rpb_pen);
    
    OptVariablesBlock* sigma = cons_rpb_pen->get_sigma();
    cons_rpb_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
  }
}

//
//thermal line limits
//
void ACOPFProblem::add_cons_thermal_li_lims(SCACOPFData& d, bool SysCond_BaseCase)
{
  //! temp
  bool useQPenLi1 = useQPen, useQPenLi2 = useQPen; //double slacks_scale=1.;

  auto v_n = variable("v_n",d);
  auto p_li1 = variable("p_li1", d), q_li1 = variable("q_li1", d);
  vector<double>& L_Rate = SysCond_BaseCase ? d.L_RateBase : d.L_RateEmer;
  {

    auto pf_line_lim1 = new PFLineLimits(con_name("line_limits1",d), d.L_Line.size(),
					 p_li1, q_li1, v_n, 
					 d.L_Nidx[0], L_Rate, slacks_scale);
    append_constraints(pf_line_lim1);
    
    //sslack_li1
    OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
    pf_line_lim1->compute_slacks(sslack_li1); sslack_li1->providesStartingPoint=true;

    if(useQPenLi1) {
      append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + sslack_li1->id,
						      sslack_li1,
						      d.P_Penalties[SCACOPFData::pS], 
						      d.P_Quantities[SCACOPFData::pS], 
						      d.PenaltyWeight, slacks_scale) );
    } else {

      PFPenaltyAffineCons* cons_li1_pen =
	new PFPenaltyAffineCons(string("pcwslin_cons_") + sslack_li1->id, sslack_li1->n, sslack_li1,
				d.P_Penalties[SCACOPFData::pS], d.P_Quantities[SCACOPFData::pS],
				d.PenaltyWeight, slacks_scale);
      append_constraints(cons_li1_pen);
	
      OptVariablesBlock* sigma = cons_li1_pen->get_sigma();
      cons_li1_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
    }
  }

  auto p_li2 = variable("p_li2", d), q_li2 = variable("q_li2", d);
  {
    auto pf_line_lim2 = new PFLineLimits(con_name("line_limits2",d), d.L_Line.size(),
					 p_li2, q_li2, v_n, 
					 d.L_Nidx[1], L_Rate, slacks_scale);
    append_constraints(pf_line_lim2);
    //sslack_li2
    OptVariablesBlock* sslack_li2 = pf_line_lim2->slacks();
    pf_line_lim2->compute_slacks(sslack_li2); sslack_li2->providesStartingPoint=true;

    if(useQPenLi2) {
      append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + sslack_li2->id,
						      sslack_li2,
						      d.P_Penalties[SCACOPFData::pS], 
						      d.P_Quantities[SCACOPFData::pS], 
						      d.PenaltyWeight, slacks_scale) );
    } else {
      PFPenaltyAffineCons* cons_li2_pen  =
	new PFPenaltyAffineCons(string("pcwslin_cons_") + sslack_li2->id, sslack_li2->n, sslack_li2,
				d.P_Penalties[SCACOPFData::pS], d.P_Quantities[SCACOPFData::pS],
				d.PenaltyWeight, slacks_scale);
      append_constraints(cons_li2_pen);
	
      
      OptVariablesBlock* sigma = cons_li2_pen->get_sigma();
      cons_li2_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
    }
  }
}

//
//thermal transformer limits
//
void ACOPFProblem::add_cons_thermal_ti_lims(SCACOPFData& d, bool SysCond_BaseCase)
{
  //! temp
  bool useQPenTi1=useQPen, useQPenTi2=useQPen; //double slacks_scale=1.;
  auto p_ti1 = variable("p_ti1", d), q_ti1 = variable("q_ti1", d);
  vector<double>& T_Rate = SysCond_BaseCase ? d.T_RateBase : d.T_RateEmer;

  {
    
    auto pf_trans_lim1 = new PFTransfLimits(con_name("trans_limits1",d), d.T_Transformer.size(),
					    p_ti1, q_ti1, 
					    d.T_Nidx[0], T_Rate, slacks_scale);
    append_constraints(pf_trans_lim1);
    //sslack_ti1
    OptVariablesBlock* sslack_ti1 = pf_trans_lim1->slacks();
    pf_trans_lim1->compute_slacks(sslack_ti1); sslack_ti1->providesStartingPoint=true;

    if(useQPenTi1) {
      append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + sslack_ti1->id,
						      sslack_ti1,
						      d.P_Penalties[SCACOPFData::pS], 
						      d.P_Quantities[SCACOPFData::pS], 
						      d.PenaltyWeight, slacks_scale) );
    } else {
      PFPenaltyAffineCons* cons_ti1_pen = 
	new PFPenaltyAffineCons(string("pcwslin_cons_") + sslack_ti1->id, sslack_ti1->n, sslack_ti1, 
				d.P_Penalties[SCACOPFData::pS], d.P_Quantities[SCACOPFData::pS],
				d.PenaltyWeight, slacks_scale);
      append_constraints(cons_ti1_pen);
	
      OptVariablesBlock* sigma = cons_ti1_pen->get_sigma();
      cons_ti1_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
    }
  }
  auto p_ti2 = variable("p_ti2", d), q_ti2 = variable("q_ti2", d);
  {
    auto pf_trans_lim2 = new PFTransfLimits(con_name("trans_limits2",d), d.T_Transformer.size(),
					    p_ti2, q_ti2,
					    d.T_Nidx[1], T_Rate, slacks_scale);
    append_constraints(pf_trans_lim2);
    //sslack_ti2
    OptVariablesBlock* sslack_ti2 = pf_trans_lim2->slacks();
    pf_trans_lim2->compute_slacks(sslack_ti2); sslack_ti2->providesStartingPoint=true;

    if(useQPenTi2) {
      append_objterm( new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + sslack_ti2->id,
						      sslack_ti2,
						      d.P_Penalties[SCACOPFData::pS], 
						      d.P_Quantities[SCACOPFData::pS], 
						      d.PenaltyWeight, slacks_scale) );
    } else {
      PFPenaltyAffineCons* cons_ti2_pen = 
	new PFPenaltyAffineCons(string("pcwslin_cons_") + sslack_ti2->id, sslack_ti2->n, sslack_ti2, 
				d.P_Penalties[SCACOPFData::pS], d.P_Quantities[SCACOPFData::pS],
				d.PenaltyWeight, slacks_scale);
      append_constraints(cons_ti2_pen);
	
      OptVariablesBlock* sigma = cons_ti2_pen->get_sigma();
      cons_ti2_pen->compute_sigma(sigma); sigma->providesStartingPoint=true;
    }
  }
}

void ACOPFProblem::add_obj_prod_cost(SCACOPFData& d)
{
  //piecewise linear objective and corresponding constraints
  //all active generators
  vector<int> gens(d.G_Generator.size()); iota(gens.begin(), gens.end(), 0);
  
  auto p_g = variable("p_g", d);

  PFProdCostAffineCons* prod_cost_cons = 
    new PFProdCostAffineCons(con_name("prodcost_cons",d), 2*gens.size(), 
			     p_g, gens, d.G_CostCi, d.G_CostPi);
  append_constraints(prod_cost_cons);

  OptVariablesBlock* t_h = prod_cost_cons->get_t_h();
  prod_cost_cons->compute_t_h(t_h); t_h->providesStartingPoint = true;
}

void ACOPFProblem::print_p_g(SCACOPFData& dB)
{
  auto p_g = variable("p_g", dB);

  printf("p_g for SC block %d\n", dB.id);
  printf("[ idx] [  id ]    p_g            lb           ub     \n");
  for(int i=0; i<dB.G_Generator.size(); i++) {
    printf("[%4d] [%4d] %12.5e  %12.5e %12.5e\n", i, dB.G_Generator[i]+1, p_g->x[i], dB.G_Plb[i], dB.G_Pub[i]);
  }
}
void ACOPFProblem::print_p_g_with_coupling_info(SCACOPFData& dB)
{
  auto p_gk = variable("p_g", dB);
  auto p_g  = variable("p_g", data_sc);
  auto delta = variable("delta", dB);
  auto rhop = variable("rhop_AGC", dB);
  auto rhom = variable("rhom_AGC", dB);
  int K_id = dB.K_Contingency[0];
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  auto ids_agc = selectfrom(data_sc.G_Generator, Gkp);

  printf("p_g for SC block %d: delta_k=%12.5e\n", dB.id, delta->x[0]);
  printf("[ idx] [  id ]         p_g     p_gk             lb            ub         rhom        rhop   |   bodies AGC\n");
  for(int i=0; i<dB.G_Generator.size(); i++) {
    int agc_idx = indexin(ids_agc, dB.G_Generator[i]); 
    int base_idx = indexin(data_sc.G_Generator, dB.G_Generator[i]);
    assert(base_idx>=0);
    if(agc_idx>=0) {
      double gb = dB.G_Pub[i]-dB.G_Plb[i];
      printf("[%4d] [%4d] %12.5e %12.5e agc %12.5e %12.5e %12.5e %12.5e | %12.5e %12.5e %12.5e \n", i, dB.G_Generator[i]+1, 
	     p_g->x[base_idx], p_gk->x[i], dB.G_Plb[i], dB.G_Pub[i], rhom->x[agc_idx], rhop->x[agc_idx],
	     p_g->x[base_idx]+dB.G_alpha[i]*delta->x[0]-p_gk->x[i]-gb*rhop->x[agc_idx]+gb*rhom->x[agc_idx],
	     (p_gk->x[i]-dB.G_Plb[i])/gb*rhom->x[agc_idx], (p_gk->x[i]-dB.G_Pub[i])/gb*rhop->x[agc_idx]);

    } else {
      printf("[%4d] [%4d] %12.5e %12.5e     %12.5e %12.5e\n", i, dB.G_Generator[i]+1, 
	     p_g->x[base_idx], p_gk->x[i], dB.G_Plb[i], dB.G_Pub[i]);
    }
  }
}

} //end namespace

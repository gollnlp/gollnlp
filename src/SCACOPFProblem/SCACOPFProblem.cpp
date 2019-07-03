#include "SCACOPFProblem.hpp"

#include "OPFConstraints.hpp"
#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

#include "OptObjTerms.hpp"

#include <numeric>
#include <typeinfo> 

using namespace std;

namespace gollnlp {

SCACOPFProblem::~SCACOPFProblem()
{
  for(auto d: data_K)
    delete d;
}

bool SCACOPFProblem::default_assembly()
{
  useQPen = true;
  //slacks_scale = 128.;

  SCACOPFData& d = data_sc; //shortcut

  //
  // base case
  //
  add_variables(d);

  add_cons_lines_pf(d);
  add_cons_transformers_pf(d);
  add_cons_active_powbal(d);
  add_cons_reactive_powbal(d);
  double L_rate_reduction=0.75;
  add_cons_thermal_li_lims(d, true, L_rate_reduction);

  double T_rate_reduction=0.75;
  add_cons_thermal_ti_lims(d, true, T_rate_reduction);

  add_obj_prod_cost(d);

  return true;
}

bool SCACOPFProblem::assembly(const std::vector<int> K_Cont)
{
  //assemble base case first
  default_assembly();
  
  SCACOPFData& d = data_sc; //shortcut
  int nK = K_Cont.size();

  for(auto K : K_Cont) {

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

    bool SysCond_BaseCase = false;
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    //coupling AGC and PVPQ; also creates delta_k
    add_cons_coupling(dK);
  }
  //print_summary();
  return true;
}

void SCACOPFProblem::add_cons_coupling(SCACOPFData& dB)
{
  int K_id = dB.K_Contingency[0];

  //indexes in data_sc.G_Generator
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  assert(Gk.size() == dB.G_Generator.size());

  if(AGC_as_nonanticip) {
    add_cons_nonanticip(dB, Gk); 
    add_cons_AGC(dB, {}); //just to print the "?!? no AGC" message
  } else {
    add_cons_nonanticip(dB, Gknop);
    add_cons_AGC(dB, Gkp);
  }




  //voltages
  if(PVPQ_as_nonanticip) {
    assert(false && "not yet implemented");
  } else {
    add_cons_PVPQ(dB, Gk);
  }
}

// Gk are the indexes of all gens other than the outgen (for generator contingencies) 
// in data_sc.G_Generator
void SCACOPFProblem::add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk)
{
  auto G_Nidx_Gk = selectfrom(data_sc.G_Nidx, Gk);
  //extra check
  assert(G_Nidx_Gk == dB.G_Nidx);

  sort(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  //printvec(G_Nidx_Gk);
  auto last = unique(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  G_Nidx_Gk.erase(last, G_Nidx_Gk.end());
  //printvec(G_Nidx_Gk);
  auto &N_PVPQ = G_Nidx_Gk; //nodes with PVPQ generators;

  //(aggregated) non-fixed q_g generator ids at each node/bus
  // with PVPQ generators that have at least one non-fixed q_g 
  vector<vector<int> > idxs_gen_agg;
  //bus indexes that have at least one non-fixed q_g
  vector<int> idxs_bus_pvpq;
  //aggregated lb and ub on reactive power at each PVPQ bus
  vector<double> Qlb, Qub;
  int nPVPQGens=0, nPVPQCons=0; 
  int num_buses_all_qgen_fixed=0, num_qgens_fixed=0;

  for(auto n: N_PVPQ) {
    assert(dB.Gn[n].size()>0);
    double Qagglb=0., Qaggub=0.;

    int numfixed = 0;
    idxs_gen_agg.push_back( vector<int>() );
    for(auto g: dB.Gn[n]) {
#ifdef DEBUG
      assert(dB.K_Contingency.size()==1);
      assert(dB.K_outidx.size()==1);
      if(dB.K_ConType[0]==SCACOPFData::kGenerator) 
	assert(data_sc.G_Generator[dB.K_outidx[0]]!=dB.G_Generator[g]);
#endif
      if(abs(dB.G_Qub[g]-dB.G_Qlb[g])<=1e-8) {
	numfixed++; num_qgens_fixed++;
	printf("PVPQ: gen ID=%d p_q at bus idx %d id %d is fixed; will not add PVPQ constraint\n",
	       dB.G_Generator[g], dB.G_Nidx[g], data_sc.N_Bus[dB.G_Nidx[g]]);
	continue;
      }
      idxs_gen_agg.back().push_back(g);
      Qagglb += dB.G_Qlb[g];
      Qaggub += dB.G_Qub[g];
    }

    assert(idxs_gen_agg.back().size()+numfixed == dB.Gn[n].size());
    nPVPQGens += idxs_gen_agg.back().size()+numfixed;

    if(idxs_gen_agg.back().size()==0) {
      assert(Qagglb==0. && Qaggub==0.);
      idxs_gen_agg.pop_back();
      num_buses_all_qgen_fixed++;
    } else {
      Qlb.push_back(Qagglb);
      Qub.push_back(Qaggub);
      idxs_bus_pvpq.push_back(n);
    }
  }
  assert(idxs_gen_agg.size() == Qlb.size());
  assert(idxs_gen_agg.size() == Qub.size());
  assert(N_PVPQ.size()  == num_buses_all_qgen_fixed+idxs_gen_agg.size());
  assert(idxs_bus_pvpq.size() == idxs_gen_agg.size());

  //for(int i=0; i<Qlb.size(); i++) {
  //  printf("%d %g %g \n", N_PVPQ[i], Qlb[i], Qub[i]);
  //  printvec(idxs_gen_agg[i]);
  //}

  auto v_n0 = variable("v_n", data_sc);
  auto v_nk = variable("v_n", dB);
  auto q_gk = variable("q_g", dB);
  if(NULL==v_n0) {
    printf("Contingency %d: v_n var not found in the base case; will NOT add PVPQ constraints.\n", dB.id);
    return;
  }
  if(NULL==v_nk) {
    printf("Contingency %d: v_n var not found in conting problem; will NOT add PVPQ constraints.\n", dB.id);
    return;
  }
  if(NULL==q_gk) {
    printf("Contingency %d: q_g var not found in the base case; will NOT add PVPQ constraints.\n", dB.id);
    return;
  }
  
  auto cons = new PVPQComplementarityCons(con_name("PVPQ", dB), 3*idxs_gen_agg.size(),
					  v_n0, v_nk, q_gk,
					  idxs_bus_pvpq, idxs_gen_agg,
					  Qlb, Qub, 
					  PVPQSmoothing);

  append_constraints(cons);

  //starting point for nup and num that were added by PVPQComplementarityCons
  auto nup=cons->get_nup(), num=cons->get_num();
  cons->compute_nus(nup, num);
  nup->providesStartingPoint=true; num->providesStartingPoint=true;

  append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+num->id, num, 1.));
  append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+nup->id, nup, 1.));
  
  //for(int i=0; i<rhop->n; i++) 
  //  printf("%g %g   %g\n", rhop->x[i], rhom->x[i], cons->gb[i]);
  //printf("\n");

  printf("PVPQ: participating %d gens at %lu buses: added %d constraints;"
	 "total PVPQ: %lu gens | %lu buses; were fixed: %d gens | %d buses with all gens fixed.\n",
	 nPVPQGens-num_qgens_fixed, idxs_bus_pvpq.size(), cons->n,
	 Gk.size(), N_PVPQ.size(),
	 num_qgens_fixed, num_buses_all_qgen_fixed);
}

void SCACOPFProblem::add_cons_nonanticip(SCACOPFData& dB, const std::vector<int>& G_idxs_no_AGC)
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
  printf("AGC: %lu gens NOT participating: added one nonanticip constraint for each\n", G_idxs_no_AGC.size());
}

void SCACOPFProblem::add_cons_AGC(SCACOPFData& dB, const std::vector<int>& G_idxs_AGC)
{
  if(G_idxs_AGC.size()==0) {
    printf("SCACOPFProblem: add_cons_AGC: NO gens participating !?! in contingency %d\n", dB.id);
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

  append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhom->id, rhom, 1.));
  append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhop->id, rhop, 1.));

  //for(int i=0; i<rhop->n; i++) 
  //  printf("%g %g   %g\n", rhop->x[i], rhom->x[i], cons->gb[i]);
  //printf("\n");

  printf("AGC: %lu gens participating: added %d constraints\n", G_idxs_AGC.size(), cons->n);
}
  
void SCACOPFProblem::add_variables(SCACOPFData& d)
{

  auto v_n = new OptVariablesBlock(data_sc.N_Bus.size(), var_name("v_n",d), 
				   data_sc.N_Vlb.data(), data_sc.N_Vub.data()); 
  append_variables(v_n);
  v_n->set_start_to(data_sc.N_v0.data());
  //v_n->print();
  //append_objterm(new DummySingleVarQuadrObjTerm("v_nsq", v_n));


  auto theta_n = new OptVariablesBlock(data_sc.N_Bus.size(), var_name("theta_n",d));
  append_variables(theta_n);
  theta_n->set_start_to(data_sc.N_theta0.data());
  int RefBus = data_sc.bus_with_largest_gen();
  //printf("RefBus=%d\n", RefBus);

  if(data_sc.N_theta0[RefBus]!=0.) {
    for(int b=0; b<theta_n->n; b++) {
      theta_n->x[b] -= data_sc.N_theta0[RefBus];
      assert( theta_n->x[b] >= theta_n->lb[b]);
      assert( theta_n->x[b] <= theta_n->ub[b]);
    }
    theta_n->lb[RefBus] = theta_n->ub[RefBus] = 0.;
    assert(theta_n->x[RefBus]==0.);
    //printf("We should set theta at RefBus to 0");
  }
  //append_objterm(new DummySingleVarQuadrObjTerm("theta_sq", theta_n));

  // LINES
  auto p_li1 = new OptVariablesBlock(d.L_Line.size(), var_name("p_li1",d));
  append_variables(p_li1);
  auto p_li2 = new OptVariablesBlock(d.L_Line.size(), var_name("p_li2",d));
  append_variables(p_li2);
  //append_objterm(new DummySingleVarQuadrObjTerm("pli1_sq", p_li1));
  //append_objterm(new DummySingleVarQuadrObjTerm("pli2_sq", p_li2));


  auto q_li1 = new OptVariablesBlock(d.L_Line.size(), var_name("q_li1",d));
  auto q_li2 = new OptVariablesBlock(d.L_Line.size(), var_name("q_li2",d));
  append_variables(q_li1); 
  append_variables(q_li2);
  //append_objterm(new DummySingleVarQuadrObjTerm("qli1_sq", q_li1));
  //append_objterm(new DummySingleVarQuadrObjTerm("qli2_sq", q_li2));
  

  //TRANSFORMERS
  auto p_ti1 = new OptVariablesBlock(d.T_Transformer.size(), var_name("p_ti1",d));
  auto p_ti2 = new OptVariablesBlock(d.T_Transformer.size(), var_name("p_ti2",d));
  append_variables(p_ti1); 
  append_variables(p_ti2); 
  //append_objterm(new DummySingleVarQuadrObjTerm("pti1_sq", p_ti1));
  //append_objterm(new DummySingleVarQuadrObjTerm("pti2_sq", p_ti2));


  auto q_ti1 = new OptVariablesBlock(d.T_Transformer.size(), var_name("q_ti1",d));
  auto q_ti2 = new OptVariablesBlock(d.T_Transformer.size(), var_name("q_ti2",d));
  append_variables(q_ti1); 
  append_variables(q_ti2); 
  //append_objterm(new DummySingleVarQuadrObjTerm("qti1_sq", q_ti1));
  //append_objterm(new DummySingleVarQuadrObjTerm("qti2_sq", q_ti2));

  auto b_s = new OptVariablesBlock(data_sc.SSh_SShunt.size(), var_name("b_s",d), 
  				   data_sc.SSh_Blb.data(), data_sc.SSh_Bub.data());
  b_s->set_start_to(data_sc.SSh_B0.data());
  append_variables(b_s);
  //append_objterm(new DummySingleVarQuadrObjTerm("b_s_sq", b_s));

  auto p_g = new OptVariablesBlock(d.G_Generator.size(), var_name("p_g",d), 
				   d.G_Plb.data(), d.G_Pub.data());
  append_variables(p_g); 
  p_g->set_start_to(d.G_p0.data());

  auto q_g = new OptVariablesBlock(d.G_Generator.size(), var_name("q_g",d), 
  				   d.G_Qlb.data(), d.G_Qub.data());
  q_g->set_start_to(d.G_q0.data());
  append_variables(q_g); 
  //append_objterm(new DummySingleVarQuadrObjTerm("q_g_sq", q_g));

}

void SCACOPFProblem::add_cons_lines_pf(SCACOPFData& d)
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
void SCACOPFProblem::add_cons_transformers_pf(SCACOPFData& d)
{
  auto v_n = variable("v_n",d), theta_n = variable("theta_n",d);
  {
    //
    // transformers active power flows
    //
    auto p_ti1 = variable("p_ti1",d), p_ti2 = variable("p_ti2",d);

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
  }
  {
    //
    // transformers reactive power flows
    //
    auto q_ti1 = variable("q_ti1",d), q_ti2 = variable("q_ti2",d);
    // i=1 addpowerflowcon!(m, q_ti[t,1], v_n[T_Nidx[t,1]], v_n[T_Nidx[t,2]],
    //		theta_n[T_Nidx[t,1]], theta_n[T_Nidx[t,2]],
    //		-T[:B][t]/T[:Tau][t]^2-T[:Bm][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], -T[:Theta][t])
    auto pf_cons1 = new PFConRectangular(con_name("q_ti1_powerflow",d), d.T_Transformer.size(), 
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
  
    double *T=pf_cons1->get_T(), *TTheta=d.T_Theta.data();
    for(int t=0; t<pf_cons1->n; t++) T[t] = -TTheta[t];
  
    // i=2 addpowerflowcon!(m, q_ti[t,2], v_n[T_Nidx[t,2]], v_n[T_Nidx[t,1]],
    //		theta_n[T_Nidx[t,2]], theta_n[T_Nidx[t,1]],
    //		-T[:B][t], T[:B][t]/T[:Tau][t], -T[:G][t]/T[:Tau][t], T[:Theta][t])
    auto pf_cons2 = new PFConRectangular(con_name("q_ti2_powerflow",d), d.T_Transformer.size(), 
					 q_ti2, v_n, theta_n,
					 d.T_Nidx[1], d.T_Nidx[0]);
  
    A=pf_cons2->get_A();
    for(int i=0; i<pf_cons2->n; i++) A[i]=-TB[i];
    DCOPY(&(pf_cons2->n), pf_cons1->get_B(), &ione, pf_cons2->get_B(), &ione);
    DCOPY(&(pf_cons2->n), pf_cons1->get_C(), &ione, pf_cons2->get_C(), &ione);
    DCOPY(&(pf_cons2->n), TTheta, &ione, pf_cons2->get_T(), &ione);

    //vector<double> vv(pf_cons2->get_B(), pf_cons2->get_B()+pf_cons2->n);
    //printvec(vv, "	T[:B] ./ T[:Tau]");

    append_constraints(pf_cons1);
    append_constraints(pf_cons2);
    pf_cons1->compute_power(q_ti1); q_ti1->providesStartingPoint=true;
    pf_cons2->compute_power(q_ti2); q_ti2->providesStartingPoint=true;
  }
}

void SCACOPFProblem::add_cons_active_powbal(SCACOPFData& d)
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

void SCACOPFProblem::add_cons_reactive_powbal(SCACOPFData& d)
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
void SCACOPFProblem::add_cons_thermal_li_lims(SCACOPFData& d, bool SysCond_BaseCase, double L_rate_reduction)
{
  //! temp
  bool useQPenLi1 = useQPen, useQPenLi2 = useQPen; //double slacks_scale=1.;

  auto v_n = variable("v_n",d);
  auto p_li1 = variable("p_li1", d), q_li1 = variable("q_li1", d);
  vector<double>& L_Rate = SysCond_BaseCase ? d.L_RateBase : d.L_RateEmer;
  {
    auto pf_line_lim1 = new PFLineLimits(con_name("line_limits1",d), d.L_Line.size(),
					 p_li1, q_li1, v_n, 
					 d.L_Nidx[0], L_Rate, slacks_scale, L_rate_reduction);
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
					 d.L_Nidx[1], L_Rate, slacks_scale, L_rate_reduction);
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
void SCACOPFProblem::add_cons_thermal_ti_lims(SCACOPFData& d, bool SysCond_BaseCase, double T_rate_reduction)
{
  //! temp
  bool useQPenTi1=useQPen, useQPenTi2=useQPen; //double slacks_scale=1.;
  
  vector<double>& T_Rate = SysCond_BaseCase ? d.T_RateBase : d.T_RateEmer;
  if(true){
    auto p_ti1 = variable("p_ti1", d), q_ti1 = variable("q_ti1", d);
    auto pf_trans_lim1 = new PFTransfLimits(con_name("trans_limits1",d), d.T_Transformer.size(),
					    p_ti1, q_ti1, 
					    T_Rate, slacks_scale);
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
  
  if(true){
    auto p_ti2 = variable("p_ti2", d), q_ti2 = variable("q_ti2", d);
    auto pf_trans_lim2 = new PFTransfLimits(con_name("trans_limits2",d), d.T_Transformer.size(),
					    p_ti2, q_ti2,
					    T_Rate, slacks_scale);
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
    	//append_objterm( new DummySingleVarQuadrObjTerm("QDu_pen_" + sslack_ti2->id, sslack_ti2));
    	//append_objterm( new DummySingleVarQuadrObjTerm("QDu_pen2_" + sslack_ti2->id, sslack_ti2));
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

void SCACOPFProblem::add_obj_prod_cost(SCACOPFData& d)
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

bool SCACOPFProblem::set_warm_start_from_base_of(SCACOPFProblem& srcProb)
{

#ifdef DEBUG
  for(auto b: vars_primal->vblocks) { b->providesStartingPoint = false; }
#endif

  //print_summary();
  //srcProb.print_summary();

  for(auto dK : data_K) {
    set_warm_start_for_cont_from_base_of(*dK, srcProb);
  }

  set_warm_start_for_base_from_base_of(srcProb);

  //vars_primal->print_summary("vars primal newer problem");
  //vars_duals_bounds_L->print_summary("vars duals bounds_L newer problem");
  //vars_duals_bounds_U->print_summary("vars duals bounds_U newer problem");
  //vars_duals_cons->print_summary("vars duals cons newer problem");

  assert(vars_primal->provides_start());
  assert(vars_duals_cons->provides_start());
  assert(vars_duals_bounds_L->provides_start());
  assert(vars_duals_bounds_U->provides_start());


  return true;
}

  bool SCACOPFProblem::set_warm_start_for_base_from_base_of(SCACOPFProblem& srcProb)
  {
    vector<string> ids = {"v_n_0" , "theta_n_0" , "p_li1_0" , "p_li2_0" , "q_li1_0" , "q_li2_0" , "p_ti1_0" , "p_ti2_0" , "q_ti1_0" , "q_ti2_0" , "b_s_0" , "p_g_0" , "q_g_0" , "pslack_n_p_balance_0" , "qslack_n_q_balance_0" , "sslack_li_line_limits1_0" , "sslack_li_line_limits2_0" , "sslack_ti_trans_limits1_0" , "sslack_ti_trans_limits2_0" , "t_h" };
    for(auto id: ids) {
      auto b = vars_primal->vars_block(id);
      auto bsrc = srcProb.vars_primal->vars_block(id);
      if(b && bsrc) b->set_start_to(*bsrc);
      else { assert(false); }
    }

    ids = {"duals_bndL_v_n_0" , "duals_bndL_theta_n_0" , "duals_bndL_p_li1_0" , "duals_bndL_p_li2_0" , "duals_bndL_q_li1_0" , "duals_bndL_q_li2_0" , "duals_bndL_p_ti1_0" , "duals_bndL_p_ti2_0" , "duals_bndL_q_ti1_0" , "duals_bndL_q_ti2_0" , "duals_bndL_b_s_0" , "duals_bndL_p_g_0" , "duals_bndL_q_g_0" , "duals_bndL_pslack_n_p_balance_0" , "duals_bndL_qslack_n_q_balance_0" , "duals_bndL_sslack_li_line_limits1_0" , "duals_bndL_sslack_li_line_limits2_0" , "duals_bndL_sslack_ti_trans_limits1_0" , "duals_bndL_sslack_ti_trans_limits2_0" , "duals_bndL_t_h"};
    for(auto id: ids) {
      auto b = vars_duals_bounds_L->vars_block(id);
      auto bsrc = srcProb.vars_duals_bounds_L->vars_block(id);
      if(b && bsrc) b->set_start_to(*bsrc);
      else { assert(false); }
    }

    ids = {"duals_bndU_v_n_0" , "duals_bndU_theta_n_0" , "duals_bndU_p_li1_0" , "duals_bndU_p_li2_0" , "duals_bndU_q_li1_0" , "duals_bndU_q_li2_0" , "duals_bndU_p_ti1_0" , "duals_bndU_p_ti2_0" , "duals_bndU_q_ti1_0" , "duals_bndU_q_ti2_0" , "duals_bndU_b_s_0" , "duals_bndU_p_g_0" , "duals_bndU_q_g_0" , "duals_bndU_pslack_n_p_balance_0" , "duals_bndU_qslack_n_q_balance_0" , "duals_bndU_sslack_li_line_limits1_0" , "duals_bndU_sslack_li_line_limits2_0" , "duals_bndU_sslack_ti_trans_limits1_0" , "duals_bndU_sslack_ti_trans_limits2_0" , "duals_bndU_t_h"};
    for(auto id: ids) {
      auto b = vars_duals_bounds_U->vars_block(id);
      auto bsrc = srcProb.vars_duals_bounds_U->vars_block(id);
      if(b && bsrc) b->set_start_to(*bsrc);
      else { assert(false); }
    }

    ids = {"duals_p_li1_powerflow_0" , "duals_p_li2_powerflow_0" , "duals_q_li1_powerflow_0" , "duals_q_li2_powerflow_0" , "duals_p_ti1_powerflow_0" , "duals_p_ti2_powerflow_0" , "duals_q_ti1_powerflow_0" , "duals_q_ti2_powerflow_0" , "duals_p_balance_0" , "duals_q_balance_0" , "duals_line_limits1_0" , "duals_line_limits2_0" , "duals_trans_limits1_0" , "duals_trans_limits2_0" , "duals_prodcost_cons_0"};
    for(auto id: ids) {
      auto b = vars_duals_cons->vars_block(id);
      auto bsrc = srcProb.vars_duals_cons->vars_block(id);
      if(b && bsrc) b->set_start_to(*bsrc);
      else { assert(false); }
    }

  }
#define SIGNED_DUALS_VAL 1.


  bool SCACOPFProblem::set_warm_start_for_cont_from_base_of(SCACOPFData& dB, SCACOPFProblem& srcProb)
  {
    SCACOPFData& dK = dB;// assert(dK.id==K_idx+1);
    int K_idx = dK.id-1;
    assert(K_idx>=0); assert(K_idx<data_sc.K_Contingency.size());

    //
    // setup for indexes used in non-anticip and AGC coupling 
    //
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    vector<int> Gk, pg0_partic_idxs, pg0_nonpartic_idxs;
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);

    vector<int> pgK_nonpartic_idxs, pgK_partic_idxs;
    // indexes in data_K (for the  contingency)
    auto ids_no_AGC = selectfrom(data_sc.G_Generator, pg0_nonpartic_idxs);
    pgK_nonpartic_idxs = indexin(dK.G_Generator, ids_no_AGC);
    pgK_nonpartic_idxs = findall(pgK_nonpartic_idxs, [](int val) {return val!=-1;});

    auto ids_AGC = selectfrom(data_sc.G_Generator, pg0_partic_idxs);
    pgK_partic_idxs = indexin(dK.G_Generator, ids_AGC);
    pgK_partic_idxs = findall(pgK_partic_idxs, [](int val) {return val!=-1;});

    // contingency indexes of lines, generators, or transformers (i.e., contingency type)
    vector<int> idxs_of_K_in_0; 

    assert(useQPen==true); assert(srcProb.useQPen==true);
    variable("v_n", dK)->set_start_to(*srcProb.variable("v_n", data_sc));
    variable("theta_n", dK)->set_start_to(*srcProb.variable("theta_n", data_sc));
    variable("b_s", dK)->set_start_to(*srcProb.variable("b_s", data_sc));

    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      auto p_gK = variable("p_g", dK), p_g0 = srcProb.variable("p_g", data_sc);
      assert(p_gK->n == p_g0->n-1);

      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
      }
      p_gK->providesStartingPoint = true;

      auto q_gK = variable("q_g", dK), q_g0 = srcProb.variable("q_g", data_sc);
      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
      }
      q_gK->providesStartingPoint = true;
      
    } else {
#ifdef DEBUG
      assert(variable("p_g", dK)->n == srcProb.variable("p_g", data_sc)->n);
      assert(variable("q_g", dK)->n == srcProb.variable("q_g", data_sc)->n);
#endif
      variable("p_g", dK)->set_start_to(*srcProb.variable("p_g", data_sc));
      variable("q_g", dK)->set_start_to(*srcProb.variable("q_g", data_sc));
    }
    
    if(dK.K_ConType[0] == SCACOPFData::kLine) {
      idxs_of_K_in_0 = indexin(dK.L_Line, data_sc.L_Line);
      size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();

      auto var_K = variable("p_li1", dK), var_0 = srcProb.variable("p_li1", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("p_li2", dK); var_0 = srcProb.variable("p_li2", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_li1", dK); var_0 = srcProb.variable("q_li1", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_li2", dK); var_0 = srcProb.variable("q_li2", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

    } else {
      assert(variable("p_li1", dK)->n == srcProb.variable("p_li1", data_sc)->n);
      assert(variable("p_li2", dK)->n == srcProb.variable("p_li2", data_sc)->n);
      assert(variable("q_li1", dK)->n == srcProb.variable("q_li1", data_sc)->n);
      assert(variable("q_li2", dK)->n == srcProb.variable("q_li2", data_sc)->n);

      variable("p_li1", dK)->set_start_to(*srcProb.variable("p_li1", data_sc));
      variable("p_li2", dK)->set_start_to(*srcProb.variable("p_li2", data_sc));
      variable("q_li1", dK)->set_start_to(*srcProb.variable("q_li1", data_sc));
      variable("q_li2", dK)->set_start_to(*srcProb.variable("q_li2", data_sc));
    }

    if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
      idxs_of_K_in_0 = indexin(dK.T_Transformer, data_sc.T_Transformer);
      size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();

      auto var_K = variable("p_ti1", dK), var_0 = srcProb.variable("p_ti1", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("p_ti2", dK); var_0 = srcProb.variable("p_ti2", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_ti1", dK); var_0 = srcProb.variable("q_ti1", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_ti2", dK); var_0 = srcProb.variable("q_ti2", data_sc);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;


    } else {
      assert(variable("p_ti1", dK)->n == srcProb.variable("p_ti1", data_sc)->n);
      assert(variable("p_ti2", dK)->n == srcProb.variable("p_ti2", data_sc)->n);
      assert(variable("q_ti1", dK)->n == srcProb.variable("q_ti1", data_sc)->n);
      assert(variable("q_ti2", dK)->n == srcProb.variable("q_ti2", data_sc)->n);

      variable("p_ti1", dK)->set_start_to(*srcProb.variable("p_ti1", data_sc));
      variable("p_ti2", dK)->set_start_to(*srcProb.variable("p_ti2", data_sc));
      variable("q_ti1", dK)->set_start_to(*srcProb.variable("q_ti1", data_sc));
      variable("q_ti2", dK)->set_start_to(*srcProb.variable("q_ti2", data_sc));
    }

    //    
    // recompute compute slacks
    //
    {
      auto cons = dynamic_cast<PVPQComplementarityCons*>(constraint("PVPQ", dK));
      if(cons) {
    	auto nup=cons->get_nup(), num=cons->get_num();
    	cons->compute_nus(nup, num);
    	nup->providesStartingPoint=true; num->providesStartingPoint=true;
      }
    }
    {
      auto cons = dynamic_cast<AGCComplementarityCons*>(constraint("AGC", dK));
      if(cons) {
    	auto rhop=cons->get_rhop(), rhom=cons->get_rhom();
    	cons->compute_rhos(rhop, rhom);
    	rhop->providesStartingPoint=true; rhom->providesStartingPoint=true;
      }
    }
    {
      auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("p_li1_powerflow", dK));
      auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("p_li2_powerflow", dK));
      auto p_li1 = variable("p_li1", dK), p_li2 = variable("p_li2", dK);
      pf_cons1->compute_power(p_li1); p_li1->providesStartingPoint=true;
      pf_cons2->compute_power(p_li2); p_li2->providesStartingPoint=true;
    }
    {
      auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("p_ti1_powerflow", dK));
      auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("p_ti2_powerflow", dK));
      auto p_ti1 = variable("p_ti1", dK), p_ti2 = variable("p_ti2", dK);
      pf_cons1->compute_power(p_ti1); p_ti1->providesStartingPoint=true;
      pf_cons2->compute_power(p_ti2); p_ti2->providesStartingPoint=true;
    }

    {
      auto pf_p_bal = dynamic_cast<PFActiveBalance*>(constraint("p_balance", dK));
      OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
      pf_p_bal->compute_slacks(pslacks_n); pslacks_n->providesStartingPoint=true;
    }
    {
      auto pf_q_bal = dynamic_cast<PFReactiveBalance*>(constraint("q_balance", dK));
      OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
      pf_q_bal->compute_slacks(qslacks_n); qslacks_n->providesStartingPoint=true;
    }
    {
      auto pf_line_lim1 = dynamic_cast<PFLineLimits*>(constraint("line_limits1",dK));
      OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
      pf_line_lim1->compute_slacks(sslack_li1); sslack_li1->providesStartingPoint=true;      
    }
    {
      auto pf_line_lim2 = dynamic_cast<PFLineLimits*>(constraint("line_limits2",dK));
      OptVariablesBlock* sslack_li2 = pf_line_lim2->slacks();
      pf_line_lim2->compute_slacks(sslack_li2); sslack_li2->providesStartingPoint=true;      
    }
    {
      auto pf_trans_lim1 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits1",dK));
      OptVariablesBlock* sslack_ti1 = pf_trans_lim1->slacks();
      pf_trans_lim1->compute_slacks(sslack_ti1); sslack_ti1->providesStartingPoint=true;
    }
    {
      auto pf_trans_lim2 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits2",dK));
      OptVariablesBlock* sslack_ti2 = pf_trans_lim2->slacks();
      pf_trans_lim2->compute_slacks(sslack_ti2); sslack_ti2->providesStartingPoint=true;
    }

    //
    //dual variables part
    //
    string prefix;
    {
      if(NULL == vars_duals_bounds_L)
	vars_duals_bounds_L = new_duals_lower_bounds();
      
      //lower bounds duals
      prefix = "duals_bndL_v_n";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      prefix = "duals_bndL_theta_n";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));

      prefix = "duals_bndL_p_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_p_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_q_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_q_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_p_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_p_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_q_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_q_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_b_s";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));

      prefix = "duals_bndL_p_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
	auto p_gK = variable_duals_lower(prefix, dK), p_g0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(p_gK->n == p_g0->n - 1);
	assert(p_g0->n == 1+pg0_nonpartic_idxs.size()+pg0_partic_idxs.size());
	assert(p_gK->n == pgK_nonpartic_idxs.size()+pgK_partic_idxs.size());
	
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
	}
	p_gK->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_q_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	auto q_gK = variable_duals_lower(prefix, dK), q_g0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(q_gK->n == q_g0->n -1);
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
	}
	q_gK->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_pslack_n_p_balance";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      prefix = "duals_bndL_qslack_n_q_balance";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));

      prefix = "duals_bndL_sslack_li_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;

      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }
      
      prefix = "duals_bndL_sslack_li_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));	
      }
      
      prefix = "duals_bndL_sslack_ti_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }

      prefix = "duals_bndL_sslack_ti_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      }
      

      if(pgK_partic_idxs.size()>0) {
	prefix = "duals_bndL_delta";
	auto v = variable_duals_lower(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);

	prefix = "duals_bndL_rhop_AGC";
	v = variable_duals_lower(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);

	prefix = "duals_bndL_rhom_AGC";
	v = variable_duals_lower(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }

      prefix = "duals_bndL_nup_PVPQ";
      //variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      variable_duals_lower(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);

      prefix = "duals_bndL_num_PVPQ";
      //variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
      variable_duals_lower(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);

      //vars_duals_bounds_L->print_summary(); 
      //assert(vars_duals_bounds_L->provides_start());
    }
    //
    //upper bounds duals
    //
    {
      if(NULL == vars_duals_bounds_U)
	vars_duals_bounds_U = new_duals_upper_bounds();
      prefix = "duals_bndU_v_n";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      prefix = "duals_bndU_theta_n";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));

      prefix = "duals_bndU_p_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }
      prefix = "duals_bndU_p_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_q_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }
      prefix = "duals_bndU_q_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_p_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_p_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_q_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_q_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_b_s";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));

      prefix = "duals_bndU_p_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	//variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
	auto p_gK = variable_duals_upper(prefix, dK), p_g0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(p_gK->n == p_g0->n - 1);
	assert(p_g0->n == 1+pg0_nonpartic_idxs.size()+pg0_partic_idxs.size());
	assert(p_gK->n == pgK_nonpartic_idxs.size()+pgK_partic_idxs.size());
	
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
	}
	p_gK->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_q_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	auto q_gK = variable_duals_upper(prefix, dK), q_g0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(q_gK->n == q_g0->n -1);
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
	}
	q_gK->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_pslack_n_p_balance";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      prefix = "duals_bndU_qslack_n_q_balance";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      
      prefix = "duals_bndU_sslack_li_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_sslack_li_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_sslack_ti_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }

      prefix = "duals_bndU_sslack_ti_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc));
      }
      
      if(pgK_partic_idxs.size()>0) {
	prefix = "duals_bndU_delta";
	auto v = variable_duals_upper(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);

	prefix = "duals_bndU_rhop_AGC";
	v = variable_duals_upper(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);

	prefix = "duals_bndU_rhom_AGC";
	v = variable_duals_upper(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      {
	prefix = "duals_bndU_nup_PVPQ";
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
	auto v = variable_duals_upper(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);

	prefix = "duals_bndU_num_PVPQ";
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
	v = variable_duals_upper(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
	//assert(vars_duals_bounds_U->provides_start());
      }
    }
    
    //
    //constraints duals
    //
    {
      if(NULL == vars_duals_cons)
	vars_duals_cons = new_duals_cons();
    
      prefix = "duals_p_li1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_p_li2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_q_li1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_q_li2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_p_ti1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_p_ti2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_q_ti1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_q_ti2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_p_balance";
      variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      prefix = "duals_q_balance";
      variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));

      prefix = "duals_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      prefix = "duals_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }


      prefix = "duals_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc));
      }

      if(pgK_partic_idxs.size()>0) {
	prefix = "duals_AGC";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      {
	prefix = "duals_non_anticip";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL); 
      }

      {
	prefix = "duals_PVPQ";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      //assert(vars_duals_cons->provides_start());
    }
    

    //srcProb.duals_bounds_lower()->print_summary("duals bounds lower");
    //srcProb.duals_bounds_upper()->print_summary("duals bounds upper");
    //srcProb.duals_constraints()->print_summary("duals constraints");

    return true;
  }

void SCACOPFProblem::print_p_g(SCACOPFData& dB)
{
  auto p_g = variable("p_g", dB);

  printf("p_g for SC block %d\n", dB.id);
  printf("[ idx] [  id ]    p_g            lb           ub     \n");
  for(int i=0; i<dB.G_Generator.size(); i++) {
    printf("[%4d] [%4d] %12.5e  %12.5e %12.5e\n", i, dB.G_Generator[i]+1, p_g->x[i], dB.G_Plb[i], dB.G_Pub[i]);
  }
}
void SCACOPFProblem::print_p_g_with_coupling_info(SCACOPFData& dB, OptVariablesBlock* p_g0)
{
  auto p_gk = variable("p_g", dB);
  auto p_g  = p_g0;
  if(p_g==NULL) p_g = variable("p_g", data_sc);

  auto delta = variable("delta", dB);
  auto rhop = variable("rhop_AGC", dB);
  auto rhom = variable("rhom_AGC", dB);

  if(NULL==delta) {
    assert(rhop==NULL); 
    assert(rhom==NULL);
    printf("print_p_g_with_coupling_info called but no AGC constraints present. will print p_g\n");
    print_p_g(dB);
    return;
  }

  int K_id = dB.K_Contingency[0];
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  auto ids_agc = selectfrom(data_sc.G_Generator, Gkp);

  printf("p_g for SC block %d: delta_k=%12.5e\n", dB.id, delta->x[0]);
  printf("[ idx] [  id ]         p_g     p_gk             lb            ub         rhom        rhop      |   bodies AGC\n");
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

void SCACOPFProblem::print_PVPQ_info(SCACOPFData& dB)
{
  //indexes in data_sc.G_Generator
  vector<int> Gk, Gkp, Gknop;
  int K_id = dB.K_Contingency[0];
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  assert(Gk.size() == dB.G_Generator.size());
  auto G_Nidx_Gk = selectfrom(data_sc.G_Nidx, Gk);
  //extra check
  assert(G_Nidx_Gk == dB.G_Nidx);

  sort(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  //printvec(G_Nidx_Gk);
  auto last = unique(G_Nidx_Gk.begin(), G_Nidx_Gk.end());
  G_Nidx_Gk.erase(last, G_Nidx_Gk.end());
  //printvec(G_Nidx_Gk);
  auto &N_PVPQ = G_Nidx_Gk; //nodes with PVPQ generators;

  //(aggregated) non-fixed q_g generator ids at each node/bus
  // with PVPQ generators that have at least one non-fixed q_g 
  vector<vector<int> > idxs_gen_agg;
  //bus indexes that have at least one non-fixed q_g
  vector<int> idxs_bus_pvpq;
  //aggregated lb and ub on reactive power at each PVPQ bus
  vector<double> Qlb, Qub;
  int nPVPQGens=0, nPVPQCons=0; 
  int num_buses_all_qgen_fixed=0, num_qgens_fixed=0;

  auto v_n0 = variable("v_n", data_sc);
  auto v_nk = variable("v_n", dB);
  auto q_gk = variable("q_g", dB);
  auto rhop = variable("nup_PVPQ", dB);
  auto rhom = variable("num_PVPQ", dB);
  printf("busidx busid     v_n0         v_nk         num          nup      | genidx q_gk qlb qub : \n");

  for(auto n: N_PVPQ) {
    assert(dB.Gn[n].size()>0);
    double Qagglb=0., Qaggub=0.;

    int numfixed = 0;
    idxs_gen_agg.push_back( vector<int>() );


    for(auto g: dB.Gn[n]) {
#ifdef DEBUG
      assert(dB.K_Contingency.size()==1);
      assert(dB.K_outidx.size()==1);
      if(dB.K_ConType[0]==SCACOPFData::kGenerator) 
	assert(data_sc.G_Generator[dB.K_outidx[0]]!=dB.G_Generator[g]);
#endif
      if(abs(dB.G_Qub[g]-dB.G_Qlb[g])<=1e-8) {
	numfixed++; num_qgens_fixed++;


	//printf("PVPQ: gen ID=%d p_q at bus %d is fixed; will not add PVPQ constraint\n",
	//       dB.G_Generator[g], dB.G_Nidx[g]);
	continue;
      }
      idxs_gen_agg.back().push_back(g);
      Qagglb += dB.G_Qlb[g];
      Qaggub += dB.G_Qub[g];
    }



    assert(idxs_gen_agg.back().size()+numfixed == dB.Gn[n].size());
    nPVPQGens += idxs_gen_agg.back().size()+numfixed;

    if(idxs_gen_agg.back().size()==0) {
      assert(Qagglb==0. && Qaggub==0.);
      idxs_gen_agg.pop_back();
      num_buses_all_qgen_fixed++;

      printf("%5d %5d %12.5e %12.5e        na      na       | ", n, data_sc.N_Bus[n], v_n0->x[n], v_nk->x[n]);
      for(auto g: dB.Gn[n])
	printf("%4d %12.5e %12.5e %12.5e : ", g, q_gk->x[g], dB.G_Qlb[g], dB.G_Qub[g]);
      printf("\n");

    } else {
      Qlb.push_back(Qagglb);
      Qub.push_back(Qaggub);
      idxs_bus_pvpq.push_back(n);

      int idx_nu = idxs_bus_pvpq.size()-1;

      printf("%5d %5d %12.5e %12.5e %12.5e %12.5e | ", n, data_sc.N_Bus[n], 
	     v_n0->x[n], v_nk->x[n], rhom->x[idx_nu], rhop->x[idx_nu]);
      for(auto g: dB.Gn[n])
	printf("%4d %12.5e %12.5e %12.5e : ", g, q_gk->x[g], dB.G_Qlb[g], dB.G_Qub[g]);
      printf("\n");
    }
  }
  assert(idxs_gen_agg.size() == Qlb.size());
  assert(idxs_gen_agg.size() == Qub.size());
  assert(N_PVPQ.size()  == num_buses_all_qgen_fixed+idxs_gen_agg.size());
  assert(idxs_bus_pvpq.size() == idxs_gen_agg.size());

}

void SCACOPFProblem::print_Transf_powers(SCACOPFData& dB, bool SysCond_BaseCase)
{
  auto p_ti1 = variable("p_ti1", dB),  p_ti2 = variable("p_ti2", dB);
  auto q_ti1 = variable("q_ti1", dB),  q_ti2 = variable("q_ti2", dB);
  vector<double>& T_Rate = SysCond_BaseCase ? dB.T_RateBase : dB.T_RateEmer;
  assert(T_Rate.size() == p_ti1->n);
  assert(T_Rate.size() == p_ti2->n);
  assert(T_Rate.size() == q_ti1->n);
  assert(T_Rate.size() == q_ti2->n);
  assert(T_Rate.size() == dB.T_Transformer.size());
  

  printf("   #    ID     pti1        pti2         qt1         qt2       rate\n");
  for(int t=0; t<dB.T_Transformer.size(); t++) {

    double aux;
    aux = p_ti1->x[t]*p_ti1->x[t] + q_ti1->x[t]*q_ti1->x[t];
    aux = sqrt(aux);

    printf("%4d %4d %12.5e %12.5e %12.5e %12.5e %12.5e\n",
	   t, dB.T_Transformer[t],
	   p_ti1->x[t], p_ti2->x[t], q_ti1->x[t], q_ti2->x[t], 
	   T_Rate[t]);
	   
  }

}

void SCACOPFProblem::write_solution_basecase()
{
  //data_sc is for the basecase
  OptVariablesBlock* p_g = variable("p_g", data_sc);
  if(NULL==p_g) {
    printf("[warning] no solution was written; p_g0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* q_g = variable("q_g", data_sc);
  if(NULL==q_g) {
    printf("[warning] no solution was written; q_g0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* v_n = variable("v_n", data_sc);
  if(NULL==v_n) {
    printf("[warning] no solution was written; v_n0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* theta_n = variable("theta_n", data_sc);
  if(NULL==theta_n) {
    printf("[warning] no solution was written; theta_n0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* b_s = variable("b_s", data_sc);
  if(NULL==b_s) {
    printf("[warning] no solution was written; b_s0 is missing from the problem\n");
    return;
  }
  auto SSh_Nidx = indexin(data_sc.SSh_Bus, data_sc.N_Bus);
  double bcsn[data_sc.N_Bus.size()];
  
  for(int i=0; i<data_sc.N_Bus.size(); i++) 
    bcsn[i] = 0.;

  for(int itssh = 0; itssh<data_sc.SSh_SShunt.size(); itssh++) {
    assert(itssh < SSh_Nidx.size());
    assert(itssh < b_s->n);
    assert(SSh_Nidx[itssh] < data_sc.N_Bus.size());
    assert(SSh_Nidx[itssh]>=0);
    
    bcsn[SSh_Nidx[itssh]] += b_s->x[itssh];
  }

  for(int i=0; i<data_sc.N_Bus.size(); i++) 
    bcsn[i] *= data_sc.MVAbase;

  string strFileName = "solution1.txt";
  FILE* file = fopen(strFileName.c_str(), "w");
  if(NULL==file) {
    printf("[warning] could not open [%s] file for writing\n", strFileName.c_str());
    return;
  }
  //
  // write bus section
  //
  fprintf(file, "--bus section\ni, v(p.u.), theta(deg), bcs(MVAR at v = 1 p.u.)\n");
  for(int n=0; n<data_sc.N_Bus.size(); n++) {
    fprintf(file, "%d, %.20f, %.20f, %.20f\n", 
	    data_sc.N_Bus[n], v_n->x[n], 180/M_PI*theta_n->x[n], bcsn[n]);
  }

  int GID = SCACOPFData::GID;
  int GI  = SCACOPFData::GI; assert(GI==0);
  assert(data_sc.generators[GID].size() == data_sc.generators[GI].size());
  assert(data_sc.generators[GID].size()>=data_sc.G_Generator.size());

  auto gmap = vector<int>(data_sc.generators[GID].size(), -1);
  for(int g=0; g<data_sc.G_Generator.size(); g++) {
    assert(g>=0);
    assert(data_sc.G_Generator[g]<data_sc.generators[GID].size());
    
    gmap[data_sc.G_Generator[g]] = g;
  }

  assert(data_sc.G_Bus.size() == data_sc.G_BusUnitNum.size());

  int g;
  //write generator section
  fprintf(file, "--generator section\ni, id, p(MW), q(MW)\n");
  for(int gi=0; gi<data_sc.generators[GI].size(); gi++) {
    g = gmap[gi];
    if(-1 == g) {
      fprintf(file, "%s, \'%s\', 0, 0\n", data_sc.generators[GI][gi].c_str(), data_sc.generators[GID][gi].c_str());
    } else {
      assert(g>=0);
      assert(g<data_sc.G_Bus.size());
      assert(g<p_g->n);
      assert(g<q_g->n);


      fprintf(file, "%d, \'%s\', %.12f, %.12f\n", 
	      data_sc.G_Bus[g], data_sc.G_BusUnitNum[g].c_str(), 
	      data_sc.MVAbase*p_g->x[g], data_sc.MVAbase*q_g->x[g]);
    }
  }
  fclose(file);
  printf("basecase solution written to file %s\n", strFileName.c_str());
}

void SCACOPFProblem::write_solution_extras_basecase()
{
  // balance slacks
  auto pf_p_bal = dynamic_cast<PFActiveBalance*>(constraint("p_balance", data_sc));
  OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
  pf_p_bal->compute_slacks(pslacks_n); pslacks_n->providesStartingPoint=true;
  auto pf_q_bal = dynamic_cast<PFReactiveBalance*>(constraint("q_balance", data_sc));
  OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
  pf_q_bal->compute_slacks(qslacks_n); qslacks_n->providesStartingPoint=true;
  
  // line flows & slacks
  OptVariablesBlock* p_li1 = variable("p_li1", data_sc);
  OptVariablesBlock* q_li1 = variable("q_li1", data_sc);
  OptVariablesBlock* p_li2 = variable("p_li2", data_sc);
  OptVariablesBlock* q_li2 = variable("q_li2", data_sc);
  if(NULL==p_li1 || NULL==q_li1 || NULL==p_li2 || NULL==q_li2) {
    printf("[warning] no solution was written; line flows are missing from the problem\n");
    return;
  }
  auto pf_line_lim1 = dynamic_cast<PFLineLimits*>(constraint("line_limits1", data_sc));
  OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
  pf_line_lim1->compute_slacks(sslack_li1); sslack_li1->providesStartingPoint=true;      
  auto pf_line_lim2 = dynamic_cast<PFLineLimits*>(constraint("line_limits2", data_sc));
  OptVariablesBlock* sslack_li2 = pf_line_lim2->slacks();
  pf_line_lim2->compute_slacks(sslack_li2); sslack_li2->providesStartingPoint=true;      
  
  // transformer flows & slacks
  OptVariablesBlock* p_ti1 = variable("p_ti1", data_sc);
  OptVariablesBlock* p_ti2 = variable("p_ti2", data_sc);
  OptVariablesBlock* q_ti1 = variable("q_ti1", data_sc);
  OptVariablesBlock* q_ti2 = variable("q_ti2", data_sc);
  OptVariablesBlock* q_g = variable("q_g", data_sc);
  if(NULL==p_ti1 || NULL==q_ti1 || NULL==p_ti2 || NULL==q_ti2) {
    printf("[warning] no solution was written; line flows are missing from the problem\n");
    return;
  }
  auto pf_trans_lim1 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits1", data_sc));
  OptVariablesBlock* sslack_ti1 = pf_trans_lim1->slacks();
  pf_trans_lim1->compute_slacks(sslack_ti1); sslack_ti1->providesStartingPoint=true;
  auto pf_trans_lim2 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits2", data_sc));
  OptVariablesBlock* sslack_ti2 = pf_trans_lim2->slacks();
  pf_trans_lim2->compute_slacks(sslack_ti2); sslack_ti2->providesStartingPoint=true;
  
  string strFileName = "solution1_extras.txt";
  FILE* file = fopen(strFileName.c_str(), "w");
  if(NULL==file) {
    printf("[warning] could not open [%s] file for writing\n", strFileName.c_str());
    return;
  }
  
  // write bus section
  fprintf(file, "--bus section\nI,pslack,qslack\n");
  int NumBuses = data_sc.N_Bus.size();
  for(int n=0; n<NumBuses; n++) {
    fprintf(file, "%d,%.20f,%.20f\n", data_sc.N_Bus[n],
            pslacks_n->x[n] - pslacks_n->x[NumBuses+n],
            qslacks_n->x[n] - qslacks_n->x[NumBuses+n]);
  }
  
  // write line section
  fprintf(file, "--line section\nI,J,CKT,p1,q1,sslack1,p2,q2,sslack2\n");
  for(int l=0; l<data_sc.L_Line.size(); l++) {
    fprintf(file, "%d,%d,%s,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f\n", 
	    data_sc.L_From[l], data_sc.L_To[l], data_sc.L_CktID[l].c_str(),
            p_li1->x[l], q_li1->x[l], sslack_li1->x[l],
            p_li2->x[l], q_li2->x[l], sslack_li2->x[l]);
  }
  
  // write transformer sections
  fprintf(file, "--transformer section\nI,J,CKT,p1,q1,sslack1,p2,q2,sslack2\n");
  for(int t=0; t<data_sc.T_Transformer.size(); t++) {
    fprintf(file, "%d,%d,%s,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f\n", 
	    data_sc.T_From[t], data_sc.T_To[t], data_sc.T_CktID[t].c_str(),
            p_ti1->x[t], q_ti1->x[t], sslack_ti1->x[t],
            p_ti2->x[t], q_ti2->x[t], sslack_ti2->x[t]);
  }
  
  fclose(file);
  printf("basecase solution extras written to file %s\n", strFileName.c_str());
}
} //end namespace

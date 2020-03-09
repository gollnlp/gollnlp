#include "SCACOPFProblem.hpp"

#include "OPFConstraints.hpp"
#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

#include "OptObjTerms.hpp"

#include "SCACOPFIO.hpp"

#include <numeric>
#include <typeinfo> 

#include <iostream>

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
  slacks_scale = 1.;//128.;

  SCACOPFData& d = data_sc; //shortcut

  //
  // base case
  //
  add_variables(d);

  add_cons_lines_pf(d);
  add_cons_transformers_pf(d);
  add_cons_active_powbal(d);
  add_cons_reactive_powbal(d);

  vector<double> rate; size_t sz; double r_emer, r_base;

  rate = d.L_RateBase; sz = d.L_RateBase.size();
  //d.L_RateBase : d.L_RateEmer;
  for(int it=0; it<sz; it++) {
    r_emer = L_rate_reduction * d.L_RateEmer[it];
    r_base =                    d.L_RateBase[it];
  }
  add_cons_thermal_li_lims(d, rate);

  rate = d.T_RateBase; sz = d.T_RateBase.size();
  //d.L_RateBase : d.L_RateEmer;
  for(int it=0; it<sz; it++) {
    r_emer = T_rate_reduction * d.T_RateEmer[it];
    r_base =                    d.T_RateBase[it];
  }
  add_cons_thermal_ti_lims(d, rate);

  add_obj_prod_cost(d);

    if(quadr_penalty_qg0) {
    OptVariablesBlock* q_g0 = variable("q_g", d); assert(q_g0);
    append_objterm(new QuadrAwayFromBoundsObjTerm(string("qg0_quadr_pen")+q_g0->id,
  						  q_g0, 1., d.G_Qlb.data(), d.G_Qub.data()));
  }			  


  // append_objterm(new GenerKPenaltyObjTerm("penalty_gener_from_conting", variable("p_g", d)));
  // append_objterm(new TransmKPenaltyObjTerm("penalty_line_from_conting",
  // 					   variable("p_li1", d), variable("q_li1", d), 
  // 					   variable("p_li2", d), variable("q_li2", d)));
  // append_objterm(new TransmKPenaltyObjTerm("penalty_transf_from_conting",
  // 					   variable("p_ti1", d), variable("q_ti1", d), 
  // 					   variable("p_ti2", d), variable("q_ti2", d)));


  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_gen_activ_power", variable("p_g", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_gen_reactiv_power", variable("q_g", d)));

  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_line_activ_power1", variable("p_li1", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_line_activ_power2", variable("p_li2", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_line_reactiv_power1", variable("q_li1", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_line_reactiv_power2", variable("q_li2", d)));

  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_transf_activ_power1", variable("p_ti1", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_transf_activ_power2", variable("p_ti2", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_transf_reactiv_power1", variable("q_ti1", d)));
  append_objterm(new QuadrBarrierPenaltyObjTerm("pen_conting_transf_reactiv_power2", variable("q_ti2", d)));

  append_objterm(new VoltageKPenaltyObjTerm("penalty_voltage_from_conting", variable("v_n", d)));

  //if(d.N_Bus.size()<=52000) 
  {
    add_agc_reserves_for_max_Lloss_Ugain();
    add_agc_reserves();
  }
  return true;
}

bool SCACOPFProblem::assembly(const std::vector<int>& K_Cont)
{
  //assemble base case first
  default_assembly();
  
  SCACOPFData& d = data_sc; //shortcut
  int nK = K_Cont.size();

  for(auto K : K_Cont) {
    add_contingency_block(K);
  }
  //print_summary();
  //printf("\n!!![best_known] initialize111 rank=%d\n\n", my_rank);
  best_known_iter.initialize(vars_primal, vars_duals_cons, vars_duals_bounds_L, vars_duals_bounds_U);

  return true;
}

bool SCACOPFProblem::add_contingency_block(const int K)
{
  SCACOPFData& d = data_sc; //shortcut
  data_K.push_back(new SCACOPFData(data_sc));

  SCACOPFData& dK = *(data_K).back(); //shortcut
  dK.rebuild_for_conting(K,1);

  //
  // update penalties for the problem
  double new_penalty_weight = (1-d.DELTA) / data_K.size();
  //double new_penalty_weight = (1-d.DELTA) / data_sc.K_Contingency.size();
  for(SCACOPFData* d : data_K) d->PenaltyWeight=new_penalty_weight;

  printf("adding blocks for contingency K=%d IDOut=%d outidx=%d Type=%s agc=%g pvpq=%g\n", 
	 K, d.K_IDout[K], d.K_outidx[K], d.cont_type_string(K).c_str(),
	 AGCSmoothing, PVPQSmoothing);
  
  bool SysCond_BaseCase = false;

  add_variables(dK, SysCond_BaseCase);
  add_cons_lines_pf(dK);
  add_cons_transformers_pf(dK);
  add_cons_active_powbal(dK);
  add_cons_reactive_powbal(dK);
  
  add_cons_thermal_li_lims(dK,SysCond_BaseCase);
  add_cons_thermal_ti_lims(dK,SysCond_BaseCase);
  
  //coupling AGC and PVPQ; also creates delta_k
  add_cons_coupling(dK);
  


  return true;
}

bool SCACOPFProblem::has_contigency(const int K_idx)
{
  for(SCACOPFData* d : data_K) 
    if(K_idx == d->id-1) {
      assert(d->K_outidx[0] == data_sc.K_outidx[K_idx]);
      return true;
    }
  return false;
}

std::vector<int> SCACOPFProblem::get_contingencies() const
{
  vector<int> v (data_K.size());
  for(int i=0; i<data_K.size(); i++)
    v[i] = data_K[i]->id-1;
  return v;
}

void SCACOPFProblem::add_cons_coupling(SCACOPFData& dB)
{
  int K_id = dB.K_Contingency[0];

  //indexes in data_sc.G_Generator
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  assert(Gk.size() == dB.G_Generator.size());

  if(AGC_simplified && AGC_as_nonanticip) {
    
    assert(false && "cannot have AGC_as_nonanticip "
	   "and AGC_simplified in the same time");
    printf("[warning] disabled AGC_simplified since AGC_as_nonanticip is enabled.\n");
    AGC_simplified = false;
  }

  if(AGC_as_nonanticip) {
    add_cons_pg_nonanticip(dB, Gk); 
    add_cons_AGC(dB, {}); //just to print the "?!? no AGC" message
  } else {
    if(AGC_simplified) {
      add_cons_pg_nonanticip(dB, Gknop);
      add_cons_AGC_simplified(dB, Gkp);
    } else {
      add_cons_pg_nonanticip(dB, Gknop);
      add_cons_AGC(dB, Gkp);
    }
  }

  //voltages
  if(PVPQ_as_nonanticip) {
    add_cons_PVPQ_as_vn_nonanticip(dB, Gk);
  } else {
    add_cons_PVPQ(dB, Gk);
  }
}
void SCACOPFProblem::get_idxs_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk,
				   vector<vector<int> >& idxs_gen_agg, vector<int>& idxs_bus_pvpq,
				   std::vector<double>& Qlb, std::vector<double>& Qub,
				   int& nPVPQGens, int &num_qgens_fixed, 
				   int& num_N_PVPQ, int& num_buses_all_qgen_fixed)
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


  int nPVPQCons=0; nPVPQGens=0; num_buses_all_qgen_fixed=0; num_qgens_fixed=0;

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
	//printf("PVPQ: gen ID=%d p_q at bus idx %d id %d is fixed; will not add PVPQ constraint\n",
	//       dB.G_Generator[g], dB.G_Nidx[g], data_sc.N_Bus[dB.G_Nidx[g]]);
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
  num_N_PVPQ = N_PVPQ.size();
}

// Gk are the indexes of all gens other than the outgen (for generator contingencies) 
// in data_sc.G_Generator
void SCACOPFProblem::add_cons_PVPQ_as_vn_nonanticip(SCACOPFData& dB, const std::vector<int>& Gk)
{
  //(aggregated) non-fixed q_g generator ids at each node/bus
  // with PVPQ generators that have at least one non-fixed q_g 
  vector<vector<int> > idxs_gen_agg;
  //bus indexes that have at least one non-fixed q_g
  vector<int> idxs_bus_pvpq;
  //aggregated lb and ub on reactive power at each PVPQ bus
  vector<double> Qlb, Qub;
  int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;

  get_idxs_PVPQ(dB, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
		nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);

  auto v_n0 = variable("v_n", data_sc);
  auto v_nk = variable("v_n", dB);

  if(NULL==v_n0) {
    printf("Contingency %d: v_n var not found in the base case; will NOT add PVPQ constraints.\n", dB.id);
    return;
  }
  if(NULL==v_nk) {
    printf("Contingency %d: v_n var not found in conting problem; will NOT add PVPQ constraints.\n", dB.id);
    return;
  }

  auto cons = new NonAnticipCons(con_name("volt_non_anticip",dB), idxs_bus_pvpq.size(), 
  				 v_n0, v_nk, idxs_bus_pvpq, idxs_bus_pvpq);
  append_constraints(cons);
  

  printf("PVPQ: participating %d gens at %lu buses: added %d NONANTICIP constraints on voltages;"
	 "total PVPQ: %lu gens | %d buses; were fixed: %d gens | %d buses with all gens fixed.\n",
	 nPVPQGens-num_qgens_fixed, idxs_bus_pvpq.size(), cons->n,
	 Gk.size(), num_N_PVPQ,
	 num_qgens_fixed, num_buses_all_qgen_fixed);

}

// Gk are the indexes of all gens other than the outgen (for generator contingencies) 
// in data_sc.G_Generator
void SCACOPFProblem::add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk)
{
  //(aggregated) non-fixed q_g generator ids at each node/bus
  // with PVPQ generators that have at least one non-fixed q_g 
  vector<vector<int> > idxs_gen_agg;
  //bus indexes that have at least one non-fixed q_g
  vector<int> idxs_bus_pvpq;
  //aggregated lb and ub on reactive power at each PVPQ bus
  vector<double> Qlb, Qub;
  int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;

  get_idxs_PVPQ(dB, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
		nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);

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
  
  printf("PVPQ: participating %d gens at %lu buses: added %d constraints; PVPQSmoothing=%g "
	 "total PVPQ: %lu gens | %d buses; were fixed: %d gens | %d buses with all gens fixed.\n",
	 nPVPQGens-num_qgens_fixed, idxs_bus_pvpq.size(), cons->n, PVPQSmoothing,
	 Gk.size(), num_N_PVPQ,
	 num_qgens_fixed, num_buses_all_qgen_fixed);
}

void SCACOPFProblem::update_PVPQ_smoothing_param(const double& val)
{
  PVPQSmoothing = val;
  for(auto d : data_K) {

    PVPQComplementarityCons* cons = dynamic_cast<PVPQComplementarityCons*>(constraint("PVPQ", *d));
    if(cons) cons->update_smoothing(val);
    //else assert(false);
  }
}

void SCACOPFProblem::add_cons_pg_nonanticip(SCACOPFData& dB, const std::vector<int>& G_idxs_no_AGC)
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
    
    auto cons = new NonAnticipCons(con_name("pg_non_anticip",dB), G_idxs_no_AGC.size(),
				   pg0, pgK, G_idxs_no_AGC, conting_matching_idxs);
    append_constraints(cons);
  }
  printf("AGC: %lu gens: added one NONANTICIP constraint for each\n", G_idxs_no_AGC.size());
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

  printf("AGC: %lu gens participating: added %d constraints "
	 "AGCSmoothing=%g\n", G_idxs_AGC.size(), cons->n, AGCSmoothing);
}

void SCACOPFProblem::update_AGC_smoothing_param(const double& val)
{
  AGCSmoothing = val;
  for(auto d : data_K) {

    AGCComplementarityCons* cons = dynamic_cast<AGCComplementarityCons*>(constraint("AGC", *d));
    if(cons) cons->update_smoothing(val);
    //else assert(false);
  }
}

void SCACOPFProblem::add_cons_AGC_simplified(SCACOPFData& dB, const std::vector<int>& G_idxs_AGC)
{
  if(G_idxs_AGC.size()==0) {
    printf("[warning] SCACOPFProblem: add_cons_AGC_simplified: NO gens participating !?! in contingency %d\n", dB.id);
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
    printf("Contingency %d: p_g var not found in the base case; will NOT add AGC simplifiedconstraints.\n", dB.id);
    return;
  }
  OptVariablesBlock* pgK = variable("p_g", dB);
  if(NULL==pgK) {
    printf("Contingency %d: p_g var not found in conting problem; will NOT add AGC simplified constraints.\n", dB.id);
    return;
  }

  OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dB));
  append_variables(deltaK);
  deltaK->set_start_to(0.);

  auto cons = new AGCSimpleCons(con_name("AGC_simple", dB), G_idxs_AGC.size(),
				pg0, pgK, deltaK, 
				G_idxs_AGC, conting_matching_idxs,
				data_sc.G_alpha);

  append_constraints(cons);

  printf("AGC: %lu gens participating: added %d SIMPLIFIED constraints\n", G_idxs_AGC.size(), cons->n);
}

void SCACOPFProblem::add_agc_reserves() 
{
  SCACOPFData& d = data_sc;  
  //indexes in K_Contingency of generator contingencies
  vector<int> idxsKGen = findall(d.K_ConType, [](int val) {return val==SCACOPFData::kGenerator;});
  vector<int> K_gens_ids = selectfrom(d.K_IDout, idxsKGen);
  //indexes in G_Generators of the generators subject to contingencies
  vector<int> K_gens_idxs= indexin(K_gens_ids, d.G_Generator);
#ifdef DEBUG
  //all must be present
  assert(K_gens_ids == selectfrom(d.G_Generator, K_gens_idxs));
#endif

  //all generators
  auto Gk = vector<int>(d.G_Generator.size()); iota(Gk.begin(), Gk.end(), 0);
  assert(d.G_Nidx.size() == d.G_Generator.size());
  //area of each generator
  auto Garea = selectfrom(d.N_Area, d.G_Nidx);
  Garea = selectfrom(Garea, Gk);
  assert(Garea.size() == Gk.size());

  auto areas = Garea;
  remove_duplicates(areas);
  //printvec(Garea);
  //printvec(areas);

  auto pg0 = variable("p_g", d);

  AGCReservesCons* loss_rsrv = new AGCReservesCons(con_name("agc_reserves_loss_Kgen", d), pg0);
  AGCReservesCons* gain_rsrv = new AGCReservesCons(con_name("agc_reserves_gain_Kgen", d), pg0);

#ifdef DEBUG
  vector<int> Kidxs_agc_loss_cons, Kidxs_agc_gain_cons;
  string str=""; char msg[1024];
#endif	
					       

  for(auto area: areas) {
    auto AGC_gens_idxs_area = findall(Garea, [&](int val) {return val==area;});
    auto K_gens_idxs_area = findall(K_gens_idxs, [&](int val) { return Garea[val]==area;});
    K_gens_idxs_area = selectfrom(K_gens_idxs, K_gens_idxs_area);

    //printf("area %d -> AGC gens\n", area);
    //printvec(AGC_gens_idxs_area);
    //printf("area %d -> generator IDs subject to contingency\n", area);
    //printvec(selectfrom(d.G_Generator,K_gens_idxs_area));

#ifdef DEBUG
    //printf("agc reserves Kgen: area %d has %d AGC gens and %d gens subj. to. contingencies\n",
    //   area, AGC_gens_idxs_area.size(), K_gens_idxs_area.size());
#endif
    if(AGC_gens_idxs_area.size()>100) continue;
    int max_num_Kgen = K_gens_idxs_area.size()>50 ? 50 : K_gens_idxs_area.size();

    //
    // power loss due to contingency
    //
    //sort based on the max capacity of the contingency generators
    auto K_gens_idxs_area_sorted = K_gens_idxs_area;
    sort(K_gens_idxs_area_sorted.begin(), K_gens_idxs_area_sorted.end(), 
	 [&](const int& a, const int& b) { return (d.G_Pub[a] > d.G_Pub[b]); });


    for(int kgi=0; kgi<max_num_Kgen; kgi++) {
      int Kgen_idx = K_gens_idxs_area_sorted[kgi];

      //power injection - skip
      if(d.G_Pub[Kgen_idx]<=0) continue;

      //remove Kgen from AGC in case it is in there
      auto responding_AGC_gens_idxs_area = AGC_gens_idxs_area;
      erase_elem_from(responding_AGC_gens_idxs_area, Kgen_idx);

      double percentage_of_loss = 1.;
      loss_rsrv->add_Kgen_loss_reserve(responding_AGC_gens_idxs_area, Kgen_idx, percentage_of_loss, d.G_Pub);
#ifdef DEBUG
      int K_idx = -1;
      for(int i=0; i<d.K_Contingency.size(); i++) 
	if(d.K_ConType[i]==SCACOPFData::kGenerator && d.K_IDout[i]==d.G_Generator[Kgen_idx]) K_idx = i;
      assert( K_idx >= 0);
      Kidxs_agc_loss_cons.push_back(K_idx);
#endif
    }

    //
    // power injection due to contingency
    //
    K_gens_idxs_area_sorted = K_gens_idxs_area;
    sort(K_gens_idxs_area_sorted.begin(), K_gens_idxs_area_sorted.end(), 
	 [&](const int& a, const int& b) { return (d.G_Plb[a] < d.G_Plb[b]); });

    for(int kgi=0; kgi<max_num_Kgen; kgi++) {
      int Kgen_idx = K_gens_idxs_area_sorted[kgi];

      if(d.G_Plb[Kgen_idx]>=0) continue;

      //remove Kgen from AGC in case it is in there
      auto responding_AGC_gens_idxs_area = AGC_gens_idxs_area;
      erase_elem_from(responding_AGC_gens_idxs_area, Kgen_idx);

      double percentage_of_loss = 1.;
      gain_rsrv->add_Kgen_gain_reserve(responding_AGC_gens_idxs_area, Kgen_idx, percentage_of_loss, d.G_Plb);
#ifdef DEBUG
      int K_idx = -1;
      for(int i=0; i<d.K_Contingency.size(); i++) 
	if(d.K_ConType[i]==SCACOPFData::kGenerator && d.K_IDout[i]==d.G_Generator[Kgen_idx]) K_idx = i;
      assert( K_idx >= 0);
      Kidxs_agc_gain_cons.push_back(K_idx);
#endif
    }
  } // end loop over areas

  if(loss_rsrv->n>0) {

    double obj_weight = (1-d.DELTA) / d.K_Contingency.size();
    //double obj_weight = 1;//(1-d.DELTA);// / 10;//d.K_Contingency.size();

    loss_rsrv->add_penalty_objterm(d.P_Penalties[SCACOPFData::pP], 
				   d.P_Quantities[SCACOPFData::pP],
				   obj_weight, 
				   slacks_scale);
    loss_rsrv->finalize_setup();

    this->append_constraints(loss_rsrv);
#ifdef DEBUG
    if(my_rank==1) {
      sprintf(msg, " -- added %d AGC loss reserves for contingencies\n", loss_rsrv->n);
      str += msg;
    }
#endif


  } else {
    delete loss_rsrv;
  }

  if(gain_rsrv->n>0) {
    double obj_weight = (1-d.DELTA) / d.K_Contingency.size();
    //double obj_weight = (1-d.DELTA) / 10;//d.K_Contingency.size();

    gain_rsrv->add_penalty_objterm(d.P_Penalties[SCACOPFData::pP], 
				   d.P_Quantities[SCACOPFData::pP],
				   (1-d.DELTA) / d.K_Contingency.size(), //weight
				   slacks_scale);
    gain_rsrv->finalize_setup();

    this->append_constraints(gain_rsrv);
#ifdef DEBUG
    if(my_rank==1) {
      sprintf(msg, " -- added %d AGC gain reserves for contingencies\n", gain_rsrv->n);
      str += msg;
    //printvec(Kidxs_agc_gain_cons, "K_idxs");
    }
#endif

  } else {
    delete gain_rsrv;
  }
#ifdef DEBUG
  if(str.size()>0) printf("add_agc_reserves\n%s", str.c_str());
#endif
}  

void SCACOPFProblem::find_AGC_infeasible_Kgens(std::vector<int>& agc_infeas_gen_idxs, 
					       std::vector<int>& agc_infeas_K_idxs)
{
  agc_infeas_gen_idxs.clear(); agc_infeas_K_idxs.clear();

  SCACOPFData& d = data_sc; 
  //! optimize - should reuse these as they are computed in add_agc_reservesXXX
  //indexes in K_Contingency of generator contingencies
  vector<int> idxsKGen = findall(d.K_ConType, [](int val) {return val==SCACOPFData::kGenerator;});
  vector<int> K_gens_ids = selectfrom(d.K_IDout, idxsKGen);
  //indexes in G_Generators of the generators subject to contingencies
  vector<int> K_gens_idxs= indexin(K_gens_ids, d.G_Generator);
#ifdef DEBUG
  //all must be present
  assert(K_gens_ids == selectfrom(d.G_Generator, K_gens_idxs));
#endif

  //compute areas and area of each generator
  auto Gk = vector<int>(d.G_Generator.size()); iota(Gk.begin(), Gk.end(), 0);
  assert(d.G_Nidx.size() == d.G_Generator.size());
  //area of each generator
  auto Garea = selectfrom(d.N_Area, d.G_Nidx);
  Garea = selectfrom(Garea, Gk);
  assert(Garea.size() == Gk.size());
  
  auto areas = Garea;
  remove_duplicates(areas);

  for(auto area: areas) {
    //AGC generators in the area
    auto AGC_gens_idxs_area = findall(Garea, [&](int val) {return val==area;});

    //gens subj. to. contingencies in the area
    auto K_gens_idxs_area = findall(K_gens_idxs, [&](int val) { return Garea[val]==area;});
    K_gens_idxs_area = selectfrom(K_gens_idxs, K_gens_idxs_area);

    for(int gkidx : K_gens_idxs_area) {
      //compute maximum possible pg ramping
      double ramp=0.;
      for(int gagcidx: AGC_gens_idxs_area) {
	if(gkidx != gagcidx) ramp += (d.G_Pub[gagcidx]-d.G_Plb[gagcidx]);
      }
      bool loss_infeas = (  d.G_Plb[gkidx] > ramp);
      bool gain_infeas = (0-d.G_Pub[gkidx] > ramp);
      if(loss_infeas || gain_infeas) {
	agc_infeas_gen_idxs.push_back(gkidx);

	int K_idx=-1; assert(d.K_outidx.size() == d.K_ConType.size());
	for(int ki=0; ki<d.K_outidx.size(); ki++) {
	  if(d.K_outidx[ki]==gkidx && d.K_ConType[ki]==SCACOPFData::kGenerator) {
	    K_idx=ki;
	    break;
	  }
	}
	assert(K_idx>=0); assert(d.K_IDout[K_idx]==d.G_Generator[gkidx]);

	agc_infeas_K_idxs.push_back(K_idx);
#ifdef DEBUG
	printf("[warning] min %s of %g from gen_idx=%d in K_idx=%d cannot be recovered"
	       " only from AGC max response of %g\n", 
	       loss_infeas ? "loss" : "gain", loss_infeas ? d.G_Plb[gkidx] : -d.G_Pub[gkidx],
	       gkidx, K_idx, ramp);
#endif
      }
	
    }
  }
}

void SCACOPFProblem::add_agc_reserves_for_max_Lloss_Ugain()
{
  SCACOPFData& d = data_sc; 
 
  //indexes in K_Contingency of generator contingencies
  vector<int> idxsKGen = findall(d.K_ConType, [](int val) {return val==SCACOPFData::kGenerator;});
  vector<int> K_gens_ids = selectfrom(d.K_IDout, idxsKGen);
  //indexes in G_Generators of the generators subject to contingencies
  vector<int> K_gens_idxs= indexin(K_gens_ids, d.G_Generator);
#ifdef DEBUG
  //all must be present
  assert(K_gens_ids == selectfrom(d.G_Generator, K_gens_idxs));
#endif

  //all generators
  auto Gk = vector<int>(d.G_Generator.size()); iota(Gk.begin(), Gk.end(), 0);
  assert(d.G_Nidx.size() == d.G_Generator.size());
  //area of each generator
  auto Garea = selectfrom(d.N_Area, d.G_Nidx);
  Garea = selectfrom(Garea, Gk);
  assert(Garea.size() == Gk.size());

  auto areas = Garea;
  remove_duplicates(areas);
  //printvec(Garea);
  //printvec(areas);

  AGCReservesCons* loss_rsrv = new AGCReservesCons(con_name("agc_reserves_loss_bnd", d),
						   variable("p_g", d));
  AGCReservesCons* gain_rsrv = new AGCReservesCons(con_name("agc_reserves_gain_bnd", d),
						   variable("p_g", d));
#ifdef DEBUG
    string str=""; char msg[1024]; 
#endif						       

  for(auto area: areas) {
    auto AGC_gens_idxs_area = findall(Garea, [&](int val) {return val==area;});
    auto K_gens_idxs_area = findall(K_gens_idxs, [&](int val) { return Garea[val]==area;});
    K_gens_idxs_area = selectfrom(K_gens_idxs, K_gens_idxs_area);


    //printf("area %d -> AGC gens\n", area);
    //printvec(AGC_gens_idxs_area);
    //printf("area %d -> generator IDs subject to contingency\n", area);
    //printvec(selectfrom(d.G_Generator,K_gens_idxs_area));

#ifdef DEBUG
    //printf("agc Lloss_Ugain reserves lb-ub: area %d has %d AGC gens and %d gens subj. to. contingencies\n",
    //	   area, AGC_gens_idxs_area.size(), K_gens_idxs_area.size());
#endif
    if(AGC_gens_idxs_area.size()>100) continue;

    double max_loss = 0., max_gain=0., aux; int max_loss_idx=-1, max_gain_idx=-1;;
    for(int Kgenidx: K_gens_idxs_area) {
      aux = std::max( d.G_Plb[Kgenidx], 0.);
      if(aux>max_loss) { max_loss=aux; max_loss_idx=Kgenidx; }
      
      aux = std::max(-d.G_Pub[Kgenidx], 0.);
      if(aux>max_gain) { max_gain=aux; max_gain_idx=Kgenidx; }
    }


#ifdef DEBUG 
    if(my_rank==1) {
      
      if(max_loss>1e-6) {
	int K_idx = -1;
	for(int i=0; i<d.K_Contingency.size(); i++) 
	  if(d.K_ConType[i]==SCACOPFData::kGenerator && d.K_IDout[i]==d.G_Generator[max_loss_idx]) K_idx = i;
	assert( K_idx >= 0);
      
	sprintf(msg, " -- max loss for area %d in the amount of %g for Kgenidx %d id %d  K_idx %d\n",
		area, max_loss, max_loss_idx, d.G_Generator[max_loss_idx], K_idx);
	str += msg;
      }
      if(max_gain>1e-6) {
	int K_idx = -1;
	for(int i=0; i<d.K_Contingency.size(); i++) 
	  if(d.K_ConType[i]==SCACOPFData::kGenerator && d.K_IDout[i]==d.G_Generator[max_gain_idx]) K_idx = i;
	assert( K_idx >= 0);
      
	sprintf(msg, " -- max gain for area %d in the amount of %g for Kgenidx %d id %d  K_idx %d\n",
	       area, max_gain, max_gain_idx, d.G_Generator[max_gain_idx], K_idx);
	str += msg;
      }
      
    }
#endif
    
    if(max_loss>1e-6) {
      assert(max_loss_idx>=0);
      auto responding_AGC_gens_idxs_area = AGC_gens_idxs_area;
      erase_elem_from(responding_AGC_gens_idxs_area, max_loss_idx);

      double percentage_of_loss = 1.;
      loss_rsrv->add_max_loss_reserve(responding_AGC_gens_idxs_area, max_loss, percentage_of_loss, d.G_Pub);
    }
    if(max_gain>1e-6) {
      assert(max_gain_idx>=0);
      auto responding_AGC_gens_idxs_area = AGC_gens_idxs_area;
      erase_elem_from(responding_AGC_gens_idxs_area, max_gain_idx);

      double percentage_of_gain = 1.;
      gain_rsrv->add_max_gain_reserve(responding_AGC_gens_idxs_area, max_gain, percentage_of_gain, d.G_Plb);
    }
  } // end of loop over areas

  if(loss_rsrv->n>0) {
    loss_rsrv->add_penalty_objterm(d.P_Penalties[SCACOPFData::pP], 
				   d.P_Quantities[SCACOPFData::pP],
				   (1-d.DELTA) / d.K_Contingency.size(), //weight
				   slacks_scale);
    loss_rsrv->finalize_setup();

    this->append_constraints(loss_rsrv);
#ifdef DEBUG
    if(my_rank==1) {
      sprintf(msg, " -- added %d AGC loss (Lloss) reserve constraints for contingencies\n", loss_rsrv->n);
      str += msg;
      //printvec(Kidxs_agc_loss_cons, "K_idxs");
    }
#endif

  } else {
    delete loss_rsrv;
  }

  if(gain_rsrv->n>0) {
    gain_rsrv->add_penalty_objterm(d.P_Penalties[SCACOPFData::pP], 
				   d.P_Quantities[SCACOPFData::pP],
				   (1-d.DELTA) / d.K_Contingency.size(), //weight
				   slacks_scale);
    gain_rsrv->finalize_setup();

    this->append_constraints(gain_rsrv);
#ifdef DEBUG
    if(my_rank==1) {
      sprintf(msg, " -- added %d AGC gain (Ugain) reserve constraints for contingencies\n", loss_rsrv->n);
      str += msg;
      //printvec(Kidxs_agc_loss_cons, "K_idxs");
    }
#endif

  } else {
    delete gain_rsrv;
  }
#ifdef DEBUG
  if(str.size()>0) printf("[rank 1] reserves_Lloss_Ugain\n%s", str.c_str());
#endif
}

// void SCACOPFProblem::add_quadr_conting_penalty_pg0(const int& idx_gen, const double& p0, const double& f_pen)
// {
//   GenerKPenaltyObjTerm* ot = dynamic_cast<GenerKPenaltyObjTerm*>(objterm("penalty_gener_from_conting"));
//   assert(ot);
//   if(ot) ot->add_quadr_penalty(idx_gen, p0, f_pen, data_sc.G_Plb[idx_gen], data_sc.G_Pub[idx_gen]);
// }
// void SCACOPFProblem::remove_quadr_conting_penalty_pg0(const int& idx_gen)
// {
//   GenerKPenaltyObjTerm* ot = dynamic_cast<GenerKPenaltyObjTerm*>(objterm("penalty_gener_from_conting"));
//   assert(ot);
//   if(ot) ot->remove_penalty(idx_gen);
// }

// void SCACOPFProblem::add_conting_penalty_line0(const int& idx_line, 
// 					       const double& pli10, const double& qli10, 
// 					       const double& pli20, const double& qli20, 
// 					       const double& f_pen)
// {
//   TransmKPenaltyObjTerm* ot =  dynamic_cast<TransmKPenaltyObjTerm*>(objterm("penalty_line_from_conting"));
//   assert(ot);
//   if(ot) ot->add_penalty(idx_line, pli10, qli10, pli20, qli20, f_pen);
// }
// void SCACOPFProblem::remove_conting_penalty_line0(const int& idx_line)
// {
//   TransmKPenaltyObjTerm* ot =  dynamic_cast<TransmKPenaltyObjTerm*>(objterm("penalty_line_from_conting"));
//   assert(ot);
//   if(ot) ot->remove_penalty(idx_line);
// }

// void SCACOPFProblem::add_conting_penalty_transf0(const int& idx_transf, 
// 						 const double& pti10, const double& qti10, 
// 						 const double& pti20, const double& qti20, 
// 						 const double& f_pen)
// {
//   TransmKPenaltyObjTerm* ot =  dynamic_cast<TransmKPenaltyObjTerm*>(objterm("penalty_transf_from_conting"));
//   assert(ot);
//   if(ot) ot->add_penalty(idx_transf, pti10, qti10, pti20, qti20, f_pen);
// }
// void SCACOPFProblem::remove_conting_penalty_transf0(const int& idx_transf)
// {
//   TransmKPenaltyObjTerm* ot =  dynamic_cast<TransmKPenaltyObjTerm*>(objterm("penalty_transf_from_conting"));
//   assert(ot);
//   if(ot) ot->remove_penalty(idx_transf);
// }

bool SCACOPFProblem::
update_conting_penalty_gener_active_power(const int& K_idx, const int& g_idx,
					  const double& pg0, const double& delta_p, const double& pen0)
{
  QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_gen_activ_power"));
  assert(ot);
  if(ot) return ot->update_term(K_idx, g_idx, pg0, pen0, delta_p);
  else   return false;
}
bool SCACOPFProblem::
update_conting_penalty_gener_reactive_power(const int& K_idx, const int& g_idx,
					    const double& qg0, const double& delta_q, const double& pen0)
{
  QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_gen_reactiv_power"));
  assert(ot);
  if(ot) return ot->update_term(K_idx, g_idx, qg0, pen0, delta_q);
  return false;
}
bool SCACOPFProblem::
update_conting_penalty_line_active_power(const int& K_idx, const int& li_idx,
					 const double& pli10, const double& pli20, 
					 const double& delta_p, const double& pen0)
{
  const double pen = pen0/2;
  bool updated=false;
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_line_activ_power1"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, li_idx, pli10, pen, delta_p)) updated=true;
  }
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_line_activ_power2"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, li_idx, pli20, pen, delta_p)) updated=true;
  }
  return updated;
}
bool SCACOPFProblem::
update_conting_penalty_transf_active_power(const int& K_idx, const int& ti_idx,
					   const double& pti10, const double& pti20, 
					   const double& delta_p, const double& pen0)
{
  const double pen = pen0/2;
  bool updated=false;
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_transf_activ_power1"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, ti_idx, pti10, pen, delta_p)) updated=true;
  }
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_transf_activ_power2"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, ti_idx, pti20, pen, delta_p)) updated=true;
  }
  return updated;
}

bool SCACOPFProblem::
update_conting_penalty_line_reactive_power(const int& K_idx, const int& li_idx,
					   const double& qli10, const double& qli20, 
					   const double& delta_q, const double& pen0)
{
  const double pen = pen0/2;
  bool updated=false;
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_line_reactiv_power1"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, li_idx, qli10, pen, delta_q)) updated=true;
  }
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_line_reactiv_power2"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, li_idx, qli20, pen, delta_q)) updated=true;
  }
  return updated;
}
bool SCACOPFProblem::
update_conting_penalty_transf_reactive_power(const int& K_idx, const int& ti_idx,
					     const double& qti10, const double& qti20, 
					     const double& delta_q, const double& pen0)
{
  const double pen = pen0/2;
  bool updated=false;
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_transf_reactiv_power1"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, ti_idx, qti10, pen, delta_q)) updated=true;
  }
  {
    QuadrBarrierPenaltyObjTerm* ot = dynamic_cast<QuadrBarrierPenaltyObjTerm*>(objterm("pen_conting_transf_reactiv_power2"));
    assert(ot);
    if(ot) 
      if(ot->update_term(K_idx, ti_idx, qti20, pen, delta_q)) updated=true;
  }
  return updated;
}

bool SCACOPFProblem::update_conting_penalty_voltage(const int& K_idx, const int& N_idx, 
						    const double& v0, const double& pen0, 
						    const double& pen0_deriv)
{
  VoltageKPenaltyObjTerm* ot = dynamic_cast<VoltageKPenaltyObjTerm*>(objterm("penalty_voltage_from_conting"));
  assert(ot);
  if(ot) return ot->update_term(K_idx, N_idx, v0, pen0, pen0_deriv);
  else   return false;
}

void SCACOPFProblem::add_variables(SCACOPFData& d, bool SysCond_BaseCase)
{
  double* vlb = SysCond_BaseCase==true ?  data_sc.N_Vlb.data() : data_sc.N_EVlb.data();
  double* vub = SysCond_BaseCase==true ?  data_sc.N_Vub.data() : data_sc.N_EVub.data();

  //for(auto& v: data_sc.N_Vub) v*=1.5;
  //for(auto& v: data_sc.N_EVub) v*=1.5;
  //printvec(data_sc.N_Vub);
  //printvec(data_sc.N_EVub);


  //printvec(data_sc.N_Vub);
  auto v_n = new OptVariablesBlock(data_sc.N_Bus.size(), var_name("v_n",d), vlb, vub);
  //data_sc.N_EVlb.data(), data_sc.N_EVub.data()); 
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
    //! TODO : fix bug -> when N_theta0[RefBus]==0, lb and ub are not set to zero
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

  //for(auto& b : d.G_Plb) b -=1e-8;
  //for(auto& b : d.G_Pub) b +=1e-8;
  //printf("!!!!!!!!!!!!!! touched it\n");

  auto p_g = new OptVariablesBlock(d.G_Generator.size(), var_name("p_g",d), 
				   d.G_Plb.data(), d.G_Pub.data());

  append_variables(p_g); 
  p_g->set_start_to(d.G_p0.data());

  //auto Qlb = d.G_Qlb, Qub = d.G_Qub;
  //if(SysCond_BaseCase==true)  
  //  for(auto& v: Qlb) v+= 0.49*fabs(v);
  
  //if(SysCond_BaseCase==true)  
  //  for(auto& v: Qub) v-= 0.49*fabs(v);


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

  assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	  (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
  if(slacks_initially_recomputed) {
    pf_p_bal->compute_slacks(pslacks_n); 
    pslacks_n->providesStartingPoint=true;
  } else {
    if(slacks_initially_to_zero)
      pslacks_n->set_start_to(0.0);
  }
  assert(pslacks_n->providesStartingPoint);

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
  assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	  (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
  if(slacks_initially_recomputed) {
    pf_q_bal->compute_slacks(qslacks_n); 
    qslacks_n->providesStartingPoint=true;
  } else {
    if(slacks_initially_to_zero)
      qslacks_n->set_start_to(0.0);
  }
  assert(qslacks_n->providesStartingPoint);

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
void SCACOPFProblem::add_cons_thermal_li_lims(SCACOPFData& d, bool SysCond_BaseCase)
{
  vector<double>& L_Rate = SysCond_BaseCase ? d.L_RateBase : d.L_RateEmer;
  add_cons_thermal_li_lims(d, L_Rate);
}
void SCACOPFProblem::add_cons_thermal_li_lims(SCACOPFData& d, 
					      const std::vector<double>& L_Rate)
{
  bool useQPenLi1 = useQPen, useQPenLi2 = useQPen; //double slacks_scale=1.;

  auto v_n = variable("v_n",d);
  auto p_li1 = variable("p_li1", d), q_li1 = variable("q_li1", d);

  {
    auto pf_line_lim1 = new PFLineLimits(con_name("line_limits1",d), d.L_Line.size(),
					 p_li1, q_li1, v_n, 
					 d.L_Nidx[0], L_Rate, slacks_scale);
    append_constraints(pf_line_lim1);
    
    //sslack_li1
    OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
    assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	    (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
    if(slacks_initially_recomputed) {
      pf_line_lim1->compute_slacks(sslack_li1); 
      sslack_li1->providesStartingPoint=true;
    } else {
      if(slacks_initially_to_zero) {
	sslack_li1->set_start_to(0.);
      }
    }
    assert(sslack_li1->providesStartingPoint);

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
    assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	    (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
    if(slacks_initially_recomputed) {
      pf_line_lim2->compute_slacks(sslack_li2); 
      sslack_li2->providesStartingPoint=true;
    } else {
      if(slacks_initially_to_zero) {
	sslack_li2->set_start_to(0.);
      }
    }
    assert(sslack_li2->providesStartingPoint);

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
  void SCACOPFProblem::add_cons_thermal_ti_lims(SCACOPFData& d, bool SysCond_BaseCase)
{
  vector<double>& T_Rate = SysCond_BaseCase ? d.T_RateBase : d.T_RateEmer;
  add_cons_thermal_ti_lims(d, T_Rate);
}

  void SCACOPFProblem::add_cons_thermal_ti_lims(SCACOPFData& d,  const std::vector<double>& T_Rate)
{
  bool useQPenTi1=useQPen, useQPenTi2=useQPen; //double slacks_scale=1.;
  
  // - removed vector<double>& T_Rate = SysCond_BaseCase ? d.T_RateBase : d.T_RateEmer;
  if(true){
    auto p_ti1 = variable("p_ti1", d), q_ti1 = variable("q_ti1", d);
    auto pf_trans_lim1 = new PFTransfLimits(con_name("trans_limits1",d), d.T_Transformer.size(),
					    p_ti1, q_ti1, 
					    T_Rate, slacks_scale);
    append_constraints(pf_trans_lim1);
    //sslack_ti1
    OptVariablesBlock* sslack_ti1 = pf_trans_lim1->slacks();

    assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	    (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
    if(slacks_initially_recomputed) {
      pf_trans_lim1->compute_slacks(sslack_ti1); 
      sslack_ti1->providesStartingPoint=true;
    } else {
      if(slacks_initially_to_zero) {
	sslack_ti1->set_start_to(0.);
      }
    }
    assert(sslack_ti1->providesStartingPoint);

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
    assert( ( slacks_initially_recomputed && !slacks_initially_to_zero) ||
	    (!slacks_initially_recomputed &&  slacks_initially_to_zero) );
    if(slacks_initially_recomputed) {
      pf_trans_lim2->compute_slacks(sslack_ti2); 
      sslack_ti2->providesStartingPoint=true;
    } else {
      if(slacks_initially_to_zero) {
	sslack_ti2->set_start_to(0.);
      }
    }
    assert(sslack_ti2->providesStartingPoint);

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

  if(linear_prod_cost) {
    PFProdCostApproxAffineObjTerm* cost_term = 
      new PFProdCostApproxAffineObjTerm("prodcost_affine_0",
					p_g, 
					gens, d.G_CostCi, d.G_CostPi);
    append_objterm(cost_term);

  } else {
    PFProdCostAffineCons* prod_cost_cons = 
      new PFProdCostAffineCons(con_name("prodcost_cons",d), 2*gens.size(), 
			       p_g, gens, d.G_CostCi, d.G_CostPi);
    append_constraints(prod_cost_cons);
    
    OptVariablesBlock* t_h = prod_cost_cons->get_t_h();
    prod_cost_cons->compute_t_h(t_h); t_h->providesStartingPoint = true;
  }
}


void SCACOPFProblem::copy_basecase_primal_variables_to(std::vector<double>& dest)
{
  dest.clear();
  for(auto b: vars_primal->vblocks) {
    int sz = b->id.size();
    if(sz>=2 && '0'==b->id[sz-1] && '_'==b->id[sz-2]) {
      for(int i=0; i<b->n; i++) 
	dest.push_back(b->x[i]);
    }
  }
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

#define SIGNED_DUALS_VAL 1.

  bool SCACOPFProblem::set_warm_start_for_base_from_base_of(SCACOPFProblem& srcProb)
  {
    vector<string> ids = {"v_n_0" , "theta_n_0" , "p_li1_0" , "p_li2_0" , "q_li1_0" , "q_li2_0" , "p_ti1_0" , "p_ti2_0" , "q_ti1_0" , "q_ti2_0" , "b_s_0" , "p_g_0" , "q_g_0" , "pslack_n_p_balance_0" , "qslack_n_q_balance_0" , "sslack_li_line_limits1_0" , "sslack_li_line_limits2_0" , "sslack_ti_trans_limits1_0" , "sslack_ti_trans_limits2_0" , "t_h_0", "sslack_agc_reserves_loss_Kgen_0", "sslack_agc_reserves_gain_Kgen_0", "sslack_agc_reserves_loss_bnd_0", "sslack_agc_reserves_gain_bnd_0" };
    for(auto id: ids) {
      auto b = vars_primal->vars_block(id);
      auto bsrc = srcProb.vars_primal->vars_block(id);
      if(bsrc) { if(b) b->set_start_to(*bsrc); }
      else { assert(false); }
    }

    ids = {"duals_bndL_v_n_0" , "duals_bndL_theta_n_0" , "duals_bndL_p_li1_0" , "duals_bndL_p_li2_0" , "duals_bndL_q_li1_0" , "duals_bndL_q_li2_0" , "duals_bndL_p_ti1_0" , "duals_bndL_p_ti2_0" , "duals_bndL_q_ti1_0" , "duals_bndL_q_ti2_0" , "duals_bndL_b_s_0" , "duals_bndL_p_g_0" , "duals_bndL_q_g_0" , "duals_bndL_pslack_n_p_balance_0" , "duals_bndL_qslack_n_q_balance_0" , "duals_bndL_sslack_li_line_limits1_0" , "duals_bndL_sslack_li_line_limits2_0" , "duals_bndL_sslack_ti_trans_limits1_0" , "duals_bndL_sslack_ti_trans_limits2_0" , "duals_bndL_t_h_0", "duals_bndL_sslack_agc_reserves_loss_Kgen_0", "duals_bndL_sslack_agc_reserves_gain_Kgen_0", "duals_bndL_sslack_agc_reserves_loss_bnd_0", "duals_bndL_sslack_agc_reserves_gain_bnd_0"};
    for(auto id: ids) {
      auto b = vars_duals_bounds_L->vars_block(id);
      auto bsrc = srcProb.vars_duals_bounds_L->vars_block(id);
      if(bsrc) { if(b) b->set_start_to(*bsrc); }
      else { assert(false); }
    }

    ids = {"duals_bndU_v_n_0" , "duals_bndU_theta_n_0" , "duals_bndU_p_li1_0" , "duals_bndU_p_li2_0" , "duals_bndU_q_li1_0" , "duals_bndU_q_li2_0" , "duals_bndU_p_ti1_0" , "duals_bndU_p_ti2_0" , "duals_bndU_q_ti1_0" , "duals_bndU_q_ti2_0" , "duals_bndU_b_s_0" , "duals_bndU_p_g_0" , "duals_bndU_q_g_0" , "duals_bndU_pslack_n_p_balance_0" , "duals_bndU_qslack_n_q_balance_0" , "duals_bndU_sslack_li_line_limits1_0" , "duals_bndU_sslack_li_line_limits2_0" , "duals_bndU_sslack_ti_trans_limits1_0" , "duals_bndU_sslack_ti_trans_limits2_0" , "duals_bndU_t_h_0", "duals_bndU_sslack_agc_reserves_loss_Kgen_0", "duals_bndU_sslack_agc_reserves_gain_Kgen_0", "duals_bndU_sslack_agc_reserves_loss_bnd_0", "duals_bndU_sslack_agc_reserves_gain_bnd_0"};
    for(auto id: ids) {
      auto b = vars_duals_bounds_U->vars_block(id);
      auto bsrc = srcProb.vars_duals_bounds_U->vars_block(id);
      if(bsrc) { if(b) b->set_start_to(*bsrc); }
      else { assert(false); }
    }

    ids = {"duals_p_li1_powerflow_0" , "duals_p_li2_powerflow_0" , "duals_q_li1_powerflow_0" , "duals_q_li2_powerflow_0" , "duals_p_ti1_powerflow_0" , "duals_p_ti2_powerflow_0" , "duals_q_ti1_powerflow_0" , "duals_q_ti2_powerflow_0" , "duals_p_balance_0" , "duals_q_balance_0" , "duals_line_limits1_0" , "duals_line_limits2_0" , "duals_trans_limits1_0" , "duals_trans_limits2_0" , "duals_prodcost_cons_0", "duals_agc_reserves_loss_Kgen_0", "duals_agc_reserves_gain_Kgen_0", "duals_agc_reserves_loss_bnd_0", "duals_agc_reserves_bnd_Kgen_0"};
    for(auto id: ids) {
      auto b = vars_duals_cons->vars_block(id);
      auto bsrc = srcProb.vars_duals_cons->vars_block(id);
      if(bsrc) { if(b) b->set_start_to(*bsrc); }
      else { assert(false); }
    }
    return true;
  }


  bool SCACOPFProblem::set_warm_start_for_cont_from_base_of(const int& K_idx, SCACOPFProblem& srcProb)
  {
    assert(K_idx>=0); assert(K_idx<data_sc.K_Contingency.size());
    for(SCACOPFData* dB: this->data_K) {
      if(dB->id-1==K_idx) {
	return set_warm_start_for_cont_from_base_of(*dB, srcProb);
      }
    }
    assert(vars_primal->provides_start());
    assert(vars_duals_cons->provides_start());
    assert(vars_duals_bounds_L->provides_start());
    assert(vars_duals_bounds_U->provides_start());
    return false;
  }
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

    auto deltak = variable("delta", dK);
    if(deltak) deltak->set_start_to(0.);
    
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

      {
	prefix = "duals_bndL_nup_PVPQ";
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
	auto v = variable_duals_lower(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
	
	prefix = "duals_bndL_num_PVPQ";
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc));
	v = variable_duals_lower(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
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

	//or try for 'AGC_simple' constraints
	prefix = "duals_AGC_simple";
	v  = variable_duals_cons(prefix, dK); 
	if(v)  v->set_start_to(0.);
      }
      {
	prefix = "duals_pg_non_anticip";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL); 
	//else assert(false);
      }

      {
	prefix = "duals_PVPQ";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      {
	prefix = "duals_volt_non_anticip";
	auto v = variable_duals_cons(prefix, dK);
	if(v) v->set_start_to(0.);
      }
      //vars_duals_cons->print_summary();
      //!assert(vars_duals_cons->provides_start());
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
    printf("[%4d] [%4d] %12.5e  %12.5e %12.5e\n", i, dB.G_Generator[i], p_g->x[i], dB.G_Plb[i], dB.G_Pub[i]);
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
    //assert(rhop==NULL); 
    //assert(rhom==NULL);
    printf("print_p_g_with_coupling_info called but no AGC constraints present. will print p_g\n");
    print_p_g(dB);
    return;
  }

  int K_id = dB.K_Contingency[0];
  vector<int> Gk, Gkp, Gknop;
  data_sc.get_AGC_participation(K_id, Gk, Gkp, Gknop);
  auto ids_agc = selectfrom(data_sc.G_Generator, Gkp);

  printf("p_g for SC block %d: delta_k=%12.5e  (indexes are withing conting)\n", dB.id, delta->x[0]);
  printf("[ idx] [  id ]         p_g     p_gk             lb            ub         rhom        rhop      |   bodies AGC\n");
  for(int i=0; i<dB.G_Generator.size(); i++) {
    int agc_idx = indexin(ids_agc, dB.G_Generator[i]); 
    int base_idx = indexin(data_sc.G_Generator, dB.G_Generator[i]);
    assert(base_idx>=0);
    if(agc_idx>=0) {
      double drhop=0., drhom=0.;
      if(rhop!=NULL) drhop = rhop->x[agc_idx];
      if(rhom!=NULL) drhom = rhom->x[agc_idx];

      double gb = dB.G_Pub[i]-dB.G_Plb[i];
      printf("[%4d] [%4d] %12.5e %12.5e agc %12.5e %12.5e %12.5e %12.5e | %12.5e %12.5e %12.5e \n", 
	     i, dB.G_Generator[i], 
	     p_g->x[base_idx], p_gk->x[i], dB.G_Plb[i], dB.G_Pub[i], drhom, drhop,
	     p_g->x[base_idx] + dB.G_alpha[i]*delta->x[0] - p_gk->x[i] - gb*drhop + gb*drhom,
	     (p_gk->x[i]-dB.G_Plb[i])/gb*drhom, (p_gk->x[i]-dB.G_Pub[i])/gb*drhop);

    } else {
      printf("[%4d] [%4d] %12.5e %12.5e     %12.5e %12.5e\n", 
	     i, dB.G_Generator[i], 
	     p_g->x[base_idx], p_gk->x[i], dB.G_Plb[i], dB.G_Pub[i]);
    }
  }
}

void SCACOPFProblem::print_PVPQ_info(SCACOPFData& dB, OptVariablesBlock* v_n0)
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

  if(v_n0 == NULL)
    v_n0 = variable("v_n", data_sc);
  auto v_nk = variable("v_n", dB);
  auto q_gk = variable("q_g", dB);
  auto rhop = variable("nup_PVPQ", dB);
  auto rhom = variable("num_PVPQ", dB);
  printf("busidx busid     v_n0         v_nk         Vlb         Vub          "
	 "EVlb         EVub         num            nup      | genidx q_gk qlb qub : \n");

  vector<double> &Vlb = data_sc.N_Vlb, &Vub = data_sc.N_Vub, 
    &EVlb = data_sc.N_EVlb, &EVub = data_sc.N_EVub;

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

      printf("%5d %5d %12.5e %12.5e %12.5e %12.5e | %12.5e %12.5e |      na        na       | ", 
	     n, data_sc.N_Bus[n], v_n0->x[n], v_nk->x[n],
	     Vlb[n], Vub[n], EVlb[n], EVub[n]);
      for(auto g: dB.Gn[n])
	printf("%4d %12.5e %12.5e %12.5e : ", g, q_gk->x[g], dB.G_Qlb[g], dB.G_Qub[g]);
      printf("\n");

    } else {
      Qlb.push_back(Qagglb);
      Qub.push_back(Qaggub);
      idxs_bus_pvpq.push_back(n);

      int idx_nu = idxs_bus_pvpq.size()-1;

      double drhom=0., drhop=0.;
      if(rhom) drhom = rhom->x[idx_nu];
      if(rhop) drhop = rhop->x[idx_nu];
      printf("%5d %5d %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e | %12.5e %12.5e | ", n, data_sc.N_Bus[n], 
	     v_n0->x[n], v_nk->x[n],
	     Vlb[n], Vub[n], EVlb[n], EVub[n], drhom, drhop);
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

void SCACOPFProblem::print_active_power_balance_info(SCACOPFData& d)
{
  auto pf_p_bal = dynamic_cast<PFActiveBalance*>(constraint("p_balance",d));
  //pslackm_n and pslackp_n
  OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
  int n = data_sc.N_Bus.size();
  assert(pslacks_n->n == 2*n);

  
  string msg; char stmp[1024];
  msg += "active power balance - large penalties\n";

  if(d.id>0) {
    
    if(d.K_ConType[0] == SCACOPFData::kTransformer) {
      int FromBusIdx = data_sc.T_Nidx[0][d.K_outidx[0]], ToBusIdx = data_sc.T_Nidx[1][d.K_outidx[0]];

      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s  FromBusIdx  ToBusIdx  = %d %d \n", 
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(), 
	     FromBusIdx, ToBusIdx);
      msg += stmp;
    }
    if(d.K_ConType[0] == SCACOPFData::kLine) {
      int FromBusIdx = data_sc.L_Nidx[0][d.K_outidx[0]], ToBusIdx = data_sc.L_Nidx[1][d.K_outidx[0]];

      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s  FromBusIdx  ToBusIdx  = %d %d   "
	      "FromBusId  ToBusId  = %d %d\n", 
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(), 
	      FromBusIdx, ToBusIdx, 
	      data_sc.L_From[d.K_IDout[0]], data_sc.L_To[d.K_IDout[0]]);
      msg += stmp;
    }
    if(d.K_ConType[0] == SCACOPFData::kGenerator) {
      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s BusIdx=%d\n",
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(),
	      data_sc.G_Bus[d.K_outidx[0]]);
      msg += stmp;
    }

  }
  bool do_print = false;
  msg += "busidx busid   pslackp     pslackm \n";
  for(int i=0; i<n; i++) {
    string neigh = "";// conn busidx:";

    if(fabs(pslacks_n->x[i])>5e-3 || fabs(pslacks_n->x[i+n])>5e-3) {

      assert(d.G_Nidx.size()==d.G_Bus.size());
      assert(d.G_Generator.size()==d.G_Bus.size());

      for(int it=0; it<d.G_Bus.size(); it++) {
	if(d.G_Nidx[it]==i) {
	  assert(d.G_Bus[it]==data_sc.N_Bus[i]);
	  neigh += "(gen id " + to_string(d.G_Generator[it]) + ") ";
	}
      }
      neigh += " conn busidx:";
      for(int it=0; it<d.L_Nidx[0].size(); it++)
	if(d.L_Nidx[0][it]==i) neigh += to_string(d.L_Nidx[1][it]) + "(l1) ";
      for(int it=0; it<d.L_Nidx[1].size(); it++)
	if(d.L_Nidx[1][it]==i) neigh += to_string(d.L_Nidx[0][it]) + "(l2) ";

      //for(int it=0; it<data_sc.L_Nidx[0].size(); it++)
      //if(data_sc.L_Nidx[0][it]==i) neigh += to_string(data_sc.L_Nidx[1][it]) + "(l10) ";
      //for(int it=0; it<d.L_Nidx[1].size(); it++)
      //if(data_sc.L_Nidx[1][it]==i) neigh += to_string(data_sc.L_Nidx[0][it]) + "(l20) ";
      
      for(int it=0; it<d.T_Nidx[0].size(); it++)
	if(d.T_Nidx[0][it]==i) neigh += to_string(d.T_Nidx[1][it]) + "(t1) ";
      for(int it=0; it<d.T_Nidx[1].size(); it++)
	if(d.T_Nidx[1][it]==i) neigh += to_string(d.T_Nidx[0][it]) + "(t2) ";
      
      sprintf(stmp, "%5d %5d %12.5e %12.5e | %s\n", 
	      i, data_sc.N_Bus[i], pslacks_n->x[i], pslacks_n->x[i+n], neigh.c_str());
      msg += stmp;
      do_print = true;
    }
  }
  if(do_print) {
    printf("%s\n", msg.c_str());
  }

}
void SCACOPFProblem::print_reactive_power_balance_info(SCACOPFData& d)
{
  auto pf_q_bal = dynamic_cast<PFReactiveBalance*>(constraint("q_balance",d));
  //pslackm_n and pslackp_n
  OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
  int n = data_sc.N_Bus.size();
  assert(qslacks_n->n == 2*n);

  
  string msg; char stmp[1024];
  msg += "reactive power balance - large penalties\n";

  if(d.id>0) {
    
    if(d.K_ConType[0] == SCACOPFData::kTransformer) {
      int FromBusIdx = data_sc.T_Nidx[0][d.K_outidx[0]], ToBusIdx = data_sc.T_Nidx[1][d.K_outidx[0]];

      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s  FromBusIdx  ToBusIdx  = %d %d \n", 
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(), 
	     FromBusIdx, ToBusIdx);
      msg += stmp;
    }
    if(d.K_ConType[0] == SCACOPFData::kLine) {
      int FromBusIdx = data_sc.L_Nidx[0][d.K_outidx[0]], ToBusIdx = data_sc.L_Nidx[1][d.K_outidx[0]];

      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s  FromBusIdx  ToBusIdx  = %d %d \n", 
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(), 
	     FromBusIdx, ToBusIdx);
      msg += stmp;
    }
    if(d.K_ConType[0] == SCACOPFData::kGenerator) {
      sprintf(stmp, "ContProb K=%d IDOut=%d outidx=%d Type=%s BusIdx=%d\n",
	     d.K_Contingency[0], d.K_IDout[0], d.K_outidx[0], d.cont_type_string(0).c_str(),
	      data_sc.G_Bus[d.K_outidx[0]]);
      msg += stmp;
    }

  }
  bool do_print = false;
  msg += "busidx busid   qslackp     qslackm \n";
  for(int i=0; i<n; i++) {
    string neigh = " conn busidx:";

    for(int it=0; it<d.L_Nidx[0].size(); it++)
      if(d.L_Nidx[0][it]==i) neigh += to_string(d.L_Nidx[1][it]) + "(l) ";
    for(int it=0; it<d.L_Nidx[1].size(); it++)
      if(d.L_Nidx[1][it]==i) neigh += to_string(d.L_Nidx[0][it]) + "(l) ";

    for(int it=0; it<d.T_Nidx[0].size(); it++)
      if(d.T_Nidx[0][it]==i) neigh += to_string(d.T_Nidx[1][it]) + "(t) ";
    for(int it=0; it<d.T_Nidx[1].size(); it++)
      if(d.T_Nidx[1][it]==i) neigh += to_string(d.T_Nidx[0][it]) + "(t) ";

    if(fabs(qslacks_n->x[i])>1e-6 || fabs(qslacks_n->x[i+n])>1e-6) {
      sprintf(stmp, "%5d %5d %12.5e %12.5e | %s\n", 
	     i, data_sc.N_Bus[i], qslacks_n->x[i], qslacks_n->x[i+n], neigh.c_str());
      msg += stmp;
      do_print = true;
    }
  }
  if(do_print) {
    printf("%s\n", msg.c_str());
  }

}

void SCACOPFProblem::print_line_limits_info(SCACOPFData& dB)
{
  string msg; char stmp[1024];
  msg += "line limits - large slacks\n";

  auto pf_line_lim1 = dynamic_cast<PFLineLimits*>(constraint("line_limits1", dB));
  OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
  
  auto pf_line_lim2 = dynamic_cast<PFLineLimits*>(constraint("line_limits2", dB));
  OptVariablesBlock* sslack_li2 = pf_line_lim2->slacks();

  auto p_li1 = variable("p_li1", dB), q_li1 = variable("q_li1", dB);
  auto p_li2 = variable("p_li2", dB), q_li2 = variable("q_li2", dB);

  assert(sslack_li1->n == dB.L_Line.size());
  assert(sslack_li2->n == dB.L_Line.size());

  bool do_print=false;


  msg += "Lineidx |     pli1         qli1       slack_li1           pli2         qli2       slack_li2  | FromBusId  ToBusId\n";
  for(int i=0; i<dB.L_Line.size(); i++) {

    if(sslack_li1->x[i]>1e-6 || sslack_li2->x[i]>1e-6) {

      sprintf(stmp, " %5d  | %12.5e %12.5e %12.5e     %12.5e %12.5e %12.5e  | %5d     %5d\n", 
	      i, p_li1->x[i], q_li1->x[i], sslack_li1->x[i], 
	      p_li2->x[i], q_li2->x[i], sslack_li2->x[i],
	      dB.L_From[i], dB.L_To[i]);
      msg += stmp;

      do_print = true;
    }
  }
  if(do_print) {
    printf("%s\n", msg.c_str()); 
  }
}
void SCACOPFProblem::print_transf_limits_info(SCACOPFData& dB)
{
  string msg; char stmp[1024];
  msg += "transformer limits - large slacks\n";

  auto pf_transf_lim1 = dynamic_cast<PFLineLimits*>(constraint("transf_limits1", dB));
  OptVariablesBlock* sslack_ti1 = pf_transf_lim1->slacks();
  
  auto pf_transf_lim2 = dynamic_cast<PFLineLimits*>(constraint("transf_limits2", dB));
  OptVariablesBlock* sslack_ti2 = pf_transf_lim2->slacks();

  assert(sslack_ti1->n == dB.T_Transformer.size());
  assert(sslack_ti2->n == dB.T_Transformer.size());

  bool do_print=false;

  msg += "Lineidx FromBusIdx ToBusIdx slack_li1 slack_li2\n";
  for(int i=0; i<dB.T_Transformer.size(); i++) {
    if(sslack_ti1->x[i]>1e-6 || sslack_ti2->x[i]>1e-6) {
      sprintf(stmp, " %5d %5d %5d %12.5e  %12.5e \n", i, dB.T_Nidx[0][i], dB.T_Nidx[1][i],
	      sslack_ti1->x[i], sslack_ti2->x[i]);
      msg += stmp;
      do_print = true;
    }
  }
  if(do_print) {
    printf("%s\n", msg.c_str()); 
  }
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

void SCACOPFProblem::write_solution_basecase(OptVariables* primal_vars)
{
  if(primal_vars==NULL) primal_vars = this->vars_primal;
  //data_sc is for the basecase
  OptVariablesBlock* p_g = primal_vars->vars_block(var_name("p_g", data_sc));
  if(NULL==p_g) {
    printf("[warning] no solution was written; p_g0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* q_g = primal_vars->vars_block(var_name("q_g", data_sc));//variable("q_g", data_sc);
  if(NULL==q_g) {
    printf("[warning] no solution was written; q_g0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* v_n = primal_vars->vars_block(var_name("v_n", data_sc));//variable("v_n", data_sc);
  if(NULL==v_n) {
    printf("[warning] no solution was written; v_n0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* theta_n = primal_vars->vars_block(var_name("theta_n", data_sc));//variable("theta_n", data_sc);
  if(NULL==theta_n) {
    printf("[warning] no solution was written; theta_n0 is missing from the problem\n");
    return;
  }
  OptVariablesBlock* b_s = primal_vars->vars_block(var_name("b_s", data_sc));//variable("b_s", data_sc);
  if(NULL==b_s) {
    printf("[warning] no solution was written; b_s0 is missing from the problem\n");
    return;
  }

  //SCACOPFIO::write_append_solution_block(v_n->x, theta_n->x, b_s->x, p_g->x, q_g->x,
  //					 data_sc, "solution1.txt", "w");
  SCACOPFIO::write_solution1(v_n->x, theta_n->x, b_s->x, p_g->x, q_g->x,
			     data_sc, "solution1.txt");  
}

void SCACOPFProblem::write_pridua_solution_basecase(OptVariables* primal_vars,
						    OptVariables* dual_con_vars,
						    OptVariables* dual_lb_vars,
						    OptVariables* dual_ub_vars)
{
  if(primal_vars==NULL) primal_vars = this->vars_primal;
  if(dual_con_vars==NULL) dual_con_vars = this->vars_duals_cons;
  if(dual_lb_vars==NULL) dual_lb_vars = this->vars_duals_bounds_L;
  if(dual_ub_vars==NULL) dual_ub_vars = this->vars_duals_bounds_U;

  string filename = "solution_b_pd.txt";
  FILE* file = fopen(filename.c_str(), "w");
  if(NULL == file) {
    printf("[warning] could not open '%s' for writing. will return\n", filename.c_str());
    return;
  }

  vector<string> vars_names={"v_n_0","theta_n_0",
			     "p_li1_0","p_li2_0","q_li1_0","q_li2_0",
			     "p_ti1_0","p_ti2_0","q_ti1_0","q_ti2_0",
			     "b_s_0","p_g_0","q_g_0", 
			     "pslack_n_p_balance_0", "qslack_n_q_balance_0",
			     "sslack_li_line_limits1_0","sslack_li_line_limits2_0",
			     "sslack_ti_trans_limits1_0","sslack_ti_trans_limits2_0"};
  //,"t_h_0","sslack_agc_reserves_loss_bnd_0","sslack_agc_reserves_loss_Kgen_0","sslack_agc_reserves_gain_Kgen_0"};
  
  //-vector<string> cons_names={"p_li1_powerflow_0","p_li2_powerflow_0","q_li1_powerflow_0","q_li2_powerflow_0",
  //-			     "p_ti1_powerflow_0","p_ti2_powerflow_0","q_ti1_powerflow_0","q_ti2_powerflow_0",
  //-			     "p_balance_0","q_balance_0",
  //-			     "line_limits1_0","line_limits2_0","trans_limits1_0","trans_limits2_0"};
  //-//,"prodcost_cons_0","agc_reserves_loss_bnd_0","agc_reserves_loss_Kgen_0","agc_reserves_gain_Kgen_0"};

  vector<string> duals_bndL_names={"duals_bndL_v_n_0","duals_bndL_theta_n_0","duals_bndL_p_li1_0","duals_bndL_p_li2_0","duals_bndL_q_li1_0","duals_bndL_q_li2_0","duals_bndL_p_ti1_0","duals_bndL_p_ti2_0","duals_bndL_q_ti1_0","duals_bndL_q_ti2_0","duals_bndL_b_s_0","duals_bndL_p_g_0","duals_bndL_q_g_0","duals_bndL_pslack_n_p_balance_0","duals_bndL_qslack_n_q_balance_0","duals_bndL_sslack_li_line_limits1_0","duals_bndL_sslack_li_line_limits2_0","duals_bndL_sslack_ti_trans_limits1_0","duals_bndL_sslack_ti_trans_limits2_0"};
  //"duals_bndL_t_h_0","duals_bndL_sslack_agc_reserves_loss_bnd_0","duals_bndL_sslack_agc_reserves_loss_Kgen_0","duals_bndL_sslack_agc_reserves_gain_Kgen_0"};


  vector<string> duals_bndU_names={"duals_bndU_v_n_0","duals_bndU_theta_n_0","duals_bndU_p_li1_0","duals_bndU_p_li2_0","duals_bndU_q_li1_0","duals_bndU_q_li2_0","duals_bndU_p_ti1_0","duals_bndU_p_ti2_0","duals_bndU_q_ti1_0","duals_bndU_q_ti2_0","duals_bndU_b_s_0","duals_bndU_p_g_0","duals_bndU_q_g_0","duals_bndU_pslack_n_p_balance_0","duals_bndU_qslack_n_q_balance_0","duals_bndU_sslack_li_line_limits1_0","duals_bndU_sslack_li_line_limits2_0","duals_bndU_sslack_ti_trans_limits1_0","duals_bndU_sslack_ti_trans_limits2_0"};
  //"duals_bndU_t_h_0","duals_bndU_sslack_agc_reserves_loss_bnd_0","duals_bndU_sslack_agc_reserves_loss_Kgen_0","duals_bndU_sslack_agc_reserves_gain_Kgen_0"};

  vector<string> duals_cons_names={"duals_p_li1_powerflow_0","duals_p_li2_powerflow_0","duals_q_li1_powerflow_0","duals_q_li2_powerflow_0","duals_p_ti1_powerflow_0","duals_p_ti2_powerflow_0","duals_q_ti1_powerflow_0","duals_q_ti2_powerflow_0","duals_p_balance_0","duals_q_balance_0","duals_line_limits1_0","duals_line_limits2_0","duals_trans_limits1_0","duals_trans_limits2_0"};
  //"duals_prodcost_cons_0","duals_agc_reserves_loss_bnd_0","duals_agc_reserves_loss_Kgen_0","duals_agc_reserves_gain_Kgen_0"};

  //vector<string> all_vars_names = vars_names;
  //all_vars_names.insert(all_vars_names.end(), duals_bndL_names.begin(), duals_bndL_names.end());
  //all_vars_names.insert(all_vars_names.end(), duals_bndU_names.begin(), duals_bndU_names.end());
  //all_vars_names.insert(all_vars_names.end(), duals_cons_names.begin(), duals_cons_names.end());

  for(string& var_name : vars_names) {
    OptVariablesBlock* var = primal_vars->vars_block(var_name);
    if(NULL==var) {
      printf("[warning] '%s' variable NOT written: is missing from the problem\n", var_name.c_str());
      continue;
    }
    SCACOPFIO::write_variable_block(var, data_sc, file);
  }
  for(string& var_name : duals_bndL_names) {
    OptVariablesBlock* var = dual_lb_vars->vars_block(var_name);//vars_block_duals_bounds_lower(var_name);
    if(NULL==var) {
      printf("[warning] '%s' variable NOT written: is missing from the problem\n", var_name.c_str());
      continue;
    }
    SCACOPFIO::write_variable_block(var, data_sc, file);
  }
  for(string& var_name : duals_bndU_names) {
    OptVariablesBlock* var = dual_ub_vars->vars_block(var_name);//vars_block_duals_bounds_upper(var_name);
    if(NULL==var) {
      printf("[warning] '%s' variable NOT written: is missing from the problem\n", var_name.c_str());
      continue;
    }
    SCACOPFIO::write_variable_block(var, data_sc, file);
  }
  for(string& var_name : duals_cons_names) {
    OptVariablesBlock* var = dual_con_vars->vars_block(var_name);//vars_block_duals_cons(var_name);
    if(NULL==var) {
      printf("[warning] '%s' variable NOT written: is missing from the problem\n", var_name.c_str());
      continue;
    }
    SCACOPFIO::write_variable_block(var, data_sc, file);
  }
  fclose(file);
}

void SCACOPFProblem::build_pd_vars_dict(std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict)
{
  for(auto& v : vars_primal->vblocks)
    dict.insert({v->id, v});
  for(auto& v : vars_duals_bounds_L->vblocks) 
    dict.insert({v->id, v});
  for(auto& v : vars_duals_bounds_U->vblocks) 
    dict.insert({v->id, v});
  for(auto& v : vars_duals_cons->vblocks) 
    dict.insert({v->id, v});
}

static void warmstart_helper(std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict,
			     OptVariables& vars)
{
  for(auto& b : vars.vblocks) {
    auto b0p = dict.find(b->id);
    if(b0p == dict.end()) {

      cout << "!!!!! could not find variable " << b->id << endl;
      //assert(false);
      b->set_start_to(0.0);
      continue;
    } else {
      b->set_start_to(*b0p->second);
    }
  }
}

void SCACOPFProblem::
warm_start_basecasevariables_from_dict(std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict)
{
  warmstart_helper(dict, *vars_primal);
  warmstart_helper(dict, *vars_duals_bounds_L);
  warmstart_helper(dict, *vars_duals_bounds_U);
  warmstart_helper(dict, *vars_duals_cons);
}

void SCACOPFProblem::write_solution_extras_basecase(OptVariables* primal_vars)
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
  
  // open file for writing
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

bool SCACOPFProblem::iterate_callback(int iter, const double& obj_value,
				      const double* primals,
				      const double& inf_pr, const double& inf_pr_orig_pr, 
				      const double& inf_du, 
				      const double& mu, 
				      const double& alpha_du, const double& alpha_pr,
				      int ls_trials, OptimizationMode mode,
				      const double* duals_con,
				      const double* duals_lb, const double* duals_ub)
{

  optim_obj_value =  obj_value;
  optim_inf_pr = inf_pr;
  optim_inf_pr_orig_pr = inf_pr_orig_pr;
  optim_inf_du = inf_du;
  optim_mu = mu;

  if(monitor.is_active) {

    //finish initialization if needed
    if(iter==0) {
      best_known_iter.initialize(vars_primal, vars_duals_cons, vars_duals_bounds_L, vars_duals_bounds_U);
      monitor.objvalue_initial=obj_value; //reset
      monitor.objvalue_last_written=1e+20;
    } else {

    }

    assert(monitor.acceptable_tol_feasib>=monitor.tol_feasib_for_write);
    if(monitor.acceptable_tol_feasib<monitor.tol_feasib_for_write) {
      
      printf("[warning] setting acceptable_tol_feasib==tol_feasib_for_write (was %g < %g)\n", monitor.acceptable_tol_feasib, monitor.tol_feasib_for_write);
      monitor.tol_feasib_for_write = monitor.acceptable_tol_feasib;
    }

    if(primals && mode!=RestorationPhaseMode) {
      assert(duals_con); assert(duals_lb); assert(duals_ub); 

      if(inf_pr_orig_pr<=monitor.acceptable_tol_feasib && best_known_iter.obj_value>=obj_value) {
	best_known_iter.copy_primal_vars_from(primals, vars_primal);
	best_known_iter.copy_dual_vars_from(duals_con, duals_lb, duals_ub);
	best_known_iter.set_iter_stats( iter, obj_value, inf_pr, inf_pr_orig_pr, inf_du, mu, mode);

	if(iter-iter_sol_written>=monitor.write_every) {
	  if(inf_pr_orig_pr <= monitor.tol_feasib_for_write &&
	     inf_du <= monitor.tol_optim_for_write &&
	     mu <= monitor.tol_mu_for_write &&
	     obj_value <= monitor.objvalue_last_written) {
	    printf("[ph1] rank %d  phase 1 attempt write solution1.txt from call_back iter=%d\n", 
		   my_rank, iter);


	    this->attempt_write_solutions(best_known_iter.vars_primal,
					  best_known_iter.vars_duals_cons,
					  best_known_iter.vars_duals_bounds_L,
					  best_known_iter.vars_duals_bounds_U,
					  true);
	    // write_solution_basecase(best_known_iter.vars_primal);
	    // write_pridua_solution_basecase(best_known_iter.vars_primal,
	    // 				   best_known_iter.vars_duals_cons,
	    // 				   best_known_iter.vars_duals_bounds_L,
	    // 				   best_known_iter.vars_duals_bounds_U);
	    iter_sol_written = iter;
	    monitor.objvalue_last_written = obj_value;
	  }
	}
      }
    } else {
      if(mode==RestorationPhaseMode) {
	monitor.emergency = true;
	printf("[stop solver][warning] restauration at iter %d optimiz at %.2f sec\n", iter, monitor.timer.measureElapsedTime());
	//do not set monitor.user_stopped=true; since doing so will look like the last solution is ok
	return false;
      }
    }

    if(!monitor.bcast_done && inf_pr_orig_pr<=1e-8 && inf_du<=1e-6 && mu<=1e-8 &&
       my_rank==1 &&
       primals && mode!=RestorationPhaseMode) {

      IterInfo v;
      v.initialize(vars_primal, vars_duals_cons, vars_duals_bounds_L, vars_duals_bounds_U);

      v.copy_primal_vars_from(primals, vars_primal);
      v.copy_dual_vars_from(duals_con, duals_lb, duals_ub);
      v.set_iter_stats( iter, obj_value, inf_pr, inf_pr_orig_pr, inf_du, mu, mode);

      printf("[ph1] rank %d  phase 1 BEFORE basecase bcast at iter %d\n", 
	     my_rank, iter);


      v.vars_primal->MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
      v.vars_duals_bounds_L->MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
      v.vars_duals_bounds_U->MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);
      v.vars_duals_cons->MPI_Bcast_x(rank_solver_rank0, comm_world, my_rank);

      double cost_basecase=obj_value;
      MPI_Bcast(&cost_basecase, 1, MPI_DOUBLE, rank_solver_rank0, comm_world);
      printf("[ph1] rank %d  phase 1 basecase bcasts done at iter %d\n", 
	     my_rank, iter);
      monitor.bcast_done = true;
    }

    if(monitor.timer.measureElapsedTime() > monitor.timeout) {
      printf("[stop solver] timeout at iter %d optimiz at %.2f sec\n", iter, monitor.timer.measureElapsedTime());
      monitor.emergency = true;
      monitor.user_stopped=true;
      return false;
    }
  }

  return true;
}


void SCACOPFProblem::attempt_write_solutions(OptVariables* primal_vars,
					     OptVariables* dual_con_vars,
					     OptVariables* dual_lb_vars,
					     OptVariables* dual_ub_vars,
					     bool opt_success)
{
  SCACOPFProblem*prob = this;
  if(prob->use_filelocks_when_writing==false) {
    //	
    // write solution1
    //
    prob->write_solution_basecase(primal_vars);
    prob->write_pridua_solution_basecase(primal_vars,
					 dual_con_vars,
					 dual_lb_vars,
					 dual_ub_vars);
    
    if(!opt_success) {
      printf("[warning] Solver rank %d: initial basecase solve failed; solution1 was written write (lock disabled)\n", my_rank);
    } else {
      printf("Solver rank %d write solution1 (lock disabled)\n", my_rank);
    }
  } else {

    bool locking_failed = false;

    {
      SolFileLocker sol_file;
      if(!sol_file.open()) {
	printf("[warning] Solver rank %d: could not open optim file (lock)\n", my_rank); 
	locking_failed = true;
      }
      if(!sol_file.lock()) {
	printf("[warning] Solver rank %d: could not lock optim file (lock)\n", my_rank); 
	locking_failed = true;
      }
    
      if(!locking_failed && sol_file.is_my_solution_better(prob->optim_obj_value, prob->optim_inf_pr_orig_pr,  prob->optim_inf_du,  prob->optim_mu)) {
	if(sol_file.write(prob->optim_obj_value, prob->optim_inf_pr_orig_pr,  prob->optim_inf_du,  prob->optim_mu)) {

	  //	
	  // write solution1
	  //
	  prob->write_solution_basecase(primal_vars);
	  prob->write_pridua_solution_basecase(primal_vars,
					       dual_con_vars,
					       dual_lb_vars,
					       dual_ub_vars);

	  if(!opt_success) {
	    printf("[warning] Solver rank %d: initial basecase solve failed; solution1 was written write (lock on)\n", my_rank);
	  } else {
	    printf("Solver rank %d write solution1 (lock on)\n", my_rank);
	  }
	} else {
	  printf("[warning] Solver rank %d: could not write optim file (lock)\n", my_rank); 
	  locking_failed = true;
	}
      } else {
	printf("Solver rank %d did not write solution1 b/c one better was already written (lock on)\n", my_rank);
      }
    
      if(!sol_file.unlock()) {
	printf("[warning] Solver rank %d: could not unlock optim file (lock)\n", my_rank); 
	locking_failed = true;
      }
      sol_file.close();
    }

    //if locking didn't work take a shot -> better than writing no solution
    if(locking_failed) {
      //write solution
      prob->write_solution_basecase(primal_vars);
      prob->write_pridua_solution_basecase(primal_vars,
					   dual_con_vars,
					   dual_lb_vars,
					   dual_ub_vars);
      if(!opt_success) {
	printf("[warning] Solver rank %d: initial basecase solve failed; solution1 was written though (also lock failed)\n", my_rank);
      } else {
	printf("Solver rank %d write solution1 (lock failed)\n", my_rank);
      }
    }
  }
}

} //end namespace

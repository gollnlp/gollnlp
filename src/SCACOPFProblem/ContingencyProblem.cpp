#include "ContingencyProblem.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"
#include "unistd.h"
using namespace std;

//#define BE_VERBOSE

namespace gollnlp {
  
  ContingencyProblem::ContingencyProblem(SCACOPFData& d_in, int K_idx_, int my_rank_) 
    : SCACOPFProblem(d_in), K_idx(K_idx_), my_rank(my_rank_)
  {
    int numK = 1; //!

    assert(0==data_K.size());
    //data_sc = d_in (member of the parent)
    data_K.push_back(new SCACOPFData(data_sc)); 
    data_K[0]->rebuild_for_conting(K_idx, numK);
    //data_K[0].PenaltyWeight = (1-d.DELTA);

    v_n0=NULL; theta_n0=NULL; b_s0=NULL; p_g0=NULL; q_g0=NULL;
    reg_vn = reg_thetan = reg_bs = reg_pg = reg_qg = false;
  }

  ContingencyProblem::~ContingencyProblem()
  {
  }

  bool ContingencyProblem::default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, 
					    OptVariablesBlock* bs0, 
					    OptVariablesBlock* pg0, OptVariablesBlock* qg0)
  {
    theta_n0=thetan0; b_s0=bs0; q_g0=qg0;
    return default_assembly(pg0, vn0);
  }

  bool ContingencyProblem::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    //printf("ContProb K_idx=%d: IDOut=%d outidx=%d Type=%s\n",
    //	   K_idx, data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
    //	   data_sc.cont_type_string(K_idx).c_str());
    //fflush(stdout);

    p_g0=pg0; v_n0=vn0;

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];

    useQPen = true;
    //slacks_scale = 1.;

    add_variables(dK,false);

    add_cons_lines_pf(dK);
    add_cons_transformers_pf(dK);
    add_cons_active_powbal(dK);
    add_cons_reactive_powbal(dK);
    bool SysCond_BaseCase = false;
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    //
    // setup for indexes used in non-anticip and AGC coupling 
    //
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);
    assert(pg0->n == Gk.size() || pg0->n == 1+Gk.size());

    //pg0_nonpartic_idxs=Gk;
    //pg0_partic_idxs={};

    // indexes in data_K (for the recourse's contingency)
    auto ids_no_AGC = selectfrom(data_sc.G_Generator, pg0_nonpartic_idxs);
    pgK_nonpartic_idxs = indexin(dK.G_Generator, ids_no_AGC);
    pgK_nonpartic_idxs = findall(pgK_nonpartic_idxs, [](int val) {return val!=-1;});

    auto ids_AGC = selectfrom(data_sc.G_Generator, pg0_partic_idxs);
    pgK_partic_idxs = indexin(dK.G_Generator, ids_AGC);
    pgK_partic_idxs = findall(pgK_partic_idxs, [](int val) {return val!=-1;});
#ifdef DEBUG
    assert(pg0_nonpartic_idxs.size() == pgK_nonpartic_idxs.size());
    for(int i0=0, iK=0; i0<pg0_nonpartic_idxs.size(); i0++, iK++) {
      //all dB.G_Generator should be in data_sc.G_Generator
      assert(pgK_nonpartic_idxs[iK]>=0); 
      //all ids should match in order
      assert(dK.G_Generator[pgK_nonpartic_idxs[iK]] ==
	     data_sc.G_Generator[pg0_nonpartic_idxs[i0]]);
    }
    assert(pg0_partic_idxs.size() == pgK_partic_idxs.size());
    for(int i=0; i<pg0_partic_idxs.size(); i++) {
      assert(pgK_partic_idxs[i]>=0); 
      //all ids should match in order
      assert(dK.G_Generator[pgK_partic_idxs[i]] ==
	     data_sc.G_Generator[pg0_partic_idxs[i]]);
    }
      
#endif
    add_cons_nonanticip_using(pg0);
    add_cons_AGC_using(pg0);
    
    //add_const_nonanticip_v_n_using(vn0, Gk);
    add_cons_PVPQ_using(vn0, Gk);
    

    //print_summary();
    //PVPQSmoothing  = 1e-2;
    //coupling AGC and PVPQ; also creates delta_k
    //add_cons_coupling(dK);

    //depending on reg_vn, reg_thetan, reg_bs, reg_pg, and reg_qg
    add_regularizations();

    return true;
  }
  bool ContingencyProblem::optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
  {
    goTimer tmrec; tmrec.start();

    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    update_cons_nonanticip_using(pg0);
    update_cons_AGC_using(pg0);

    f = -1e+20;
    if(!OptProblem::optimize("ipopt")) {
      return false;
    }

    // objective value
    f = this->obj_value;

    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProb K_idx=%d: optimize took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif

    return true;

  }
  bool ContingencyProblem::eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
  {
    goTimer tmrec; tmrec.start();

    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    update_cons_nonanticip_using(pg0);
    update_cons_AGC_using(pg0);

    f = -1e+20;
    //if(!optimize("ipopt")) {
    if(!reoptimize(OptProblem::primalDualRestart)) {
      //if(!reoptimize(OptProblem::primalRestart)) {
      f = 1e+6;
      return false;
    }

    // objective value
    f = this->obj_value;

    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProb K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif
    return true;
  }

  void ContingencyProblem::get_solution_simplicial_vectorized(std::vector<double>& vsln)
  {
    SCACOPFData& dK = *data_K[0];

    int n_sln = v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n + 1;
    vsln = vector<double>(n_sln);

    auto v_nk = variable("v_n", dK);
    assert(v_nk->n == v_n0->n);
    memcpy(&vsln[0], v_nk->x, v_nk->n*sizeof(double));

    auto theta_nk = variable("theta_n", dK);
    assert(theta_nk->n == theta_n0->n);
    memcpy(&vsln[v_nk->n], theta_nk->x, theta_nk->n*sizeof(double));

    auto b_sk = variable("b_s", dK);
    assert(b_sk->n == b_s0->n);
    memcpy(&vsln[v_nk->n + theta_nk->n], b_sk->x, b_sk->n*sizeof(double));

    auto p_gk = variable("p_g", dK);
    auto q_gk = variable("q_g", dK);

    assert(dK.K_Contingency.size()==1); assert(dK.K_IDout.size()==1); 
    assert(dK.K_ConType.size()==1);
    
    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      int offset = v_nk->n + theta_nk->n + b_sk->n;
#ifdef DEBUG
      for(int i=0; i<p_g0->n + q_g0->n; i++) vsln[offset+i]= -10117.;
#endif
      int ngk = Gk.size(); assert(ngk==p_g0->n-1);
      for(int i=0; i<ngk; i++) {
	assert(Gk[i]>=0 && Gk[i]<p_g0->n);
	vsln[offset+Gk[i]] = p_gk->x[i];
      }
      assert(vsln[offset+dK.K_outidx[0]]==-10117.);
      vsln[offset+dK.K_outidx[0]] = 0.;

      offset += p_g0->n;
      for(int i=0; i<ngk; i++) {
	assert(Gk[i]>=0 && Gk[i]<q_g0->n);
	vsln[offset+Gk[i]] = q_gk->x[i];
      }
#ifdef DEBUG
      assert(vsln[offset+dK.K_outidx[0]]==-10117.);
#endif
      vsln[offset+dK.K_outidx[0]] = 0.;

    } else {
      assert(p_gk->n == p_g0->n);
      assert(q_gk->n == q_g0->n);
      memcpy(&vsln[v_nk->n + theta_nk->n + b_sk->n], p_gk->x, p_gk->n*sizeof(double));
      memcpy(&vsln[v_nk->n + theta_nk->n + b_sk->n + p_gk->n], q_gk->x, q_gk->n*sizeof(double));
    }

    auto delta_k = variable("delta", dK);
    assert(delta_k != NULL);
    vsln[n_sln-1] = delta_k->x[0];
  }

  //
  // PVPQ
  //
  void ContingencyProblem::add_const_nonanticip_v_n_using(OptVariablesBlock* v_n0, 
							  const vector<int>& Gk) 
  {
    assert(data_K.size()==1);
    SCACOPFData& dB = *data_K[0];
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
#ifdef BE_VERBOSE
	  //printf("PVPQ: gen ID=%d p_q at bus idx %d id %d is fixed; will not enforce voltage nonanticip constraint\n",
	  //	 dB.G_Generator[g], dB.G_Nidx[g], data_sc.N_Bus[dB.G_Nidx[g]]);
#endif
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

    auto v_nk = variable("v_n", dB);
    assert(v_n0->n == v_nk->n);

    assert(v_n0->xref == v_n0->x);

    for(int b : idxs_bus_pvpq) {
      assert(b>=0);
      assert(b<v_nk->n);

      //v_nk->lb[b] = 0.99*v_n0->xref[b];
      //v_nk->ub[b] = 1.01*v_n0->xref[b];
      v_nk->lb[b] = v_n0->xref[b];
      v_nk->ub[b] = v_n0->xref[b];

    }

  }
  void ContingencyProblem::add_cons_PVPQ_using(OptVariablesBlock* vn0, 
					       const vector<int>& Gk) 
  {
    assert(data_K.size()==1); assert(v_n0 == vn0);
    SCACOPFData& dB = *data_K[0];

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

    auto v_nk = variable("v_n", dB);
    auto q_gk = variable("q_g", dB);

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
					    PVPQSmoothing,
					    false, 0.,
					    true); //fixed v_n0
    
    append_constraints(cons);
    
    //starting point for nup and num that were added by PVPQComplementarityCons
    auto nup=cons->get_nup(), num=cons->get_num();
    cons->compute_nus(nup, num);
    nup->providesStartingPoint=true; num->providesStartingPoint=true;
    
    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+num->id, num, 1.));
    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+nup->id, nup, 1.));
#ifdef BE_VERBOSE
    printf("ContProb K_idx=%d: PVPQ: participating %d gens at %lu buses: "
	   "added %d constraints; PVPQSmoothing=%g; "
	   "total PVPQ: %lu gens | %d buses; fixed: %d gens | %d buses with all gens fixed.\n",
	   K_idx,
	   nPVPQGens-num_qgens_fixed, idxs_bus_pvpq.size(), cons->n, PVPQSmoothing,
	   Gk.size(), num_N_PVPQ,
	   num_qgens_fixed, num_buses_all_qgen_fixed);
#endif
  }
  void ContingencyProblem::update_cons_PVPQ_using(OptVariablesBlock* vn0, 
						  const vector<int>& Gk) {
    assert(false);
  }

  void ContingencyProblem::add_cons_nonanticip_using(OptVariablesBlock* pg0) {
    bodyof_cons_nonanticip_using(pg0);
#ifdef BE_VERBOSE
    printf("ContProb K_idx=%d on rank %d: "
	   "AGC: %lu gens NOT participating: fixed all of them.\n",
	   K_idx, my_rank, pg0_nonpartic_idxs.size());
#endif
  }
  void ContingencyProblem::bodyof_cons_nonanticip_using(OptVariablesBlock* pg0)
  {
    SCACOPFData& dK = *data_K[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblem K_idx=%d: p_g var not found in contingency  "
	     "problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    
    int sz = pgK_nonpartic_idxs.size();
    assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    int idxK; double pg0_val, lb, ub; 

#ifdef DEBUG
    assert(pg0->xref == pg0->x);
#endif

    for(int i=0; i<sz; i++) {
      assert(pg0_idxs[i]<pg0->n && pg0_idxs[i]>=0);
      assert(pgK_idxs[i]<pgK->n && pgK_idxs[i]>=0);
      idxK = pgK_idxs[i];
      pgK->lb[idxK] = pgK->ub[idxK] = pg0->xref[pg0_idxs[i]];
      
      //printf("%g lb[%g %g]\n",  pg0->x[pg0_idxs[i]], pgK->lb[idxK], pgK->ub[idxK]);
    }
  }

  void ContingencyProblem::add_cons_AGC_using(OptVariablesBlock* pg0)
  {
    if(pgK_partic_idxs.size()==0) {
      //assert(pg0_partic_idxs.size()==0);
#ifdef BE_VERBOSE
      printf("ContingencyProblem: add_cons_AGC_using: NO gens participating !?!\n");
#endif
      return;
    }

    SCACOPFData& dK = *data_K[0];
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblem K_idx=%d: p_g var not found in contingency  "
	     "recourse problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      assert(false);
      return;
    }
    OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK));
    append_variables(deltaK);
    deltaK->set_start_to(0.);
    
    //AGCSmoothing = 1e-3;
    auto cons = new AGCComplementarityCons(con_name("AGC", dK), 3*pgK_partic_idxs.size(),
					   pg0, pgK, deltaK, 
					   pg0_partic_idxs, pgK_partic_idxs, 
					   selectfrom(data_sc.G_Plb, pg0_partic_idxs), 
					   selectfrom(data_sc.G_Pub, pg0_partic_idxs),
					   data_sc.G_alpha, AGCSmoothing,
					   false, 0., //no internal penalty
					   true); //fixed p_g0 
    append_constraints(cons);

    //starting point for rhop and rhom that were added by AGCComplementarityCons
    auto rhop=cons->get_rhop(), rhom=cons->get_rhom();
    cons->compute_rhos(rhop, rhom);
    rhop->providesStartingPoint=true; rhom->providesStartingPoint=true;

    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhom->id, rhom, 1));
    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhop->id, rhop, 1));
#ifdef BE_VERBOSE
    printf("ContProb K_idx=%d: AGC %lu gens participating (out of %d) AGCSmoothing=%g\n", 
    	   K_idx, pgK_partic_idxs.size(), pgK->n, AGCSmoothing);
#endif
    //printvec(pg0_partic_idxs, "partic idxs");
  }
  void ContingencyProblem::update_cons_AGC_using(OptVariablesBlock* pg0)
  {
    if(pgK_partic_idxs.size()==0) {
      return;
    }
    //pg0 pointer that AGCComplementarityCons should not change
#ifdef DEBUG
    SCACOPFData& dK = *data_K[0];
    auto cons_AGC = dynamic_cast<AGCComplementarityCons*>(constraint("AGC", dK));
    assert(cons_AGC);
    if(pg0 != cons_AGC->get_p_g0()) {
      assert(false);
    }
#endif
  }



  void ContingencyProblem::add_regularizations()
  {
    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0]; assert(dK.id==K_idx+1);

    if(reg_vn) {
      assert(variable("v_n", dK)->n == v_n0->n);
      append_objterm(new QuadrRegularizationObjTerm("regul_vn", variable("v_n", dK),
						    1e-4, v_n0->x));
      //printf("added regularization term for v_n\n");
    }

    if(reg_thetan) {
      append_objterm(new QuadrRegularizationObjTerm("regul_thetan", variable("theta_n", dK),
						    1e-4, theta_n0->x));
      //printf("added regularization term for theta_n\n");

    }
    if(reg_bs) {
      append_objterm(new QuadrRegularizationObjTerm("regul_bs", variable("b_s", dK),
						    1e-4, b_s0->x));
      //printf("added regularization term for b_s\n");

    }
    if(reg_pg) {
      assert(Gk.size() == variable("p_g", dK)->n);
      assert(Gk.size() == p_g0->n || Gk.size() == p_g0->n-1);
      auto pg0_vec = selectfrom(p_g0->x, p_g0->n, Gk);
      append_objterm(new QuadrRegularizationObjTerm("regul_pg", variable("p_g", dK),
						    1e-4, pg0_vec.data()));
      //printf("added regularization term for p_g\n");
    }
    if(reg_qg) {
      assert(Gk.size() == variable("q_g", dK)->n);
      assert(Gk.size() == q_g0->n || Gk.size() == q_g0->n-1);
      auto qg0_vec = selectfrom(q_g0->x, q_g0->n, Gk);
      append_objterm(new QuadrRegularizationObjTerm("regul_qg", variable("q_g", dK),
						    1e-4, qg0_vec.data()));
      //printf("added regularization term for q_g\n");
    }

  }

#define SIGNED_DUALS_VAL 0.

  bool ContingencyProblem::set_warm_start_from_base_of(SCACOPFProblem& srcProb)
  {
    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0]; assert(dK.id==K_idx+1);

    // contingency indexes of lines, generators, or transformers (i.e., contingency type)
    vector<int> idxs_of_K_in_0; 

    assert(useQPen==true); assert(srcProb.useQPen==true);
    variable("v_n", dK)->set_start_to(*srcProb.variable("v_n", data_sc));
    variable("theta_n", dK)->set_start_to(*srcProb.variable("theta_n", data_sc));
    variable("b_s", dK)->set_start_to(*srcProb.variable("b_s", data_sc));

    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      auto p_gK = variable("p_g", dK), p_g0 = srcProb.variable("p_g", data_sc);
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
	variable_duals_lower(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndL_rhop_AGC";
	variable_duals_lower(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndL_rhom_AGC";
	variable_duals_lower(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
      }

      {
	prefix = "duals_bndL_nup_PVPQ";
	auto v = variable_duals_lower(prefix, dK); if(v) v->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndL_num_PVPQ";
	v = variable_duals_lower(prefix, dK); if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      //vars_duals_bounds_L->print_summary();
      assert(vars_duals_bounds_L->provides_start());
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
	variable_duals_upper(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndU_rhop_AGC";
	variable_duals_upper(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndU_rhom_AGC";
	variable_duals_upper(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
      }
      {
	prefix = "duals_bndU_nup_PVPQ";
	auto v = variable_duals_upper(prefix, dK); if(v) v->set_start_to(SIGNED_DUALS_VAL);
	prefix = "duals_bndU_num_PVPQ";
	v = variable_duals_upper(prefix, dK); if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      
      assert(vars_duals_bounds_U->provides_start());
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
	variable_duals_cons(prefix, dK)->set_start_to(SIGNED_DUALS_VAL);
      }
      {
	prefix = "duals_PVPQ";
	auto v = variable_duals_cons(prefix, dK); if(v) v->set_start_to(SIGNED_DUALS_VAL);
      }
      //vars_duals_cons->print_summary();
      assert(vars_duals_cons->provides_start());
    }
    

    //srcProb.duals_bounds_lower()->print_summary("duals bounds lower");
    //srcProb.duals_bounds_upper()->print_summary("duals bounds upper");
    //srcProb.duals_constraints()->print_summary("duals constraints");

    return true;
  }
  bool ContingencyProblem::
  set_warm_start_from_contingency_of(SCACOPFProblem& srcProb)
  {
    assert(data_K.size()==1); 
    SCACOPFData& dK = *data_K[0]; assert(dK.id==K_idx+1);
    bool bfound = false;
    for(auto d : srcProb.data_K) if(d->id == dK.id) bfound=true;
    if(!bfound) {
      printf("set_warm_start_from_contingency_of SCACOPFProblem: src does not have "
	     "the contingency id %d required by destination\n", dK.id);
      return false;
    }
      
    for(auto v : vars_primal->vblocks) {
      auto vsrc = srcProb.vars_block(v->id);
      if(!vsrc) return false;
      assert(v->n == vsrc->n);
      v->set_start_to(*vsrc);
    }
    return true;
  }
} //end of namespace

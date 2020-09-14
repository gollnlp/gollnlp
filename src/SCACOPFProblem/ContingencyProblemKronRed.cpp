#include "ContingencyProblemKronRed.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"

#include "ContingencyProblemKronRedWithFixingCode1.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"
#include "unistd.h"
using namespace std;

//#define BE_VERBOSE

namespace gollnlp {
  
  ContingencyProblemKronRed::ContingencyProblemKronRed(SCACOPFData& d_in, int K_idx_, int my_rank_) 
    : ACOPFProblemKronRed(d_in), K_idx(K_idx_), my_rank(my_rank_), cc_callbacks_(NULL)
  {
    int numK = 1; //!

    assert(0==data_K_.size());
    //data_sc_ = d_in (member of the parent)
    data_K_.push_back(new SCACOPFData(data_sc_)); 
    data_K_[0]->rebuild_for_conting(K_idx, numK);
    //data_K_[0].PenaltyWeight = (1-d.DELTA);

    v_n0=NULL; theta_n0=NULL; b_s0=NULL; p_g0=NULL; q_g0=NULL;
  }

  ContingencyProblemKronRed::~ContingencyProblemKronRed()
  {
    assert(data_K_.size() == 1);
    for(auto p : data_K_)
      delete p;
  }

  bool ContingencyProblemKronRed::assemble()
  {
    assert(false && "this method has not been tested");

    assert(data_K_.size() == 1);
    SCACOPFData& dK = *data_K_[0];
    
    add_variables(dK);
    add_cons_pf(dK);

    //objective
    add_obj_prod_cost(dK);
    
    print_summary();

    return true;
  }
  
  
  bool ContingencyProblemKronRed::default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, 
						   OptVariablesBlock* bs0, 
						   OptVariablesBlock* pg0, OptVariablesBlock* qg0)
  {
    assert(false); //assembled outside the class
    theta_n0=thetan0; b_s0=bs0; q_g0=qg0;
    return default_assembly(pg0, vn0);
  }

  bool ContingencyProblemKronRed::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    assert(false); //assembled outside the class


//     p_g0=pg0; v_n0=vn0;

//     assert(data_K_.size()==1);
//     SCACOPFData& dK = *data_K_[0];

//     useQPen = true;
//     //slacks_scale = 1.;

//     add_variables(dK,false);

//     add_cons_lines_pf(dK);
//     add_cons_transformers_pf(dK);
//     add_cons_active_powbal(dK);
//     add_cons_reactive_powbal(dK);
//     bool SysCond_BaseCase = false;
//     add_cons_thermal_li_lims(dK,SysCond_BaseCase);
//     add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

//     //
//     // setup for indexes used in non-anticip and AGC coupling 
//     //
//     //indexes in data_sc_.G_Generator; exclude 'outidx' if K_idx is a generator contingency
//     data_sc_.get_AGC_participation(K_idx, Gk_, pg0_partic_idxs, pg0_nonpartic_idxs);
//     assert(pg0->n == Gk_.size() || pg0->n == 1+Gk_.size());

//     //pg0_nonpartic_idxs=Gk;
//     //pg0_partic_idxs={};

//     // indexes in data_K_ (for the recourse's contingency)
//     auto ids_no_AGC = selectfrom(data_sc_.G_Generator, pg0_nonpartic_idxs);
//     pgK_nonpartic_idxs = indexin(dK.G_Generator, ids_no_AGC);
//     pgK_nonpartic_idxs = findall(pgK_nonpartic_idxs, [](int val) {return val!=-1;});

//     auto ids_AGC = selectfrom(data_sc_.G_Generator, pg0_partic_idxs);
//     pgK_partic_idxs = indexin(dK.G_Generator, ids_AGC);
//     pgK_partic_idxs = findall(pgK_partic_idxs, [](int val) {return val!=-1;});
// #ifdef DEBUG
//     assert(pg0_nonpartic_idxs.size() == pgK_nonpartic_idxs.size());
//     for(int i0=0, iK=0; i0<pg0_nonpartic_idxs.size(); i0++, iK++) {
//       //all dB.G_Generator should be in data_sc_.G_Generator
//       assert(pgK_nonpartic_idxs[iK]>=0); 
//       //all ids should match in order
//       assert(dK.G_Generator[pgK_nonpartic_idxs[iK]] ==
// 	     data_sc_.G_Generator[pg0_nonpartic_idxs[i0]]);
//     }
//     assert(pg0_partic_idxs.size() == pgK_partic_idxs.size());
//     for(int i=0; i<pg0_partic_idxs.size(); i++) {
//       assert(pgK_partic_idxs[i]>=0); 
//       //all ids should match in order
//       assert(dK.G_Generator[pgK_partic_idxs[i]] ==
// 	     data_sc_.G_Generator[pg0_partic_idxs[i]]);
//     }
      
// #endif
//     add_cons_nonanticip_using(pg0);
//     add_cons_AGC_using(pg0);
    
//     //add_const_nonanticip_v_n_using(vn0, Gk);
//     add_cons_PVPQ_using(vn0, Gk_);
    
    return true;
  }
//   bool ContingencyProblemKronRed::optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
//   {
//     assert(false && "not yet supported");
//     goTimer tmrec; tmrec.start();

//     assert(p_g0 == pg0); assert(v_n0 == vn0);
//     p_g0 = pg0; v_n0=vn0;

//     update_cons_nonanticip_using(pg0);
//     update_cons_AGC_using(pg0);

//     f = -1e+20;
//     if(!OptProblem::optimize("ipopt")) {
//       return false;
//     }

//     // objective value
//     f = this->obj_value;

//     tmrec.stop();
// #ifdef BE_VERBOSE
//     printf("ContProbKron K_idx=%d: optimize took %g sec  %d iterations on rank=%d\n", 
// 	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
//     fflush(stdout);
// #endif

//     return true;

//   }
  bool ContingencyProblemKronRed::eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
  {
    goTimer tmrec; tmrec.start();

    assert(false && "not tested");
    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    update_cons_nonanticip_using(pg0);
    update_cons_AGC_using(pg0);

    f = -1e+20;
    //if(!optimize("ipopt")) {
    if(!reoptimize(OptProblem::primalDualRestart)) {
      //if(!reoptimize(OptProblem::primalRestart)) {
      //if(!monitor.user_stopped) {
      assert(cc_callbacks_);
      if(cc_callbacks_) {
	if(!cc_callbacks_->monitor.user_stopped) {
	  f = 1e+6;
	  return false;
	}
      }
    }
    
    // objective value
    f = this->obj_value;

    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProbKron K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif
    return true;
  }

  void ContingencyProblemKronRed::get_solution_simplicial_vectorized(std::vector<double>& vsln)
  {
    SCACOPFData& dK = *data_K_[0];
    
    assert(v_n0);
    //assert(theta_n0);
    assert(b_s0);
    assert(p_g0);
    assert(q_g0);

    int n_sln = v_n0->n + theta_n0->n + b_s0->n + p_g0->n + q_g0->n + 1;
    vsln = vector<double>(n_sln);

    //auto v_nk = variable("v_n", dK);
    vector<double> v_nk, theta_nk;
    compute_v_and_theta_from_v_complex(v_n_all_complex_, v_nk, theta_nk);
    //
    assert(v_nk.size() == v_n0->n);
    //memcpy(&vsln[0], v_nk->x, v_nk->n*sizeof(double));
    std::copy(v_nk.begin(), v_nk.end(), vsln.begin());
    
    //auto theta_nk = variable("theta_n", dK);
    assert(theta_nk.size() == theta_n0->n);
    //memcpy(&vsln[v_nk->n], theta_nk->x, theta_nk->n*sizeof(double));
    std::copy(theta_nk.begin(), theta_nk.end(), vsln.begin()+v_n0->n);

    auto b_sk = variable("b_s", dK);
    assert(b_sk->n == b_s0->n);
    memcpy(&vsln[v_nk.size() + theta_nk.size()], b_sk->x, b_sk->n*sizeof(double));

    auto p_gk = variable("p_g", dK);
    auto q_gk = variable("q_g", dK);

    assert(dK.K_Contingency.size()==1); assert(dK.K_IDout.size()==1); 
    assert(dK.K_ConType.size()==1);
    
    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      int offset = v_nk.size() + theta_nk.size() + b_sk->n;
#ifdef DEBUG
      for(int i=0; i<p_g0->n + q_g0->n; i++) vsln[offset+i]= -10117.;
#endif

      int ngk = Gk_.size(); assert(ngk==p_g0->n-1);
      for(int i=0; i<ngk; i++) {
	assert(Gk_[i]>=0 && Gk_[i]<p_g0->n);
	vsln[offset+Gk_[i]] = p_gk->x[i];
      }
      assert(vsln[offset+dK.K_outidx[0]]==-10117.);
      vsln[offset+dK.K_outidx[0]] = 0.;

      offset += p_g0->n;
      for(int i=0; i<ngk; i++) {
	assert(Gk_[i]>=0 && Gk_[i]<q_g0->n);
	vsln[offset+Gk_[i]] = q_gk->x[i];
      }
#ifdef DEBUG
      assert(vsln[offset+dK.K_outidx[0]]==-10117.);
#endif
      vsln[offset+dK.K_outidx[0]] = 0.;

    } else {
      assert(p_gk->n == p_g0->n);
      assert(q_gk->n == q_g0->n);
      memcpy(&vsln[v_nk.size() + theta_nk.size() + b_sk->n], p_gk->x, p_gk->n*sizeof(double));
      memcpy(&vsln[v_nk.size() + theta_nk.size() + b_sk->n + p_gk->n], q_gk->x, q_gk->n*sizeof(double));
    }

    auto delta_k = variable("delta", dK);
    assert(delta_k != NULL);
    vsln[n_sln-1] = delta_k->x[0];
  }

  //
  // PVPQ
  //

  void ContingencyProblemKronRed::add_const_nonanticip_v_n_using(OptVariablesBlock* v_n0, 
								 const vector<int>& Gk) 
  {
    assert(Gk_ == Gk);
    assert(data_K_.size()==1);
    SCACOPFData& dB = *data_K_[0];
    auto G_Nidx_Gk = selectfrom(data_sc_.G_Nidx, Gk);
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
	  assert(data_sc_.G_Generator[dB.K_outidx[0]]!=dB.G_Generator[g]);
#endif
	if(abs(dB.G_Qub[g]-dB.G_Qlb[g])<=1e-8) {
	  numfixed++; num_qgens_fixed++;
#ifdef BE_VERBOSE
	  //printf("PVPQ: gen ID=%d p_q at bus idx %d id %d is fixed; will not enforce voltage nonanticip constraint\n",
	  //	 dB.G_Generator[g], dB.G_Nidx[g], data_sc_.N_Bus[dB.G_Nidx[g]]);
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
    auto v_aux_nk = variable("v_aux_n", dB);
    assert(v_n0->n >= v_nk->n);

    assert(v_n0->xref == v_n0->x);

    assert(map_idxbuses_idxsoptimiz_.size() == v_n0->n);
    
    for(int bidx : idxs_bus_pvpq) {
      assert(bidx>=0);
      assert(bidx<v_n0->n);

      //printf("[warning] nothing really: bidx=%d K_idx=%d\n", bidx, K_idx); 
      
      if(map_idxbuses_idxsoptimiz_[bidx]==-1) {
	printf("[warning] bus idx %d is PVPQ but not in v_n or v_n_aux of Kron\n", bidx);
	continue;
      } else {
	if(map_idxbuses_idxsoptimiz_[bidx]>=0) {
	  const int idx_nonaux = map_idxbuses_idxsoptimiz_[bidx];
	  assert(idx_nonaux < v_nk->n);
	  v_nk->lb[idx_nonaux] = v_n0->xref[bidx];
	  v_nk->ub[idx_nonaux] = v_n0->xref[bidx];
	} else {
	  const int idx_aux = -map_idxbuses_idxsoptimiz_[bidx]-2;
	  assert(idx_aux < v_aux_nk->n);
	  assert(idx_aux >= 0);
	  v_aux_nk->lb[idx_aux] = v_n0->xref[bidx];
	  v_aux_nk->ub[idx_aux] = v_n0->xref[bidx];
	}
      }
      //v_nk->lb[b] = v_n0->xref[b];
      //v_nk->ub[b] = v_n0->xref[b]; 
    }
  }
  /*
  void ContingencyProblemKronRed::add_cons_PVPQ_using(OptVariablesBlock* vn0, 
						      const vector<int>& Gk) 
  {
    assert(Gk_ == Gk);
    assert(data_K_.size()==1);
    assert(v_n0 == vn0);
    SCACOPFData& dB = *data_K_[0];

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
      printf("Contingency(Kron) %d: v_n var not found in conting problem; will NOT add PVPQ constraints.\n",
	     dB.id);
      return;
    }
    if(NULL==q_gk) {
      printf("Contingency(Kron) %d: q_g var not found in the base case; will NOT add PVPQ constraints.\n",
	     dB.id);
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
    printf("ContProbKron K_idx=%d: PVPQ: participating %d gens at %lu buses: "
	   "added %d constraints; PVPQSmoothing=%g; "
	   "total PVPQ: %lu gens | %d buses; fixed: %d gens | %d buses with all gens fixed.\n",
	   K_idx,
	   nPVPQGens-num_qgens_fixed, idxs_bus_pvpq.size(), cons->n, PVPQSmoothing,
	   Gk.size(), num_N_PVPQ,
	   num_qgens_fixed, num_buses_all_qgen_fixed);
#endif
  }
  */
  void ContingencyProblemKronRed::update_cons_PVPQ_using(OptVariablesBlock* vn0, 
						  const vector<int>& Gk) {
    assert(false);
  }

  void ContingencyProblemKronRed::add_cons_nonanticip_using(OptVariablesBlock* pg0) {
    bodyof_cons_nonanticip_using(pg0);
#ifdef BE_VERBOSE
    printf("ContProbKron K_idx=%d on rank %d: "
	   "AGC: %lu gens NOT participating: fixed all of them.\n",
	   K_idx, my_rank, pg0_nonpartic_idxs.size());
#endif
  }
  void ContingencyProblemKronRed::bodyof_cons_nonanticip_using(OptVariablesBlock* pg0)
  {
    assert(1==data_K_.size());
    SCACOPFData& dK = *data_K_[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblemKron K_idx=%d: p_g var not found in contingency  "
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
  /*
  void ContingencyProblemKronRed::add_cons_AGC_using(OptVariablesBlock* pg0)
  {
    if(pgK_partic_idxs.size()==0) {
      //assert(pg0_partic_idxs.size()==0);
#ifdef BE_VERBOSE
      printf("ContingencyProblemKron: add_cons_AGC_using: NO gens participating !?!\n");
#endif
      return;
    }

    SCACOPFData& dK = *data_K_[0];
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblemKron K_idx=%d: p_g var not found in contingency  "
	     "recourse problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      assert(false);
      return;
    }
    OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK));
    append_varsblock(deltaK);
    deltaK->set_start_to(0.);
    
    //AGCSmoothing = 1e-3;
    auto cons = new AGCComplementarityCons(con_name("AGC", dK), 3*pgK_partic_idxs.size(),
					   pg0, pgK, deltaK, 
					   pg0_partic_idxs, pgK_partic_idxs, 
					   selectfrom(data_sc_.G_Plb, pg0_partic_idxs), 
					   selectfrom(data_sc_.G_Pub, pg0_partic_idxs),
					   data_sc_.G_alpha, AGCSmoothing,
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
    printf("ContProbKron K_idx=%d: AGC %lu gens participating (out of %d) AGCSmoothing=%g\n", 
    	   K_idx, pgK_partic_idxs.size(), pgK->n, AGCSmoothing);
#endif
    //printvec(pg0_partic_idxs, "partic idxs");
  }
  */


  void ContingencyProblemKronRed::update_cons_AGC_using(OptVariablesBlock* pg0)
  {
    if(pgK_partic_idxs.size()==0) {
      return;
    }
    //pg0 pointer that AGCComplementarityCons should not change
#ifdef DEBUG
    SCACOPFData& dK = *data_K_[0];
    auto cons_AGC = dynamic_cast<AGCComplementarityCons*>(constraint("AGC", dK));
    assert(cons_AGC);
    if(pg0 != cons_AGC->get_p_g0()) {
      assert(false);
    }
#endif
  }

  void ContingencyProblemKronRed::regularize_vn(const double& gamma)
  {
    assert(false);
    assert(data_K_.size()==1);

    SCACOPFData& dK = *data_K_[0];
    assert(dK.id==K_idx+1);

    assert(variable("v_n", dK));
    assert(v_n0);
    //! todo: map v_n0 into v_n nonaux
    assert(variable("v_n", dK)->n <= v_n0->n);
    
    OptObjectiveTerm* ot = obj->objterm("regul_vn");
    if(NULL==ot) {
      append_objterm(new QuadrRegularizationObjTerm("regul_vn", variable("v_n", dK),
						    gamma, v_n0->x));
      primal_problem_changed();
    } else {
      QuadrRegularizationObjTerm* rot = dynamic_cast<QuadrRegularizationObjTerm*>(ot);
      rot->gamma = gamma;
    }
  }
  void ContingencyProblemKronRed::regularize_thetan(const double& gamma)
  {
    assert(false);
    assert(data_K_.size()==1);
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);
    assert(variable("theta_n", dK)->n <= v_n0->n);
    
    //! todo: map theta_n0 into theta_n nonaux
    OptObjectiveTerm* ot = obj->objterm("regul_thetan");
    if(NULL==ot) {
      append_objterm(new QuadrRegularizationObjTerm("regul_thetan", variable("theta_n", dK),
						    gamma, v_n0->x));
      primal_problem_changed();
    } else {
      QuadrRegularizationObjTerm* rot = dynamic_cast<QuadrRegularizationObjTerm*>(ot);
      rot->gamma = gamma;
    }
  }
  void ContingencyProblemKronRed::regularize_bs(const double& gamma)
  {
    assert(false);
    assert(data_K_.size()==1);
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);

    OptObjectiveTerm* ot = obj->objterm("regul_bs");
    if(NULL==ot) {
      //! todo should not use v_n0
      append_objterm(new QuadrRegularizationObjTerm("regul_bs", variable("b_s", dK),
						    gamma, v_n0->x));
      primal_problem_changed();
    } else {
      QuadrRegularizationObjTerm* rot = dynamic_cast<QuadrRegularizationObjTerm*>(ot);
      rot->gamma = gamma;
    }
  }
  void ContingencyProblemKronRed::regularize_pg(const double& gamma)
  {
    assert(data_K_.size()==1);
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);

    OptObjectiveTerm* ot = obj->objterm("regul_pg");
    if(NULL==ot) {
      //! todo should not use v_n0
      append_objterm(new QuadrRegularizationObjTerm("regul_pg", variable("p_g", dK),
						    gamma, v_n0->x));
      primal_problem_changed();
    } else {
      QuadrRegularizationObjTerm* rot = dynamic_cast<QuadrRegularizationObjTerm*>(ot);
      rot->gamma = gamma;
    }
  }
  void ContingencyProblemKronRed::regularize_qg(const double& gamma)
  {
    assert(data_K_.size()==1);
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);

    OptObjectiveTerm* ot = obj->objterm("regul_qg");
    if(NULL==ot) {
      //! todo should not use v_n0
      append_objterm(new QuadrRegularizationObjTerm("regul_qg", variable("q_g", dK),
						    gamma, v_n0->x));
      primal_problem_changed();
    } else {
      QuadrRegularizationObjTerm* rot = dynamic_cast<QuadrRegularizationObjTerm*>(ot);
      rot->gamma = gamma;
    }
  }

  void ContingencyProblemKronRed::update_regularizations(const double& gamma)
  {
    assert(false);
    //regularize_vn(gamma);
    //regularize_thetan(gamma);
    //regularize_bs(gamma);
    regularize_pg(gamma);
    regularize_qg(gamma);
  }
  //   if(reg_thetan) {
  //     append_objterm(new QuadrRegularizationObjTerm("regul_thetan", variable("theta_n", dK),
  // 						    1e-4, theta_n0->x));
  //     //printf("added regularization term for theta_n\n");

  //   }
  //   if(reg_bs) {
  //     append_objterm(new QuadrRegularizationObjTerm("regul_bs", variable("b_s", dK),
  // 						    1e-4, b_s0->x));
  //     //printf("added regularization term for b_s\n");

  //   }
  //   if(reg_pg) {
  //     assert(Gk.size() == variable("p_g", dK)->n);
  //     assert(Gk.size() == p_g0->n || Gk.size() == p_g0->n-1);
  //     auto pg0_vec = selectfrom(p_g0->x, p_g0->n, Gk);
  //     append_objterm(new QuadrRegularizationObjTerm("regul_pg", variable("p_g", dK),
  // 						    1e-4, pg0_vec.data()));
  //     //printf("added regularization term for p_g\n");
  //   }
  //   if(reg_qg) {
  //     assert(Gk.size() == variable("q_g", dK)->n);
  //     assert(Gk.size() == q_g0->n || Gk.size() == q_g0->n-1);
  //     auto qg0_vec = selectfrom(q_g0->x, q_g0->n, Gk);
  //     append_objterm(new QuadrRegularizationObjTerm("regul_qg", variable("q_g", dK),
  // 						    1e-4, qg0_vec.data()));
  //     //printf("added regularization term for q_g\n");
  //   }
  // }


#define SIGNED_DUALS_VAL 0.

  bool ContingencyProblemKronRed::set_warm_start_from_base_of(SCACOPFProblem& srcProb)
  {
    assert(data_K_.size()==1);
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);

    // contingency indexes of lines, generators, or transformers (i.e., contingency type)
    vector<int> idxs_of_K_in_0; 

    assert(useQPen==true); assert(srcProb.useQPen==true);
    variable("v_n", dK)->set_start_to(*srcProb.variable("v_n", data_sc_));
    variable("theta_n", dK)->set_start_to(*srcProb.variable("theta_n", data_sc_));
    variable("b_s", dK)->set_start_to(*srcProb.variable("b_s", data_sc_));

    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      auto p_gK = variable("p_g", dK), p_g0 = srcProb.variable("p_g", data_sc_);
      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
      }
      p_gK->providesStartingPoint = true;

      auto q_gK = variable("q_g", dK), q_g0 = srcProb.variable("q_g", data_sc_);
      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
      }
      q_gK->providesStartingPoint = true;
      
    } else {
#ifdef DEBUG
      assert(variable("p_g", dK)->n == srcProb.variable("p_g", data_sc_)->n);
      assert(variable("q_g", dK)->n == srcProb.variable("q_g", data_sc_)->n);
#endif
      variable("p_g", dK)->set_start_to(*srcProb.variable("p_g", data_sc_));
      variable("q_g", dK)->set_start_to(*srcProb.variable("q_g", data_sc_));
    }
    
    if(dK.K_ConType[0] == SCACOPFData::kLine) {
      idxs_of_K_in_0 = indexin(dK.L_Line, data_sc_.L_Line);
      size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();

      auto var_K = variable("p_li1", dK), var_0 = srcProb.variable("p_li1", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("p_li2", dK); var_0 = srcProb.variable("p_li2", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_li1", dK); var_0 = srcProb.variable("q_li1", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_li2", dK); var_0 = srcProb.variable("q_li2", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

    } else {
      assert(variable("p_li1", dK)->n == srcProb.variable("p_li1", data_sc_)->n);
      assert(variable("p_li2", dK)->n == srcProb.variable("p_li2", data_sc_)->n);
      assert(variable("q_li1", dK)->n == srcProb.variable("q_li1", data_sc_)->n);
      assert(variable("q_li2", dK)->n == srcProb.variable("q_li2", data_sc_)->n);

      variable("p_li1", dK)->set_start_to(*srcProb.variable("p_li1", data_sc_));
      variable("p_li2", dK)->set_start_to(*srcProb.variable("p_li2", data_sc_));
      variable("q_li1", dK)->set_start_to(*srcProb.variable("q_li1", data_sc_));
      variable("q_li2", dK)->set_start_to(*srcProb.variable("q_li2", data_sc_));
    }

    if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
      idxs_of_K_in_0 = indexin(dK.T_Transformer, data_sc_.T_Transformer);
      size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();

      auto var_K = variable("p_ti1", dK), var_0 = srcProb.variable("p_ti1", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("p_ti2", dK); var_0 = srcProb.variable("p_ti2", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_ti1", dK); var_0 = srcProb.variable("q_ti1", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;

      var_K = variable("q_ti2", dK); var_0 = srcProb.variable("q_ti2", data_sc_);
      assert(var_K->n+1 == var_0->n);
      assert(sz == var_K->n);
      for(i=0; i<sz; i++) {
    	assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
    	var_K->x[i] = var_0->x[idxs_in_0[i]];
      }
      var_K->providesStartingPoint = true;


    } else {
      assert(variable("p_ti1", dK)->n == srcProb.variable("p_ti1", data_sc_)->n);
      assert(variable("p_ti2", dK)->n == srcProb.variable("p_ti2", data_sc_)->n);
      assert(variable("q_ti1", dK)->n == srcProb.variable("q_ti1", data_sc_)->n);
      assert(variable("q_ti2", dK)->n == srcProb.variable("q_ti2", data_sc_)->n);

      variable("p_ti1", dK)->set_start_to(*srcProb.variable("p_ti1", data_sc_));
      variable("p_ti2", dK)->set_start_to(*srcProb.variable("p_ti2", data_sc_));
      variable("q_ti1", dK)->set_start_to(*srcProb.variable("q_ti1", data_sc_));
      variable("q_ti2", dK)->set_start_to(*srcProb.variable("q_ti2", data_sc_));
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
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      prefix = "duals_bndL_theta_n";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));

      prefix = "duals_bndL_p_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_p_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_q_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_q_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_p_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_p_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_q_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_q_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_b_s";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));

      prefix = "duals_bndL_p_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	//variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
	auto p_gK = variable_duals_lower(prefix, dK), p_g0 = srcProb.variable_duals_lower(prefix, data_sc_);
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
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_q_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	auto q_gK = variable_duals_lower(prefix, dK), q_g0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(q_gK->n == q_g0->n -1);
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
	}
	q_gK->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_pslack_n_p_balance";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      prefix = "duals_bndL_qslack_n_q_balance";
      variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));

      prefix = "duals_bndL_sslack_li_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;

      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }
      
      prefix = "duals_bndL_sslack_li_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));	
      }
      
      prefix = "duals_bndL_sslack_ti_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
      }

      prefix = "duals_bndL_sslack_ti_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_lower(prefix, dK), var_0 = srcProb.variable_duals_lower(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_lower(prefix, dK)->set_start_to(*srcProb.variable_duals_lower(prefix, data_sc_));
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
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      prefix = "duals_bndU_theta_n";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));

      prefix = "duals_bndU_p_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }
      prefix = "duals_bndU_p_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_q_li1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }
      prefix = "duals_bndU_q_li2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_p_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_p_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_q_ti1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_q_ti2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_b_s";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));

      prefix = "duals_bndU_p_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	//variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
	auto p_gK = variable_duals_upper(prefix, dK), p_g0 = srcProb.variable_duals_upper(prefix, data_sc_);
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
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_q_g";
      if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	auto q_gK = variable_duals_upper(prefix, dK), q_g0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(q_gK->n == q_g0->n -1);
	for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	  q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
	}
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
	}
	q_gK->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_pslack_n_p_balance";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      prefix = "duals_bndU_qslack_n_q_balance";
      variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      
      prefix = "duals_bndU_sslack_li_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_sslack_li_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_sslack_ti_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
      }

      prefix = "duals_bndU_sslack_ti_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_upper(prefix, dK), var_0 = srcProb.variable_duals_upper(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;	
      } else {
	variable_duals_upper(prefix, dK)->set_start_to(*srcProb.variable_duals_upper(prefix, data_sc_));
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
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_p_li2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_q_li1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_q_li2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_p_ti1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_p_ti2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_q_ti1_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_q_ti2_powerflow";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_p_balance";
      variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      prefix = "duals_q_balance";
      variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));

      prefix = "duals_line_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_line_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kLine) {
	assert(idxs_of_K_in_0.size() == dK.L_Line.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }

      prefix = "duals_trans_limits1";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
      }


      prefix = "duals_trans_limits2";
      if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	assert(idxs_of_K_in_0.size() == dK.T_Transformer.size());
	size_t sz = idxs_of_K_in_0.size(); int i, *idxs_in_0 = idxs_of_K_in_0.data();
	auto var_K = variable_duals_cons(prefix, dK), var_0 = srcProb.variable_duals_cons(prefix, data_sc_);
	assert(var_K->n+1 == var_0->n);
	assert(sz == var_K->n);
	for(i=0; i<sz; i++) {
	  assert(idxs_in_0[i]>=0 && idxs_in_0[i]<var_0->n);
	  var_K->x[i] = var_0->x[idxs_in_0[i]];
	}
	var_K->providesStartingPoint = true;
      } else {
	variable_duals_cons(prefix, dK)->set_start_to(*srcProb.variable_duals_cons(prefix, data_sc_));
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
  bool ContingencyProblemKronRed::
  set_warm_start_from_contingency_of(SCACOPFProblem& srcProb)
  {
    assert(data_K_.size()==1); 
    SCACOPFData& dK = *data_K_[0]; assert(dK.id==K_idx+1);
    bool bfound = false;
    for(auto d : srcProb.data_K) if(d->id == dK.id) bfound=true;
    if(!bfound) {
      printf("(Kron) set_warm_start_from_contingency_of SCACOPFProblem: src does not have "
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

  bool ContingencyProblemKronRed::
  iterate_callback(int iter, const double& obj_value,
		   const double* primals,
		   const double& inf_pr, const double& inf_pr_orig_pr, 
		   const double& inf_du, 
		   const double& mu, 
		   const double& alpha_du, const double& alpha_pr,
		   int ls_trials, OptimizationMode mode,
		   const double* duals_con,
		   const double* duals_lb, const double* duals_ub)
  {
    if(cc_callbacks_) {
      return cc_callbacks_->
	iterate_callback(iter, obj_value, primals, inf_pr, inf_pr_orig_pr, inf_du, mu,
			 alpha_du, alpha_pr, ls_trials, mode, duals_con, duals_lb, duals_ub);
    }
    
    return true;
  }
  
} //end of namespace
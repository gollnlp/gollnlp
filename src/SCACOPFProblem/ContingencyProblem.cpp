#include "ContingencyProblem.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"
#include "unistd.h"
using namespace std;

namespace gollnlp {
  
  ContingencyProblem::ContingencyProblem(SCACOPFData& d_in, int K_idx_, int my_rank_) 
    : SCACOPFProblem(d_in), K_idx(K_idx_), my_rank(my_rank_)
  {
    int numK = 1; //!

    assert(0==data_K.size());
    //data_sc = d_in (member of the parent)
    data_K.push_back(new SCACOPFData(data_sc)); 
    data_K[0]->rebuild_for_conting(K_idx, numK);
  }

  ContingencyProblem::~ContingencyProblem()
  {
  }
  
  bool ContingencyProblem::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    printf("ContingencyProblem K_id %d: assembly IDOut=%d outidx=%d Type=%s\n",
	   K_idx, data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str());

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];

    useQPen = true;
    //slacks_scale = 1.;

    add_variables(dK);

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
    vector<int> Gk;
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
    
    //print_summary();
    //PVPQSmoothing = AGCSmoothing = 1e-2;
    //coupling AGC and PVPQ; also creates delta_k
    //add_cons_coupling(dK);
    return true;
  }

  bool ContingencyProblem::eval_obj(OptVariablesBlock* pg0,
				   OptVariablesBlock* vn0,
				   double& f)
  {
#ifdef DEBUG
    goTimer tmrec; tmrec.start();
#endif
    update_cons_nonanticip_using(pg0);
    update_cons_AGC_using(pg0);
    //!update_cons_PVPQ_using(vn0);

    use_nlp_solver("ipopt");
    set_solver_option("print_frequency_iter", 1);
    set_solver_option("linear_solver", "ma57"); 
    set_solver_option("print_level", 2);
    set_solver_option("mu_init", 1e-4);
    set_solver_option("mu_target", 1e-9);
    //if(!optimize("ipopt")) {
    if(!reoptimize(OptProblem::primalDualRestart)) {
      //if(!reoptimize(OptProblem::primalRestart)) {
      return false;
    }

    // objective value
    f = this->obj_value;
#ifdef DEBUG
    tmrec.stop();
    printf("ContingencyProblem K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    //printf("ContingencyProblem K_id %d: recourse obj_value %g\n", K_idx, this->obj_value);

    if(false) {
      int dim = pg0->n;
      printf("p_g0 in\n");
      for(int i=0; i<pg0->n; i++)
	printf("%12.5e ", pg0->x[i]);
      printf("\n\n");
    }
#endif
    return true;
  }

  void ContingencyProblem::bodyof_cons_nonanticip_using(OptVariablesBlock* pg0)
  {
    SCACOPFData& dK = *data_K[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblem K_id %d: p_g var not found in contingency  "
	     "problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    
    int sz = pgK_nonpartic_idxs.size();
    assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    int idxK; double pg0_val, lb, ub; 

#ifdef DEBUG
    assert(pg0->xref == pg0->x);

    //usleep(1e6*my_rank);
    // printf("cont %d rank %d\n", K_idx, my_rank);
    // for(int i=0; i<sz; i++) {
    //   assert(pg0_idxs[i]<pg0->n && pg0_idxs[i]>=0);
    //   assert(pgK_idxs[i]<pgK->n && pgK_idxs[i]>=0);
    //   idxK = pgK_idxs[i];
    //   pgK->lb[idxK] = pgK->ub[idxK] = pg0->xref[pg0_idxs[i]];

    //   printf("%g %g\n",  pg0->x[pg0_idxs[i]],  pg0->xref[pg0_idxs[i]]);
    // }
    // printf("-----------------\n\n");
#endif
  }

  void ContingencyProblem::add_cons_AGC_using(OptVariablesBlock* pg0)
  {
    if(pgK_partic_idxs.size()==0) {
      //assert(pg0_partic_idxs.size()==0);
      printf("ContingencyProblem: add_cons_AGC_using: NO gens participating !?!\n");
      return;
    }

    SCACOPFData& dK = *data_K[0];
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblem K_id %d: p_g var not found in contingency  "
	     "recourse problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      assert(false);
      return;
    }
    OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK));
    append_variables(deltaK);
    deltaK->set_start_to(0.);
    
    AGCSmoothing = 1e-2;
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

    //append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhom->id, rhom, 1e-2));
    //append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhop->id, rhop, 1e-2));

    printf("ContingencyProblem K_id %d: AGC %lu gens participating (out of %d)\n", 
	   K_idx, pgK_partic_idxs.size(), pgK->n);
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

#define SIGNED_DUALS_VAL 1.

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
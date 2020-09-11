#include "ContingencyProblemKronRedWithFixingCode1.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"


#include "goSignalHandling.hpp"

using namespace std;

//definitions in ContingencyProblemWithFixing
extern const int max_mem_ma57_normal = 1000; //MB
extern const int max_mem_ma57_safem = 1500; //MB
extern const int alarm_ma57_normal = 30; //seconds
extern const int alarm_ma57_safem = 30; //M
extern const int max_mem_ma27_normal = 1000; //MB
extern const int max_mem_ma27_safem = 1500; //MB
extern const int alarm_ma27_normal = 45; //seconds
extern const int alarm_ma27_safem = 45; //MB


extern volatile sig_atomic_t g_solve_watch_ma57;
extern volatile sig_atomic_t g_alarm_duration_ma57;
extern volatile sig_atomic_t g_max_memory_ma57;
extern volatile int g_my_rank_ma57;
extern volatile int g_my_K_idx_ma57;
void set_timer_message_ma57(const char* msg);

extern volatile sig_atomic_t g_solve_watch_ma27;
extern volatile sig_atomic_t g_alarm_duration_ma27;
extern volatile sig_atomic_t g_max_memory_ma27;
extern volatile int g_my_rank_ma27;
extern volatile int g_my_K_idx_ma27;
void set_timer_message_ma27(const char* msg);

#define BE_VERBOSE 1

namespace gollnlp {

  bool ContingencyProblemKronRedWithFixingCode1::
  default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0)
  {
    double K_avg_time_so_far = time_so_far / std::max(num_K_done,1);
    string ssfm=" ";
    if(safe_mode) ssfm=" [safe mode] ";
    printf("ContProbKronWithFixing K_idx=%d%sIDOut=%d outidx=%d Type=%s avgtm=%.2f rank=%d\n",
	   K_idx, ssfm.c_str(), data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str(), K_avg_time_so_far, my_rank); fflush(stdout);

    p_g0=pg0; v_n0=vn0;
    prob_mds_->v_n0 = vn0;
    prob_mds_->p_g0 = pg0;
    

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];
    
    useQPen = true;
    prob_mds_->useQPen = true;

    ////////////////////////////////////////////////////////////
    // setup for indexes used in non-anticip and AGC coupling 
    ////////////////////////////////////////////////////////////
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);
    assert(pg0->n == Gk.size() || pg0->n == 1+Gk.size());

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

    prob_mds_->Gk_ = Gk;

    prob_mds_->initialize(false);
    
    prob_mds_->add_variables(dK, false);

    if(!warm_start_variable_from_basecase_dict(*prob_mds_->vars_primal)) {
      assert(false);
      return false;
    }

    auto pgK = prob_mds_->variable("p_g", dK); assert(pgK!=NULL);
    //find AGC generators that are "blocking" and fix them; update particip and non-particip indexes
    solv1_pg0_partic_idxs=pg0_partic_idxs; solv1_pgK_partic_idxs=pgK_partic_idxs;
    solv1_pgK_nonpartic_idxs=pgK_nonpartic_idxs; solv1_pg0_nonpartic_idxs=pg0_nonpartic_idxs;
    double gen_K_diff=0.01;//default for transmission contingencies; surplus or deficit of generation
    double residual_Pg;
    solv1_delta_out=0.; solv1_delta_lb=-1e+20; solv1_delta_ub=1e+20; 
    solv1_delta_blocking=0.; solv1_delta_needed=0.; solv1_gens_pushed = 0.;
    if(dK.K_ConType[0]==SCACOPFData::kGenerator || gen_K_diff>0.) {
      
      if(dK.K_ConType[0]==SCACOPFData::kGenerator) {
	assert(data_sc.K_outidx[K_idx]<pg0->n);
	gen_K_diff = pg0->x[data_sc.K_outidx[K_idx]];
	
      }

      solv1_Pg_was_enough = push_and_fix_AGCgen(dK, gen_K_diff, 0., 
				   solv1_pg0_partic_idxs, solv1_pgK_partic_idxs, 
				   solv1_pg0_nonpartic_idxs, solv1_pgK_nonpartic_idxs,
				   pg0, pgK, 
				   data_sc.G_Plb, data_sc.G_Pub, data_sc.G_alpha,
				   solv1_delta_out, solv1_delta_needed, solv1_delta_blocking, 
				   solv1_delta_lb, solv1_delta_ub, residual_Pg);
      //alter starting points 
      assert(solv1_pg0_partic_idxs.size() == solv1_pgK_partic_idxs.size());
      for(int it=0; it<solv1_pg0_partic_idxs.size(); it++) {
	const int& i0 = solv1_pg0_partic_idxs[it]; 
	pgK->x[solv1_pgK_partic_idxs[it]] = pg0->x[i0]+data_sc.G_alpha[i0] * solv1_delta_out;
      }
      solv1_gens_pushed = pg0_partic_idxs.size()-solv1_pg0_partic_idxs.size();
#ifdef BE_VERBOSE
      printf("ContProbKronWithFixing K_idx=%d def_ass (extra gener) %.8f gen missing; "
	     "fixed %lu gens; delta out=%g needed=%g blocking=%g residualPg=%g feasib=%d\n",
	     K_idx, gen_K_diff, pg0_partic_idxs.size()-solv1_pg0_partic_idxs.size(),
	     solv1_delta_out, solv1_delta_needed, solv1_delta_blocking, residual_Pg, solv1_Pg_was_enough);
#endif
    }

    //! removed prob_mds_->add_cons_lines_pf(dK);
    //! removed prob_mds_->add_cons_transformers_pf(dK);

    
    //! replaced prob_mds_->add_cons_active_powbal(dK);
    //! replaced prob_mds_->add_cons_reactive_powbal(dK);
    prob_mds_->add_cons_pf(dK);
    
    //! removed bool SysCond_BaseCase = false;
    //! removed prob_mds_->add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    //! removed prob_mds_->add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    this->add_cons_pg_nonanticip_using(pg0, solv1_pg0_nonpartic_idxs, solv1_pgK_nonpartic_idxs);
    //add_cons_AGC_using(pg0);
    
    if(solv1_pg0_partic_idxs.size() > 0) {
      this->add_cons_AGC_simplified(dK, solv1_pg0_partic_idxs, solv1_pgK_partic_idxs, pg0);
      auto deltav = prob_mds_->variable("delta", dK); assert(deltav);
      if(deltav) { //it may happen that all AGC gens were fixed
	deltav->set_start_to(solv1_delta_out);
	deltav->lb[0] = solv1_delta_lb; deltav->ub[0] = solv1_delta_ub; 
      } 
    } else { 
      //all AGC gens were fixed; add fixed variable delta 
      OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK)); 
      prob_mds_->append_varsblock(deltaK);
      deltaK->set_start_to(solv1_delta_out);
      deltaK->lb[0] = deltaK->ub[0] = solv1_delta_out;
      prob_mds_->append_objterm(new QuadrRegularizationObjTerm("delta_regul", deltaK, 1., solv1_delta_out));
    }

    prob_mds_->add_const_nonanticip_v_n_using(vn0, Gk);
    //PVPQSmoothing=1e-8;
    //add_cons_PVPQ_using(vn0, Gk);

    assert(prob_mds_->vars_primal->provides_start());

    const double gamma = 1e-6;
    //prob_mds_->regularize_vn(gamma);
    //prob_mds_->regularize_thetan(gamma);
    prob_mds_->regularize_bs(gamma);
    prob_mds_->regularize_pg(gamma);
    prob_mds_->regularize_qg(gamma);
    
    if(NULL==prob_mds_->vars_duals_bounds_L ||
       NULL==prob_mds_->vars_duals_bounds_U ||
       NULL==prob_mds_->vars_duals_cons) {
      //force allocation of duals
      prob_mds_->dual_problem_changed();
    }

    if(!warm_start_variable_from_basecase_dict(*prob_mds_->vars_duals_bounds_L)) {
      assert(false);
      return false;
    }
    
    if(prob_mds_->variable_duals_lower("duals_bndL_delta", dK)) {
      prob_mds_->variable_duals_lower("duals_bndL_delta", dK)->set_start_to(0.0);
    } else {
      assert(false);
    }
    assert(prob_mds_->vars_duals_bounds_L->provides_start());

    if(!warm_start_variable_from_basecase_dict(*prob_mds_->vars_duals_bounds_U))  {
      assert(false);
      return false;
    }
    if(prob_mds_->variable_duals_upper("duals_bndU_delta", dK))
      prob_mds_->variable_duals_upper("duals_bndU_delta", dK)->set_start_to(0.0);
    assert(prob_mds_->vars_duals_bounds_U->provides_start());
    
    //AGC_simple_fixedpg0
    if(!warm_start_variable_from_basecase_dict(*prob_mds_->vars_duals_cons))  {
      assert(false);
      return false;
    }

    //! no obvious way to start the duals from the base case.
    prob_mds_->duals_constraints()->set_start_to(0.0);
    //if(prob_mds_->variable_duals_cons("duals_AGC_simple_fixedpg0", dK))
    //prob_mds_->variable_duals_cons("duals_AGC_simple_fixedpg0", dK)->set_start_to(0.0);
    
    assert(prob_mds_->vars_duals_cons->provides_start());


#ifdef GOLLNLP_FAULT_HANDLING
    string msg =
      "[timer] ma57 timeout rank=" + std::to_string(my_rank) +
      " for K_idx=" + std::to_string(K_idx) + " occured!\n";
    set_timer_message_ma57(msg.c_str());

    msg =
      "[timer] ma27 timeout rank=" + std::to_string(my_rank) +
      " for K_idx=" + std::to_string(K_idx) + " occured!\n";
    set_timer_message_ma27(msg.c_str());

    assert(my_rank>=1);
#endif
    vars_last = prob_mds_->vars_primal->new_copy();
    vars_ini  = prob_mds_->vars_primal->new_copy();
    
    double* x = new double[prob_mds_->vars_primal->n()];
    double obj;
    prob_mds_->vars_primal->copy_to(x);
    //will copy the values
    best_known_iter.initialize(prob_mds_->vars_primal);
    if(prob_mds_->OptProblemMDS::eval_obj(x, true, obj)) {
      best_known_iter.set_objective(obj);
    } else {
      assert(false);
      best_known_iter.set_objective(1e+10);
    }
    delete [] x;
    return true;
  }
  
  bool ContingencyProblemKronRedWithFixingCode1::eval_obj(OptVariablesBlock* pg0,
						   OptVariablesBlock* vn0,
						   double& f,
						   double* data_for_master)
  {
    //prob_mds_->print_summary(); fflush(stdout);

    goTimer tmrec; tmrec.start();
    SCACOPFData& d = *data_K[0];
    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    set_no_recourse_action(data_for_master);

    //!
    //this is disabled for now
    if(false) {
    if(best_known_iter.obj_value <= pen_accept_initpt) {
      assert(prob_mds_->vars_primal->n() == best_known_iter.vars_primal->n());
      prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
      prob_mds_->obj_value = best_known_iter.obj_value;

#ifdef BE_VERBOSE
      printf("ContProbKron_wfix K_idx=%d opt1 ini point is acceptable on rank=%d\n", K_idx, my_rank);
      fflush(stdout);
#endif
      f = prob_mds_->obj_value;
      set_no_recourse_action(data_for_master, f);
      return true;
    }
    }
    //!
    best_known_iter.set_objective(1e+10);
    
    printf("!!!!!!!!!!!!!!!!!!!!   DO SOLVE 1 !!!!!!!!!!!!\n");
    
    bool bFirstSolveOK = do_solve1();

    printf("!!!!!!!!!!!!!!!!!!!!   DO SOLVE 1 DONE !!!!!!!!!!!!\n");
    
    f = prob_mds_->obj_value;
    
    if(prob_mds_->variable("delta", d))
      solv1_delta_optim = prob_mds_->variable("delta", d)->x[0];
    else
      solv1_delta_optim = 0.;

    //roundoff or bounds relaxation can creep in
    if(solv1_delta_optim<solv1_delta_lb) solv1_delta_optim = solv1_delta_lb;
    if(solv1_delta_optim>solv1_delta_ub) solv1_delta_optim = solv1_delta_ub;
    
    if(solv1_delta_lb <= -1e+20 && solv1_delta_ub >= +1e+20)
      if(solv1_gens_pushed==0 && solv1_delta_out==0.)
	if(fabs(solv1_delta_optim)<1e-8) solv1_delta_optim = 0.;

    double acceptable_penalty = safe_mode ?  pen_accept_safemode : pen_accept_solve1;

    bool skip_2nd_solve = false;
    
    if(!bFirstSolveOK) skip_2nd_solve=false;
    
    if(tmTotal.measureElapsedTime() > 0.95*timeout) {
      skip_2nd_solve = true;
      if(bFirstSolveOK) {
	printf("ContProbKron_wfix K_idx=%d premature exit opt1 too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
      } else {
	printf("ContProbKron_wfix K_idx=%d premature exit inipt returned opt1 took too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
	//return ini point to make sure we stay feasible
	prob_mds_->vars_primal->set_start_to(*vars_ini);
      }
    } else {
      if(f>=acceptable_penalty)
	determine_recourse_action(data_for_master);
    }

    if(monitor.emergency) acceptable_penalty = std::max(acceptable_penalty, pen_accept_emer);
    
    if(prob_mds_->obj_value>acceptable_penalty && !skip_2nd_solve) {

      //#ifdef BE_VERBOSE
      //print_objterms_evals();
      //print_p_g_with_coupling_info(*data_K[0], pg0);
      //printf("ContProbKron_wfix K_idx=%d first pass resulted in high pen; delta=%g\n", K_idx, solv1_delta_optim);
      //#endif

      double pplus, pminus, poverall;
      estimate_active_power_deficit(pplus, pminus, poverall);
#ifdef BE_VERBOSE
      printf("ContProbKron_wfix K_idx=%d (after solv1) act pow imbalances p+ p- poveral %g %g %g; delta=%g\n",
	     K_idx, pplus, pminus, poverall, solv1_delta_optim);
#endif

      bool one_more_push_and_fix=false; double gen_K_diff=0.;
      if(fabs(solv1_delta_optim-solv1_delta_blocking)<1e-2 && 
	 d.K_ConType[0]==SCACOPFData::kGenerator && solv1_Pg_was_enough) {
	one_more_push_and_fix = true;
	if(pg0->x[data_sc.K_outidx[K_idx]]>1e-6 )  gen_K_diff = std::max(0., 1.2*poverall);
	else if(pg0->x[data_sc.K_outidx[K_idx]]<-1e-6)  gen_K_diff = std::min(0., poverall);
	else one_more_push_and_fix = false;
      }

      if(solv1_delta_optim>=0) assert(gen_K_diff>=0);
      if(solv1_delta_optim<=0) assert(gen_K_diff<=0);
      
      if(solv1_delta_optim * gen_K_diff<0) {
	one_more_push_and_fix = false;
	gen_K_diff=0.;
      }
      
      if(fabs(poverall)>1e-4) {// && d.K_ConType[0]!=SCACOPFData::kGenerator) {
	double rpa = fabs(pplus) / fabs(poverall);
	double rma = fabs(pminus) / fabs(poverall);

	//solv1_delta_optim=0.;//!

	if( (rpa>0.85 && rpa<1.15) || (rma>0.85 && rma <1.15) ) {
	  if(poverall*solv1_delta_optim>=0 && poverall*solv1_delta_out>=0) {
	    gen_K_diff = 0.;
	    one_more_push_and_fix = true;
	    if(poverall<0)
	      gen_K_diff = poverall;
	    else if(poverall>0)
	      gen_K_diff = 1.2*poverall;
	    else one_more_push_and_fix = false;

	    
	    //if our first attempt to ramp up resulted in a active power balance deficit, then be more agressive this time
	    if(d.K_ConType[0]==SCACOPFData::kGenerator) {
	      double pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits;
	      get_objective_penalties(pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits);
	      if(pen_p_balance > 100.*pen_q_balance && 
		 pen_p_balance > 100.*pen_line_limits && 
		 pen_p_balance > 100.*pen_trans_limits) {
		
		double gK = pg0->x[data_sc.K_outidx[K_idx]];
		if(gK < -1e-6) assert(false); //would love to run into this case to double check things
		
		//double gen_deficit = pg0->x[data_sc.K_outidx[K_idx]];
		if(pen_p_balance > 3e5) {
		  if(poverall<0 && gK<=-1e-6) gen_K_diff = poverall;
		  else if(poverall>0 && gK>=+1e-6) gen_K_diff = 2.75*poverall;
		  else { gen_K_diff = 0.; one_more_push_and_fix = false; }
		} else if(pen_p_balance > 5e4) {
		  if(poverall<0 && gK<=-1e-6) gen_K_diff = poverall;
		  else if(poverall>0 && gK>=+1e-6) gen_K_diff = 1.75*poverall; 
		  else { gen_K_diff = 0.; one_more_push_and_fix = false; }
		} else { 
		  if(poverall<0 && gK<=-1e-6) gen_K_diff = poverall; 
		  else if(poverall>0 && gK>=+1e-6) gen_K_diff = 1.25*poverall;
		  else { gen_K_diff = 0.; one_more_push_and_fix = false; }
		}
		//if(pg0->x[data_sc.K_outidx[K_idx]] < -1e-6) assert(false);
		
		//double gen_deficit = pg0->x[data_sc.K_outidx[K_idx]];
		//if(pen_p_balance > 2e5)
		//  gen_K_diff = 3*poverall;
		//else if(pen_p_balance > 5e4)
		//  gen_K_diff = 2*poverall;
		//else 
		//  gen_K_diff = 1.5*poverall;
	      }
	    }
	  }
	}
      }
      
      if(one_more_push_and_fix) {
 	//apparently we need to further unblock generation
 	auto pgK = prob_mds_->variable("p_g", d); assert(pgK!=NULL);
 	//find AGC generators that are "blocking" and fix them; update particip and non-particip indexes
 	vector<int> pg0_partic_idxs_u=solv1_pg0_partic_idxs, pgK_partic_idxs_u=solv1_pgK_partic_idxs;
 	vector<int> pgK_nonpartic_idxs_u = solv1_pgK_nonpartic_idxs; 
	vector<int> pg0_nonpartic_idxs_u = solv1_pg0_nonpartic_idxs;

 	double delta_out=0., delta_needed=0., delta_blocking=0., delta_lb, delta_ub; 
	double residual_Pg;
 	bool bfeasib;

	if(fabs(gen_K_diff)>1e-6) {
	  //solv1_delta_optim and gen_K_diff must have same sign at this point
	  if(poverall*solv1_delta_optim<0 || poverall*solv1_delta_out<0) {
	    assert(false);
	    //last moment bail out
	    gen_K_diff=0.; //push_and_fix will do nothing 
	  } 

	  bfeasib = push_and_fix_AGCgen(d, gen_K_diff, solv1_delta_optim, 
					pg0_partic_idxs_u, pgK_partic_idxs_u,
					pg0_nonpartic_idxs_u, pgK_nonpartic_idxs_u,
					pg0, pgK, 
					data_sc.G_Plb, data_sc.G_Pub, data_sc.G_alpha,
					delta_out, delta_needed, delta_blocking,
					delta_lb, delta_ub,
					residual_Pg);
 	  //alter starting points 
	  assert(pg0_partic_idxs_u.size() == pgK_partic_idxs_u.size());
	  for(int it=0; it<pg0_partic_idxs_u.size(); it++) {
	    const int& i0 = pg0_partic_idxs_u[it];
	    pgK->x[pgK_partic_idxs_u[it]] = pg0->x[i0]+data_sc.G_alpha[i0]*delta_out;
	  }
#ifdef BE_VERBOSE
	  printf("ContProbKron_wfix K_idx=%d (gener)(after solv1) fixed %lu gens; "
		 "adtl deltas out=%g needed=%g blocking=%g "
		 "residualPg=%g feasib=%d\n",
		 K_idx, solv1_pg0_partic_idxs.size()-pg0_partic_idxs_u.size(),
		 delta_out, delta_needed, delta_blocking, residual_Pg, bfeasib);
	  //printvec(solv1_pgK_partic_idxs, "solv1_pgK_partic_idxs");
	  //printvec(pgK_partic_idxs_u, "pgK_partic_idxs_u");
#endif
	  
	  prob_mds_->delete_constraint_block(con_name("AGC_simple_fixedpg0", d));
	  prob_mds_->delete_duals_constraint(con_name("AGC_simple_fixedpg0", d));
	  
	  if(pg0_partic_idxs_u.size()>0) {
	    this->add_cons_AGC_simplified(d, pg0_partic_idxs_u, pgK_partic_idxs_u, pg0);
	    prob_mds_->append_duals_constraint(con_name("AGC_simple_fixedpg0", d));
	    prob_mds_->variable_duals_cons("duals_AGC_simple_fixedpg0", d)->set_start_to(0.0);
	    
	    prob_mds_->variable("delta", d)->set_start_to(delta_out);
	  }
	  
	  prob_mds_->primal_problem_changed();
	}
      } // else of if(one_more_push_and_fix)

      //
      {
	auto v = prob_mds_->variable("v_n", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }
      {
	auto v = prob_mds_->variable("q_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }

      {
	auto v = prob_mds_->variable("p_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }

      this->do_qgen_fixing_for_PVPQ(prob_mds_->variable("v_n", d), prob_mds_->variable("q_g", d));

#ifdef DEBUG
      if(bFirstSolveOK) {
	if(!prob_mds_->vars_duals_bounds_L->provides_start()) {
	  assert(false);
	}
	assert(prob_mds_->vars_duals_bounds_L->provides_start());
	assert(prob_mds_->vars_duals_bounds_U->provides_start()); 	
	assert(prob_mds_->vars_duals_cons->provides_start());
      }
      assert(prob_mds_->vars_primal->n() == vars_last->n());
#endif
      
      //
      // --- SOLVE 2 --- 
      //
      bool opt2_ok = do_solve2(bFirstSolveOK);
      f = prob_mds_->obj_value;
      if(!opt2_ok) {
	if(bFirstSolveOK) {
	  //sln = sln_solve1;
	  f = obj_solve1;
	  //recourse actions were already determined
	} else {
	  printf("[warning][panic] ContProbKron_wfix K_idx=%d return bestknown; "
		 "opt1 and opt2 failed on rank=%d\n", K_idx, my_rank);
	  prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
	  //get_solution_simplicial_vectorized(sln_solve1);
	  //sln = sln_solve1;
	  f = best_known_iter.obj_value;
	  //no recourse actions when both solve1 and solve2 fail
	}

      } else { //opt2_ok
	obj_solve2 = prob_mds_->obj_value;
	if(obj_solve1<obj_solve2) {
	  //sln = sln_solve1;
	  f = obj_solve1;
	  //recourse actions were already determined
	} else {
	  //sln = sln_solve2;
	  f = obj_solve2;
	  determine_recourse_action(data_for_master);
	}

      } //end of opt2_ok
      
      if(obj_solve2>pen_accept) { 
	double delta_optim = 0.;//
	if(prob_mds_->variable("delta", d))
	  delta_optim = prob_mds_->variable("delta", d)->x[0];
#ifdef BE_VERBOSE
	//print_objterms_evals();
	//print_line_limits_info(*data_K[0]);
	//print_active_power_balance_info(*data_K[0]);
	//print_reactive_power_balance_info(*data_K[0]);
	//print_p_g_with_coupling_info(*data_K[0], pg0);
	printf("ContProbKron_wfix K_idx=%d opt1 opt2 resulted in high pen delta=%g\n", K_idx, delta_optim);
#endif
      }  
    } else {
      //sln = sln_solve1;
      f = obj_solve1;
      if(prob_mds_->obj_value>acceptable_penalty && skip_2nd_solve)
	printf("ContProbKron_wfix K_idx=%d opt2 needed but not done insufic time rank=%d\n", K_idx, my_rank);
      if(prob_mds_->obj_value>acceptable_penalty)
	determine_recourse_action(data_for_master);
    }
    
    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProbKron_wfix K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif
    
    return true;
  }

  bool ContingencyProblemKronRedWithFixingCode1::do_solve1()
  {
    goTimer tmrec; tmrec.start();
    //! "ma27_ignore_singularity" 
    //set_solver_option("ma27_meminc_factor", 1.1);
#ifdef GOLLNLP_FAULT_HANDLING    
    g_solve_watch_ma57=true;
#else
    g_solve_watch_ma57=false;
#endif
    g_alarm_duration_ma57=alarm_ma57_normal;
    g_max_memory_ma57=max_mem_ma57_normal;
    g_my_rank_ma57=my_rank;
    g_my_K_idx_ma57=K_idx;
    
#ifdef GOLLNLP_FAULT_HANDLING    
    g_solve_watch_ma27=true;
#else
    g_solve_watch_ma27=false;
#endif
    g_alarm_duration_ma27=alarm_ma27_normal;
    g_max_memory_ma27=max_mem_ma27_normal;
    g_my_rank_ma27=my_rank;
    g_my_K_idx_ma27=K_idx;

    vector<int> hist_iter, hist_obj; 
    prob_mds_->obj_value = 1e+20;
    bool done = false; bool bret = true;
    OptimizationStatus last_opt_status = Solve_Succeeded; //be positive
    bool solve1_emer_mode=false;
    int n_solves=0; 
    while(!done) {

      printf("ContProbKron_wfix - do_solve1: K_idx=%d nsolves=%d\n", K_idx, n_solves);
      //prob_mds_->set_solver_option("derivative_test", "only-second-order");
      
      bool opt_ok=false; bool PDRestart=true;

      solve1_emer_mode=false;

      assert(prob_mds_);
      
      switch(n_solves) {
      case 0: 
	{
	  PDRestart=false;
	  prob_mds_->set_solver_option("mu_target", 5e-9);
	  prob_mds_->set_solver_option("mu_init", 1e-1);
	  prob_mds_->set_solver_option("tol", 5e-8);
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  prob_mds_->set_solver_option("linear_scaling_on_demand", "yes");
	  
	  const double gamma = 1e-3;
	  //prob_mds_->regularize_vn(gamma);
	  //prob_mds_->regularize_thetan(gamma);
	  prob_mds_->regularize_bs(gamma);
	  prob_mds_->regularize_pg(gamma);
	  prob_mds_->regularize_qg(gamma);
	}
	break;
      case 1: 
	{
	  PDRestart=false;
	  solve1_emer_mode=true; //keep emergency mode off
	  if(last_opt_status!=User_Requested_Stop && 
	     last_opt_status!=Unrecoverable_Exception && 
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	  }

	  prob_mds_->set_solver_option("ma57_small_pivot_flag", 1);

	  prob_mds_->set_solver_option("mu_target", 5e-8);
	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.25);
	  
	  const double gamma = 1e-3;
	  //prob_mds_->regularize_vn(gamma);
	  //prob_mds_->regularize_thetan(gamma);
	  prob_mds_->regularize_bs(gamma);
	  prob_mds_->regularize_pg(gamma);
	  prob_mds_->regularize_qg(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;

	}
	break;
      case 2: //MA27
	{
	  PDRestart=false;
	  solve1_emer_mode=true;
	  prob_mds_->reallocate_nlp_solver();
	  printf("[warning] ContProbKronWithFixing K_idx=%d opt1 will switch to ma27 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  prob_mds_->set_solver_option("linear_solver", "ma27"); 

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	  }

	  prob_mds_->set_solver_option("mu_init", 1e-2); 
	  prob_mds_->set_solver_option("mu_target", 5e-8);

	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  prob_mds_->set_solver_option("linear_scaling_on_demand", "yes");

	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-3;
	  prob_mds_->update_regularizations(gamma);
	}
	break;
      case 3: //MA27
	{
	  PDRestart=false;
	  solve1_emer_mode=true;
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	  }
	  prob_mds_->set_solver_option("mu_target", 5e-8);
	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma27=alarm_ma27_safem;
	  g_max_memory_ma27=max_mem_ma27_safem;
	}
	break;
      case 4: 
	{
	  PDRestart=false;
	  solve1_emer_mode=true;
	  prob_mds_->reallocate_nlp_solver();

	  prob_mds_->vars_primal->set_start_to(*vars_ini);
	  prob_mds_->set_solver_option("mu_init", 1.);
	  prob_mds_->set_solver_option("mu_target", 5e-8);

	  printf("[warning] ContProbKronWithFixing K_idx=%d opt1 will switch to ma57 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("ma57_automatic_scaling", "yes");
	  prob_mds_->set_solver_option("ma57_small_pivot_flag", 1);

	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  prob_mds_->set_solver_option("linear_scaling_on_demand", "yes");

	  prob_mds_->set_solver_option("tol", 1e-6);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;
	}
	break;
      default:
	{
	  PDRestart=false;
	  solve1_emer_mode=true;
	  prob_mds_->set_solver_option("mu_init", 1.);
	  prob_mds_->set_solver_option("mu_target", 1e-7);
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("ma57_automatic_scaling", "yes");
	  prob_mds_->set_solver_option("tol", 5e-6);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-2 + 0.1*n_solves;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;
	}
      }
      prob_mds_->set_solver_option("print_user_options", "no");
      //!prob_mds_->set_solver_option("print_level", 2);
      prob_mds_->set_solver_option("sb","yes");

      prob_mds_->set_solver_option("max_iter", 300);
      prob_mds_->set_solver_option("acceptable_tol", 1e-3);
      prob_mds_->set_solver_option("acceptable_constr_viol_tol", 1e-6);
      prob_mds_->set_solver_option("acceptable_iter", 5);

      prob_mds_->set_solver_option("fixed_variable_treatment", "relax_bounds");
      prob_mds_->set_solver_option("honor_original_bounds", "yes");
      double relax_factor = 1e-8;//std::min(1e-8, pow(10., 3*n_solves-16));
      prob_mds_->set_solver_option("bound_relax_factor", relax_factor);
      double bound_push = 1e-2;//std::min(1e-2, pow(10., 3*n_solves-12));
      prob_mds_->set_solver_option("bound_push", bound_push);
      prob_mds_->set_solver_option("slack_bound_push", bound_push); 
      double bound_frac = 1e-2;//std::min(1e-2, pow(10., 3*n_solves-10));
      prob_mds_->set_solver_option("bound_frac", bound_frac);
      prob_mds_->set_solver_option("slack_bound_frac", bound_frac);

      //default yes ->ChiangZavala primal regularization
      prob_mds_->set_solver_option("neg_curv_test_reg", "no"); 

      monitor.timer.restart();
      monitor.hist_tm.clear();
      monitor.user_stopped = false;
      monitor.timeout = std::max(1., timeout - tmTotal.measureElapsedTime());
      
      if(safe_mode) {
	monitor.emergency=true;
	monitor.pen_accept = pen_accept_safemode;
	monitor.pen_accept_emer = pen_accept_safemode;
      } else {
	monitor.pen_accept = pen_accept_solve1;
	monitor.pen_accept_emer = pen_accept_emer;
	monitor.emergency = solve1_emer_mode;
      }

      bool ok_to_exit = false;
      if(best_known_iter.obj_value <= monitor.pen_accept) {
	ok_to_exit = true;
	prob_mds_->obj_value = best_known_iter.obj_value;
	prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
	printf("[warning] ContProbKronWithFixing K_idx=%d opt1 exit "
	       "best_known < pen_accept(%g) rank=%d  %g sec\n", 
	       K_idx,  monitor.pen_accept, my_rank, tmrec.measureElapsedTime()); 
      }
      if(monitor.emergency && best_known_iter.obj_value <= monitor.pen_accept_emer) {
	ok_to_exit = true;
	prob_mds_->obj_value = best_known_iter.obj_value;
	prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
	printf("[warning] ContProbKronWithFixing K_idx=%d opt1 exit best_known < pen_accept_emer(%g) "
	       "rank=%d  %g sec\n", 
	       K_idx,  monitor.pen_accept_emer, my_rank, tmrec.measureElapsedTime()); 
      }
	
      if(ok_to_exit) {
	done = true;
	bret = true;
      } else {

	prob_mds_->obj_value = 1e+20;
	if(PDRestart) {
	  //opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
	  opt_ok = prob_mds_->reoptimize(OptProblem::primalDualRestart);
	} else {
	  //opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);
	  opt_ok = prob_mds_->reoptimize(OptProblem::primalRestart);
	}

	n_solves++;
	last_opt_status = prob_mds_->OptProblem::optimization_status();

	hist_iter.push_back(prob_mds_->number_of_iterations());
	hist_obj.push_back(prob_mds_->obj_value);
	
	if(opt_ok) {
	  done = true;
	} else {
	  if(monitor.user_stopped) {
	    assert(last_opt_status == User_Requested_Stop);
	    done = true;
	  } else {
	    //something bad happened, will resolve
	    printf("[warning] ContProbKronRedWithFixing K_idx=%d opt1 failed at try %d rank=%d  %g sec\n", 
		   K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
	  }
	}
      
	if(n_solves>9) done = true;
	if(tmTotal.measureElapsedTime() > timeout) {
	  printf("[warning] ContProbKronRedWithFixing K_idx=%d opt1 timeout  %g sec; rank=%d; tries %d\n", 
		 K_idx, tmTotal.measureElapsedTime(), my_rank, n_solves);
	  done = true;
	  bret = false;
	}
      } //end of else best_know_iter<monitor.pen_accept
    } //end of outer while
    
    if(prob_mds_->obj_value > best_known_iter.obj_value) {
      prob_mds_->obj_value = best_known_iter.obj_value;
      prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
      printf("ContProbWithKronRedFixing K_idx=%d opt1 return best_known obj=%g on rank=%d\n", 
	     K_idx, prob_mds_->obj_value, my_rank);
    }

    prob_mds_->get_solution_simplicial_vectorized(sln_solve1);
    obj_solve1 = prob_mds_->obj_value;
    
#ifdef BE_VERBOSE
    string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbKronWithFixing K_idx=%d opt1 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
    fflush(stdout);
#endif
   
    return bret;
  }
  //
  // solve2
  //
  bool ContingencyProblemKronRedWithFixingCode1::do_solve2(bool bFirstSolveOK)
  {
    goTimer tmrec; tmrec.start();

    if(bFirstSolveOK)
      vars_ini->set_start_to(*prob_mds_->vars_primal);
#ifdef GOLLNLP_FAULT_HANDLING
    g_solve_watch_ma57=true;
#else
    g_solve_watch_ma57=false;
#endif
    g_alarm_duration_ma57=alarm_ma57_normal;
    g_max_memory_ma57=max_mem_ma57_normal;
    g_my_rank_ma57=my_rank;
    g_my_K_idx_ma57=K_idx;
#ifdef GOLLNLP_FAULT_HANDLING
    g_solve_watch_ma27=true;
#else
    g_solve_watch_ma27=false;
#endif
    g_alarm_duration_ma27=alarm_ma27_normal;
    g_max_memory_ma27=max_mem_ma27_normal;//Mbytes
    g_my_rank_ma27=my_rank;
    g_my_K_idx_ma27=K_idx;

    vector<int> hist_iter, hist_obj;
    prob_mds_->obj_value = 1e+20;
    bool bret = true, done = false; 
    OptimizationStatus last_opt_status = Solve_Succeeded; //be positive
    bool solve2_emer_mode=false;
    int n_solves=0; 
    while(!done) {
      bool opt_ok=false; bool PDRestart=true;
      solve2_emer_mode=false;
      printf("ContProbKron_wfix - do_solve2: K_idx=%d nsolves=%d\n", K_idx, n_solves);
      switch(n_solves) {
      case 0: 
	{ 
	  if(bFirstSolveOK) {
	    PDRestart=false;
	    prob_mds_->set_solver_option("mu_target", 5e-9);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	  } else {
	    PDRestart=false;
	    prob_mds_->set_solver_option("mu_init", 1e-1);
	  }
	  prob_mds_->set_solver_option("tol", 5e-8);
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  prob_mds_->set_solver_option("linear_scaling_on_demand", "yes");

	  const double gamma = 1e-3;
	  //prob_mds_->regularize_vn(gamma);
	  //prob_mds_->regularize_thetan(gamma);
	  prob_mds_->regularize_bs(gamma);
	  prob_mds_->regularize_pg(gamma);
	  prob_mds_->regularize_qg(gamma);
	}
	break;
      case 1: 
	{
	  solve2_emer_mode=true; //keep it off at the second solve
	  prob_mds_->set_solver_option("mu_target", 5e-8);
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	    PDRestart=false;
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	    PDRestart=false;
	  }

	  prob_mds_->set_solver_option("ma57_small_pivot_flag", 1);

	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.25);

	  const double gamma = 1e-3;
	  //prob_mds_->regularize_vn(gamma);
	  //prob_mds_->regularize_thetan(gamma);
	  prob_mds_->regularize_bs(gamma);
	  prob_mds_->regularize_pg(gamma);
	  prob_mds_->regularize_qg(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;
	}
	break;
      case 2: //MA27
	{
	  PDRestart=false;
	  solve2_emer_mode=true;
	  prob_mds_->reallocate_nlp_solver();
	  printf("[warning] ContProbKronWithFixing K_idx=%d opt2 will switch to ma27 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  prob_mds_->set_solver_option("linear_solver", "ma27"); 

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	  }

	  prob_mds_->set_solver_option("mu_init", 1e-2); 
	  prob_mds_->set_solver_option("mu_target", 5e-8);

	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");

	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-3;
	  prob_mds_->update_regularizations(gamma);
	}
	break;
      case 3: //MA27
	{	  
	  solve2_emer_mode=true;
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	    PDRestart=false;
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	    PDRestart=false;
	  }
	  prob_mds_->set_solver_option("mu_target", 5e-8);
	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma27=alarm_ma27_safem;
	  g_max_memory_ma27=max_mem_ma27_safem;
	}
	break;
      case 4: 
	{
	  PDRestart=false;
	  solve2_emer_mode=true;
	  prob_mds_->reallocate_nlp_solver();

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	  }

	  prob_mds_->set_solver_option("mu_target", 5e-8);

	  printf("[warning] ContProbKronWithFixing K_idx=%d opt2 will switch to ma57 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("ma57_automatic_scaling", "yes");
	  prob_mds_->set_solver_option("ma57_small_pivot_flag", 1);

	  prob_mds_->set_solver_option("linear_system_scaling", "mc19");
	  prob_mds_->set_solver_option("linear_scaling_on_demand", "yes");

	  prob_mds_->set_solver_option("tol", 5e-7);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);
	  
	  const double gamma = 1e-2;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;
	}
	break;
      default:
	{
	  PDRestart=false;
	  solve2_emer_mode=true;

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    prob_mds_->vars_primal->set_start_to(*vars_ini);
	    prob_mds_->set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    prob_mds_->vars_primal->set_start_to(*vars_last);
	    prob_mds_->set_solver_option("mu_init", 1e-2);
	  }
	  
	  prob_mds_->set_solver_option("mu_target", 1e-7);
	  prob_mds_->set_solver_option("linear_solver", "ma57"); 
	  prob_mds_->set_solver_option("ma57_automatic_scaling", "yes");
	  prob_mds_->set_solver_option("tol", 5e-6);
	  prob_mds_->set_solver_option("mu_linear_decrease_factor", 0.4);
	  prob_mds_->set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-2 + 0.1*n_solves;
	  prob_mds_->update_regularizations(gamma);

	  g_alarm_duration_ma57=alarm_ma57_safem;
	  g_max_memory_ma57=max_mem_ma57_safem;
	}
      }
      prob_mds_->set_solver_option("print_user_options", "no");
      //!prob_mds_->set_solver_option("print_level", 2);
      prob_mds_->set_solver_option("sb","yes");

      prob_mds_->set_solver_option("max_iter", 500);
      prob_mds_->set_solver_option("acceptable_tol", 1e-3);
      prob_mds_->set_solver_option("acceptable_constr_viol_tol", 1e-6);
      prob_mds_->set_solver_option("acceptable_iter", 2);

      prob_mds_->set_solver_option("fixed_variable_treatment", "relax_bounds");
      prob_mds_->set_solver_option("honor_original_bounds", "yes");
      double relax_factor = 1e-8;//std::min(1e-8, pow(10., 3*n_solves-16));
      prob_mds_->set_solver_option("bound_relax_factor", relax_factor);
      double bound_push = 1e-2;//std::min(1e-2, pow(10., 3*n_solves-12));
      prob_mds_->set_solver_option("bound_push", bound_push);
      prob_mds_->set_solver_option("slack_bound_push", bound_push); 
      double bound_frac = 1e-2;//std::min(1e-2, pow(10., 3*n_solves-10));
      prob_mds_->set_solver_option("bound_frac", bound_frac);
      prob_mds_->set_solver_option("slack_bound_frac", bound_frac);

      //default yes ->ChiangZavala primal regularization
      prob_mds_->set_solver_option("neg_curv_test_reg", "no"); 

      monitor.timer.restart();
      monitor.hist_tm.clear();
      monitor.user_stopped = false;
      monitor.timeout = std::max(1., timeout - tmTotal.measureElapsedTime());
      
      if(safe_mode) {
	monitor.emergency = true;
	monitor.pen_accept = pen_accept_safemode;
	monitor.pen_accept_emer = pen_accept_safemode;
      } else {
	monitor.emergency = solve2_emer_mode;
	monitor.pen_accept = pen_accept;
	monitor.pen_accept_emer = pen_accept_emer;
      }

      bool ok_to_exit = false;
      if(best_known_iter.obj_value <= monitor.pen_accept) {
	ok_to_exit = true;
	prob_mds_->obj_value = best_known_iter.obj_value;
	prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
	printf("[warning] ContProbKronWithFixing K_idx=%d opt2 exit best_known < pen_accept(%g) "
	       "rank=%d  %g sec\n", 
	       K_idx,  monitor.pen_accept, my_rank, tmrec.measureElapsedTime()); 
      }

      if(monitor.emergency && best_known_iter.obj_value <= monitor.pen_accept_emer) {
	ok_to_exit = true;
	prob_mds_->obj_value = best_known_iter.obj_value;
	prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
	printf("[warning] ContProbKronWithFixing K_idx=%d opt2 exit best_known < pen_accept_emer(%g) "
	       "rank=%d  %g sec\n", 
	       K_idx,  monitor.pen_accept_emer, my_rank, tmrec.measureElapsedTime()); 
      }

      if(ok_to_exit) {
	done = true;
	bret = true;
      } else {

	prob_mds_->obj_value = 1e+20;	
	if(PDRestart) {
	  opt_ok = prob_mds_->reoptimize(OptProblem::primalDualRestart);
	} else {
	  opt_ok = prob_mds_->reoptimize(OptProblem::primalRestart);
	}
	
	n_solves++;
	last_opt_status = prob_mds_->OptProblem::optimization_status();
	
	hist_iter.push_back(prob_mds_->number_of_iterations());
	hist_obj.push_back(prob_mds_->obj_value);
      
	if(opt_ok) {
	  done = true; 
	} else {
	  if(monitor.user_stopped) {
	    assert(last_opt_status == User_Requested_Stop);
	    done = true; 
	  } else {
	    //something bad happened, will resolve
	    printf("[warning] ContProbKronWithFixing K_idx=%d opt2 failed at try %d rank=%d time %g\n", 
		   K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
	  }
	}
	
	if(n_solves>9) done = true;
	if(tmTotal.measureElapsedTime() > timeout) {
	  printf("[warning] ContProbKronWithFixing K_idx=%d opt2 timeout  rank=%d; tries %d took %g sec\n", 
		 K_idx, my_rank, n_solves, tmTotal.measureElapsedTime());
	  done = true;
	  bret = false;
	}
      } //end of else 
    } //end of outer while

    if(prob_mds_->obj_value > best_known_iter.obj_value) {
      
      prob_mds_->obj_value = best_known_iter.obj_value;
      prob_mds_->vars_primal->set_start_to(*best_known_iter.vars_primal);
      printf("ContProbKronWithFixing K_idx=%d opt2 return best_known obj=%g on rank=%d\n", 
	     K_idx, prob_mds_->obj_value, my_rank);
    }
    prob_mds_->get_solution_simplicial_vectorized(sln_solve2);
    obj_solve2 = prob_mds_->obj_value;
    
    //print_active_power_balance_info(*data_K[0]);
    //print_reactive_power_balance_info(*data_K[0]);
#ifdef BE_VERBOSE
    string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbKronWithFixing K_idx=%d opt2 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
    fflush(stdout);
#endif
    return bret;
  }

  
  
  void ContingencyProblemKronRedWithFixingCode1::
  get_objective_penalties(double& pen_p_balance, double& pen_q_balance, 
			  double& pen_line_limits, double& pen_trans_limits)
  {
    pen_p_balance = pen_q_balance = pen_line_limits = pen_trans_limits = 0;
    double pen; SCACOPFData& d = *data_K[0]; bool new_x = false;
    {
      auto ot = prob_mds_->obj->objterm(objterm_name("quadr_pen_pslack_n_p_balance", d));
      if(NULL==ot || !ot->eval_f(*prob_mds_->vars_primal, new_x, pen_p_balance))
	pen_p_balance = 0.;
    }
    {
      auto ot = prob_mds_->obj->objterm(objterm_name("quadr_pen_qslack_n_q_balance", d));
      if(NULL==ot || !ot->eval_f(*prob_mds_->vars_primal, new_x, pen_q_balance))
	pen_q_balance = 0.;
    }
    {
      pen_line_limits = 0.;
      auto ot1 = prob_mds_->obj->objterm(objterm_name("quadr_pen_sslack_li_line_limits1", d));
      if(ot1) ot1->eval_f(*prob_mds_->vars_primal, new_x, pen_line_limits);
      else assert(false);

      auto ot2 = prob_mds_->obj->objterm(objterm_name("quadr_pen_sslack_li_line_limits2", d));
      if(ot2) ot2->eval_f(*prob_mds_->vars_primal, new_x, pen_line_limits);
      else assert(false);
      
      pen_line_limits /= 2.;
    }
    {
      pen_trans_limits = 0.;
      auto ot1 = prob_mds_->obj->objterm(objterm_name("quadr_pen_sslack_ti_trans_limits1", d));
      if(ot1) ot1->eval_f(*prob_mds_->vars_primal, new_x, pen_trans_limits);
      
      auto ot2 = prob_mds_->obj->objterm(objterm_name("quadr_pen_sslack_ti_trans_limits2", d));
      if(ot2) ot2->eval_f(*prob_mds_->vars_primal, new_x, pen_trans_limits);
      pen_trans_limits /= 2.;
    }
  }
  void ContingencyProblemKronRedWithFixingCode1::
  estimate_active_power_deficit(double& p_plus, double& p_minus, double& p_overall)
  {
    p_plus = p_minus = p_overall = 0.;
    auto pf_p_bal = dynamic_cast<PFActiveBalance*>(prob_mds_->constraint("p_balance",*data_K[0]));
    assert(pf_p_bal);
    OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
    int n = data_sc.N_Bus.size(); assert(pslacks_n->n == 2*n);
    for(int i=n; i<2*n; i++) { p_plus  += pslacks_n->x[i]; p_overall += pslacks_n->x[i]; }
    for(int i=0; i<n; i++)   { p_minus -= pslacks_n->x[i]; p_overall -= pslacks_n->x[i]; }
  }
  
  void ContingencyProblemKronRedWithFixingCode1::
  estimate_reactive_power_deficit(double& q_plus, double& q_minus, double& q_overall)
  {
    q_plus = q_minus = q_overall = 0.;

    auto pf_q_bal = dynamic_cast<PFReactiveBalance*>(prob_mds_->constraint("q_balance",*data_K[0]));
    OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
    int n = data_sc.N_Bus.size(); assert(qslacks_n->n == 2*n);
    for(int i=n; i<2*n; i++) { q_plus  += qslacks_n->x[i]; q_overall += qslacks_n->x[i]; }
    for(int i=0; i<n; i++)   { q_minus -= qslacks_n->x[i]; q_overall -= qslacks_n->x[i]; }
  }
  
  bool ContingencyProblemKronRedWithFixingCode1::determine_recourse_action(double* info_out)
  {
    SCACOPFData& data = data_sc;
    //
    // prepare info for master rank
    //
    info_out[0]=prob_mds_->obj_value;
    double pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits;
    get_objective_penalties(pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits);

    double p_imbalance = 0;
    {
      double bal_p_plus, bal_p_minus, bal_p_overall;
      estimate_active_power_deficit(bal_p_plus,  bal_p_minus,  bal_p_overall);
      p_imbalance = fabs(bal_p_plus) > fabs(bal_p_minus) ? bal_p_plus : bal_p_minus;
    }

    double q_imbalance = 0;
    {
      double bal_q_plus, bal_q_minus, bal_q_overall;
      estimate_reactive_power_deficit(bal_q_plus,  bal_q_minus,  bal_q_overall);
      q_imbalance = fabs(bal_q_plus) > fabs(bal_q_minus) ? bal_q_plus : bal_q_minus;
    }

    
    if(data.K_ConType[K_idx] == SCACOPFData::kGenerator) {

      assert(p_g0->n == data.G_Generator.size());
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      assert(data.K_outidx.size() == data.K_Contingency.size());
      
      int idx_gen = data.K_outidx[K_idx];
      assert(idx_gen>=0 && idx_gen<p_g0->n);

      info_out[1]=info_out[2]=info_out[3]=info_out[4]=0.;

      string msg = "penalizing ";
#ifdef BE_VERBOSE
      printf("ContingencyProblemKron_wfix K_idx=%d recourse_generator:  penalties p=%.4e q=%.4e ll=%.4e tl=%.4e imbalance p=%.4e q=%.4e  rank=%d\n",
	     K_idx, pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits,
	     -p_imbalance, -q_imbalance, my_rank);
#endif
      if(pen_p_balance>5e4) {
	double delta = fabs(p_imbalance);
	delta = ((int)(delta*10000))/10000.;
	info_out[1]=1e8*delta + fabs(p_g0->x[idx_gen]);

	//printf("!!!!!!!!!!! delta = %16.8e info1=%.16f  info2=%.16f\n", delta, info_out[1], info_out[2]);
	msg += "pg ";
      }
      if(pen_q_balance>1e5 && pen_q_balance>pen_p_balance) {
	double delta = fabs(q_imbalance);
	delta = ((int)(delta*10000))/10000.;
	info_out[2]=1e8*delta + fabs(q_g0->x[idx_gen]);
	//info_out[4]=1e8*delta + fabs(q_li20->x[idx]);
	msg += "qg ";
      }
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_line: %s  rank=%d\n", K_idx, msg.c_str(), my_rank);
#endif
 
    } else if(data.K_ConType[K_idx] == SCACOPFData::kLine) {

      assert(p_li10); assert(q_li10); assert(p_li20); assert(q_li20);
      assert(data.L_Line.size() == q_li10->n);
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      
      int idx = data.K_outidx[K_idx];
       
      assert(idx>=0 && idx<q_li10->n);

      info_out[1]=info_out[2]=info_out[3]=info_out[4]=0.;

      string msg = "penalizing ";
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_line:  "
	     "penalties p=%.4e q=%.4e ll=%.4e tl=%.4e imbalance p=%.4e q=%.4e  rank=%d\n",
	     K_idx, pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits,
	     -p_imbalance, -q_imbalance, my_rank);
#endif
      if(pen_p_balance>5e4) {
	double delta = fabs(p_imbalance);
	delta = ((int)(delta*10000))/10000.;

	info_out[1]=1e8*delta + fabs(p_li10->x[idx]);
	info_out[3]=1e8*delta + fabs(p_li20->x[idx]);

	msg += "pli ";
      }
      if(pen_q_balance>5e4) {
	double delta = fabs(q_imbalance);
	delta = ((int)(delta*10000))/10000.;
	info_out[2]=1e8*delta + fabs(q_li10->x[idx]);
	info_out[4]=1e8*delta + fabs(q_li20->x[idx]);
	msg += "qli ";
      }
      if(pen_q_balance>5*pen_p_balance) {
	if(!recourse_action_from_voltages(idx, true, info_out)) { 
	  
	} else {
	  msg = "penalizing voltages";
	}   
      }  
      printf("ContingencyProblem_wfix K_idx=%d recourse_line: %s  rank=%d\n", K_idx, msg.c_str(), my_rank);

    } else if(data.K_ConType[K_idx] == SCACOPFData::kTransformer) {

      assert(data.T_Transformer.size() == q_ti10->n);
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      int idx = data.K_outidx[K_idx];
      assert(idx>=0 && idx<q_ti10->n);

      info_out[1]=info_out[2]=info_out[3]=info_out[4]=0.;

      string msg = "penalizing ";
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_transf:  penalties p=%.4e q=%.4e ll=%.4e tl=%.4e imbalance p=%.4e q=%.4e  rank=%d\n",
	     K_idx, pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits,
	     -p_imbalance, -q_imbalance, my_rank);
#endif
      if(pen_p_balance>5e4) {
	double delta = fabs(p_imbalance);
	delta = ((int)(delta*10000))/10000.;
	info_out[1]=1e8*delta + fabs(p_ti10->x[idx]);
	info_out[3]=1e8*delta + fabs(p_ti20->x[idx]);
	msg += "pli ";
      }
      if(pen_q_balance>5e4) {
	double delta = fabs(q_imbalance);
	delta = ((int)(delta*10000))/10000.;
	info_out[2]=1e8*delta + fabs(q_ti10->x[idx]);
	info_out[4]=1e8*delta + fabs(q_ti20->x[idx]);
	msg += "qli ";
      }

      if(pen_q_balance>5*pen_p_balance) {
	if(!recourse_action_from_voltages(idx, false, info_out)) { 
	} else {
	  msg = "penalizing voltages";
	}
      }
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_transf: %s  rank=%d\n", K_idx, msg.c_str(), my_rank);
#endif
    }
    return true;
  }
  bool ContingencyProblemKronRedWithFixingCode1::
  recourse_action_from_voltages(int outidx, bool isLine, double* info_out)
  {
    int NidxFrom = isLine ? data_sc.L_Nidx[0][outidx] : data_sc.T_Nidx[0][outidx];
    int NidxTo   = isLine ? data_sc.L_Nidx[1][outidx] : data_sc.T_Nidx[1][outidx];

    vector<int> dualsidx_vnk_lb(data_sc.N_Bus.size()), dualsidx_vnk_ub(data_sc.N_Bus.size());
    iota(dualsidx_vnk_lb.begin(), dualsidx_vnk_lb.end(), 0);
    iota(dualsidx_vnk_ub.begin(), dualsidx_vnk_ub.end(), 0);

    auto vnk_duals_ub = prob_mds_->variable_duals_upper("duals_bndU_v_n", *data_K[0]); 
    if(!vnk_duals_ub) {assert(false); return false; }
    auto vnk_duals_lb = prob_mds_->variable_duals_lower("duals_bndL_v_n", *data_K[0]); 
    if(!vnk_duals_lb) {assert(false); return false; }
    auto v_nk = variable("v_n", *data_K[0]);

    sort(dualsidx_vnk_lb.begin(), dualsidx_vnk_lb.end(), 
	 [&](const int& a, const int& b) { return fabs(vnk_duals_lb->x[a]) > fabs(vnk_duals_lb->x[b]); } );
    sort(dualsidx_vnk_ub.begin(), dualsidx_vnk_ub.end(), 
	 [&](const int& a, const int& b) { return fabs(vnk_duals_ub->x[a]) > fabs(vnk_duals_ub->x[b]); } );
    
#ifdef BE_VERBOSE2
    printf("ContingencyProblem_wfix largest duals for K_idx=%d recourse_%s outidx=%d busidxs from=%d to %d\n",
	   K_idx, isLine ? "line" : "transf", outidx, NidxFrom, NidxTo);

    printf("[[largest lower]]\n");
    for(int i=0; i<10; i++) {
      printf("\t{vn0,lb,ub[%5d]=%9.6e %9.6e %9.6e}  "
	     "{vnk,lb,duallb[%5d]=%9.6e %9.6e %9.6e} "
	     "{vnk,ub,dualub[%5d]=%9.6e %9.6e %9.6e}\n",
	     dualsidx_vnk_lb[i], v_n0->x[dualsidx_vnk_lb[i]],
	     v_n0->lb[dualsidx_vnk_lb[i]], v_n0->ub[dualsidx_vnk_lb[i]], 
	     dualsidx_vnk_lb[i], v_nk->x[dualsidx_vnk_lb[i]],
	     v_nk->lb[dualsidx_vnk_lb[i]], vnk_duals_lb->x[dualsidx_vnk_lb[i]],
	     dualsidx_vnk_lb[i], v_nk->x[dualsidx_vnk_lb[i]],
	     v_nk->ub[dualsidx_vnk_lb[i]], vnk_duals_ub->x[dualsidx_vnk_lb[i]]);
	     
    }
    printf("[[largest upper]]\n");
    for(int i=0; i<10; i++) {
      printf("\t{vn0,lb,ub[%5d]=%9.6e %9.6e %9.6e}  "
	     "{vnk,lb,duallb[%5d]=%9.6e %9.6e %9.6e} "
	     "{vnk,ub,dualub[%5d]=%9.6e %9.6e %9.6e}\n",
	     dualsidx_vnk_ub[i], v_n0->x[dualsidx_vnk_ub[i]],
	     v_n0->lb[dualsidx_vnk_ub[i]], v_n0->ub[dualsidx_vnk_ub[i]], 
	     dualsidx_vnk_ub[i], v_nk->x[dualsidx_vnk_ub[i]],
	     v_nk->lb[dualsidx_vnk_ub[i]], vnk_duals_lb->x[dualsidx_vnk_ub[i]],
	     dualsidx_vnk_ub[i], v_nk->x[dualsidx_vnk_ub[i]],
	     v_nk->ub[dualsidx_vnk_ub[i]], vnk_duals_ub->x[dualsidx_vnk_ub[i]]);
    }
#endif

    int Nidx_from_upper = -1, Nidx_from_lower = -1; 
    {
      int idx_in_Nfrom = indexin(dualsidx_vnk_ub, NidxFrom); assert(idx_in_Nfrom>=0);
      int idx_in_Nto   = indexin(dualsidx_vnk_ub, NidxTo);   assert(idx_in_Nto  >=0);
      
      if(idx_in_Nfrom < 10 &&
	 fabs(v_n0->x[dualsidx_vnk_ub[idx_in_Nfrom]] - v_n0->ub[dualsidx_vnk_ub[idx_in_Nfrom]])<1e-2) {
	Nidx_from_upper = dualsidx_vnk_ub[idx_in_Nfrom];
      }
      if(idx_in_Nto   < 10 &&
	 fabs(v_n0->x[dualsidx_vnk_ub[idx_in_Nto  ]] - v_n0->ub[dualsidx_vnk_ub[idx_in_Nto  ]])<1e-2) {
	if(Nidx_from_upper>=0) {
	  if(fabs(vnk_duals_ub->x[idx_in_Nto]) > fabs(vnk_duals_ub->x[Nidx_from_upper]))
	    Nidx_from_upper = dualsidx_vnk_ub[idx_in_Nto];
	} else {
	  Nidx_from_upper = dualsidx_vnk_ub[idx_in_Nto];
	}
      }      
    }
    {
      int idx_in_Nfrom = indexin(dualsidx_vnk_lb, NidxFrom); assert(idx_in_Nfrom>=0);
      int idx_in_Nto   = indexin(dualsidx_vnk_lb, NidxTo);   assert(idx_in_Nto  >=0);
      
      if(idx_in_Nfrom < 10 &&
	 fabs(v_n0->x[dualsidx_vnk_lb[idx_in_Nfrom]] - v_n0->lb[dualsidx_vnk_lb[idx_in_Nfrom]])<1e-2) {
	Nidx_from_lower = dualsidx_vnk_lb[idx_in_Nfrom];
      }
      if(idx_in_Nto   < 10 &&
	 fabs(v_n0->x[dualsidx_vnk_lb[idx_in_Nto  ]] - v_n0->lb[dualsidx_vnk_lb[idx_in_Nto  ]])<1e-2) {
	if(Nidx_from_lower>=0) {
	  if(fabs(vnk_duals_lb->x[idx_in_Nto]) > fabs(vnk_duals_lb->x[Nidx_from_upper]))
	    Nidx_from_lower = dualsidx_vnk_lb[idx_in_Nto];
	} else {
	  Nidx_from_lower = dualsidx_vnk_lb[idx_in_Nto];
	}
      }      
    }

    if(Nidx_from_upper>=0 && Nidx_from_lower<0) {

      const int Nidx = Nidx_from_upper;
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_%s voltage pen at busidx=%d (voltage at upper)\n",
	     K_idx, isLine ? "line" : "transf", Nidx);
#endif
      info_out[0] = prob_mds_->obj_value;
      info_out[0+1] = 1000+v_n0->x[Nidx];
      info_out[1+1] = 1e+20; //upper is 1e+20, lower is -1e+20; fabs>=1e+20 indicates a voltage penalty
      info_out[2+1] = vnk_duals_ub->x[Nidx];
      info_out[3+1] = (double)(1+Nidx); //it will be -1-Nidx for lower

      return true;      
    }
    if(Nidx_from_lower>=0 && Nidx_from_upper<0) {
      printf("[warning] code1 in an UNTESTED case (voltage at lower) K_idx=%d\n", K_idx);
      const int Nidx = Nidx_from_lower;
#ifdef BE_VERBOSE
      printf("ContingencyProblem_wfix K_idx=%d recourse_%s voltage pen at busidx=%d (voltage at lower)\n",
	     K_idx, isLine ? "line" : "transf", Nidx);
#endif
      info_out[0] = prob_mds_->obj_value;
      info_out[0+1] = 1000+v_n0->x[Nidx];
      info_out[1+1] = -1e+20; //upper is 1e+20, lower is -1e+20; fabs>=1e+20 indicates a voltage penalty
      info_out[2+1] = vnk_duals_lb->x[Nidx];
      info_out[3+1] = (double)(-1-Nidx); // -1-Nidx for lower
      return true;
    }

    if(Nidx_from_lower>=0 && Nidx_from_upper>=0) {
      if(Nidx_from_lower == Nidx_from_upper) {
	printf("[warning] code1 in an UNTESTED case (voltage at lower and upper at the same bus %d) K_idx=%d will do nothing\n", Nidx_from_lower, K_idx);
	return false;
      } else {
	printf("[warning] code1 in an UNTESTED case (voltage at lower and upper) K_idx=%d will do use upper\n", K_idx);
	const int Nidx = Nidx_from_upper;
	info_out[0] = prob_mds_->obj_value;
	info_out[0+1] = 1000+v_n0->x[Nidx];
	info_out[1+1] = 1e+20; //upper is 1e+20, lower is -1e+20; fabs>=1e+20 indicates a voltage penalty
	info_out[2+1] = vnk_duals_ub->x[Nidx];
	info_out[3+1] = (double)(1+Nidx); //it will be -1-Nidx for lower
	return true;
      }
    }

    if(Nidx_from_upper>=0 || Nidx_from_lower>=0) return false;

    return false;
  }


  void ContingencyProblemKronRedWithFixingCode1::
  add_cons_pg_nonanticip_using(OptVariablesBlock* pg0,
			       const std::vector<int>& idxs_pg0_nonparticip, 
			       const std::vector<int>& idxs_pgK_nonparticip)
  {
    assert(pg0 == p_g0);
    assert(pg0 == prob_mds_->p_g0);
    SCACOPFData& dK = *data_K[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = prob_mds_->variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] ContingencyProblemWithFixing K_idx=%d p_g var not found in contingency  "
	     "problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    int sz = pgK_nonpartic_idxs.size();  assert(sz == pg0_nonpartic_idxs.size());
    const int *pgK_idxs = idxs_pgK_nonparticip.data(), *pg0_idxs = idxs_pg0_nonparticip.data();
    int idxK; //double pg0_val, lb, ub; 

#ifdef DEBUG
    assert(pg0->xref == pg0->x);
#endif

    for(int i=0; i<sz; i++) {
      assert(pg0_idxs[i]<pg0->n && pg0_idxs[i]>=0); assert(pgK_idxs[i]<pgK->n && pgK_idxs[i]>=0);
      idxK = pgK_idxs[i];
      pgK->lb[idxK] = pgK->ub[idxK] = pg0->xref[pg0_idxs[i]];
    }
  }

  bool ContingencyProblemKronRedWithFixingCode1::
  add_cons_AGC_simplified(SCACOPFData& dB, 
			  const std::vector<int>& idxs0_AGC_particip, 
			  const std::vector<int>& idxsK_AGC_particip,
			  OptVariablesBlock* pg0)
  {
    assert(pg0 == p_g0);
    assert(pg0 == prob_mds_->p_g0);
    assert(idxs0_AGC_particip.size()==idxsK_AGC_particip.size());
    assert(prob_mds_->variable("p_g", dB));

    if(idxs0_AGC_particip.size()==0) {
      printf("[warning] ContingencyProblemWithFixing add_cons_AGC_simplified: "
	     "NO gens participating !?! in contingency %d\n", dB.id);
      return true;
    }

    OptVariablesBlock* deltaK = prob_mds_->variable("delta", dB);
    if(deltaK==NULL) {
      deltaK = new OptVariablesBlock(1, var_name("delta", dB));
      prob_mds_->append_varsblock(deltaK);
      deltaK->set_start_to(0.);
    }

    auto cons = new AGCSimpleCons_pg0Fixed(con_name("AGC_simple_fixedpg0", dB),
					   idxs0_AGC_particip.size(), 
					   pg0,
					   prob_mds_->variable("p_g", dB),
					   deltaK, 
					   idxs0_AGC_particip, idxsK_AGC_particip, 
					   data_sc.G_alpha);
    prob_mds_->append_constraints(cons);

    return true;
  }

  bool ContingencyProblemKronRedWithFixingCode1::
  warm_start_variable_from_basecase_dict(OptVariables& v)
  {
    SCACOPFData& dK = *data_K[0];

    OptVariablesBlock *v_n_0 = NULL, *theta_n_0 = NULL; //source, fs, from dict_basecase_vars
    OptVariablesBlock *v_n_k = NULL, *theta_n_k = NULL; //destination, reduced-space, from 'v'
    OptVariablesBlock *v_aux_n_k = NULL, *theta_aux_n_k = NULL; //destination, reduced-space, from 'v'
    
    for(auto& b : v.vblocks) {

      if(b->id.find("balance_kron")!=string::npos) {
	b->set_start_to(0.);
	continue;
      }
      
      size_t pos = b->id.find_last_of("_");
      if(pos == string::npos) { 
	assert(false);
	b->set_start_to(0.0);
	b->providesStartingPoint = false; 
	continue; 
      }

      const string b0_name = b->id.substr(0, pos+1) + "0";
      auto b0p = dict_basecase_vars.find(b0_name);
      if(b0p == dict_basecase_vars.end()) {

	//if(b->id.find("delta") == string::npos && 
	//   b->id.find("AGC")   == string::npos) return false;

	//these will be done the same warm-start procedure with v_n and theta_n
	//if(b->id.find("v_aux") != string::npos ||
	//   b->id.find("theta_aux") != string::npos) continue;
	
	//assert(b->id.find("delta") != string::npos || 
	//       b->id.find("AGC") != string::npos); //!remove agc later
	continue;
      }

      //
      //v_n and theta_n are done separately (this includes primal and Low/Upp dual bounds)
      //
      if(b->id.find("v_n") != string::npos) {
	assert(v_n_0 == NULL);
	assert(v_n_k == NULL);
	v_n_0 =  b0p->second;
	v_n_k =  b;
	continue;
      }
      if(b->id.find("theta_n") != string::npos) {
	assert(theta_n_0 == NULL);
	assert(theta_n_k == NULL);
	theta_n_0 = b0p->second;
	theta_n_k = b;
	continue;
      }
      if(b->id.find("v_aux") != string::npos) {
	assert(v_aux_n_k == NULL);
	v_aux_n_k = b;
      }
      if(b->id.find("theta_aux") != string::npos) {
	assert(v_aux_n_k == NULL);
	v_aux_n_k = b;
      }
     
      auto b0 = b0p->second; assert(b0);

      if(b0->n == b->n) {
	b->set_start_to(*b0);
      } else {

	if(b0->n - 1 != b->n) {
	  printf("b0 [%s]=%d    b [%s]=%d\n", b0->id.c_str(), b0->n, b->id.c_str(), b->n);
	}
	
	assert(b0->n - 1 == b->n);
	if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	  assert(b->id.find("_g_") != string::npos);
	  for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	    assert(pgK_nonpartic_idxs[i] < b->n);
	    assert(pg0_nonpartic_idxs[i] < b0->n);
	    b->x[pgK_nonpartic_idxs[i]] = b0->x[pg0_nonpartic_idxs[i]];
	  }

	  for(int i=0; i<pg0_partic_idxs.size(); i++) {
	    assert(pgK_partic_idxs[i] < b->n);
	    assert(pg0_partic_idxs[i] < b0->n);
	    b->x[pgK_partic_idxs[i]] = b0->x[pg0_partic_idxs[i]];
	  }
	  b->providesStartingPoint = true; 

	} else if(dK.K_ConType[0] == SCACOPFData::kLine) {
	  assert(b->id.find("_li") != string::npos);
	  int i=0, i0=0;
	  for(; i0<b0->n; i0++) {
	    if(i0 != dK.K_outidx[0]) {
	      b->x[i] = b0->x[i0];
	      i++;
	    }
	  }
	  assert(i0 == b0->n);
	  assert(i  == b->n);
	  b->providesStartingPoint = true; 

	} else if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	  assert(b->id.find("_ti") != string::npos || b->id.find("_trans_") != string::npos);
	  int i=0, i0=0;
	  for(; i0<b0->n; i0++) {
	    if(i0 != dK.K_outidx[0]) {
	      b->x[i] = b0->x[i0];
	      i++;
	    }
	  }
	  assert(i0 == b0->n);
	  assert(i  == b->n);
	  b->providesStartingPoint = true; 

	} else { assert(false); }
      }
    }

    //take care of voltages and angles
    if(v_n_0 && theta_n_0) prob_mds_->v_and_theta_start_from_fs(dK,
								v_n_k,
								theta_n_k,
								v_aux_n_k,
								theta_aux_n_k,
								*v_n_0,
								*theta_n_0);
    
    return true;
  }
  
  

  // bool ContingencyProblemKronRedWithFixingCode1::
  // do_qgen_fixing_for_PVPQ(OptVariablesBlock* vnk, OptVariablesBlock* qgk)
  // {
  //   SCACOPFData& d = *data_K[0];

  //   //(aggregated) non-fixed q_g generator ids at each node/bus
  //   // with PVPQ generators that have at least one non-fixed q_g 
  //   vector<vector<int> > idxs_gen_agg;
  //   //bus indexes that have at least one non-fixed q_g
  //   vector<int> idxs_bus_pvpq;
  //   //aggregated lb and ub on reactive power at each PVPQ bus
  //   vector<double> Qlb, Qub;
  //   int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;
    
  //   get_idxs_PVPQ(d, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
  // 		  nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);
  //   assert(idxs_gen_agg.size() == idxs_bus_pvpq.size());
  //   assert(vnk->n == v_n0->n);

  //   for(int itpvpq=0; itpvpq<idxs_bus_pvpq.size(); itpvpq++) {
  //     const int busidx = idxs_bus_pvpq[itpvpq];
  //     double vdev = (vnk->x[busidx]-v_n0->x[busidx]) / std::max(1., fabs(v_n0->x[busidx]));
  //     double Qlbn=0., Qubn=0., qapprox_nk=0.;
  //     for(int gidx : idxs_gen_agg[itpvpq]) {
  // 	Qlbn += d.G_Qlb[gidx];
  // 	Qubn += d.G_Qub[gidx];
  // 	qapprox_nk += qgk->x[gidx];
  //     }

  //     double gen_band = Qubn - Qlbn; 
  //     double dist_lower = (qapprox_nk - Qlbn)/gen_band; 
  //     double dist_upper = (Qubn - qapprox_nk)/gen_band; 

  //     //if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
  //     //printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
  //     //     busidx, gen_band, dist_lower,  dist_upper,
  //     //     Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);
  //     dist_lower = std::max(dist_lower, 0.);
  //     dist_upper = std::max(dist_upper, 0.);

  //     assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
  //     assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

  //     const double rtol = 1e-2, rtolv=1e-3;
  //     if(dist_lower > rtol && dist_upper > rtol) {
  // 	//inside -> fix v_nk

  // 	//!	assert(fabs(vnk->ub[busidx] - vnk->lb[busidx])<1e-8);
  // 	//!assert(fabs(vnk->ub[busidx] - vnk->x[busidx]) <1e-8);

  //     } else if(dist_lower <= rtol) {
  // 	if(vdev >= rtolv) {
  // 	  //strict complementarity -> fix q_gk     to Qlb               
  //         //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

  // 	  vnk->lb[busidx] = v_n0->x[busidx] - g_bounds_abuse;; 
  // 	  vnk->ub[busidx] = data_sc.N_EVub[busidx] + g_bounds_abuse;
  // 	  for(int g : idxs_gen_agg[itpvpq]) {
  // 	    qgk->lb[g] =  d.G_Qlb[g] - g_bounds_abuse;
  // 	    qgk->ub[g] =  d.G_Qlb[g] + g_bounds_abuse;
  // 	  }
  // 	}  else {
  // 	  //degenerate complementarity 
	  
  // 	  //if(fixVoltage) {
  // 	  //  printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will fix voltage\n", busidx); 
  // 	  //  vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
  // 	  //} else {
  // 	  //printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will put q_g close to lower\n", busidx); 
  // 	  vnk->lb[busidx] = v_n0->x[busidx] - g_bounds_abuse; 
  // 	  vnk->ub[busidx] = data_sc.N_EVub[busidx] + g_bounds_abuse;
  // 	  for(int g : idxs_gen_agg[itpvpq]){
  // 	    qgk->lb[g] =  d.G_Qlb[g] - g_bounds_abuse;
  // 	    qgk->ub[g] =  d.G_Qlb[g] + g_bounds_abuse;
  // 	  }
	  
  // 	}
  //     } else { // if(dist_upper <= rtol)
  // 	assert(dist_upper <= rtol);
  // 	if(vdev <= - rtolv) {
  // 	  //strict complementarity -> fix q_gk to Qub 
  // 	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
  // 	  vnk->ub[busidx] = v_n0->x[busidx] + g_bounds_abuse; 
  // 	  vnk->lb[busidx] = data_sc.N_EVlb[busidx] - g_bounds_abuse;

  // 	  for(int g : idxs_gen_agg[itpvpq]) {
  // 	    qgk->lb[g] = d.G_Qub[g] - g_bounds_abuse;
  // 	    qgk->ub[g] = d.G_Qub[g] + g_bounds_abuse;
  // 	  }
  // 	} else {
  // 	  //degenerate complementarity 
  // 	  vnk->ub[busidx] = v_n0->x[busidx] + g_bounds_abuse; 
  // 	  vnk->lb[busidx] = data_sc.N_EVlb[busidx] - g_bounds_abuse;
  // 	  for(int g : idxs_gen_agg[itpvpq]){
  // 	    qgk->lb[g] = d.G_Qub[g] - g_bounds_abuse;
  // 	    qgk->ub[g] = d.G_Qub[g] + g_bounds_abuse;
  // 	  }
  // 	}
  //     }
      
  //   }

  //   return true;
  // }
}

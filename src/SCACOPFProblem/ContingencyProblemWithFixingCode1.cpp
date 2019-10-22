#include "ContingencyProblemWithFixingCode1.hpp"

#include "goSignalHandling.hpp"

using namespace std;

//definitions in ContingencyProblemWithFixing
extern const int max_mem_ma57_normal;// = 1000; //MB
extern const int max_mem_ma57_safem;// = 1500; //MB
extern const int alarm_ma57_normal;// = 30; //seconds
extern const int alarm_ma57_safem;// = 30; //M
extern const int max_mem_ma27_normal;// = 1000; //MB
extern const int max_mem_ma27_safem;// = 1500; //MB
extern const int alarm_ma27_normal;// = 45; //seconds
extern const int alarm_ma27_safem;// = 45; //MB


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

  bool ContingencyProblemWithFixingCode1::eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, double* data_for_master)
  {
    goTimer tmrec; tmrec.start();
    SCACOPFData& d = *data_K[0];
    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    set_no_recourse_action(data_for_master);

    if(best_known_iter.obj_value <= pen_accept_initpt) {
      assert(vars_primal->n() == best_known_iter.vars_primal->n());
      vars_primal->set_start_to(*best_known_iter.vars_primal);
      this->obj_value = best_known_iter.obj_value;

#ifdef BE_VERBOSE
      printf("ContProb_wfix K_idx=%d opt1 ini point is acceptable on rank=%d\n", K_idx, my_rank);
      fflush(stdout);
#endif
      f = this->obj_value;
      set_no_recourse_action(data_for_master, f);
      return true;
    }
    
    bool bFirstSolveOK = do_solve1();
    f = this->obj_value;
    
    if(variable("delta", d)) solv1_delta_optim = variable("delta", d)->x[0];
    else                     solv1_delta_optim = 0.;

    double acceptable_penalty = safe_mode ?  pen_accept_safemode : pen_accept_solve1;

    bool skip_2nd_solve = false;
    
    if(!bFirstSolveOK) skip_2nd_solve=false;
    
    if(tmTotal.measureElapsedTime() > 0.95*timeout) {
      skip_2nd_solve = true;
      if(bFirstSolveOK) {
	printf("ContProb_wfix K_idx=%d premature exit opt1 too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
      } else {
	printf("ContProb_wfix K_idx=%d premature exit inipt returned opt1 took too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
	//return ini point to make sure we stay feasible
	vars_primal->set_start_to(*vars_ini);
      }
    } else {
      if(f>=acceptable_penalty)
	determine_recourse_action(data_for_master);
    }

    if(monitor.emergency) acceptable_penalty = std::max(acceptable_penalty, pen_accept_emer);
    
    if(this->obj_value>acceptable_penalty && !skip_2nd_solve) {

 #ifdef BE_VERBOSE
      print_objterms_evals();
      //print_p_g_with_coupling_info(*data_K[0], pg0);
      printf("ContProb_wfix K_idx=%d first pass resulted in high pen; delta=%g\n", K_idx, solv1_delta_optim);
#endif

      double pplus, pminus, poverall;
      estimate_active_power_deficit(pplus, pminus, poverall);
#ifdef BE_VERBOSE
      printf("ContProb_wfix K_idx=%d (after solv1) act pow imbalances p+ p- poveral %g %g %g\n",
	     K_idx, pplus, pminus, poverall);
#endif

      bool one_more_push_and_fix=false; double gen_K_diff=0.;
      if(fabs(solv1_delta_optim-solv1_delta_blocking)<1e-2 && 
	 d.K_ConType[0]==SCACOPFData::kGenerator && solv1_Pg_was_enough) {
	one_more_push_and_fix = true;
	if(pg0->x[data_sc.K_outidx[K_idx]]>1e-6 )  gen_K_diff = std::max(0., 1.2*poverall);
	else if(pg0->x[data_sc.K_outidx[K_idx]]<-1e-6)  gen_K_diff = std::min(0., poverall);
	else one_more_push_and_fix = false;
      }

      if(fabs(poverall)>1e-4) {// && d.K_ConType[0]!=SCACOPFData::kGenerator) {
	double rpa = fabs(pplus) / fabs(poverall);
	double rma = fabs(pminus) / fabs(poverall);

	//solv1_delta_optim=0.;//!

	if( (rpa>0.85 && rpa<1.15) || (rma>0.85 && rma <1.15) ) {
	  one_more_push_and_fix = true;
	  gen_K_diff = 1.2*poverall;

	  //ignore small delta for transmission contingencies since they're really optimization noise
	  if(d.K_ConType[0]!=SCACOPFData::kGenerator && fabs(solv1_delta_optim)<1e-6) {
	    solv1_delta_optim=0.;
	  }

	  //if our first attempt to ramp up resulted in a active power balance deficit, then be more agressive this time
	  if(d.K_ConType[0]==SCACOPFData::kGenerator) {
	    double pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits;
	    get_objective_penalties(pen_p_balance, pen_q_balance, pen_line_limits, pen_trans_limits);
	    if(pen_p_balance > 100.*pen_q_balance && 
	       pen_p_balance > 100.*pen_line_limits && 
	       pen_p_balance > 100.*pen_trans_limits) {

	      if(pg0->x[data_sc.K_outidx[K_idx]] < -1e-6) assert(false);

	      //double gen_deficit = pg0->x[data_sc.K_outidx[K_idx]];
	      if(pen_p_balance > 2e5)
		gen_K_diff = 3*poverall;
	      else if(pen_p_balance > 5e4)
		gen_K_diff = 2*poverall;
	      else 
		gen_K_diff = 1.5*poverall;
	    }
	  }
	}
      }

      if(one_more_push_and_fix) {
 	//apparently we need to further unblock generation
 	auto pgK = variable("p_g", d); assert(pgK!=NULL);
 	//find AGC generators that are "blocking" and fix them; update particip and non-particip indexes
 	vector<int> pg0_partic_idxs_u=solv1_pg0_partic_idxs, pgK_partic_idxs_u=solv1_pgK_partic_idxs;
 	vector<int> pgK_nonpartic_idxs_u=solv1_pgK_nonpartic_idxs, pg0_nonpartic_idxs_u=solv1_pg0_nonpartic_idxs;

 	double delta_out=0., delta_needed=0., delta_blocking=0., delta_lb, delta_ub; 
	double residual_Pg;
 	bool bfeasib;

	if(fabs(gen_K_diff)>1e-6) {
	  if(K_idx==11827) printf("!!!!K_idx=%d gen_K_diff=%g solv1_delta_optim=%g\n", K_idx, gen_K_diff, solv1_delta_optim);
	  //solv1_delta_optim and gen_K_diff must have same sign at this point
	  if(solv1_delta_optim * gen_K_diff < 0) gen_K_diff=0.;
	  bfeasib = push_and_fix_AGCgen(d, gen_K_diff, solv1_delta_optim, 
					pg0_partic_idxs_u, pgK_partic_idxs_u, pg0_nonpartic_idxs_u, pgK_nonpartic_idxs_u,
					pg0, pgK, 
					data_sc.G_Plb, data_sc.G_Pub, data_sc.G_alpha,
					delta_out, delta_needed, delta_blocking, delta_lb, delta_ub, residual_Pg);
 	  //alter starting points 
	  assert(pg0_partic_idxs_u.size() == pgK_partic_idxs_u.size());
	  for(int it=0; it<pg0_partic_idxs_u.size(); it++) {
	    const int& i0 = pg0_partic_idxs_u[it];
	    pgK->x[pgK_partic_idxs_u[it]] = pg0->x[i0]+data_sc.G_alpha[i0]*delta_out;
	  }
#ifdef BE_VERBOSE
	  printf("ContProb_wfix K_idx=%d (gener)(after solv1) fixed %lu gens; adtl deltas out=%g needed=%g blocking=%g "
		 "residualPg=%g feasib=%d\n",
		 K_idx, solv1_pg0_partic_idxs.size()-pg0_partic_idxs_u.size(),
		 delta_out, delta_needed, delta_blocking, residual_Pg, bfeasib);
	  //printvec(solv1_pgK_partic_idxs, "solv1_pgK_partic_idxs");
	  //printvec(pgK_partic_idxs_u, "pgK_partic_idxs_u");
#endif
	  
	  delete_constraint_block(con_name("AGC_simple_fixedpg0", d));
	  delete_duals_constraint(con_name("AGC_simple_fixedpg0", d));
	  
	  if(pg0_partic_idxs_u.size()>0) {
	    add_cons_AGC_simplified(d, pg0_partic_idxs_u, pgK_partic_idxs_u, pg0);
	    append_duals_constraint(con_name("AGC_simple_fixedpg0", d));
	    variable_duals_cons("duals_AGC_simple_fixedpg0", d)->set_start_to(0.0);
	    
	    variable("delta", d)->set_start_to(delta_out);
	  }
	  
	  primal_problem_changed();
	}
      } // else of if(one_more_push_and_fix)

      //
      {
	auto v = variable("v_n", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }
      {
	auto v = variable("q_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }

      {
	auto v = variable("p_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse;
	  v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }

      do_qgen_fixing_for_PVPQ(variable("v_n", d), variable("q_g", d));

#ifdef DEBUG
      if(bFirstSolveOK) {
	if(!vars_duals_bounds_L->provides_start()) print_summary();
	assert(vars_duals_bounds_L->provides_start()); 	assert(vars_duals_bounds_U->provides_start()); 	
	assert(vars_duals_cons->provides_start());
      }
      assert(vars_primal->n() == vars_last->n());
#endif

      //
      // --- SOLVE 2 --- 
      //
      bool opt2_ok = do_solve2(bFirstSolveOK);
      f = this->obj_value;
      if(!opt2_ok) {
	if(bFirstSolveOK) {
	  //sln = sln_solve1;
	  f = obj_solve1;
	  //recourse actions were already determined
	} else {
	  printf("[warning][panic] ContProb_wfix K_idx=%d return bestknown; opt1 and opt2 failed on rank=%d\n", K_idx, my_rank);
	  vars_primal->set_start_to(*best_known_iter.vars_primal);
	  //get_solution_simplicial_vectorized(sln_solve1);
	  //sln = sln_solve1;
	  f = best_known_iter.obj_value;
	  //no recourse actions when both solve1 and solve2 fail
	}

      } else { //opt2_ok
	obj_solve2 = this->obj_value;
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
	if(variable("delta", d)) delta_optim = variable("delta", d)->x[0];
#ifdef BE_VERBOSE
	print_objterms_evals();
	//print_p_g_with_coupling_info(*data_K[0], pg0);
	printf("ContProb_wfix K_idx=%d opt1 opt2 resulted in high pen delta=%g\n", K_idx, delta_optim);
#endif
      }  
    } else {
      //sln = sln_solve1;
      f = obj_solve1;
      if(this->obj_value>acceptable_penalty && skip_2nd_solve)
	printf("ContProb_wfix K_idx=%d opt2 needed but not done insufic time rank=%d\n", K_idx, my_rank);
      if(this->obj_value>acceptable_penalty)
	determine_recourse_action(data_for_master);
    }
    
    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProb_wfix K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif
    return true;
  }

  bool ContingencyProblemWithFixingCode1::determine_recourse_action(double* info_out)
  {
    SCACOPFData& data = data_sc;
    //
    // prepare info for master rank
    //
    info_out[0]=this->obj_value;

    if(data.K_ConType[K_idx] == SCACOPFData::kGenerator) {

      assert(p_g0->n == data.G_Generator.size());
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      assert(data.K_outidx.size() == data.K_Contingency.size());
      
      int idx_gen = data.K_outidx[K_idx];
      assert(idx_gen>=0 && idx_gen<p_g0->n);
      
      info_out[1]=p_g0->x[idx_gen];
      info_out[2]=info_out[3]=info_out[4]=0.;
      
    } else if(data.K_ConType[K_idx] == SCACOPFData::kLine) {

      assert(p_li10); assert(q_li10); assert(p_li20); assert(q_li20);
      assert(data.L_Line.size() == q_li10->n);
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      
      int idx = data.K_outidx[K_idx];
       
      assert(idx>=0 && idx<q_li10->n);
      if(!recourse_action_from_voltages(idx, true, info_out)) { 
	//use penalization of the powers through the transformer
	info_out[1]=p_li10->x[idx];
	info_out[2]=q_li10->x[idx];
	info_out[3]=p_li20->x[idx];
	info_out[4]=q_li20->x[idx];
      }      
      
    } else if(data.K_ConType[K_idx] == SCACOPFData::kTransformer) {

      assert(data.T_Transformer.size() == q_ti10->n);
      assert(K_idx>=0 && K_idx<data.K_outidx.size());
      int idx = data.K_outidx[K_idx];
      assert(idx>=0 && idx<q_ti10->n);

      if(!recourse_action_from_voltages(idx, false, info_out)) { 
	//use penalization of the powers through the transformer
	info_out[1]=p_ti10->x[idx];
	info_out[2]=q_ti10->x[idx];
	info_out[3]=p_ti20->x[idx];
	info_out[4]=q_ti20->x[idx];
      }
    }
    return true;
  }
  bool ContingencyProblemWithFixingCode1::recourse_action_from_voltages(int outidx, bool isLine, double* info_out)
  {
    int NidxFrom = isLine ? data_sc.L_Nidx[0][outidx] : data_sc.T_Nidx[0][outidx];
    int NidxTo   = isLine ? data_sc.L_Nidx[1][outidx] : data_sc.T_Nidx[1][outidx];

    vector<int> dualsidx_vnk_lb(data_sc.N_Bus.size()), dualsidx_vnk_ub(data_sc.N_Bus.size());
    iota(dualsidx_vnk_lb.begin(), dualsidx_vnk_lb.end(), 0);
    iota(dualsidx_vnk_ub.begin(), dualsidx_vnk_ub.end(), 0);

    auto vnk_duals_ub = variable_duals_upper("duals_bndU_v_n", *data_K[0]); 
    if(!vnk_duals_ub) {assert(false); return false; }
    auto vnk_duals_lb = variable_duals_lower("duals_bndL_v_n", *data_K[0]); 
    if(!vnk_duals_lb) {assert(false); return false; }
    auto v_nk = variable("v_n", *data_K[0]);

    sort(dualsidx_vnk_lb.begin(), dualsidx_vnk_lb.end(), 
	 [&](const int& a, const int& b) { return fabs(vnk_duals_lb->x[a]) > fabs(vnk_duals_lb->x[b]); } );
    sort(dualsidx_vnk_ub.begin(), dualsidx_vnk_ub.end(), 
	 [&](const int& a, const int& b) { return fabs(vnk_duals_ub->x[a]) > fabs(vnk_duals_ub->x[b]); } );
#ifdef BE_VERBOSE
    printf("ContingencyProblem_wfix largest duals for K_idx=%d recourse_%s outidx=%d busidxs from=%d to %d\n",
	   K_idx, isLine ? "line" : "transf", outidx, NidxFrom, NidxTo);

    printf("[[largest lower]]\n");
    for(int i=0; i<10; i++) {
      printf("\t{vn0,lb,ub[%5d]=%9.6e %9.6e %9.6e}  {vnk,lb,duallb[%5d]=%9.6e %9.6e %9.6e} {vnk,ub,dualub[%5d]=%9.6e %9.6e %9.6e}\n",
	     dualsidx_vnk_lb[i], v_n0->x[dualsidx_vnk_lb[i]], v_n0->lb[dualsidx_vnk_lb[i]], v_n0->ub[dualsidx_vnk_lb[i]], 
	     dualsidx_vnk_lb[i], v_nk->x[dualsidx_vnk_lb[i]], v_nk->lb[dualsidx_vnk_lb[i]], vnk_duals_lb->x[dualsidx_vnk_lb[i]],
	     dualsidx_vnk_lb[i], v_nk->x[dualsidx_vnk_lb[i]], v_nk->ub[dualsidx_vnk_lb[i]], vnk_duals_ub->x[dualsidx_vnk_lb[i]]);
	     
    }
    printf("[[largest upper]]\n");
    for(int i=0; i<10; i++) {
      printf("\t{vn0,lb,ub[%5d]=%9.6e %9.6e %9.6e}  {vnk,lb,duallb[%5d]=%9.6e %9.6e %9.6e} {vnk,ub,dualub[%5d]=%9.6e %9.6e %9.6e}\n",
	     dualsidx_vnk_ub[i], v_n0->x[dualsidx_vnk_ub[i]], v_n0->lb[dualsidx_vnk_ub[i]], v_n0->ub[dualsidx_vnk_ub[i]], 
	     dualsidx_vnk_ub[i], v_nk->x[dualsidx_vnk_ub[i]], v_nk->lb[dualsidx_vnk_ub[i]], vnk_duals_lb->x[dualsidx_vnk_ub[i]],
	     dualsidx_vnk_ub[i], v_nk->x[dualsidx_vnk_ub[i]], v_nk->ub[dualsidx_vnk_ub[i]], vnk_duals_ub->x[dualsidx_vnk_ub[i]]);
	     
    }
#endif

    int Nidx_from_upper = -1, Nidx_from_lower = -1; 
    {
      int idx_in_Nfrom = indexin(dualsidx_vnk_ub, NidxFrom); assert(idx_in_Nfrom>=0);
      int idx_in_Nto   = indexin(dualsidx_vnk_ub, NidxTo);   assert(idx_in_Nto  >=0);
      
      if(idx_in_Nfrom < 10 && fabs(v_nk->x[dualsidx_vnk_ub[idx_in_Nfrom]] - v_nk->ub[dualsidx_vnk_ub[idx_in_Nfrom]])<1e-2) {
	Nidx_from_upper = dualsidx_vnk_ub[idx_in_Nfrom];
      }
      if(idx_in_Nto   < 10 && fabs(v_nk->x[dualsidx_vnk_ub[idx_in_Nto  ]] - v_nk->ub[dualsidx_vnk_ub[idx_in_Nto  ]])<1e-2) {
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
      
      if(idx_in_Nfrom < 10 && fabs(v_nk->x[dualsidx_vnk_lb[idx_in_Nfrom]] - v_nk->lb[dualsidx_vnk_lb[idx_in_Nfrom]])<1e-2) {
	Nidx_from_lower = dualsidx_vnk_lb[idx_in_Nfrom];
      }
      if(idx_in_Nto   < 10 && fabs(v_nk->x[dualsidx_vnk_lb[idx_in_Nto  ]] - v_nk->lb[dualsidx_vnk_lb[idx_in_Nto  ]])<1e-2) {
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
      info_out[0] = obj_value;
      info_out[0+1] = 1000+v_n0->x[Nidx];
      info_out[1+1] = 1e+8; //upper is 1e+8, lower is -1e+8; fabs>=1e+8 indicates a voltage penalty
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
      info_out[0] = obj_value;
      info_out[0+1] = 1000+v_n0->x[Nidx];
      info_out[1+1] = -1e+8; //upper is 1e+8, lower is -1e+8; fabs>=1e+8 indicates a voltage penalty
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
	info_out[0] = obj_value;
	info_out[0+1] = 1000+v_n0->x[Nidx];
	info_out[1+1] = 1e+8; //upper is 1e+8, lower is -1e+8; fabs>=1e+8 indicates a voltage penalty
	info_out[2+1] = vnk_duals_ub->x[Nidx];
	info_out[3+1] = (double)(1+Nidx); //it will be -1-Nidx for lower
	return true;
      }
    }

    if(Nidx_from_upper>=0 || Nidx_from_lower>=0) return false;



    return false;
  }

    //      if(K_idx==11763){// || 925==K_idx) {
    //   print_active_power_balance_info(*data_K[0]);
    //   print_reactive_power_balance_info(*data_K[0]);
    //   print_PVPQ_info(*data_K[0], v_n0);

    //   auto duals_ub = variable_duals_upper("duals_bndU_v_n", *data_K[0]);
    //   auto duals_lb = variable_duals_lower("duals_bndL_v_n", *data_K[0]);
    //   auto v_n = variable("v_n", *data_K[0]);

    //   assert(duals_lb); assert(duals_ub); assert(v_n);
    //   vector<int> idxs = {1, 4090, 7454, 7476, 7477, 16677};
    //   for(auto idx: idxs) 
    // 	printf("!!!! idx=%d v_n0=%.5e v_nk=[%.5e < %.5e < %.5e] dual_L=%g dual_U=%g\n", 
    // 	       idx, v_n0->x[idx], v_n->lb[idx], v_n->x[idx], v_n->ub[idx], duals_lb->x[idx], duals_ub->x[idx]);
    // }
    // if(K_idx==11763){// || 925==K_idx) {
    //   print_active_power_balance_info(*data_K[0]);
    //   print_reactive_power_balance_info(*data_K[0]);
    //   print_PVPQ_info(*data_K[0], v_n0);

    //   auto duals_ub = variable_duals_upper("duals_bndU_v_n", *data_K[0]);
    //   auto duals_lb = variable_duals_lower("duals_bndL_v_n", *data_K[0]);
    //   auto v_n = variable("v_n", *data_K[0]);

    //   assert(duals_lb); assert(duals_ub); assert(v_n);

    //   assert(duals_lb); assert(duals_ub); assert(v_n);
    //   vector<int> idxs = {1, 4090, 7454, 7476, 7477, 16677};
    //   for(auto idx: idxs) 
    // 	printf("!!!! idx=%d v_n0=%.5e v_nk=[%.5e < %.5e < %.5e] dual_L=%g dual_U=%g\n", 
    // 	       idx, v_n0->x[idx], v_n->lb[idx], v_n->x[idx], v_n->ub[idx], duals_lb->x[idx], duals_ub->x[idx]);
    // }

}


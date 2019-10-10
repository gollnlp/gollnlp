#include "ContingencyProblemWithFixingCode1.hpp"

using namespace std;

static const int max_mem_ma57_normal = 1000; //MB
static const int max_mem_ma57_safem = 1500; //MB
static const int alarm_ma57_normal = 30; //seconds
static const int alarm_ma57_safem = 30; //M

static const int max_mem_ma27_normal = 1000; //MB
static const int max_mem_ma27_safem = 1500; //MB
static const int alarm_ma27_normal = 45; //seconds
static const int alarm_ma27_safem = 45; //MB


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

  bool ContingencyProblemWithFixingCode1::eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
  {
    goTimer tmrec; tmrec.start();
    SCACOPFData& d = *data_K[0];
    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;
    
    if(best_known_iter.obj_value <= pen_accept_initpt) {
      assert(vars_primal->n() == best_known_iter.vars_primal->n());
      vars_primal->set_start_to(*best_known_iter.vars_primal);
      this->obj_value = best_known_iter.obj_value;

#ifdef BE_VERBOSE
      printf("ContProbWithFixing K_idx=%d opt1 ini point is acceptable on rank=%d\n", K_idx, my_rank);
      fflush(stdout);
#endif
      f = this->obj_value;
      return true;
    }
    
    printf("solve 1 before!!!!!!!!!!!!!!!\n");
    bool bFirstSolveOK = do_solve1();
    f = this->obj_value;
    printf("solve 1 after!!!!!!!!!!!!!!!\n");

    
    if(variable("delta", d)) solv1_delta_optim = variable("delta", d)->x[0];
    else                     solv1_delta_optim = 0.;

    bool skip_2nd_solve = false;
    
    if(!bFirstSolveOK) skip_2nd_solve=false;
    
    if(tmTotal.measureElapsedTime() > 0.95*timeout) {
      	skip_2nd_solve = true;
      if(bFirstSolveOK) {
	printf("ContProbWithFixing K_idx=%d premature exit opt1 too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
      } else {
	printf("ContProbWithFixing K_idx=%d premature exit inipt returned opt1 took too long %g sec on rank=%d\n", 
	       K_idx, tmrec.measureElapsedTime(), my_rank);
	//return ini point to make sure we stay feasible
	vars_primal->set_start_to(*vars_ini);
      }
    }

    double acceptable_penalty = safe_mode ?  pen_accept_safemode : pen_accept_solve1;
    if(monitor.emergency) acceptable_penalty = std::max(acceptable_penalty, pen_accept_emer);
    
    if(this->obj_value>acceptable_penalty && !skip_2nd_solve) {

 #ifdef BE_VERBOSE
      print_objterms_evals();
      //print_p_g_with_coupling_info(*data_K[0], pg0);
      printf("ContProbWithFixing K_idx=%d first pass resulted in high pen; delta=%g\n", K_idx, solv1_delta_optim);
#endif

      double pplus, pminus, poverall;
      estimate_active_power_deficit(pplus, pminus, poverall);
#ifdef BE_VERBOSE
      printf("ContProbWithFixing K_idx=%d (after solv1) act pow imbalances p+ p- poveral %g %g %g\n",
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
	    if(pen_p_balance > 500.*pen_q_balance && 
	       pen_p_balance > 500.*pen_line_limits && 
	       pen_p_balance > 500.*pen_trans_limits) {

	      if(pg0->x[data_sc.K_outidx[K_idx]] < -1e-6) assert(false);

	      //double gen_deficit = pg0->x[data_sc.K_outidx[K_idx]];
	      if(pen_p_balance > 2e5)
		gen_K_diff = 10*poverall;
	      else if(pen_p_balance > 5e4)
		gen_K_diff = 5*poverall;
	      else 
		gen_K_diff = 2.5*poverall;
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
	  printf("ContProbWithFixing K_idx=%d (gener)(after solv1) fixed %lu gens; adtl deltas out=%g needed=%g blocking=%g "
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
	} else {
	  printf("[warning][panic] ContProbWithFixing K_idx=%d return bestknown; opt1 and opt2 failed on rank=%d\n", K_idx, my_rank);
	  vars_primal->set_start_to(*best_known_iter.vars_primal);
	  //get_solution_simplicial_vectorized(sln_solve1);
	  //sln = sln_solve1;
	  f = best_known_iter.obj_value;
	}
      } else { //opt2_ok

	obj_solve2 = this->obj_value;
	if(obj_solve1<obj_solve2) {
	  //sln = sln_solve1;
	  f = obj_solve1;
	} else {
	  //sln = sln_solve2;
	  f = obj_solve2;
	}
	//if(!bFirstSolveOK) sln = sln_solve2;
      }
      
      if(obj_solve2>pen_accept) { 
	double delta_optim = 0.;//
	if(variable("delta", d)) delta_optim = variable("delta", d)->x[0];
#ifdef BE_VERBOSE
	print_objterms_evals();
	//print_p_g_with_coupling_info(*data_K[0], pg0);
	printf("ContProbWithFixing K_idx=%d opt1 opt2 resulted in high pen delta=%g\n", K_idx, delta_optim);
#endif
      }  
    } else {
      //sln = sln_solve1;
      f = obj_solve1;
      if(this->obj_value>acceptable_penalty && skip_2nd_solve)
	printf("ContProbWithFixing K_idx=%d opt2 needed but not done insufic time rank=%d\n", K_idx, my_rank);
    }
      

























    f = -1e+20;
    //if(!optimize("ipopt")) {
    if(!reoptimize(OptProblem::primalDualRestart)) {
      //if(!reoptimize(OptProblem::primalRestart)) {
      if(!monitor.user_stopped) {
	f = 1e+6;
	return false;
      }
    }
    
    // objective value
    f = this->obj_value;
    
    tmrec.stop();
#ifdef BE_VERBOSE
    printf("ContProb_wfix K_id %d: eval_obj took %g sec  %d iterations on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), number_of_iterations(), my_rank);
    fflush(stdout);
#endif
    return true;
  }
}


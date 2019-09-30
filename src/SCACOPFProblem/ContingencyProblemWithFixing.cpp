#include "ContingencyProblemWithFixing.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"

#include <string>

#include "goUtils.hpp"

#ifdef GOLLNLP_FAULT_HANDLING
#include "goSignalHandling.hpp"
#endif

#include "goTimer.hpp"
#include "unistd.h"
using namespace std;

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

// #ifdef GOLLNLP_FAULT_HANDLING
// #define MSG_MAX_SZ 128
// static char msg_timer[MSG_MAX_SZ];
// static int sz_msg_timer = 0;

// static volatile sig_atomic_t solve_watch=false;
// static volatile sig_atomic_t alarm_duration=15; //seconds

// //this is used by the optimiz solver's callback
// volatile sig_atomic_t solve_is_alive=false;

// static sigjmp_buf jump_env;

// //only use approved/safe functions
// extern "C" void solve_timer_handler(int nsignum)
// {
//   if(solve_watch) {
//     //write(2, "watch\n", 6);
//     if(!solve_is_alive) {
//       //write(2, "notalive\n", 9);
//       write(2, msg_timer, sz_msg_timer);
//       siglongjmp(jump_env,1);
//     } else {
//       //write(2, "alive\n", 6);
//       solve_is_alive=false;
//       alarm(alarm_duration);
//     }
//   }
// };

// static void set_timer_message(const char* msg)
// {
//   strncpy(msg_timer, msg,  MSG_MAX_SZ-2);
//   msg_timer[MSG_MAX_SZ-2] = '\n';
//   msg_timer[MSG_MAX_SZ-1] = '\0';
//   sz_msg_timer = strlen(msg_timer);
// };
// #endif

namespace gollnlp {

  double ContingencyProblemWithFixing::g_bounds_abuse=9e-5;

  ContingencyProblemWithFixing::ContingencyProblemWithFixing(SCACOPFData& d_in, int K_idx_, 
							     int my_rank, int comm_size_,
							     std::unordered_map<std::string, 
							     gollnlp::OptVariablesBlock*>& dict_basecase_vars_,
							     const int& num_K_done_, const double& time_so_far_,
							     bool safe_mode_)
    : ContingencyProblem(d_in, K_idx_, my_rank),  comm_size(comm_size_),
      dict_basecase_vars(dict_basecase_vars_), solv1_Pg_was_enough(true),
      num_K_done(num_K_done_), time_so_far(time_so_far_), safe_mode(safe_mode_)
    { 
      pen_threshold=1e+3; 
      obj_solve1 = obj_solve2 = 1e+20; 
      vars_ini = vars_last = NULL;
    };

  ContingencyProblemWithFixing::~ContingencyProblemWithFixing()
  {
    delete vars_ini;
    delete vars_last;
  }

  
  bool ContingencyProblemWithFixing::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    double K_avg_time_so_far = time_so_far / std::max(num_K_done,1);
    string ssfm=" ";
    if(safe_mode) ssfm=" [safe mode] ";
    printf("ContProbWithFixing K_idx=%d%sIDOut=%d outidx=%d Type=%s avgtm=%.2f rank=%d\n",
	   K_idx, ssfm.c_str(), data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str(), K_avg_time_so_far, my_rank); fflush(stdout);

    p_g0=pg0; v_n0=vn0;

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];
    useQPen = true;

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

    add_variables(dK,false);

    if(!warm_start_variable_from_basecase(*vars_primal)) {
      assert(false);
      return false;
    }

    auto pgK = variable("p_g", dK); assert(pgK!=NULL);
    //find AGC generators that are "blocking" and fix them; update particip and non-particip indexes
    solv1_pg0_partic_idxs=pg0_partic_idxs; solv1_pgK_partic_idxs=pgK_partic_idxs;
    solv1_pgK_nonpartic_idxs=pgK_nonpartic_idxs; solv1_pg0_nonpartic_idxs=pg0_nonpartic_idxs;
    double gen_K_diff=0.;//default for transmission contingencies; surplus or deficit of generation
    double residual_Pg;
    solv1_delta_out=0.; solv1_delta_lb=-1e+20; solv1_delta_ub=1e+20; 
    solv1_delta_blocking=0.; solv1_delta_needed=0.;
    if(dK.K_ConType[0]==SCACOPFData::kGenerator) {
      assert(data_sc.K_outidx[K_idx]<pg0->n);
      gen_K_diff = pg0->x[data_sc.K_outidx[K_idx]];

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
#ifdef BE_VERBOSE
      printf("ContProbWithFixing K_idx=%d (gener) %g gen missing; fixed %lu gens; deltas out=%g needed=%g blocking=%g "
	     "residualPg=%g feasib=%d\n",
	     K_idx, gen_K_diff, pg0_partic_idxs.size()-solv1_pg0_partic_idxs.size(),
	     solv1_delta_out, solv1_delta_needed, solv1_delta_blocking, residual_Pg, solv1_Pg_was_enough);
#endif
    }

    add_cons_lines_pf(dK);
    add_cons_transformers_pf(dK);
    add_cons_active_powbal(dK);
    add_cons_reactive_powbal(dK);
    bool SysCond_BaseCase = false;
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    add_cons_pg_nonanticip_using(pg0, solv1_pg0_nonpartic_idxs, solv1_pgK_nonpartic_idxs);
    //add_cons_AGC_using(pg0);
    if(solv1_pg0_partic_idxs.size() > 0) {
      add_cons_AGC_simplified(dK, solv1_pg0_partic_idxs, solv1_pgK_partic_idxs, pg0);
      auto deltav = variable("delta", dK); assert(deltav);
      if(deltav) { //it may happen that all AGC gens were fixed
	deltav->set_start_to(solv1_delta_out);
	deltav->lb[0] = solv1_delta_lb; deltav->ub[0] = solv1_delta_ub; 
      } 
    } else {
      //all AGC gens were fixed; add fixed variable delta 
      OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK)); 
      append_variables(deltaK);
      deltaK->set_start_to(solv1_delta_out);
      deltaK->lb[0] = deltaK->ub[0] = solv1_delta_out;
      append_objterm(new QuadrRegularizationObjTerm("delta_regul", deltaK, 1., solv1_delta_out));
    }
    
    add_const_nonanticip_v_n_using(vn0, Gk);
    //add_cons_PVPQ_using(vn0, Gk);

    assert(vars_primal->provides_start());

    if(NULL==vars_duals_bounds_L || NULL==vars_duals_bounds_U || NULL==vars_duals_cons) {
      //force allocation of duals
      dual_problem_changed();
    }

    if(!warm_start_variable_from_basecase(*vars_duals_bounds_L)) return false;
    if( variable_duals_lower("duals_bndL_delta", dK) )
      variable_duals_lower("duals_bndL_delta", dK)->set_start_to(0.0);
    assert(vars_duals_bounds_L->provides_start());

    if(!warm_start_variable_from_basecase(*vars_duals_bounds_U)) return false;
    if(variable_duals_upper("duals_bndU_delta", dK))
      variable_duals_upper("duals_bndU_delta", dK)->set_start_to(0.0);
    assert(vars_duals_bounds_U->provides_start());

    //AGC_simple_fixedpg0
    if(!warm_start_variable_from_basecase(*vars_duals_cons)) return false;
    if(variable_duals_cons("duals_AGC_simple_fixedpg0", dK))
      variable_duals_cons("duals_AGC_simple_fixedpg0", dK)->set_start_to(0.0);
    assert(vars_duals_cons->provides_start());


#ifdef GOLLNLP_FAULT_HANDLING
    string msg = "[timer] ma57 timeout rank=" + std::to_string(my_rank) +" for K_idx=" + std::to_string(K_idx) + " occured!\n";
    set_timer_message_ma57(msg.c_str());

    msg = "[timer] ma27 timeout rank=" + std::to_string(my_rank) +" for K_idx=" + std::to_string(K_idx) + " occured!\n";
    set_timer_message_ma27(msg.c_str());
    
    assert(my_rank>=1);

    vars_last = vars_primal->new_copy();
    vars_ini  = vars_primal->new_copy();
#endif

    return true;
  }


  bool ContingencyProblemWithFixing::do_solve1()
  {
    //! "ma27_ignore_singularity" 
    //set_solver_option("ma27_meminc_factor", 1.1);

    g_solve_watch_ma57=true;
    g_alarm_duration_ma57=6;//seconds
    g_max_memory_ma57=300;//Mbytes
    g_my_rank_ma57=my_rank;
    g_my_K_idx_ma57=K_idx;

    g_solve_watch_ma27=true;
    g_alarm_duration_ma27=8;//seconds
    g_max_memory_ma27=400;//Mbytes
    g_my_rank_ma27=my_rank;
    g_my_K_idx_ma27=K_idx;

    goTimer tmrec; tmrec.start();
    vector<int> hist_iter, hist_obj;
    bool bret = true, done = false; 
    OptimizationStatus last_opt_status = Solve_Succeeded; //be positive
    bool solve1_safe_mode=safe_mode;
    int n_solves=0; 
    while(!done) {

      bool opt_ok=false; bool PDRestart=true;

      switch(n_solves) {
      case 0: 
	{ 
	  PDRestart=true;
	  //set_solver_option("mu_target", 1e-9);
	  set_solver_option("mu_init", 1e-4);
	  set_solver_option("tol", 1e-8);
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");
	}
	break;
      case 1: 
	{
	  PDRestart=false;
	  set_solver_option("mu_target", 1e-8);
	  if(last_opt_status!=User_Requested_Stop && 
	     last_opt_status!=Unrecoverable_Exception && 
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-4);
	  }

	  set_solver_option("ma57_small_pivot_flag", 1);

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.25);
	  
	  const double gamma = 1e-3;
	  regularize_vn(gamma);
	  regularize_thetan(gamma);
	  regularize_bs(gamma);
	  regularize_pg(gamma);
	  regularize_qg(gamma);

	  g_alarm_duration_ma57=10;//seconds
	  g_max_memory_ma57=500;//Mbytes

	}
	break;
      case 2: //MA27
	{
	  PDRestart=false;
	  reallocate_nlp_solver();
	  printf("[warning] ContProbWithFixing K_idx=%d opt1 will switch to ma27 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  set_solver_option("linear_solver", "ma27"); 

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	  }

	  set_solver_option("mu_init", 1e-4); 
	  set_solver_option("mu_target", 1e-8);

	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-3;
	  update_regularizations(gamma);
	}
	break;
      case 3: //MA27
	{
	  PDRestart=false;
	  solve1_safe_mode=true;
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-4);
	  }
	  set_solver_option("mu_target", 1e-8);
	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  update_regularizations(gamma);

	  g_alarm_duration_ma27=15;//seconds
	  g_max_memory_ma27=700;//Mbytes
	}
	break;
      case 4: 
	{
	  PDRestart=false;
	  solve1_safe_mode=true;
	  reallocate_nlp_solver();

	  vars_primal->set_start_to(*vars_ini);
	  set_solver_option("mu_init", 1.);
	  set_solver_option("mu_target", 1e-8);

	  printf("[warning] ContProbWithFixing K_idx=%d opt1 will switch to ma57 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("ma57_automatic_scaling", "yes");
	  set_solver_option("ma57_small_pivot_flag", 1);

	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  update_regularizations(gamma);

	  g_alarm_duration_ma57=12;//seconds
	  g_max_memory_ma57=600;//Mbytes
	}
	break;
      default:
	{
	  PDRestart=false;
	  solve1_safe_mode=true;
	  set_solver_option("mu_init", 1.);
	  set_solver_option("mu_target", 1e-8);
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("ma57_automatic_scaling", "yes");
	  set_solver_option("tol", 1e-6);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-2 + 0.1*n_solves;
	  update_regularizations(gamma);

	  g_alarm_duration_ma57=12;//seconds
	  g_max_memory_ma57=600;//Mbytes
	}
      }
      set_solver_option("print_user_options", "no");
      set_solver_option("print_level", 2);
      set_solver_option("sb","yes");

      set_solver_option("max_iter", 300);
      set_solver_option("acceptable_tol", 1e-3);
      set_solver_option("acceptable_constr_viol_tol", 1e-6);
      set_solver_option("acceptable_iter", 5);

      set_solver_option("fixed_variable_treatment", "relax_bounds");
      set_solver_option("honor_original_bounds", "yes");
      double relax_factor = std::min(1e-8, pow(10., 3*n_solves-16));
      set_solver_option("bound_relax_factor", relax_factor);
      double bound_push = std::min(1e-2, pow(10., 3*n_solves-12));
      set_solver_option("bound_push", bound_push);
      set_solver_option("slack_bound_push", bound_push); 
      double bound_frac = std::min(1e-2, pow(10., 3*n_solves-10));
      set_solver_option("bound_frac", bound_frac);
      set_solver_option("slack_bound_frac", bound_frac);
      
      set_solver_option("neg_curv_test_reg", "no"); //default yes ->ChiangZavala primal regularization


      monitor.safe_mode=solve1_safe_mode; 
      monitor.timer.restart();
      monitor.hist_tm.clear();
      monitor.user_stopped = false;
      
      if(data_sc.N_Bus.size()>8999) {
	if(solve1_safe_mode)
	  monitor.bailout_allowed=true;//! probably not needed when watching timeouts
	else 
	  monitor.bailout_allowed=false;
      }

      if(PDRestart) {
	opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
      } else {
	opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);
      }

      n_solves++;
      last_opt_status = OptProblem::optimization_status();

      hist_iter.push_back(number_of_iterations());
      hist_obj.push_back(this->obj_value);

      if(opt_ok) {
	done = true;
      } else {
	if(monitor.user_stopped) {
	  assert(last_opt_status == User_Requested_Stop);
	  done = true;
	} else {
	  //something bad happened, will resolve
	  printf("[warning] ContProbWithFixing K_idx=%d opt1 failed at try %d rank=%d time %g\n", 
		 K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
	}
      }
      
      if(n_solves>9) done = true;
      if(tmrec.measureElapsedTime()>800) {
	printf("[warning] ContProbWithFixing K_idx=%d opt1 taking too long on rank=%d; tries %d time %g\n", 
	       K_idx, my_rank, n_solves, tmrec.measureElapsedTime());
	done = true;
	bret = false;
      }
      
    } //end of outer while
#ifdef BE_VERBOSE
    string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbWithFixing K_idx=%d opt1 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
    fflush(stdout);
#endif
    get_solution_simplicial_vectorized(sln_solve1);
    obj_solve1 = this->obj_value;
    return bret;
  }
  //
  // solve2
  //
  bool ContingencyProblemWithFixing::do_solve2(bool bFirstSolveOK)
  {
    goTimer tmrec; tmrec.start();

#ifdef GOLLNLP_FAULT_HANDLING
    if(bFirstSolveOK)
      vars_ini->set_start_to(*vars_primal);
#endif

    g_solve_watch_ma57=true;
    g_alarm_duration_ma57=6;//seconds
    g_max_memory_ma57=300;//Mbytes
    g_my_rank_ma57=my_rank;
    g_my_K_idx_ma57=K_idx;

    g_solve_watch_ma27=true;
    g_alarm_duration_ma27=8;//seconds
    g_max_memory_ma27=400;//Mbytes
    g_my_rank_ma27=my_rank;
    g_my_K_idx_ma27=K_idx;

    vector<int> hist_iter, hist_obj;
    bool bret = true, done = false; 
    OptimizationStatus last_opt_status = Solve_Succeeded; //be positive
    bool solve2_safe_mode=safe_mode;
    int n_solves=0; 
    while(!done) {
      bool opt_ok=false; bool PDRestart=true;

      switch(n_solves) {
      case 0: 
	{ 
	  if(bFirstSolveOK) {
	    PDRestart=true;
	    //set_solver_option("mu_target", 1e-9);
	    set_solver_option("mu_init", 1e-4);
	  } else {
	    PDRestart=false;
	    set_solver_option("mu_init", 1e-1);
	  }
	  set_solver_option("tol", 1e-8);
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");
	}
	break;
      case 1: 
	{	  
	  set_solver_option("mu_target", 1e-8);
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	    PDRestart=false;
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-4);
	    PDRestart=true;
	  }

	  set_solver_option("ma57_small_pivot_flag", 1);

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.25);

	  const double gamma = 1e-3;
	  regularize_vn(gamma);
	  regularize_thetan(gamma);
	  regularize_bs(gamma);
	  regularize_pg(gamma);
	  regularize_qg(gamma);

	  g_alarm_duration_ma57=10;//seconds
	  g_max_memory_ma57=500;//Mbytes

	}
	break;
      case 2: //MA27
	{
	  PDRestart=false;
	  reallocate_nlp_solver();
	  printf("[warning] ContProbWithFixing K_idx=%d opt2 will switch to ma27 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  set_solver_option("linear_solver", "ma27"); 

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	  }

	  set_solver_option("mu_init", 1e-4); 
	  set_solver_option("mu_target", 1e-8);

	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-3;
	  update_regularizations(gamma);
	}
	break;
      case 3: //MA27
	{	  
	  solve2_safe_mode=true;
	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	    PDRestart=false;
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-4);
	    PDRestart=true;
	  }
	  set_solver_option("mu_target", 1e-8);
	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 1e-2;
	  update_regularizations(gamma);

	  g_alarm_duration_ma27=15;//seconds
	  g_max_memory_ma27=700;//Mbytes
	}
	break;
      case 4: 
	{
	  PDRestart=false;
	  solve2_safe_mode=true;
	  reallocate_nlp_solver();

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	    last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-3);
	  }

	  set_solver_option("mu_target", 1e-8);

	  printf("[warning] ContProbWithFixing K_idx=%d opt2 will switch to ma57 at try %d rank=%d\n", 
		 K_idx, n_solves+1, my_rank); 
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("ma57_automatic_scaling", "yes");
	  set_solver_option("ma57_small_pivot_flag", 1);

	  set_solver_option("linear_system_scaling", "mc19");
	  set_solver_option("linear_scaling_on_demand", "yes");

	  set_solver_option("tol", 1e-7);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);
	  
	  const double gamma = 1e-2;
	  update_regularizations(gamma);

	  g_alarm_duration_ma57=12;//seconds
	  g_max_memory_ma57=600;//Mbytes
	}
	break;
      default:
	{
	  PDRestart=false;
	  solve2_safe_mode=true;

	  if(last_opt_status!=User_Requested_Stop && last_opt_status!=Unrecoverable_Exception &&
	     last_opt_status!=Maximum_Iterations_Exceeded) {
	    assert(last_opt_status!=Solve_Succeeded && last_opt_status!=Solved_To_Acceptable_Level);
	    //restauration or something bad happened
	    vars_primal->set_start_to(*vars_ini);
	    set_solver_option("mu_init", 1.);
	  } else {
	    //we do a primal restart only since restarting duals didn't work (and tends to pose issues)
	    vars_primal->set_start_to(*vars_last);
	    set_solver_option("mu_init", 1e-3);
	  }
	  
	  set_solver_option("mu_target", 1e-8);
	  set_solver_option("linear_solver", "ma57"); 
	  set_solver_option("ma57_automatic_scaling", "yes");
	  set_solver_option("tol", 1e-6);
	  set_solver_option("mu_linear_decrease_factor", 0.4);
	  set_solver_option("mu_superlinear_decrease_power", 1.2);

	  const double gamma = 5e-2 + 0.1*n_solves;
	  update_regularizations(gamma);


	  g_alarm_duration_ma57=12;//seconds
	  g_max_memory_ma57=600;//Mbytes
	}
      }
      set_solver_option("print_user_options", "no");
      set_solver_option("print_level", 2);
      set_solver_option("sb","yes");

      set_solver_option("max_iter", 500);
      set_solver_option("acceptable_tol", 1e-3);
      set_solver_option("acceptable_constr_viol_tol", 1e-6);
      set_solver_option("acceptable_iter", 4);

      set_solver_option("fixed_variable_treatment", "relax_bounds");
      set_solver_option("honor_original_bounds", "yes");
      double relax_factor = std::min(1e-8, pow(10., 3*n_solves-16));
      set_solver_option("bound_relax_factor", relax_factor);
      double bound_push = std::min(1e-2, pow(10., 3*n_solves-12));
      set_solver_option("bound_push", bound_push);
      set_solver_option("slack_bound_push", bound_push); 
      double bound_frac = std::min(1e-2, pow(10., 3*n_solves-10));
      set_solver_option("bound_frac", bound_frac);
      set_solver_option("slack_bound_frac", bound_frac);
      
      set_solver_option("neg_curv_test_reg", "no"); //default yes ->ChiangZavala primal regularization


      monitor.safe_mode=solve2_safe_mode; 
      monitor.timer.restart();
      monitor.hist_tm.clear();
      monitor.user_stopped = false;
      
      if(data_sc.N_Bus.size()>8999) {
	if(solve2_safe_mode)
	  monitor.bailout_allowed=true;//! probably not needed when watching timeouts
	else 
	  monitor.bailout_allowed=false;
      }

      if(PDRestart) {
	opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
      } else {
	opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);
      }

      n_solves++;
      last_opt_status = OptProblem::optimization_status();

      hist_iter.push_back(number_of_iterations());
      hist_obj.push_back(this->obj_value);
      
      if(opt_ok) {
	done = true; 
      } else {
	if(monitor.user_stopped) {
	  assert(last_opt_status == User_Requested_Stop);
	  done = true; 
	} else {
	  //something bad happened, will resolve
	  printf("[warning] ContProbWithFixing K_idx=%d opt2 failed at try %d rank=%d time %g\n", 
		 K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
	}
      }

      if(n_solves>9) done = true;
      if(tmrec.measureElapsedTime()>800) {
	printf("[warning] ContProbWithFixing K_idx=%d opt2 taking too long on rank=%d; tries %d time %g\n", 
	       K_idx, my_rank, n_solves, tmrec.measureElapsedTime());
	done = true;
	bret = false;
      }

    } //end of outer while
#ifdef BE_VERBOSE
    string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbWithFixing K_idx=%d opt2 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
    fflush(stdout);
#endif
    get_solution_simplicial_vectorized(sln_solve2);
    obj_solve2 = this->obj_value;
    return bret;
  }

  bool ContingencyProblemWithFixing::optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, vector<double>& sln)
  {
    goTimer tmrec; tmrec.start();
    SCACOPFData& d = *data_K[0];

    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    bool bFirstSolveOK = do_solve1();
    f = this->obj_value;

#ifdef DEBUG
    if(bFirstSolveOK) {
      auto pgK = variable("p_g", d); assert(pgK!=NULL); 
      if(variable("delta", d)) {
	auto delta = variable("delta", d)->x[0]; 
	for(int i=0; i<pg0_partic_idxs.size(); i++) {
	  const double gen = pg0->x[pg0_partic_idxs[i]] + delta * data_sc.G_alpha[pg0_partic_idxs[i]];
	  if(gen >= data_sc.G_Pub[pg0_partic_idxs[i]]) 
	    assert(fabs(pgK->x[pgK_partic_idxs[i]] - data_sc.G_Pub[pg0_partic_idxs[i]]) < 9e-5);
	  if(gen <= data_sc.G_Plb[pg0_partic_idxs[i]]) 
	    assert(fabs(pgK->x[pgK_partic_idxs[i]] - data_sc.G_Plb[pg0_partic_idxs[i]]) < 9e-5);
	}
      }
    }
#endif

    
    if(variable("delta", d)) solv1_delta_optim = variable("delta", d)->x[0];
    else                     solv1_delta_optim = 0.;

    if(num_K_done<comm_size-1) num_K_done=comm_size-1;

    double K_avg_time_so_far = time_so_far  / num_K_done;

    if(K_avg_time_so_far > 0.91*2.) monitor.is_late=true;

    bool skip_2nd_solve = monitor.is_late;

    if(time_so_far < 0.085*2.*data_sc.K_Contingency.size()) skip_2nd_solve=false;

    if(this->obj_value>=5e5 && K_avg_time_so_far < 0.950*2.) skip_2nd_solve=false;
    if(this->obj_value>=1e6 && K_avg_time_so_far < 1.025*2.) skip_2nd_solve=false;

    if(!bFirstSolveOK) skip_2nd_solve=false;

    if(bFirstSolveOK && tmrec.measureElapsedTime()>800.) {
      skip_2nd_solve=true;
      printf("ContProbWithFixing K_idx=%d will exit prematuraly b/c first solves took long %g sec on rank=%d\n", 
	     K_idx, tmrec.measureElapsedTime(), my_rank);
    }

    if(this->obj_value>pen_threshold && !skip_2nd_solve) {

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
	if(pg0->x[data_sc.K_outidx[K_idx]]>1e-6 )  gen_K_diff = std::max(0., 1.1*poverall);
	else if(pg0->x[data_sc.K_outidx[K_idx]]<-1e-6)  gen_K_diff = std::min(0., poverall);
	else one_more_push_and_fix = false;
      }

      if(fabs(poverall)>1e-4) {// && d.K_ConType[0]!=SCACOPFData::kGenerator) {
	double rpa = fabs(pplus) / fabs(poverall);
	double rma = fabs(pminus) / fabs(poverall);

	//solv1_delta_optim=0.;//!

	if( (rpa>0.85 && rpa<1.15) || (rma>0.85 && rma <1.15) ) {	  
	  one_more_push_and_fix = true;
	  gen_K_diff = poverall;

	  //ignore small delta for transmission contingencies since they're really optimization noise
	  if(d.K_ConType[0]!=SCACOPFData::kGenerator && fabs(solv1_delta_optim)<1e-6) {
	    solv1_delta_optim=0.;
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
	  v->lb[i] = v->lb[i] - g_bounds_abuse; v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }{
	auto v = variable("q_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse; v->ub[i] = v->ub[i] + g_bounds_abuse;
	}
      }{
	auto v = variable("p_g", d);
	for(int i=0; i<v->n; i++) {
	  v->lb[i] = v->lb[i] - g_bounds_abuse; v->ub[i] = v->ub[i] + g_bounds_abuse;
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
	  sln = sln_solve1;
	  f = obj_solve1;
	} else {
	  printf("[warning][panic] ContProbWithFixing K_idx=%d opt1 and opt2 both failed on rank=%d\n", K_idx, my_rank);
	  sln = sln_solve1;
	  f = obj_solve1;
	}
      } else { //opt2_ok

	obj_solve2 = this->obj_value;
	if(obj_solve1<obj_solve2) {
	  sln = sln_solve1;
	  f = obj_solve1;
	} else {
	  sln = sln_solve2;
	  f = obj_solve2;
	}
	if(!bFirstSolveOK) sln = sln_solve2;
      }
      
      if(obj_solve2>pen_threshold) {
	double delta_optim = 0.;//
	if(variable("delta", d)) delta_optim = variable("delta", d)->x[0];
#ifdef BE_VERBOSE
	print_objterms_evals();
	//print_p_g_with_coupling_info(*data_K[0], pg0);
	printf("ContProbWithFixing K_idx=%d  pass 1-2 resulted in high pen delta=%g\n", K_idx, delta_optim);
#endif
      }  
    } else {
      sln = sln_solve1;
      f = obj_solve1;
      if(this->obj_value>pen_threshold && skip_2nd_solve) 
	printf("ContProbWithFixing K_idx=%d pass2 needed but not done - time restrictions\n", K_idx);
    }
      
    tmrec.stop();
#ifdef BE_VERBOSE
    //string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    //string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbWithFixing K_idx=%d optimize took %g sec rank=%d\n", K_idx, tmrec.getElapsedTime(), my_rank);
    fflush(stdout);
#endif
    return true;

  }




  bool ContingencyProblemWithFixing::warm_start_variable_from_basecase(OptVariables& v)
  {
    SCACOPFData& dK = *data_K[0];
    for(auto& b : v.vblocks) {
      
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
	assert(b->id.find("delta") != string::npos || 
	       b->id.find("AGC") != string::npos); //!remove agc later
	continue;
      }
      auto b0 = b0p->second; assert(b0);

      //printf("from %s to %s  sizes: %d %d\n", b0->id.c_str(), b->id.c_str(), b0->n, b->n);

      if(b0->n == b->n) {
	b->set_start_to(*b0);
      } else {
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
    return true;
  }
//   bool ContingencyProblemWithFixing::set_warm_start_from_basecase()
//   {
//     assert(false && "do not use");
//     SCACOPFData& dK = *data_K[0];
// if(false) {
    

//     if(!warm_start_variable_from_basecase(*vars_primal)) return false;

//     if(NULL==vars_duals_bounds_L || NULL==vars_duals_bounds_U || NULL==vars_duals_cons) {
//       //force allocation of duals
//       dual_problem_changed();
//     }
//     if(!warm_start_variable_from_basecase(*vars_duals_bounds_L)) return false;
//     if(!warm_start_variable_from_basecase(*vars_duals_bounds_U)) return false;
//     if(!warm_start_variable_from_basecase(*vars_duals_cons)) return false;
//     return true;

// } else {
//     variable("v_n", dK)->set_start_to(*v_n0);
//     variable("theta_n", dK)->set_start_to(*theta_n0);
//     variable("b_s", dK)->set_start_to(*b_s0);

//     if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
//       auto p_gK = variable("p_g", dK);
//       for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
// 	p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
//       }
//       for(int i=0; i<pg0_partic_idxs.size(); i++) {
// 	p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
//       }
//       p_gK->providesStartingPoint = true;
      
//       auto q_gK = variable("q_g", dK);
//       for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
// 	q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
//       }
//       for(int i=0; i<pg0_partic_idxs.size(); i++) {
// 	q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
//       }
//       q_gK->providesStartingPoint = true;
      
//     } else {
// #ifdef DEBUG
//       assert(variable("p_g", dK)->n == p_g0->n);
//       assert(variable("q_g", dK)->n == q_g0->n);
// #endif
//       variable("p_g", dK)->set_start_to(*p_g0);
//       variable("q_g", dK)->set_start_to(*q_g0);
//     }
// }

//   }

  bool ContingencyProblemWithFixing::
  add_cons_AGC_simplified(SCACOPFData& dB, 
			  const std::vector<int>& idxs0_AGC_particip, 
			  const std::vector<int>& idxsK_AGC_particip,
			  OptVariablesBlock* pg0)
  {
    assert(idxs0_AGC_particip.size()==idxsK_AGC_particip.size());
    assert(variable("p_g", dB));

    if(idxs0_AGC_particip.size()==0) {
      printf("[warning] ContingencyProblemWithFixing add_cons_AGC_simplified: NO gens participating !?! in contingency %d\n", dB.id);
      return true;
    }

    OptVariablesBlock* deltaK = variable("delta", dB);
    if(deltaK==NULL) {
      deltaK = new OptVariablesBlock(1, var_name("delta", dB));
      append_variables(deltaK);
      deltaK->set_start_to(0.);
    }

    auto cons = new AGCSimpleCons_pg0Fixed(con_name("AGC_simple_fixedpg0", dB), idxs0_AGC_particip.size(), 
					   pg0, variable("p_g", dB), deltaK, 
					   idxs0_AGC_particip, idxsK_AGC_particip, 
					   data_sc.G_alpha);
    append_constraints(cons);

    return true;
  }

  // Make up for excess or surplus 'P' in the contingency by ramping up or down AGC generation.
  // Iterate over gens and fix them when the bounds are hit, till 'P' can be apparently made up for
  // or it cannot because there is not enough generation (return false in this case).
  
  // Needed ramp : delta1 * sum { alpha[i] : i in particip } = P  (needed delta)
  // Blocking  delta
  // delta2 = min { (Pub[i]-pg0[i])/alpha[i] : i in particip }  if P>0 (in which case delta>0)
  //        or
  //        = max { (Plb[i]-pg0[i])/alpha[i] : i in particip }  if P<0 (in which case delta<0)
  // if delta1<=delta2 it appears to be enough generation
  // else 
  //   fix pgK[i] to upper (P>0) or lower bounds (P<0) for those 'i' that block
  //   decrease P accordingly and repeat
  bool ContingencyProblemWithFixing::
  push_and_fix_AGCgen(SCACOPFData& dB, const double& P_in, const double& delta_in, 
		      std::vector<int>& idxs0_AGCparticip, std::vector<int>& idxsK_AGCparticip,
		      std::vector<int>& idxs0_nonparticip, std::vector<int>& idxsK_nonparticip,
		      OptVariablesBlock* pg0, OptVariablesBlock* pgK,
		      std::vector<double>& Plb, std::vector<double>& Pub, std::vector<double>& alpha,
		      double& delta, double& delta1, double& delta2, double& delta_lb, double& delta_ub, 
		      double& P_out)
  {
    P_out = P_in;

    double sum, dist;
    delta1=delta2=0.;
    const bool Pispos = (P_in > 0);

    if(fabs(P_in) < 1e-12) {
      printf("[warning] push_and_fix_AGCgen P_in came in too small; K_idx=%d P_in=%g delta_in=%g rank=%d\n",
	     K_idx, P_in, delta_in, my_rank);
      return true;
    }
    
#ifdef BE_VERBOSE
    printf("push_and_fix_AGCgen K_idx=%d P_in=%g delta_in=%g rank=%d\n", K_idx, P_in, delta_in, my_rank);
#endif
    if(!Pispos && delta_in>0) printf("K_idx=%d !!!!!!!!\n", K_idx);

    if(Pispos) { delta_lb=0.;     delta_ub=1e+20; assert(delta_in>=0); }
    else { 
      delta_lb=-1e+20; delta_ub=0.;    
      if(P_in<0) assert(delta_in<=0); 
  }

    if(idxs0_AGCparticip.size()==0) P_out=0.; //force exit

    if(fabs(P_out)<1e-6) { delta=delta_in; delta1=delta2=0.; return true; }

    while(true) {
      //delta1 = P / sum { alpha[i] : i in particip };
      sum=0.; for(int& idx : idxs0_AGCparticip) { assert(alpha[idx]>=1e-6); sum += alpha[idx]; }; assert(sum>=1e-6);
      
      delta1 = P_out/sum;
      
      //compute delta2 = blocking
      
      if(Pispos) {
	delta2 = 1e+20;
	for(int it=0; it<idxs0_AGCparticip.size(); it++) {
	  const int &i0=idxs0_AGCparticip[it], &iK = idxsK_AGCparticip[it];
	  const double r = (Pub[i0]-pgK->x[iK])/alpha[i0]; 
	  if(r<delta2) delta2=r; 
	}
	assert(delta>=0);
      } else { assert(P_out<0);
	delta2 = -1e+20;
	for(int it=0; it<idxs0_AGCparticip.size(); it++) {
	  const int &i0=idxs0_AGCparticip[it], &iK = idxsK_AGCparticip[it];
	  const double r = (Plb[i0]-pgK->x[iK])/alpha[i0]; 
	  if(r>delta2) delta2=r; 
	}
	assert(delta<=0);
      }

      //printf("aaaaaa delta2=%g delta1=%g P=%g\n", delta2, delta1, P_out); fflush(stdout);

      //if( (Pispos && (delta1 < delta2 + 1e-6)) || (!Pispos && (delta1 > delta2 - 1e-6)))  {
      if( (Pispos && (delta1 < delta2 )) || (!Pispos && (delta1 > delta2 )))  {
	//enough to cover for P
	delta = delta1 = delta_in+delta1; 
	delta2 = delta_in+delta2; 
	if(Pispos) { delta_lb = delta_in; delta_ub=1e+20; } 
	else       { delta_lb = -1e+20;   delta_ub=delta_in; } 
	return true;

      } else {
	assert(delta2 > -1e+20 && delta2 < +1e+20);
	assert(idxs0_AGCparticip.size()==idxsK_AGCparticip.size());
	assert(idxs0_nonparticip.size()==idxsK_nonparticip.size());
	//how much ramping can be done at this iteration? extract it from 'P'
	const double ramp=sum*delta2; P_out = P_out - ramp;
        if(Pispos) { assert(P_out>0); } else { assert(P_out<0); }

	//find blocking G indexes
	vector<int> idxs0_to_remove, idxsK_to_remove; //participating idxs to remove
	if(Pispos) { assert(delta2>=0);
	  for(int it=0; it<idxs0_AGCparticip.size(); it++) {
	    const int &i0=idxs0_AGCparticip[it], &iK = idxsK_AGCparticip[it];
	    dist = Pub[i0] - pgK->x[iK]; assert(dist>=0);

	    assert(dist>=alpha[i0]*delta2- 1e-6);
	    if( fabs(dist - alpha[i0]*delta2) < 1e-5 ) { 
	      idxs0_to_remove.push_back(i0); idxsK_to_remove.push_back(iK); 
	      assert(pgK->ub[iK]==Pub[i0]); assert(pgK->lb[iK]==Plb[i0]);
	      pgK->x[iK]  = Pub[i0];
	      pgK->lb[iK] = Pub[i0]-1e-6;
	      pgK->ub[iK] = Pub[i0]+1e-6;
	    }
	  }
	} else { assert(P_out<0); assert(delta2<=0);
	  for(int it=0; it<idxs0_AGCparticip.size(); it++) {
	    const int &i0=idxs0_AGCparticip[it], &iK = idxsK_AGCparticip[it];
	    dist = pgK->x[i0] - Plb[i0]; assert(dist>=0);
	    //if(dist < 0-alpha[i0]*delta) 
	    if(dist < 0-alpha[i0]*delta2 - 1e-6)
	      printf("---- dist=%g alpha[i0=%d]=%g delta2=%g [%12.5e] \n", dist, i0, alpha[i0], delta2, dist + alpha[i0]*delta2);
	    fflush(stdout);
	    assert(dist >= 0-alpha[i0]*delta2 - 1e-6);
	    if( fabs(dist + alpha[i0]*delta2) < 1e-5) { 
	      idxs0_to_remove.push_back(i0); idxsK_to_remove.push_back(iK); 
	      assert(pgK->ub[iK]==Pub[i0]); assert(pgK->lb[iK]==Plb[i0]);
	      pgK->x[iK]  = Plb[i0];
	      pgK->lb[iK] = Plb[i0]-1e-6;
	      pgK->ub[iK] = Plb[i0]+1e-6;
	    }
	  }
	} // end of if / else Pispos 

	//printvec(idxs0_to_remove, "idx0 at upper");
	//printvec(idxsK_to_remove, "idxK at upper");

	bool ret;
	for(int& i0 : idxs0_to_remove) { 
	  ret = erase_elem_from(idxs0_AGCparticip, i0); assert(ret); 
	  idxs0_nonparticip.push_back(i0);
	}
	for(int& iK : idxsK_to_remove) { 
	  ret = erase_elem_from(idxsK_AGCparticip, iK); assert(ret); 
	  idxsK_nonparticip.push_back(iK);
	}
	assert(idxs0_AGCparticip.size()==idxsK_AGCparticip.size());
	assert(idxs0_nonparticip.size()==idxsK_nonparticip.size());

	if(idxs0_AGCparticip.size()==0) {
	  //we maxed out
	  delta = delta2 = delta_in+delta2; 
	  delta1 = delta_in+delta1; 
	  if(Pispos) { delta_lb = delta_in; delta_ub=1e+20; } 
	  else       { delta_lb = -1e+20;   delta_ub=delta_in; } 

	  return fabs(P_out)<1e-6;
	}
	
      }
    }
    return false;
  }
  void ContingencyProblemWithFixing::
  add_cons_pg_nonanticip_using(OptVariablesBlock* pg0,
			       const std::vector<int>& idxs_pg0_nonparticip, 
			       const std::vector<int>& idxs_pgK_nonparticip)
  {
    SCACOPFData& dK = *data_K[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = variable("p_g", dK);
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
  void ContingencyProblemWithFixing::estimate_active_power_deficit(double& p_plus, double& p_minus, double& p_overall)
  {
    p_plus = p_minus = p_overall = 0.;
    auto pf_p_bal = dynamic_cast<PFActiveBalance*>(constraint("p_balance",*data_K[0]));
    OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
    int n = data_sc.N_Bus.size(); assert(pslacks_n->n == 2*n);
    for(int i=n; i<2*n; i++) { p_plus  += pslacks_n->x[i]; p_overall += pslacks_n->x[i]; }
    for(int i=0; i<n; i++)   { p_minus -= pslacks_n->x[i]; p_overall -= pslacks_n->x[i]; }

  }

  // bool ContingencyProblemWithFixing::do_fixing_for_AGC(const double& smoothing, bool fixVoltage, 
  // 						       OptVariablesBlock* pgk, OptVariablesBlock* delta)
  // {
  //   SCACOPFData& d = *data_K[0];
  //   assert(delta); assert(delta->n==1);
  //   const double deltak = delta->x[0];

  //   vector<int> Gk_simplified;

  //   double rtol  = sqrt(smoothing);
  //   double rtolp = sqrt(smoothing)/10.;

  //   for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
  //     const int iK=pgK_nonpartic_idxs[i];
  //     const int i0=pg0_nonpartic_idxs[i];

  //     double gen_band = d.G_Pub[iK] - d.G_Plb[iK];
  //     assert(d.G_Pub[iK] == data_sc.G_Pub[i0]); assert(d.G_Plb[iK] == data_sc.G_Plb[i0]);
  //     assert(gen_band>=1e-6);

  //     double dist_lower = (pgk->x[iK]-d.G_Plb[iK])/gen_band;
  //     double dist_upper = (d.G_Pub[iK]-pgk->x[iK])/gen_band;
	
  //     assert(dist_lower<0); assert(dist_upper<0);

  //     double distp = (pgk->x[iK] - p_g0->x[i0] - d.G_alpha[i0]*deltak)/gen_band;

  //     if(dist_lower > rtol && dist_upper > rtol) {
  // 	//p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
  // 	//
  // 	//!@constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
  // 	printf("  inside, enf. eq\n");
  //     } else if(dist_lower <= rtol) {
  // 	if(distp > rtolp) { // && deltaIsFeas) {
  // 	  //strict complementarity and delta is feasible -> fixing is clear
  // 	  pgk->ub[iK] = d.G_Plb[iK];
  // 	  printf("  fixed p_gk at lower\n");
  // 	} else {
  // 	  //degenerate complementarity or delta is not feasible , enforce equality

  // 	  //p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
	  
  //         //! @constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
  // 	  Gk_simplified.push_back(i0);
  // 	  printf("  p_gk at lower but enforcing eq.; degen compl\n");
  // 	}  
  //     } else {
  // 	//#dist_upper <= rtol
  // 	if(distp < -rtolp) { // && deltaIsFeas
  // 	  //strict complementarity and delta is feasible, fixing is clear
  // 	  pgk->lb[iK] = d.G_Pub[iK];
  // 	  printf("  fixed p_gk at upper\n");
  // 	} else {
  // 	  //degenerate complementarity or delta is infeasible, enforce equality
  // 	  //p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
  // 	  //!@constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
  // 	  Gk_simplified.push_back(i0);
  // 	  printf("  p_gk at upper but enforcing eq.; degen compl\n");
  // 	}
  //     }
  //   }
  

  //   if(Gk_simplified.size() > 0) {
  //     //!add_cons_AGC_simplified(d, Gk_simplified);
  //   }

  //   return true;
  // }

  bool ContingencyProblemWithFixing::do_qgen_fixing_for_PVPQ(OptVariablesBlock* vnk, OptVariablesBlock* qgk)
  {
    SCACOPFData& d = *data_K[0];

    //(aggregated) non-fixed q_g generator ids at each node/bus
    // with PVPQ generators that have at least one non-fixed q_g 
    vector<vector<int> > idxs_gen_agg;
    //bus indexes that have at least one non-fixed q_g
    vector<int> idxs_bus_pvpq;
    //aggregated lb and ub on reactive power at each PVPQ bus
    vector<double> Qlb, Qub;
    int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;
    
    get_idxs_PVPQ(d, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
		  nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);
    assert(idxs_gen_agg.size() == idxs_bus_pvpq.size());
    assert(vnk->n == v_n0->n);

    for(int itpvpq=0; itpvpq<idxs_bus_pvpq.size(); itpvpq++) {
      const int busidx = idxs_bus_pvpq[itpvpq];
      double vdev = (vnk->x[busidx]-v_n0->x[busidx]) / std::max(1., fabs(v_n0->x[busidx]));
      double Qlbn=0., Qubn=0., qapprox_nk=0.;
      for(int gidx : idxs_gen_agg[itpvpq]) {
	Qlbn += d.G_Qlb[gidx];
	Qubn += d.G_Qub[gidx];
	qapprox_nk += qgk->x[gidx];
      }

      double gen_band = Qubn - Qlbn; 
      double dist_lower = (qapprox_nk - Qlbn)/gen_band; 
      double dist_upper = (Qubn - qapprox_nk)/gen_band; 
      

      //if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
      //printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
      //       busidx, gen_band, dist_lower,  dist_upper,
      //       Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);

      assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
      assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

      const double rtol = 1e-2, rtolv=1e-3;
      if(dist_lower > rtol && dist_upper > rtol) {
	//inside -> fix v_nk

	//!	assert(fabs(vnk->ub[busidx] - vnk->lb[busidx])<1e-8);
	//!assert(fabs(vnk->ub[busidx] - vnk->x[busidx]) <1e-8);

      } else if(dist_lower <= rtol) {
	if(vdev >= rtolv) {
	  //strict complementarity -> fix q_gk     to Qlb               
          //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

	  vnk->lb[busidx] = v_n0->x[busidx] - g_bounds_abuse;; 
	  vnk->ub[busidx] = data_sc.N_EVub[busidx] + g_bounds_abuse;
	  for(int g : idxs_gen_agg[itpvpq]) {
	    qgk->lb[g] =  d.G_Qlb[g] - g_bounds_abuse;
	    qgk->ub[g] =  d.G_Qlb[g] + g_bounds_abuse;
	  }
	}  else {
	  //degenerate complementarity 
	  
	  //if(fixVoltage) {
	  //  printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will fix voltage\n", busidx); 
	  //  vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
	  //} else {
	  //printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will put q_g close to lower\n", busidx); 
	  vnk->lb[busidx] = v_n0->x[busidx] - g_bounds_abuse; 
	  vnk->ub[busidx] = data_sc.N_EVub[busidx] + g_bounds_abuse;
	  for(int g : idxs_gen_agg[itpvpq]){
	    qgk->lb[g] =  d.G_Qlb[g] - g_bounds_abuse;
	    qgk->ub[g] =  d.G_Qlb[g] + g_bounds_abuse;
	  }
	  
	}
      } else { // if(dist_upper <= rtol)
	assert(dist_upper <= rtol);
	if(vdev <= - rtolv) {
	  //strict complementarity -> fix q_gk to Qub 
	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
	  vnk->ub[busidx] = v_n0->x[busidx] + g_bounds_abuse; 
	  vnk->lb[busidx] = data_sc.N_EVlb[busidx] - g_bounds_abuse;

	  for(int g : idxs_gen_agg[itpvpq]) {
	    qgk->lb[g] = d.G_Qub[g] - g_bounds_abuse;
	    qgk->ub[g] = d.G_Qub[g] + g_bounds_abuse;
	  }
	} else {
	  //degenerate complementarity 
	  
	  //if(fixVoltage) 
	  //  {
	  //    printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will fix voltage\n", busidx); 
	  //    vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
	  //  } else {
	  //printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will put q_g to the upper\n", busidx); 
	  vnk->ub[busidx] = v_n0->x[busidx] + g_bounds_abuse; 
	  vnk->lb[busidx] = data_sc.N_EVlb[busidx] - g_bounds_abuse;
	  for(int g : idxs_gen_agg[itpvpq]){
	    qgk->lb[g] = d.G_Qub[g] - g_bounds_abuse;
	    qgk->ub[g] = d.G_Qub[g] + g_bounds_abuse;
	  }
	}
      }
      
    }

    return true;
  }

//   bool ContingencyProblemWithFixing::do_solve1_old()
//   {
//     g_solve_watch_ma57=true;
//     g_alarm_duration_ma57=3;//seconds
//     g_max_memory_ma57=200;//Mbytes
//     g_my_rank_ma57=my_rank;
//     g_my_K_idx_ma57=K_idx;



//     goTimer tmrec; tmrec.start();
//     vector<int> hist_iter, hist_obj;
//     bool bret = true, done = false; 
//     int n_solves=0; 
//     while(!done) {

//       double mu_init; bool opt_ok=false;

//       if(n_solves>2) safe_mode = true;

//       monitor.safe_mode=safe_mode; 
//       monitor.timer.restart();
//       monitor.hist_tm.clear();
//       monitor.user_stopped = false;

//        if(n_solves==2) {
// 	reallocate_nlp_solver();

// 	vars_primal->set_start_to(*vars_last);

// 	set_solver_option("linear_solver", "ma27"); 
// 	printf("[warning] ContProbWithFixing K_idx=%d opt1 will use ma27 for try %d\n", K_idx, n_solves+1); 
//       } else {
// 	set_solver_option("linear_solver", "ma57"); 
//       }

//      if(n_solves==1) {
// 	set_solver_option("ma57_pivot_order", 4); //enforce metis
//       } else {
// 	set_solver_option("ma57_pivot_order", 5); //automatic
//       }

//       if(n_solves>=3) {
// 	set_solver_option("ma57_pivot_order", 4); 
// 	set_solver_option("ma57_automatic_scaling", "yes");
//       } else {
// 	set_solver_option("ma57_automatic_scaling", "no");
//       }

//       set_solver_option("print_user_options", "yes");

//       set_solver_option("sb","yes");
//       set_solver_option("print_level", 5);
//       set_solver_option("max_iter", 300);
//       set_solver_option("acceptable_tol", 1e-3);
//       set_solver_option("acceptable_constr_viol_tol", 1e-6);
//       set_solver_option("acceptable_iter", 5);
      

//       set_solver_option("neg_curv_test_reg", "yes"); //default yes ->ChiangZavala primal regularization
//       set_solver_option("linear_system_scaling", "mc19");
//       set_solver_option("linear_scaling_on_demand", "yes");

//       if(data_sc.N_Bus.size()>8999) {
// 	if(safe_mode)
// 	  monitor.bailout_allowed=true;//! probably not needed when watching timeouts
// 	else 
// 	  monitor.bailout_allowed=false;
// 	set_solver_option("mu_init", 1e-1);
//       } else {
// 	set_solver_option("mu_init", 1e-4);
//       }

//       //if(n_solves>0) 
// 	set_solver_option("fixed_variable_treatment", "relax_bounds");
//       double relax_factor = std::min(1e-8, pow(10., 3*n_solves-16));
//       set_solver_option("bound_relax_factor", relax_factor);
 

//       double bound_push = std::min(1e-2, pow(10., 3*n_solves-12));
//       set_solver_option("bound_push", bound_push);
//       set_solver_option("slack_bound_push", bound_push);

//       double bound_frac = std::min(1e-2, pow(10., 3*n_solves-10));
//       set_solver_option("bound_frac", bound_frac);
//       set_solver_option("slack_bound_frac", bound_frac);

//       if(n_solves>=1) {
// 	set_solver_option("tol", 1e-7);
// 	set_solver_option("mu_linear_decrease_factor", 0.4);
// 	set_solver_option("mu_superlinear_decrease_power", 1.2);
//       }

//       //  opt_ok = OptProblem::optimize("ipopt");
//       if(n_solves==0) {
//        	//medium and small problems default primal-dual restart
//        	//if(data_sc.N_Bus.size()<9000)	  

// 	opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);

// 	  //else 
//        	  //opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);	
//       } else {
//        	if(n_solves<=2)
// 	  vars_primal->set_start_to(*vars_last);
//        	else 
//        	  vars_primal->set_start_to(*vars_ini);

//        	opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);
//       }

//       n_solves++;

//       hist_iter.push_back(number_of_iterations());
//       hist_obj.push_back(this->obj_value);

//       if(opt_ok) {
// 	done = true;
//       } else {
// 	if(monitor.user_stopped) {
// 	  done = true;
// 	} else {
// 	  //something bad happened, will resolve
// 	  printf("[warning] ContProbWithFixing K_idx=%d opt1 failed at try %d rank=%d time %g\n", 
// 		 K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
// 	}
//       }
      
//       if(n_solves>9) done = true;
//       if(tmrec.measureElapsedTime()>800) {
// 	printf("[warning] ContProbWithFixing K_idx=%d opt1 taking too long on rank=%d; tries %d time %g\n", 
// 	       K_idx, my_rank, n_solves, tmrec.measureElapsedTime());
// 	done = true;
// 	bret = false;
//       }
      
//     } //end of outer while
// #ifdef BE_VERBOSE
//     string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
//     string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
//     printf("ContProbWithFixing K_idx=%d opt1 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
// 	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
//     fflush(stdout);
// #endif
//     get_solution_simplicial_vectorized(sln_solve1);
//     obj_solve1 = this->obj_value;
//     return bret;
//   }
//   //
//   // solve2
//   //
//   bool ContingencyProblemWithFixing::do_solve2_old(bool bFirstSolveOK)
//   {
//     goTimer tmrec; tmrec.start();

// #ifdef GOLLNLP_FAULT_HANDLING
//     if(bFirstSolveOK)
//       vars_ini->set_start_to(*vars_primal);
// #endif

//     vector<int> hist_iter, hist_obj;
//     bool bret = true, done = false; 
//     int volatile n_solves=0; 
//     while(!done) {
//       double mu_init; bool opt_ok=false;

//       if(n_solves>2) safe_mode = true;

//       monitor.safe_mode=safe_mode; 
//       monitor.timer.restart();
//       monitor.hist_tm.clear();
//       monitor.user_stopped = false;

//       if(n_solves==2) {
// 	reallocate_nlp_solver();
	
// 	vars_primal->set_start_to(*vars_last);
	
// 	set_solver_option("linear_solver", "ma27"); 
// 	printf("[warning] ContProbWithFixing K_idx=%d opt1 will use ma27 for try %d\n", K_idx, n_solves+1); 
//       } else {
// 	set_solver_option("linear_solver", "ma57"); 
//       }

//      if(n_solves==1) {
// 	set_solver_option("ma57_pivot_order", 4); //enforce metis
//       } else {
// 	set_solver_option("ma57_pivot_order", 5); //automatic
//       }

//       if(n_solves>=3) {
// 	set_solver_option("ma57_pivot_order", 4); 
// 	set_solver_option("ma57_automatic_scaling", "yes");
//       } else {
// 	set_solver_option("ma57_automatic_scaling", "no");
//       }

//       set_solver_option("max_iter", 500);
//       set_solver_option("acceptable_tol", 1e-3);
//       set_solver_option("acceptable_constr_viol_tol", 1e-6);
//       set_solver_option("acceptable_iter", 5);
      

//       if(data_sc.N_Bus.size()>8999) {
// 	if(safe_mode)
// 	  monitor.bailout_allowed=true;//! probably not needed when watching timeouts
// 	else 
// 	  monitor.bailout_allowed=false;
// 	set_solver_option("mu_init", 1e-1);
// 	//opt_ok = OptProblem::optimize("ipopt");
//       } else {
// 	set_solver_option("mu_init", 1e-4);
// 	//opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
//       }
//       double relax_factor = std::min(1e-8, pow(10., 3*n_solves-16));
//       set_solver_option("bound_relax_factor", relax_factor);
//       if(n_solves>0) 
// 	set_solver_option("fixed_variable_treatment", "relax_bounds");

//       double bound_push = std::min(1e-2, pow(10., 3*n_solves-12));
//       set_solver_option("bound_push", bound_push);
//       set_solver_option("slack_bound_push", bound_push);

//       double bound_frac = std::min(1e-2, pow(10., 3*n_solves-10));
//       set_solver_option("bound_frac", bound_frac);
//       set_solver_option("slack_bound_frac", bound_frac);

//       set_solver_option("mu_linear_decrease_factor", 0.5);
//       set_solver_option("mu_superlinear_decrease_power", 1.2);

//       if(n_solves>=1) { 
// 	set_solver_option("tol", 1e-7);
// 	set_solver_option("mu_linear_decrease_factor", 0.4);
// 	set_solver_option("mu_superlinear_decrease_power", 1.2);
//       }

//       if(bFirstSolveOK) {
// 	//medium and small problems default primal-dual restart
// 	opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
//       } else {
// 	if(n_solves<=2)
// 	  vars_primal->set_start_to(*vars_last);
//        	else 
//        	  vars_primal->set_start_to(*vars_ini);
	
// 	opt_ok = OptProblem::reoptimize(OptProblem::primalRestart);
//       }

//       n_solves++;
//       hist_iter.push_back(number_of_iterations());
//       hist_obj.push_back(this->obj_value);
      
//       if(opt_ok) {
// 	done = true; 
//       } else {
// 	if(monitor.user_stopped) {
// 	  done = true; 
// 	} else {
// 	  //something bad happened, will resolve
// 	  printf("[warning] ContProbWithFixing K_idx=%d opt2 failed at try %d rank=%d time %g\n", 
// 		 K_idx, n_solves, my_rank, tmrec.measureElapsedTime()); 
// 	}
//       }

//       if(n_solves>9) done = true;
//       if(tmrec.measureElapsedTime()>800) {
// 	printf("[warning] ContProbWithFixing K_idx=%d opt2 taking too long on rank=%d; tries %d time %g\n", 
// 	       K_idx, my_rank, n_solves, tmrec.measureElapsedTime());
// 	done = true;
// 	bret = false;
//       }

//     } //end of outer while
// #ifdef BE_VERBOSE
//     string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
//     string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
//     printf("ContProbWithFixing K_idx=%d opt2 took %g sec - iters %s objs %s tries %d on rank=%d\n", 
// 	   K_idx, tmrec.measureElapsedTime(), sit.c_str(), sobj.c_str(), n_solves, my_rank);
//     fflush(stdout);
// #endif
//     get_solution_simplicial_vectorized(sln_solve2);
//     obj_solve2 = this->obj_value;
//     return bret;
//   }




  // bool ContingencyProblemWithFixing::do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
  // 							OptVariablesBlock* vnk, OptVariablesBlock* qgk)
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
      

  //     if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
  // 	printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
  // 	       busidx, gen_band, dist_lower,  dist_upper,
  // 	       Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);

  //     assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
  //     assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

  //     const double rtol = 1e-2, rtolv=1e-3;
  //     if(dist_lower > rtol && dist_upper > rtol) {
  // 	//inside -> fix v_nk
  // 	if(fabs(vdev)>=sqrt(smoothing)) {
  // 	  printf("[warning] sum(q_gk) in is inside the bounds (dist_lower,dist_upper)=(%g,%g), "
  // 		 "but volt dev is large %g. %d gens at the busidx %d.", 
  // 		 dist_lower, dist_upper, vdev, idxs_gen_agg[itpvpq].size(), busidx);
  // 	}
  // 	//printf("  fixing v_nk\n");
  // 	vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
  //       //for g=Gnk
  // 	//    q_gk[g] = @variable(m, lower_bound=G[:Qlb][g], upper_bound=G[:Qub][g], start=qapprox_gk[g])
  // 	//end

  //     } else if(dist_lower <= rtol) {
  // 	if(vdev >= rtolv) {
  // 	  //strict complementarity -> fix q_gk     to Qlb               
	  
  //         //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

  // 	  vnk->lb[busidx] = v_n0->x[busidx]; 
  // 	  for(int g : idxs_gen_agg[itpvpq])
  // 	    qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
  // 	}  else {
  // 	  //degenerate complementarity 
	  
  // 	  if(fixVoltage) {
  // 	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will fix voltage\n", busidx); 
  // 	    vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
  // 	  } else {
  // 	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will put q_g close to lower\n", busidx); 
  // 	    vnk->lb[busidx] = v_n0->x[busidx]; 
  // 	    for(int g : idxs_gen_agg[itpvpq])
  // 	      qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
  // 	  }
  // 	}
  //     } else { // if(dist_upper <= rtol)
  // 	assert(dist_upper <= rtol);
  // 	if(vdev <= - rtolv) {
  // 	  //strict complementarity -> fix q_gk to Qub 
  // 	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
  // 	  vnk->ub[busidx] = v_n0->x[busidx]; 
  // 	    for(int g : idxs_gen_agg[itpvpq])
  // 	      qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
  // 	} else {
  // 	  //degenerate complementarity 
	  
  // 	  if(fixVoltage) 
  // 	    {
  // 	      printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will fix voltage\n", busidx); 
  // 	      vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
  // 	    } else {
  // 	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will put q_g to the upper\n", busidx); 
  // 	    vnk->ub[busidx] = v_n0->x[busidx]; 
  // 	    for(int g : idxs_gen_agg[itpvpq])
  // 		qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
  // 	  }
  // 	}
  //     }
      
  //   }

  //   return true;
  // }

  // bool ContingencyProblemWithFixing::attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
  // 							     OptVariablesBlock* vnk, OptVariablesBlock* qgk)
  // {
  //   SCACOPFData& d = *data_K[0];
  //   bool needs_another_fixing=false;

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
      

  //     if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
  // 	printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
  // 	       busidx, gen_band, dist_lower,  dist_upper,
  // 	       Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);

  //     assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
  //     assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

  //     const double rtol = 1e-2, rtolv=1e-3;
  //     if(dist_lower > rtol && dist_upper > rtol) {
  // 	//inside -> fix v_nk
  // 	if(fabs(vdev)>=sqrt(smoothing)) {
  // 	  printf("[warning] sum(q_gk) in is inside the bounds (dist_lower,dist_upper)=(%g,%g), "
  // 		 "but volt dev is large %g. %d gens at the busidx %d.", 
  // 		 dist_lower, dist_upper, vdev, idxs_gen_agg[itpvpq].size(), busidx);
  // 	}
  // 	//printf("  fixing v_nk\n");
  // 	//!vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
  //       //for g=Gnk
  // 	//    q_gk[g] = @variable(m, lower_bound=G[:Qlb][g], upper_bound=G[:Qub][g], start=qapprox_gk[g])
  // 	//end

  //     } else if(dist_lower <= rtol) {
  // 	if(vdev >= rtolv) {
  // 	  //strict complementarity -> fix q_gk     to Qlb               
	  
  //         //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

  // 	  //!vnk->lb[busidx] = v_n0->x[busidx]; 
  // 	  //!for(int g : idxs_gen_agg[itpvpq])
  // 	  //!  qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
  // 	}  else {
  // 	  //degenerate complementarity 
  // 	  needs_another_fixing=true;
  // 	  if(fixVoltage) {
  // 	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d;\n", busidx); 
  // 	    //!!vnk->lb[busidx] = v_n0->x[busidx];
  // 	    //!1for(int g : idxs_gen_agg[itpvpq])
  // 	    //!!  qgk->ub[g] = qgk->x[g];
  // 	  } else {
  // 	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d\n", busidx); 
  // 	    //!!vnk->lb[busidx] = v_n0->x[busidx]; 
  // 	    //!!for(int g : idxs_gen_agg[itpvpq])
  // 	    //!!  qgk->ub[g] = qgk->x[g];
  // 	  }
  // 	}
  //     } else { // if(dist_upper <= rtol)
  // 	assert(dist_upper <= rtol);
  // 	if(vdev <= - rtolv) {
  // 	  //strict complementarity -> fix q_gk to Qub 
  // 	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
  // 	  //!vnk->ub[busidx] = v_n0->x[busidx]; 
  // 	  //!for(int g : idxs_gen_agg[itpvpq])
  // 	  //!  qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
  // 	} else {
  // 	  //degenerate complementarity 
  // 	  needs_another_fixing=true;
  // 	  if(fixVoltage) {
  // 	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d\n", busidx); 
  // 	    //!!vnk->ub[busidx] = v_n0->x[busidx];
  // 	    //!!for(int g : idxs_gen_agg[itpvpq])
  // 	//!!	qgk->lb[g] = qgk->x[g];
  // 	  } else {
  // 	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d\n", busidx); 
  // 	    //!!vnk->ub[busidx] = v_n0->x[busidx]; 
  // 	    //!!for(int g : idxs_gen_agg[itpvpq])
  // 	//!!	qgk->lb[g] = qgk->x[g];
  // 	  }
  // 	}
  //     }
      
  //   }

  //   return needs_another_fixing;
  // }

//   bool ContingencyProblemWithFixing::optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, vector<double>& sln)
//   {
//     goTimer tmrec; tmrec.start();
//     SCACOPFData& d = *data_K[0];

//     assert(p_g0 == pg0); assert(v_n0 == vn0);
//     p_g0 = pg0; v_n0=vn0;

//     bool bFirstSolveOK=true; 
//     vector<int> hist_iter, hist_obj;

//     bFirstSolveOK = do_solve1();

//     double mu_init; bool opt_ok=false;

//     monitor.safe_mode=false; 
//     monitor.timer.restart();
//     monitor.hist_tm.clear();
//     set_solver_option("max_iter", 250);

//     //default_primal_start();
//     //print_summary();

//     if(data_sc.N_Bus.size()>8999) {
//       monitor.bailout_allowed=true;
//       set_solver_option("mu_init", 1e-1);
//       opt_ok = OptProblem::optimize("ipopt");
//     } else {
//       set_solver_option("mu_init", 1e-4);
//       opt_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
//     }

//     if(!opt_ok) {
//       if(!monitor.user_stopped) {
// 	monitor.user_stopped=false;
// 	monitor.safe_mode=true;
// 	monitor.bailout_allowed=false;
// 	monitor.timer.restart();
// 	monitor.hist_tm.clear();
// 	printf("[warning] ContProbWithFixing K_idx=%d opt1 failed\n", K_idx); 
// 	bFirstSolveOK=false;
// 	hist_iter.push_back(number_of_iterations());
// 	hist_obj.push_back(this->obj_value);
	
// 	set_solver_option("mu_init", 1e-1);
// 	set_solver_option("max_iter", 300);

// 	set_solver_option("bound_relax_factor", 1e-8);
// 	set_solver_option("bound_push", 0.01);
// 	set_solver_option("slack_bound_push", 0.01);
// 	set_solver_option("mu_linear_decrease_factor", 0.5);
// 	set_solver_option("mu_superlinear_decrease_power", 1.2);
// 	set_solver_option("tol", 1e-7);

// 	if(!OptProblem::reoptimize(OptProblem::primalRestart)) {
// 	  printf("[warning] ContProbWithFixing K_idx=%d opt11 failed user[stop]=%d\n", K_idx, monitor.user_stopped);
// 	  //default_primal_start();
	  
// 	  //get a solution even if it failed
// 	  get_solution_simplicial_vectorized(sln);
// 	  if(!monitor.user_stopped)
// 	    bFirstSolveOK=false;
// 	  else 
// 	    bFirstSolveOK=true;
// 	  monitor.user_stopped=false;
// 	} else {
// 	  bFirstSolveOK=true;
// 	}
//       } else { //if(!monitor.user_stopped) 

// 	//solution OK

// 	monitor.user_stopped=false;
//       }
//     } 
//     //else 
//     {
//       get_solution_simplicial_vectorized(sln);

// #ifdef DEBUG
//       auto pgK = variable("p_g", d); assert(pgK!=NULL); 
//       double delta=0.; assert(variable("delta", d));
//       if(variable("delta", d)) {
// 	auto delta = variable("delta", d)->x[0]; 
	
// 	for(int i=0; i<pg0_partic_idxs.size(); i++) {
// 	  const double gen = pg0->x[pg0_partic_idxs[i]] + delta * data_sc.G_alpha[pg0_partic_idxs[i]];
// 	  if(gen >= data_sc.G_Pub[pg0_partic_idxs[i]]) 
// 	    assert(fabs(pgK->x[pgK_partic_idxs[i]] - data_sc.G_Pub[pg0_partic_idxs[i]]) < 9e-5);
// 	  if(gen <= data_sc.G_Plb[pg0_partic_idxs[i]]) 
// 	    assert(fabs(pgK->x[pgK_partic_idxs[i]] - data_sc.G_Plb[pg0_partic_idxs[i]]) < 9e-5);
// 	}
//       }

// #endif
//     }
//     f = this->obj_value;
//     hist_iter.push_back(number_of_iterations());
//     hist_obj.push_back(this->obj_value);

//     if(variable("delta", d)) solv1_delta_optim = variable("delta", d)->x[0];
//     else                     solv1_delta_optim = 0.;

//     if(num_K_done<comm_size-1) num_K_done=comm_size-1;

//     double K_avg_time_so_far = time_so_far  / num_K_done;

//     if(K_avg_time_so_far > 0.91*2.) monitor.is_late=true;

//     bool skip_2nd_solve = monitor.is_late;

//     if(time_so_far < 0.085*2.*data_sc.K_Contingency.size()) skip_2nd_solve=false;

//     if(this->obj_value>=5e5 && K_avg_time_so_far < 0.950*2.) skip_2nd_solve=false;
//     if(this->obj_value>=1e6 && K_avg_time_so_far < 1.025*2.) skip_2nd_solve=false;

//     if(!bFirstSolveOK) skip_2nd_solve=false;

//     if(bFirstSolveOK && tmrec.measureElapsedTime()>800.) {
//       skip_2nd_solve=true;
//       printf("ContProbWithFixing K_idx=%d will exit prematuraly b/c first solves took long %g sec on rank=%d\n", 
// 	     K_idx, tmrec.measureElapsedTime(), my_rank);
//     }

//     if(this->obj_value>pen_threshold && !skip_2nd_solve) {

//  #ifdef BE_VERBOSE
//       print_objterms_evals();
//       //print_p_g_with_coupling_info(*data_K[0], pg0);
//       printf("ContProbWithFixing K_idx=%d first pass resulted in high pen delta=%g\n", K_idx, solv1_delta_optim);
// #endif

//       double pplus, pminus, poverall;
//       estimate_active_power_deficit(pplus, pminus, poverall);
// #ifdef BE_VERBOSE
//       printf("ContProbWithFixing K_idx=%d (after solv1) act pow imbalances p+ p- poveral %g %g %g\n",
// 	     K_idx, pplus, pminus, poverall);
// #endif

//       bool one_more_push_and_fix=false; double gen_K_diff=0.;
//       if(fabs(solv1_delta_optim-solv1_delta_blocking)<1e-2 && 
// 	 d.K_ConType[0]==SCACOPFData::kGenerator && solv1_Pg_was_enough) {
// 	one_more_push_and_fix = true;
// 	if(pg0->x[data_sc.K_outidx[K_idx]]>1e-6 )  gen_K_diff = std::max(0., 1.1*poverall);
// 	else if(pg0->x[data_sc.K_outidx[K_idx]]<-1e-6)  gen_K_diff = std::min(0., poverall);
// 	else one_more_push_and_fix = false;
//       }

//       if(fabs(poverall)>1e-4) {// && d.K_ConType[0]!=SCACOPFData::kGenerator) {
// 	double rpa = fabs(pplus) / fabs(poverall);
// 	double rma = fabs(pminus) / fabs(poverall);

// 	//solv1_delta_optim=0.;//!

// 	if( (rpa>0.85 && rpa<1.15) || (rma>0.85 && rma <1.15) ) {	  
// 	  one_more_push_and_fix = true;
// 	  gen_K_diff = poverall;

// 	  //ignore small delta for transmission contingencies since they're really optimization noise
// 	  if(d.K_ConType[0]!=SCACOPFData::kGenerator && fabs(solv1_delta_optim)<1e-6) {
// 	    solv1_delta_optim=0.;
// 	  }
// 	}
//       }

//       if(one_more_push_and_fix) {
//  	//apparently we need to further unblock generation
//  	auto pgK = variable("p_g", d); assert(pgK!=NULL);
//  	//find AGC generators that are "blocking" and fix them; update particip and non-particip indexes
//  	vector<int> pg0_partic_idxs_u=solv1_pg0_partic_idxs, pgK_partic_idxs_u=solv1_pgK_partic_idxs;
//  	vector<int> pgK_nonpartic_idxs_u=solv1_pgK_nonpartic_idxs, pg0_nonpartic_idxs_u=solv1_pg0_nonpartic_idxs;

//  	double delta_out=0., delta_needed=0., delta_blocking=0., delta_lb, delta_ub; 
// 	double residual_Pg;
//  	bool bfeasib;

// 	if(fabs(gen_K_diff)>1e-6) {
// 	  //solv1_delta_optim and gen_K_diff must have same sign at this point
// 	  if(solv1_delta_optim * gen_K_diff < 0) gen_K_diff=0.;
// 	  bfeasib = push_and_fix_AGCgen(d, gen_K_diff, solv1_delta_optim, 
// 					pg0_partic_idxs_u, pgK_partic_idxs_u, pg0_nonpartic_idxs_u, pgK_nonpartic_idxs_u,
// 					pg0, pgK, 
// 					data_sc.G_Plb, data_sc.G_Pub, data_sc.G_alpha,
// 					delta_out, delta_needed, delta_blocking, delta_lb, delta_ub, residual_Pg);
//  	  //alter starting points 
// 	  assert(pg0_partic_idxs_u.size() == pgK_partic_idxs_u.size());
// 	  for(int it=0; it<pg0_partic_idxs_u.size(); it++) {
// 	    const int& i0 = pg0_partic_idxs_u[it];
// 	    pgK->x[pgK_partic_idxs_u[it]] = pg0->x[i0]+data_sc.G_alpha[i0]*delta_out;
// 	  }
// #ifdef BE_VERBOSE
// 	  printf("ContProbWithFixing K_idx=%d (gener)(after solv1) fixed %lu gens; adtl deltas out=%g needed=%g blocking=%g "
// 		 "residualPg=%g feasib=%d\n",
// 		 K_idx, solv1_pg0_partic_idxs.size()-pg0_partic_idxs_u.size(),
// 		 delta_out, delta_needed, delta_blocking, residual_Pg, bfeasib);
// 	  //printvec(solv1_pgK_partic_idxs, "solv1_pgK_partic_idxs");
// 	  //printvec(pgK_partic_idxs_u, "pgK_partic_idxs_u");
// #endif
	  
// 	  delete_constraint_block(con_name("AGC_simple_fixedpg0", d));
// 	  delete_duals_constraint(con_name("AGC_simple_fixedpg0", d));
	  
// 	  if(pg0_partic_idxs_u.size()>0) {
// 	    add_cons_AGC_simplified(d, pg0_partic_idxs_u, pgK_partic_idxs_u, pg0);
// 	    append_duals_constraint(con_name("AGC_simple_fixedpg0", d));
// 	    variable_duals_cons("duals_AGC_simple_fixedpg0", d)->set_start_to(0.0);
	    
// 	    variable("delta", d)->set_start_to(delta_out);
// 	  }
	  
// 	  primal_problem_changed();
// 	}
//       } // else of if(one_more_push_and_fix)

//       //
//       {
// 	auto v = variable("v_n", d);
// 	for(int i=0; i<v->n; i++) {
// 	  v->lb[i] = v->lb[i] - g_bounds_abuse;
// 	  v->ub[i] = v->ub[i] + g_bounds_abuse;
// 	}
//       }
//       if(true){
// 	auto v = variable("q_g", d);
// 	for(int i=0; i<v->n; i++) {
// 	  v->lb[i] = v->lb[i] - g_bounds_abuse;
// 	  v->ub[i] = v->ub[i] + g_bounds_abuse;
// 	}
//       }

//       if(true){
// 	auto v = variable("p_g", d);
// 	for(int i=0; i<v->n; i++) {
// 	  v->lb[i] = v->lb[i] - g_bounds_abuse;
// 	  v->ub[i] = v->ub[i] + g_bounds_abuse;
// 	}
//       }

//       do_qgen_fixing_for_PVPQ(variable("v_n", d), variable("q_g", d));

// #ifdef DEBUG
//       if(bFirstSolveOK) {
// 	if(!vars_duals_bounds_L->provides_start()) print_summary();
// 	assert(vars_duals_bounds_L->provides_start()); 	assert(vars_duals_bounds_U->provides_start()); 	
// 	assert(vars_duals_cons->provides_start());
//       }
// #endif

//       //second solve
//       monitor.safe_mode = false;
//       monitor.user_stopped=false;
//       if(bFirstSolveOK) monitor.bailout_allowed=true;
//       else              monitor.bailout_allowed=false;
//       monitor.timer.restart();
//       monitor.hist_tm.clear();
//       this->set_solver_option("max_iter", 250);
//       if(data_sc.N_Bus.size()>8999) {
// 	set_solver_option("mu_init", 1e-1);
//       }
//       bool opt2_ok = false;
//       if(bFirstSolveOK) {
// 	opt2_ok = OptProblem::reoptimize(OptProblem::primalDualRestart);
//       } else {
// 	opt2_ok = OptProblem::reoptimize(OptProblem::primalRestart);
//       }

//       if(!opt2_ok) {
// 	if(bFirstSolveOK && data_sc.N_Bus.size()>9999) {
// 	  //first solve is good enough 

// 	  //some reporting 
// 	  f = this->obj_value;
// 	  hist_iter.push_back(number_of_iterations());
// 	  hist_obj.push_back(this->obj_value);

// 	  if(monitor.user_stopped) {
// 	    //we can assume that solution is OK
// 	    if(bFirstSolveOK) { if(hist_obj.back() < hist_obj[0]) get_solution_simplicial_vectorized(sln); }
// 	    else get_solution_simplicial_vectorized(sln);
// 	  }	  

// 	} else { //first solves failed or the network is small 
// 	  if(!monitor.user_stopped) {
// 	    hist_iter.push_back(number_of_iterations());
// 	    hist_obj.push_back(this->obj_value);
// 	    monitor.bailout_allowed=false;
// 	    monitor.user_stopped=false;
// 	    monitor.safe_mode = true;
// 	    monitor.timer.restart();
// 	    monitor.hist_tm.clear();
// 	    printf("[warning] ContProbWithFixing K_idx=%d opt2 failed\n", K_idx); 
	    
// 	    set_solver_option("mu_init", 1e-1);
// 	    set_solver_option("max_iter", 500);
	    
// 	    set_solver_option("bound_relax_factor", 1e-8);
// 	    set_solver_option("bound_push", 0.01);
// 	    set_solver_option("slack_bound_push", 0.01);
// 	    set_solver_option("mu_linear_decrease_factor", 0.5);
// 	    set_solver_option("mu_superlinear_decrease_power", 1.2);
// 	    set_solver_option("tol", 1e-7);
	    
// 	    if(!OptProblem::reoptimize(OptProblem::primalRestart)) {
// 	      printf("[warning] ContProbWithFixing K_idx=%d opt22 failed user[stop]=%d\n", K_idx, monitor.user_stopped); 
// 	      if(!bFirstSolveOK) get_solution_simplicial_vectorized(sln);
// 	    } else {
// 	      get_solution_simplicial_vectorized(sln);
// 	    }
// 	    f = this->obj_value;
// 	    hist_iter.push_back(number_of_iterations());
// 	    hist_obj.push_back(this->obj_value);


// 	  } else { //true == monitor.user_stopped
// 	    //we can assume that solution is OK

// 	    f = this->obj_value;
// 	    hist_iter.push_back(number_of_iterations());
// 	    hist_obj.push_back(this->obj_value);

// 	    if(bFirstSolveOK) { if(hist_obj.back() < hist_obj[0]) get_solution_simplicial_vectorized(sln); }
// 	    else get_solution_simplicial_vectorized(sln);
// 	  }	    
// 	}
//       } else { //opt2_ok

// 	f = this->obj_value;
// 	hist_iter.push_back(number_of_iterations());
// 	hist_obj.push_back(this->obj_value);
	
// 	assert(hist_iter.size()>=1);
// 	assert(hist_obj.size()>=1);
// 	if(hist_obj.back() < hist_obj[0]) {
// 	  get_solution_simplicial_vectorized(sln);
// 	}
// 	if(!bFirstSolveOK) get_solution_simplicial_vectorized(sln);
//       }
      
//       if(this->obj_value>pen_threshold) {
// 	double delta_optim = 0.;//
// 	if(variable("delta", d)) delta_optim = variable("delta", d)->x[0];
// #ifdef BE_VERBOSE
// 	print_objterms_evals();
// 	//print_p_g_with_coupling_info(*data_K[0], pg0);
// 	printf("ContProbWithFixing K_idx=%d  pass 1-2 resulted in high pen delta=%g\n", K_idx, delta_optim);
// #endif
//       }  
//     } else {
//       if(this->obj_value>pen_threshold && skip_2nd_solve) 
// 	printf("ContProbWithFixing K_idx=%d pass2 needed but not done - time restrictions\n", K_idx);
//     }
      
//     tmrec.stop();
// #ifdef BE_VERBOSE
//     string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
//     string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
//     printf("ContProbWithFixing K_idx=%d optimize took %g sec - iters %s objs %s on rank=%d\n", 
// 	   K_idx, tmrec.getElapsedTime(), sit.c_str(), sobj.c_str(), my_rank);
//     fflush(stdout);
// #endif
//     return true;

//   }

  // void ContingencyProblemWithFixing::default_primal_start()
  // {
  //   assert(false);
  //   //for(auto b: vars_primal->vblocks) b->providesStartingPoint=false; 

  //   SCACOPFData& dK = *data_K[0]; //aaa
  //   auto v = variable("v_n", dK);
  //   //v->set_start_to(data_sc.N_v0.data());
  //   v->set_start_to(*v_n0);

  //   v = variable("theta_n", dK);
  //   //v->set_start_to(data_sc.N_theta0.data());
  //   v->set_start_to(*theta_n0);

  //   v = variable("b_s", dK);
  //   //v->set_start_to(data_sc.SSh_B0.data());
  //   v->set_start_to(*b_s0);

  //   v = variable("p_g", dK); assert(v->n == dK.G_p0.size());
  //   //v->set_start_to(dK.G_p0.data());

  //   v = variable("q_g", dK); 
  //   //v->set_start_to(dK.G_q0.data());

  //   //compute starting points: p_li1_powerflow, p_li2_powerflow
  //   if(true){
  //     auto p_li1 = variable("p_li1",dK), p_li2 = variable("p_li2",dK);
  //     auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("p_li1_powerflow", dK));
  //     auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("p_li2_powerflow", dK));
  //     pf_cons1->compute_power(p_li1); p_li1->providesStartingPoint=true;
  //     pf_cons2->compute_power(p_li2); p_li2->providesStartingPoint=true;
  //   }

  //   //q_li1_powerflow, q_li2_powerflow
  //   if(true){
  //     auto q_li1 = variable("q_li1",dK), q_li2 = variable("q_li2",dK);
  //     auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("q_li1_powerflow", dK));
  //     auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("q_li2_powerflow", dK));
  //     pf_cons1->compute_power(q_li1); q_li1->providesStartingPoint=true;
  //     pf_cons2->compute_power(q_li2); q_li2->providesStartingPoint=true;
  //   }


  //   // // transformers
  //   if(true){
  //     auto p_ti1 = variable("p_ti1",dK), p_ti2 = variable("p_ti2",dK);
  //     auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("p_ti1_powerflow", dK));
  //     auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("p_ti2_powerflow", dK));
  //     pf_cons1->compute_power(p_ti1); p_ti1->providesStartingPoint=true;
  //     pf_cons2->compute_power(p_ti2); p_ti2->providesStartingPoint=true;
  //   }

  //   //q_li1_powerflow, q_li2_powerflow
  //   if(true){
  //     auto q_ti1 = variable("q_ti1",dK), q_ti2 = variable("q_ti2",dK);
  //     auto pf_cons1 = dynamic_cast<PFConRectangular*>(constraint("q_ti1_powerflow", dK));
  //     auto pf_cons2 = dynamic_cast<PFConRectangular*>(constraint("q_ti2_powerflow", dK));
  //     pf_cons1->compute_power(q_ti1); q_ti1->providesStartingPoint=true;
  //     pf_cons2->compute_power(q_ti2); q_ti2->providesStartingPoint=true;
  //   }

  //   //active balance slacks
  //   if(true) {
  //     auto pf_p_bal = dynamic_cast<PFActiveBalance*>(constraint("p_balance", dK));
  //     OptVariablesBlock* pslacks_n = pf_p_bal->slacks();
  //     pf_p_bal->compute_slacks(pslacks_n); pslacks_n->providesStartingPoint=true;
      
  //     //reactive power balance slacks
  //     auto pf_q_bal = dynamic_cast<PFReactiveBalance*>(constraint("q_balance", dK));
  //     OptVariablesBlock* qslacks_n = pf_q_bal->slacks();
  //     pf_q_bal->compute_slacks(qslacks_n); qslacks_n->providesStartingPoint=true;
  //   }

  //   //line limits
  //   if(true){
  //     auto pf_line_lim1 = dynamic_cast<PFLineLimits*>(constraint("line_limits1", dK));
  //     OptVariablesBlock* sslack_li1 = pf_line_lim1->slacks();
  //     pf_line_lim1->compute_slacks(sslack_li1); sslack_li1->providesStartingPoint=true;
  //     auto pf_line_lim2 = dynamic_cast<PFLineLimits*>(constraint("line_limits2", dK));
  //     OptVariablesBlock* sslack_li2 = pf_line_lim2->slacks();
  //     pf_line_lim1->compute_slacks(sslack_li2); sslack_li2->providesStartingPoint=true;

  //   }

  //   //transformer limits
  //   if(true){
  //     auto pf_trans_lim1 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits1", dK));
  //     OptVariablesBlock* sslack_ti1 = pf_trans_lim1->slacks();
  //     pf_trans_lim1->compute_slacks(sslack_ti1); sslack_ti1->providesStartingPoint=true;
  //     auto pf_trans_lim2 = dynamic_cast<PFTransfLimits*>(constraint("trans_limits2", dK));
  //     OptVariablesBlock* sslack_ti2 = pf_trans_lim2->slacks();
  //     pf_trans_lim2->compute_slacks(sslack_ti2); sslack_ti2->providesStartingPoint=true;

  //   }
  //  auto deltav = variable("delta", dK);
  //  //if(deltav) deltav->set_start_to(1.41203e-06);
  //  //print_summary();
  //  assert(vars_primal->provides_start());
  // }

} //end of namespace

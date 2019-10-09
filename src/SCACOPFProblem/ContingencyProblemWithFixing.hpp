#ifndef SC_CONTINGENCY_PROBLEM_WITH_FIXING
#define SC_CONTINGENCY_PROBLEM_WITH_FIXING

#include "ContingencyProblem.hpp"

#include <vector>
#include <unordered_map>
#include "goTimer.hpp"

namespace gollnlp {

  class ContingencyProblemWithFixing : public ContingencyProblem
  {
  public:
    ContingencyProblemWithFixing(SCACOPFData& d_in, int K_idx_, 
				 int my_rank, int comm_size_,
				 std::unordered_map<std::string, 
				 gollnlp::OptVariablesBlock*>& dict_basecase_vars_,
				 const int& num_K_done_, const double& time_so_far_,
				 bool safe_mode=false);
    virtual ~ContingencyProblemWithFixing();

    //see go_code2.cpp, 'solve_contingency' method for explanations
    double pen_accept, pen_accept_initpt, pen_accept_solve1;
    double pen_accept_emer, pen_accept_safemode;
    double timeout; //of the entire 2-step, possibly multi-solve solve of this class
  public:
    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0)
    {
      return ContingencyProblem::default_assembly(vn0, thetan0, bs0, pg0, qg0);
    }
    virtual bool optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, 
			  std::vector<double>& sln);

  protected:
    bool add_cons_AGC_simplified(SCACOPFData& dB, 
				 const std::vector<int>& idxs_pg0_AGC_particip, 
				 const std::vector<int>& idxs_pgK_AGC_particip,
				 OptVariablesBlock* pg0);

    void add_cons_pg_nonanticip_using(OptVariablesBlock* pg0,
				      const std::vector<int>& idxs_pg0_nonparticip, 
				      const std::vector<int>& idxs_pgK_nonparticip);

    // Make up for excess or surplus 'P' in the contingency by ramping up or down AGC generation.
    // Iterates over gens and fix them when the bounds are hit, till 'P' can be apparently made up for
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
    bool push_and_fix_AGCgen(SCACOPFData& dB, const double& P, const double& delta_in,
			     std::vector<int>& idxs_pg0_AGC_particip, std::vector<int>& idxs_pgK_AGC_particip,
			     std::vector<int>& idxs_pg0_nonparticip, std::vector<int>& idxs_pgK_nonparticip,
			     OptVariablesBlock* pg0, OptVariablesBlock* pgK,
			     std::vector<double>& Plb, std::vector<double>& Pub, std::vector<double>& alpha,
			     double& delta_out, double& delta_needed, double& delta_blocking, 
			     double& delta_lb, double& delta_ub,
			     double& residual_Pg);

     bool do_qgen_fixing_for_PVPQ(OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
			    OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
				 OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_AGC(const double& smoothing, bool fixVoltage, OptVariablesBlock* pgk, OptVariablesBlock* delta);

    //uses 'dict_basecase_vars'
    //bool set_warm_start_from_basecase_dictb();
    //also looks up 'dict_basecase_vars'
    bool warm_start_variable_from_basecase_dict(OptVariables& v);

    void default_primal_start();
    
    void estimate_active_power_deficit(double& p_plus, double& p_minus, double& p_overall);
    void get_objective_penalties(double& pen_p_balance, double& pen_q_balance, 
				 double& pen_line_limits, double& pen_trans_limits);

    bool do_solve1();
    bool do_solve2(bool first_solve_OK);
  protected:
    std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict_basecase_vars;
    std::vector<int> solv1_pg0_partic_idxs, solv1_pgK_partic_idxs, solv1_pgK_nonpartic_idxs, solv1_pg0_nonpartic_idxs;
    double solv1_delta_out, solv1_delta_needed, solv1_delta_blocking, solv1_delta_lb, solv1_delta_ub, solv1_delta_optim;
    bool solv1_Pg_was_enough;
    bool safe_mode; //true if ContingencyProblemWithFixing::optimize went AWOL on a different rank
    std::vector<double> sln_solve1, sln_solve2;
    double obj_solve1, obj_solve2;
  public:
    static double g_bounds_abuse;
  protected:
    int num_K_done;
    double time_so_far;

    int comm_size;
    goTimer tmTotal;
  public:
#ifdef GOLLNLP_FAULT_HANDLING
    virtual bool iterate_callback(int iter, const double& obj_value,
				  const double* primals,
				  const double& inf_pr, const double& inf_pr_orig_pr, 
				  const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials, OptimizationMode mode)
    {
      if(primals && mode!=RestorationPhaseMode) {
	if(vars_last) vars_last->copy_from(primals);

	if(inf_pr_orig_pr<=1e-6 && best_known_iter.obj_value>=obj_value) {
	  best_known_iter.copy_primal_vars_from(primals, vars_primal);
	  best_known_iter.set_iter_stats( iter, obj_value, inf_pr, inf_pr_orig_pr, inf_du, mu, mode);
	}
      } else {
	if(mode==RestorationPhaseMode) {
	  monitor.emergency = true;
	  //do not set monitor.user_stopped=true; since doing so will look like the last solution is ok
	  if(best_known_iter.obj_value<=monitor.pen_accept_emer) {
	    printf("[stop]rest   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e "
		   "inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  }
	  return false;
	}
      }

      if(monitor.is_active) {
	monitor.hist_tm.push_back(monitor.timer.measureElapsedTime());
	
	const double tmavg =  monitor.hist_tm.back() / monitor.hist_tm.size();
	
	const int over_n_last = 3; double tmavg_over_last = tmavg;
	if(monitor.hist_tm.size() > over_n_last) {
	  const int idx_ref = monitor.hist_tm.size()-over_n_last-1;
	  tmavg_over_last = (monitor.hist_tm.back()-monitor.hist_tm[idx_ref])/over_n_last;
	}
	
	if(obj_value<monitor.pen_accept && inf_pr_orig_pr<1e-6) {
	  monitor.user_stopped=true;
	  printf("[stop]acpt   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e "
		 "inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		 K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;
	}

	if(tmavg_over_last>4.*tmavg) {
	  monitor.emergency = true;
	  if(obj_value<pen_accept_emer && inf_pr_orig_pr<1e-6) {
	    monitor.user_stopped=true;
	    printf("[stop]slo1   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e "
		   "inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;	 
	  }
	}

	if(monitor.timer.measureElapsedTime() > monitor.timeout) {
	  monitor.emergency = true;
	  if(obj_value<pen_accept_emer && inf_pr_orig_pr<1e-6) {
	    monitor.user_stopped=true;
	    printf("[stop]time   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e "
		   "inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;	 
	  }
	}

	if(monitor.emergency) {
	  if(obj_value<pen_accept_emer && inf_pr_orig_pr<1e-6) {
	    monitor.user_stopped=true;
	    printf("[stop]emer   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e "
		   "inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;	 
	  }
	}
      }
      return true;
      //ContingencyProblem::iterate_callback(iter, obj_value, primals, inf_pr, inf_pr_orig_pr, inf_du,
      //					  mu, alpha_du, alpha_pr, ls_trials, mode);
    }
  public:
    struct IterInfo
    {
      IterInfo() 
	: obj_value(1e+20), vars_primal(NULL), inf_pr(1e+20), inf_pr_orig_pr(1e+20), inf_du(1e+20), mu(1000.), iter(-1)
      { }
      virtual ~IterInfo()
      {
	delete vars_primal;
      }
      
      inline void initialize( OptVariables* primal_vars_template ) {
	if(NULL==vars_primal)
	  vars_primal = primal_vars_template->new_copy();
	else if(vars_primal->n() != primal_vars_template->n()) {
	  assert(false);
	  delete vars_primal;
	  vars_primal = NULL;
	  vars_primal = primal_vars_template->new_copy();
	}
      }

      inline void set_objective(const double& obj) { obj_value = obj; }
      
      inline void copy_primal_vars_from(const double* opt_vars_values, OptVariables* primal_vars_template) {
	if(NULL!=vars_primal) 
	  vars_primal->copy_from(opt_vars_values);
	else 
	  assert(false);
      }
      
      inline void set_iter_stats(int iter_, const double& obj_value_,
				 const double& inf_pr_, const double& inf_pr_orig_pr_, 
				 const double& inf_du_, 
				 const double& mu_, OptimizationMode mode_)
      {
	iter=iter_;
	obj_value=obj_value_;
	inf_pr=inf_pr_;
	inf_pr_orig_pr=inf_pr_orig_pr_;
	inf_du=inf_du_;
	mu=mu_;
	mode=mode_;
      }
      
      OptVariables* vars_primal;
      double obj_value, inf_pr, inf_pr_orig_pr, inf_du, mu;
      int iter;
      OptimizationMode mode;
    };
  protected:
    OptVariables *vars_ini, *vars_last;
    IterInfo best_known_iter;
#endif
  };

} //end namespace
#endif

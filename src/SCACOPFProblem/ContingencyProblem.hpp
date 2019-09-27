#ifndef SC_CONTINGENCY_PROBLEM
#define SC_CONTINGENCY_PROBLEM

#include "SCACOPFProblem.hpp"
#include <vector>
#include "goUtils.hpp"

#ifdef GOLLNLP_FAULT_HANDLING
#include "goSignalHandling.hpp"
extern volatile sig_atomic_t solve_is_alive;
#endif

namespace gollnlp {

  class ContingencyProblem : public SCACOPFProblem
  {
  public:
    ContingencyProblem(SCACOPFData& d_in, int K_idx_, int my_rank);
    virtual ~ContingencyProblem();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0);

    //evaluates objective/penalty given pg0 and vn0 (these are 'in' arguments)
    virtual bool eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);
    
    //similar to the above, but may use a different warm-starting procedure
    virtual bool optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);

    virtual void get_solution_simplicial_vectorized(std::vector<double>& vsln);

    //
    // warm-starting
    //
    // these functions return false whenever there is a discrepancy between 
    // this' and srcProb's variables

    // warm-starts both primal and dual variables 
    //safe to call "reoptimize" with PD warm-start
    bool set_warm_start_from_base_of(SCACOPFProblem& srcProb);
    bool set_warm_start_from_contingency_of(SCACOPFProblem& srcProb);
    bool set_warm_start_from(ContingencyProblem& srcProb)
    {
      assert(false && "not implemented yet");
      return false;
    }

    //
    // non-anticipativity
    //
    // simply set lb = ub =pg0 for p_gK  for non-AGC generators
    void bodyof_cons_nonanticip_using(OptVariablesBlock* pg0);
    void add_cons_nonanticip_using(OptVariablesBlock* pg0);
    inline void update_cons_nonanticip_using(OptVariablesBlock* pg0) {
      bodyof_cons_nonanticip_using(pg0);
    }
  protected:
    //
    // AGC
    //

    //indexes of non-participating AGC generators in data_K[0].G_Generator and data_sc.G_Generator, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_nonpartic_idxs, pg0_nonpartic_idxs;
    //indexes of participating AGC generators in data_K[0].G_Generator and data_sc.G_Generator, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_partic_idxs, pg0_partic_idxs;
    //indexes of data_K[0].G_Generator in data_sc.G_Generator
    //these indexes exclude 'outidx' when K_idx is a generator contingency; otherwise Gk=0,1,2,...
    std::vector<int> Gk;

    void add_cons_AGC_using(OptVariablesBlock* pg0);
    void update_cons_AGC_using(OptVariablesBlock* pg0);

    //
    // PVPQ
    //
    void add_const_nonanticip_v_n_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    void add_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    void update_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);

    void add_regularizations();
  public:
    int K_idx;
    int my_rank;

    //regularizations: gamma* || x - xbasecase]]^2
    bool reg_vn, reg_thetan, reg_bs, reg_pg, reg_qg;
  protected:
    OptVariablesBlock *v_n0, *theta_n0, *b_s0, *p_g0, *q_g0;
  public:
    virtual bool iterate_callback(int iter, const double& obj_value,
				  const double* primals,
				  const double& inf_pr, const double& inf_pr_orig_pr, 
				  const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials)
    {
#ifdef GOLLNLP_FAULT_HANDLING
      solve_is_alive = true;
#endif
      if(monitor.is_active) {
	monitor.hist_tm.push_back(monitor.timer.measureElapsedTime());

	const double tmavg =  monitor.hist_tm.back() / monitor.hist_tm.size();

	const int over_n_last = 3; double tmavg_over_last = tmavg;
	if(monitor.hist_tm.size() > over_n_last) {
	  const int idx_ref = monitor.hist_tm.size()-over_n_last-1;
	  tmavg_over_last = (monitor.hist_tm.back()-monitor.hist_tm[idx_ref])/over_n_last;
	}

	//if(K_idx==4023 || K_idx==3738) {
	if(K_idx==3197 || my_rank==54){
	  printf("[stop]????   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e "
	     "rank=%d time=%g avg=[%.2f %.2f(%d)] is_late=%d safe_mode=%d bailout_allowed=%d\n",
	       K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank, monitor.timer.measureElapsedTime(),
	       tmavg, tmavg_over_last, over_n_last, monitor.is_late, monitor.safe_mode, monitor.bailout_allowed);
	}
	if(monitor.is_late) {
	  if(obj_value<2*monitor.pen_threshold && inf_pr_orig_pr<2e-6 && mu<=1e-5) {
	    monitor.user_stopped=true;
	    printf("[stop]late   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;
	  } 
	} else {
	  if(obj_value<monitor.pen_threshold && inf_pr_orig_pr<1e-6 && mu<=5e-6) {
	    monitor.user_stopped=true;
	    printf("[stop]norm   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;
	  }
	}
      
	if(tmavg_over_last>5.*tmavg && obj_value<20*monitor.pen_threshold && inf_pr_orig_pr<1e-6 && mu<=5e-6) {
	  monitor.user_stopped=true;
	  printf("[stop]slo1   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		 K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;	 
	}
	if(tmavg_over_last>3.*tmavg && obj_value<10*monitor.pen_threshold && inf_pr_orig_pr<1e-6 && mu<=5e-6) {
	  monitor.user_stopped=true;
	  printf("[stop]slo2   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		 K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;	 
	}
	
	if(tmavg_over_last>2.*tmavg && obj_value< 2.*monitor.pen_threshold && inf_pr_orig_pr<1e-6 && mu<=5e-6) {
	  monitor.user_stopped=true;
	  printf("[stop]slo3   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		 K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;	 
	}

	if(tmavg_over_last>1.5*tmavg && obj_value<10.*monitor.pen_threshold && inf_pr_orig_pr<1e-7 && inf_du <1e-6 && mu<=1e-6) {
	  monitor.user_stopped=true;
	  printf("[stop]slo4   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		 K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;	 
	}

	if(true==monitor.bailout_allowed) {
	  // do not set monitor.user_stopped=true;

	  bool bret = true;

	  if(tmavg_over_last>30. && iter>3) 
	    if(data_sc.N_Bus.size() <= 35000) bret = false;
	  if(tmavg_over_last>40. && iter>3) 
	    bret = false;

	  if(tmavg>5. && iter>5)
	    if(data_sc.N_Bus.size() <= 35000) bret = false;

	  if(tmavg>9. && iter>4) bret = false;

	  if(!bret) {
	    printf("[stop]bail   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    return false;	 
	  }
	}

	if(!monitor.safe_mode) {
	  if(monitor.timer.measureElapsedTime() > 350.) {
	    printf("[stop]time   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e "
		   "rank=%d avg=[%.2f %.2f(%d)]\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank,
		   tmavg, tmavg_over_last,over_n_last);
	    
	    // do not set monitor.user_stopped=true;
	    
	    return false;
	  }
	  
	} else { //this is in safe_mode
	  if(obj_value<20*monitor.pen_threshold && inf_pr_orig_pr<2e-6 && mu<=1e-5) {
	    printf("[stop]safe   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	    monitor.user_stopped=true;
	    return false;
	  }
	  
	  if(tmavg_over_last>3.*tmavg && obj_value<1e+6*monitor.pen_threshold && inf_pr_orig_pr<2e-6 && mu<=1e-5) {
	    monitor.user_stopped=true;
	    printf("[stop]slos   K_idx=%d iter %d : obj=%12.5e inf_pr_o=%12.5e mu=%12.5e inf_du=%12.5e a_du=%12.5e a_pr=%12.5e rank=%d\n",
		   K_idx, iter, obj_value, inf_pr_orig_pr, mu, inf_du,  alpha_du, alpha_pr, my_rank);
	  return false;	 
	  }

	}
      }

      return true; 
    }
    
  };

} //end namespace
#endif

#ifndef SCACOPF_PROBLEM
#define SCACOPF_PROBLEM

#include "OptProblem.hpp"

#include "SCACOPFData.hpp"

#include <cstring>
#include "blasdefs.hpp"
#include "goTimer.hpp"
//this class is for ACOPF base case and is inherited by ACOPFContingencyProblem
#include <unistd.h>

#include <unordered_map>

namespace gollnlp {
  
  class SCACOPFProblem : public OptProblem
  {
  public:
    SCACOPFProblem(SCACOPFData& d_in) 
      : data_sc(d_in), 
	useQPen(false), slacks_scale(1.),  PVPQSmoothing(0.01), AGCSmoothing(1e-2),
	AGC_as_nonanticip(false), AGC_simplified(false), PVPQ_as_nonanticip(false),
	linear_prod_cost(false), quadr_penalty_qg0(false)
    {
      L_rate_reduction = T_rate_reduction = 1.;
      my_rank=-1; rank_solver_rank0 = 1;
      comm_world = MPI_COMM_NULL;
      iter_sol_written=-10;
    }
    virtual ~SCACOPFProblem();

    //only base case case, no contingencies and no coupling
    virtual bool default_assembly();
    //base case + the variables and blocks needed by contingencies specified by 'K_idxs'
    virtual bool assembly(const std::vector<int>& K_idxs);

    //add block for contingency K_idx
    virtual bool add_contingency_block(const int K_idx);
    virtual bool has_contigency(const int K_idx);
    
    // contingencies considered in the SCACOPF; returns idx in data.K_Contingency
    virtual std::vector<int> get_contingencies() const;


    //add reserve constraints on AGC so that AGC generators can ramp up and down to the largest 
    //power loss and gain (among all generator contingencies in each area). 
    //
    // The power loss 'max_loss' is taken to be the max over area's Kgens of max(Plb,0)
    //  - this will add at most one set of constraints in the form
    //       sum(Pub[i]-pg[i]: i in AGC) - max_loss + splus >=0 
    //       splus >=0  and a quadratic penalty obj term of splus
    //
    // The power gain 'max_gain' is taken to be the max over area's Kgens of max(0,-Pub)
    //  - this will add at most one set of constraints in the form
    //       sum(pg[i]-Plb[i]: i in AGC) - max_gain + sminus >=0 
    //       sminus >=0  and a quadratic penalty obj term on sminus
    virtual void add_agc_reserves_for_max_Lloss_Ugain();

    //same as above but the power loss 'max_loss' or the power gain 'max_gain'
    //is actually the active generation pg[Kgen_idx]
    virtual void add_agc_reserves();

    //vector of Kgen idxs for which there appears to be insufficient active power contingency
    //response from the participating AGC generators even in the most optimistic case (AGC
    //gens can respond at full capacity and the Kgen is at lower (for loss) or upper (for 
    //gain/injection) limit
    //
    //essentially  returns Kgen K_idxs (indexes in data.K_Contingency) for which the Kgen
    // i. power loss situation
    //   Glb[Kgen_idx] > sum {Gub[gidx]-Glb[gidx] : gidx in responding AGC} 
    // ii. power gain/injection situation
    //  -Gub[Kgen_idx] > sum {Gub[gidx]-Glb[gidx] : gidx in responding AGC} 
    //
    // the second vector is of the corresponding K_idxs
    virtual void find_AGC_infeasible_Kgens(std::vector<int>& agc_infeas_gen_idxs, 
					   std::vector<int>& agc_infeas_K_idxs);

    //controllers of how AGC and PVPQ constraints are enforced
    inline void set_AGC_as_nonanticip(bool onOrOff)
    { AGC_as_nonanticip = onOrOff; }
    inline void set_AGC_simplified(bool onOrOff)
    { AGC_simplified = onOrOff; }
    void update_AGC_smoothing_param(const double& val);

    inline void set_linear_prod_cost(const bool onOrOff)
    { linear_prod_cost = onOrOff; assert(onOrOff && "can only be switched on");}

    inline void set_PVPQ_as_nonanticip(bool onOrOff)
    { PVPQ_as_nonanticip = onOrOff; }
    void update_PVPQ_smoothing_param(const double& val);

    inline void set_basecase_L_rate_reduction(const double& rate) { L_rate_reduction = rate; }
    inline void set_basecase_T_rate_reduction(const double& rate) { T_rate_reduction = rate; }

    inline void set_quadr_penalty_qg0(bool onOrOff) { quadr_penalty_qg0 = onOrOff; }

    void add_quadr_conting_penalty_pg0(const int& idx_gen, const double& p0, const double& f_pen);
    void remove_quadr_conting_penalty_pg0(const int& idx_gen);

    void add_conting_penalty_line0(const int& idx_line, 
				   const double& pli10, const double& qli10, 
				   const double& pli20, const double& qli20, 
				   const double& f_pen);
    void remove_conting_penalty_line0(const int& idx_line);

    void add_conting_penalty_transf0(const int& idx_transf, 
				   const double& pti10, const double& qti10, 
				   const double& pti20, const double& qti20, 
				   const double& f_pen);
    void remove_conting_penalty_transf0(const int& idx_line);


    bool set_warm_start_from_base_of(SCACOPFProblem& srcProb);
  protected:
    bool set_warm_start_for_base_from_base_of(SCACOPFProblem& srcProb);
    bool set_warm_start_for_cont_from_base_of(SCACOPFData& dB, SCACOPFProblem& srcProb);
  public:
    bool set_warm_start_for_cont_from_base_of(const int& K_idx, SCACOPFProblem& srcProb);
  protected:
    // for all add_ methods, dB is the block data (base case or contingency)
    // all these methods use 'd' SCACOPFData as well since dB contains only a 
    // subset, contingency-specific data from SCACOPFData
    void add_variables(SCACOPFData& dB, bool SysCond_BaseCase = true);
    void add_cons_lines_pf(SCACOPFData& dB);
    void add_cons_transformers_pf(SCACOPFData& dB);
    void add_cons_active_powbal(SCACOPFData& dB);
    void add_cons_reactive_powbal(SCACOPFData& dB);

    // 'SysCond_BaseCase' decides whether to use RateBase or RateEmer
    //  SysCond_BaseCase=true -> base case; =false -> contingency
    void add_cons_thermal_li_lims(SCACOPFData& dB, bool SysCond_BaseCase=true);
    void add_cons_thermal_ti_lims(SCACOPFData& dB, bool SysCond_BaseCase=true);
    // same as above, but with explicit argument for T_Rate
    void add_cons_thermal_li_lims(SCACOPFData& dB, const std::vector<double>& rate);
    void add_cons_thermal_ti_lims(SCACOPFData& dB, const std::vector<double>& rate);

    void add_obj_prod_cost(SCACOPFData& dB);

    void add_cons_coupling(SCACOPFData& dB);

    void add_cons_pg_nonanticip(SCACOPFData& dB, const std::vector<int>& Gk_no_AGC);
    void add_cons_AGC(SCACOPFData& dB, const std::vector<int>& Gk_AGC);
    void add_cons_AGC_simplified(SCACOPFData& dB, const std::vector<int>& Gk_AGC);
    void add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk);
    void add_cons_PVPQ_as_vn_nonanticip(SCACOPFData& dB, const std::vector<int>& Gk);
  public: 
    //contingencies' SCACOPFData
    std::vector<SCACOPFData*> data_K;
  protected: 
    SCACOPFData& data_sc;
  public:
    //options
    bool useQPen;
    double slacks_scale;
    int my_rank, rank_solver_rank0;
    MPI_Comm comm_world;
  protected:
    double AGCSmoothing, PVPQSmoothing;
    bool quadr_penalty_qg0;
  public:
    bool AGC_as_nonanticip, AGC_simplified, PVPQ_as_nonanticip;
    bool linear_prod_cost;
    double L_rate_reduction, T_rate_reduction;
  public:
    //variables and constraints accessers
    inline static std::string var_name(const std::string& prefix, const SCACOPFData& d) { 
      return var_name(prefix, d.id); 
    }
    inline static std::string var_name(const std::string& prefix, int Kid) { 
      return prefix+"_"+std::to_string(Kid); 
    }
    inline OptVariablesBlock* variable(const std::string& prefix, const SCACOPFData& d) { 
      return vars_block(var_name(prefix, d));
    }
    inline OptVariablesBlock* variable(const std::string& prefix, int Kid) { 
      return vars_block(var_name(prefix, Kid));
    }
    inline static std::string con_name(const std::string& prefix, int Kid) { 
      return prefix+"_"+std::to_string(Kid); 
    }
    inline static std::string con_name(const std::string& prefix, const SCACOPFData& d) { 
      return con_name(prefix, d.id);
    }
    inline static std::string objterm_name(const std::string& prefix, const SCACOPFData& d) { 
      return prefix+"_"+std::to_string(d.id); 
    }

    inline OptConstraintsBlock* constraint(const std::string& prefix, const SCACOPFData& d) { 
      return constraints_block(con_name(prefix, d));
    }
    inline OptConstraintsBlock* constraint(const std::string& prefix, int Kid) {
      return constraints_block(con_name(prefix, Kid));
    }
    
    inline OptVariablesBlock* variable_duals_lower(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_bounds_lower(prefix+"_"+std::to_string(d.id));
    }
    inline OptVariablesBlock* variable_duals_upper(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_bounds_upper(prefix+"_"+std::to_string(d.id));
    }
    inline OptVariablesBlock* variable_duals_cons(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_cons(prefix+"_"+std::to_string(d.id));
    }

    //grows dest as needed
    void copy_basecase_primal_variables_to(std::vector<double>& dest);

    //copy values from the dictionary to the blocks of 'vars'
    //this is for testing
    void warm_start_basecasevariables_from_dict(std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict);
    void build_pd_vars_dict(std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict);

    // returns the idxs of PVPQ gens and corresponding buses
    // generators at the same PVPQ bus are aggregated
    //
    // Gk are the indexes of all gens other than the outgen (for generator contingencies) 
    // in data_sc.G_Generator
    void get_idxs_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk,
		       std::vector<std::vector<int> >& idxs_gen_agg, std::vector<int>& idxs_bus_pvpq,
		       std::vector<double>& Qlb, std::vector<double>& Qub,
		       int& nPVPQGens, int &num_qgens_fixed, 
		       int& num_N_PVPQ, int& num_buses_all_qgen_fixed);

    //printing
    void print_p_g(SCACOPFData& dB);
    void print_p_g_with_coupling_info(SCACOPFData& dB, OptVariablesBlock* p_g0=NULL);
    void print_PVPQ_info(SCACOPFData& dB, OptVariablesBlock* v_n0=NULL);
    void print_Transf_powers(SCACOPFData& dB, bool SysCond_BaseCase = true);

    void print_active_power_balance_info(SCACOPFData& dB);
    void print_reactive_power_balance_info(SCACOPFData& dB);

    void print_line_limits_info(SCACOPFData& dB);
    void print_transf_limits_info(SCACOPFData& dB);
    void write_solution_basecase(OptVariables* primal_vars=NULL);
    void write_pridua_solution_basecase(OptVariables* primal_vars=NULL,
					OptVariables* dual_con_vars=NULL,
					OptVariables* dual_lb_vars=NULL,
					OptVariables* dual_ub_vars=NULL);
    void write_solution_extras_basecase(OptVariables* primal_vars=NULL);

  public:
    virtual bool iterate_callback(int iter, const double& obj_value,
				  const double* primals,
				  const double& inf_pr, const double& inf_pr_orig_pr, 
				  const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials, OptimizationMode mode,
				  const double* duals_con=NULL,
				  const double* duals_lb=NULL, const double* duals_ub=NULL);
    
    struct ConvMonitor
    {
      ConvMonitor() : is_active(false), user_stopped(false), emergency(false),
		      pen_accept(1.), pen_accept_emer(1000.), timeout(500), bcast_done(false)
      {
	timer.start();
      };

      bool is_active;
      bool user_stopped;
      bool emergency;
      double pen_accept, pen_accept_emer; //under normal and emergency
      double timeout; //max time spent 
      goTimer timer;
      std::vector<double> hist_tm;
      bool bcast_done;
    };
    ConvMonitor monitor;
  public:
    struct IterInfo
    {
      IterInfo() 
	: obj_value(1e+20), vars_primal(NULL), inf_pr(1e+20), inf_pr_orig_pr(1e+20), inf_du(1e+20), mu(1000.), iter(-1),
	  vars_duals_cons(NULL), vars_duals_bounds_L(NULL), vars_duals_bounds_U(NULL), my_rank(-1)
      {
      }
      virtual ~IterInfo()
      {
	delete vars_primal;
	delete vars_duals_cons;
	delete vars_duals_bounds_L;
	delete vars_duals_bounds_U;
      }
      
      inline void initialize( OptVariables* primal_vars_template,
			      OptVariables* duals_cons_vars_template=NULL,
			      OptVariables* duals_lb_vars_template=NULL,
			      OptVariables* duals_ub_vars_template=NULL) {
	if(NULL==vars_primal) {
	  //printf("\n!!![best_known] primal_vars_created rank=%d\n\n", my_rank);
	  vars_primal = primal_vars_template->new_copy();
	}
	else if(vars_primal->n() != primal_vars_template->n()) {
	  assert(false);
	  delete vars_primal;
	  vars_primal = NULL;
	  vars_primal = primal_vars_template->new_copy();
	}

	if(duals_cons_vars_template) {
	  if(vars_duals_cons==NULL) {
	    vars_duals_cons = duals_cons_vars_template->new_copy();
	    //printf("\n!!![best_known] duals_cons_vars_created rank=%d\n\n", my_rank);
	  }
	  else if(vars_duals_cons->n() != duals_cons_vars_template->n()) {
	    assert(false);
	    delete vars_duals_cons;
	    vars_duals_cons = NULL;
	    vars_duals_cons = duals_cons_vars_template->new_copy();
	  }
	}
	if(duals_lb_vars_template) {
	  assert(duals_ub_vars_template);
	  if(vars_duals_bounds_L==NULL) {
	    assert(vars_duals_bounds_U==NULL);
	    //printf("\n!!![best_known] dual_lb_vars_created rank=%d\n\n", my_rank);
	    vars_duals_bounds_L = duals_lb_vars_template->new_copy();
	    vars_duals_bounds_U = duals_ub_vars_template->new_copy();
	  } else if(vars_duals_bounds_L->n() != duals_lb_vars_template->n()) {
	    assert(false);
	    delete vars_duals_bounds_L;
	    vars_duals_bounds_L=NULL;
	    vars_duals_bounds_L = duals_lb_vars_template->new_copy();
	    delete vars_duals_bounds_U;
	    vars_duals_bounds_U=NULL;
	    vars_duals_bounds_U=duals_ub_vars_template->new_copy();
	  }
	}		
      }

      inline void set_objective(const double& obj) { obj_value = obj; }
      
      inline void copy_primal_vars_from(const double* opt_vars_values, OptVariables* primal_vars_template) {
	if(NULL!=vars_primal) 
	  vars_primal->copy_from(opt_vars_values);
	else 
	  assert(false);
      }
      inline void copy_dual_vars_from(const double* duals_con, 
				      const double* duals_lb,
				      const double* duals_ub) {
	if(NULL!=vars_duals_cons) 
	  vars_duals_cons->copy_from(duals_con);
	else 
	  assert(false);

	if(NULL!=vars_duals_bounds_L) {
	  vars_duals_bounds_L->copy_from(duals_lb);
	  vars_duals_bounds_L->set_inactive_duals_lb_to(0., *vars_primal);
	}else assert(false);

	if(NULL!=vars_duals_bounds_U) {
	  vars_duals_bounds_U->copy_from(duals_ub);
	  vars_duals_bounds_U->set_inactive_duals_ub_to(0., *vars_primal);
	} else assert(false);
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
      OptVariables *vars_duals_cons, *vars_duals_bounds_L, *vars_duals_bounds_U;
      int my_rank;
    };
    IterInfo best_known_iter;
    int iter_sol_written;
  }; // end of SCACOPFProblem



}

#endif

#ifndef SCACOPF_PROBLEM
#define SCACOPF_PROBLEM

#include "OptProblem.hpp"

#include "SCACOPFData.hpp"

#include <cstring>
#include "blasdefs.hpp"
//this class is for ACOPF base case and is inherited by ACOPFContingencyProblem

namespace gollnlp {
  
  class SCACOPFProblem : public OptProblem
  {
  public:
    SCACOPFProblem(SCACOPFData& d_in) 
      : data_sc(d_in), 
	useQPen(false), slacks_scale(1.),  PVPQSmoothing(0.01), AGCSmoothing(1e-2),
	AGC_as_nonanticip(false), AGC_simplified(false), PVPQ_as_nonanticip(false),
	quadr_penalty_qg0(false)
    {
      L_rate_reduction = T_rate_reduction = 1.;
    }
    virtual ~SCACOPFProblem();

    //overwrites of OptProblem
    virtual bool iterate_callback(int iter, const double& obj_value, const double* primals,
				  const double& inf_pr, const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials) 
    { return true; }

    //only base case case, no contingencies and no coupling
    virtual bool default_assembly();
    //base case + the variables and blocks needed by contingencies specified by 'K_idxs'
    virtual bool assembly(const std::vector<int>& K_idxs);

    virtual void add_agc_reserves();

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

    //controllers of how AGC and PVPQ constraints are enforced
    inline void set_AGC_as_nonanticip(bool onOrOff)
    { AGC_as_nonanticip = onOrOff; }
    inline void set_AGC_simplified(bool onOrOff)
    { AGC_simplified = onOrOff; }
    void update_AGC_smoothing_param(const double& val);

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
  protected:
    double AGCSmoothing, PVPQSmoothing;
    bool quadr_penalty_qg0;
  public:
    bool AGC_as_nonanticip, AGC_simplified, PVPQ_as_nonanticip;
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
    void print_PVPQ_info(SCACOPFData& dB);
    void print_Transf_powers(SCACOPFData& dB, bool SysCond_BaseCase = true);

    void print_active_power_balance_info(SCACOPFData& dB);
    void print_reactive_power_balance_info(SCACOPFData& dB);

    void print_line_limits_info(SCACOPFData& dB);
    void print_transf_limits_info(SCACOPFData& dB);
    void write_solution_basecase();
    void write_solution_extras_basecase();
  }; // end of SCACOPFProblem



}

#endif

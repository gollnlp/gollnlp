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
	useQPen(false), slacks_scale(1.),  PVPQSmoothing(0.01), AGCSmoothing(1e-4),
	AGC_as_nonanticip(false), AGC_simplified(false), PVPQ_as_nonanticip(false)
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
    virtual bool assembly(const std::vector<int> K_idxs);

    //controllers of how AGC and PVPQ constraints are enforced
    inline void set_AGC_as_nonanticip(bool onOrOff)
    { AGC_as_nonanticip = onOrOff; }
    inline void set_AGC_simplified(bool onOrOff)
    { AGC_simplified = onOrOff; }
    inline void set_PVPQ_as_nonanticip(bool onOrOff)
    { PVPQ_as_nonanticip = onOrOff; }
    inline void set_basecase_L_rate_reduction(const double& rate) { L_rate_reduction = rate; }
    inline void set_basecase_T_rate_reduction(const double& rate) { T_rate_reduction = rate; }

    bool set_warm_start_from_base_of(SCACOPFProblem& srcProb);
  protected:
    bool set_warm_start_for_base_from_base_of(SCACOPFProblem& srcProb);
    bool set_warm_start_for_cont_from_base_of(SCACOPFData& dB, SCACOPFProblem& srcProb);
  protected:
    // for all add_ methods, dB is the block data (base case or contingency)
    // all these methods use 'd' SCACOPFData as well since dB contains only a 
    // subset, contingency-specific data from SCACOPFData
    void add_variables(SCACOPFData& dB);
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
    void add_cons_nonanticip(SCACOPFData& dB, const std::vector<int>& Gk_no_AGC);
    void add_cons_AGC(SCACOPFData& dB, const std::vector<int>& Gk_AGC);
    void add_cons_AGC_simplified(SCACOPFData& dB, const std::vector<int>& Gk_AGC);
    void add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk);
  public: 
    //contingencies' SCACOPFData
    std::vector<SCACOPFData*> data_K;
  protected: 
    SCACOPFData& data_sc;
  public:
    //options
    bool useQPen;
    double slacks_scale;
    double AGCSmoothing, PVPQSmoothing;
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

    void write_solution_basecase();
    void write_solution_extras_basecase();
  }; // end of SCACOPFProblem



}

#endif

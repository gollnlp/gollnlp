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
	useQPen(false), slacks_scale(1.),  PVPQSmoothing(0.01), AGCSmoothing(0.01){}
    virtual ~SCACOPFProblem();
    
    virtual bool default_assembly();

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

    void add_obj_prod_cost(SCACOPFData& dB);

    void add_cons_coupling(SCACOPFData& dB);
    void add_cons_nonanticip(SCACOPFData& dB, const std::vector<int>& Gk_no_AGC);
    void add_cons_AGC(SCACOPFData& dB, const std::vector<int>& Gk_AGC);
    void add_cons_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk);
  protected: 
    SCACOPFData& data_sc;
    //contingencies' SCACOPFData
    std::vector<SCACOPFData*> data_K;
    
    //options
    bool useQPen;
    double slacks_scale;
    double AGCSmoothing, PVPQSmoothing;
  protected:
    //variables and constraints accessers
    inline std::string var_name(const std::string& prefix, const SCACOPFData& d) { 
      return prefix+"_"+std::to_string(d.id); 
    }
    inline OptVariablesBlock* variable(const std::string& prefix, const SCACOPFData& d) { 
      return vars_block(var_name(prefix, d));
    }
    inline std::string con_name(const std::string& prefix, const SCACOPFData& d) { 
      return prefix+"_"+std::to_string(d.id); 
    }
    //printing
    void print_p_g(SCACOPFData& dB);
    void print_p_g_with_coupling_info(SCACOPFData& dB);
    void print_PVPQ_info(SCACOPFData& dB);
    void print_Transf_powers(SCACOPFData& dB, bool SysCond_BaseCase = true);
  };

}

#endif
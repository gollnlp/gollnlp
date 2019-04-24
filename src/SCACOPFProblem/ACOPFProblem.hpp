#ifndef ACOPF_PROBLEM
#define ACOPF_PROBLEM

#include "OptProblem.hpp"

#include "SCACOPFData.hpp"
#include "OPFConstraints.hpp"
#include "OPFObjectiveTerms.hpp"

#include <cstring>
#include "blasdefs.hpp"
//this class is for ACOPF base case and is inherited by ACOPFContingencyProblem

namespace gollnlp {
  
  class ACOPFProblem : public OptProblem
  {
  public:
    ACOPFProblem(SCACOPFData& d_in) : data_sc(d_in), useQPen(false), slacks_scale(1.) {}
    OptProblem opt_prob;
    
    virtual bool default_assembly();

  protected:
    // for all add_ methods, dB is the block data (base case or contingency), but they
    // use 'd' SCACOPFData as well
    void add_variables(SCACOPFData& dB);
    void add_cons_lines_pf(SCACOPFData& dB);
    void add_cons_transformers_pf(SCACOPFData& dB);
    void add_cons_active_powbal(SCACOPFData& dB);
    void add_cons_reactive_powbal(SCACOPFData& dB);
    void add_cons_thermal_li_lims(SCACOPFData& dB);
    void add_cons_thermal_ti_lims(SCACOPFData& dB);
    void add_obj_prod_cost(SCACOPFData& dB);
  protected: 
    SCACOPFData& data_sc;
    //contingencies' SCACOPFData
    std::vector<SCACOPFData*> data_K;
    
    //options
    bool useQPen;
    double slacks_scale;

  protected:
    //utilities
    inline std::string var_name(const std::string& prefix, const SCACOPFData& d) { 
      return prefix+"_"+std::to_string(d.id); 
    }
    inline OptVariablesBlock* variable(const std::string& prefix, const SCACOPFData& d) { 
      return vars_block(var_name(prefix, d));
    }
    inline std::string con_name(const std::string& prefix, const SCACOPFData& d) { 
      return prefix+"_"+std::to_string(d.id); 
    }
  };

}

#endif

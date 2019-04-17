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
    ACOPFProblem(SCACOPFData& d_in) : d(d_in) {}
    OptProblem opt_prob;
    
    virtual bool default_assembly();
    virtual bool append_penalty_blocks(OptVariablesBlock* pg, const std::vector<int>& Gidxs);


  protected: 
    SCACOPFData& d;
  };

}

#endif

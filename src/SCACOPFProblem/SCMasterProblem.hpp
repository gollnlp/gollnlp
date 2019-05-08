#ifndef SCACOPF_MASTER_PROBLEM
#define SCACOPF_MASTER_PROBLEM

#include "SCACOPFProblem.hpp"

namespace gollnlp {

  class SCMasterProblem : public SCACOPFProblem
  {
  public:
    SCMasterProblem(SCACOPFData& d_in, const std::vector<int> K_Cont) 
      : SCACOPFProblem(d_in), Contingencies(K_Cont)
    {}
    SCMasterProblem(SCACOPFData& d_in)  : SCMasterProblem(d_in, d_in.K_Contingency) {}
    virtual ~SCMasterProblem() {}

    virtual bool default_assembly();
  public:
    std::vector<int> Contingencies;
  };

} //end namespace
#endif

#ifndef SC_CONTINGENCY_PROBLEM_CODE1_WITH_FIXING
#define SC_CONTINGENCY_PROBLEM_CODE1_WITH_FIXING

#include "ContingencyProblemWithFixing.hpp"

namespace gollnlp {
  class ContingencyProblemWithFixingCode1 : public ContingencyProblemWithFixing
  {
  public:
    ContingencyProblemWithFixingCode1(SCACOPFData& d_in, int K_idx_, 
				      int my_rank, int comm_size_,
				      std::unordered_map<std::string, 
				      gollnlp::OptVariablesBlock*>& dict_,
				      const int& num_K_done_, const double& time_so_far_,
				      bool safe_mode=false)
      : ContingencyProblemWithFixing(d_in, K_idx_, my_rank, comm_size_, dict_, 
				     num_K_done_, time_so_far_, safe_mode)
    {

    }

    virtual ~ContingencyProblemWithFixingCode1() {};

    //evaluates objective/penalty given pg0 and vn0 (these are 'in' arguments)
    virtual bool eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);
  };

} //end namespace
#endif

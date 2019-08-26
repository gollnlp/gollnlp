#ifndef SC_CONTINGENCY_PROBLEM_WITH_FIXING
#define SC_CONTINGENCY_PROBLEM_WITH_FIXING

#include "ContingencyProblem.hpp"
#include <vector>
#include <unordered_map>

namespace gollnlp {

  class ContingencyProblemWithFixing : public ContingencyProblem
  {
  public:
    ContingencyProblemWithFixing(SCACOPFData& d_in, int K_idx_, 
				 int my_rank,
				 std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict_basecase_vars_)
      : ContingencyProblem(d_in, K_idx_, my_rank), 
	dict_basecase_vars(dict_basecase_vars_)
    { };
    virtual ~ContingencyProblemWithFixing();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0)
    {
      return ContingencyProblem::default_assembly(vn0, thetan0, bs0, pg0, qg0);
    }
    virtual bool optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);

  protected:
    bool add_cons_AGC_simplified(SCACOPFData& dB, 
				 const std::vector<int>& idxs_pg0_AGC_particip, 
				 const std::vector<int>& idxs_pgK_AGC_particip,
				 OptVariablesBlock* pg0);

    bool do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
			    OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
				 OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_AGC(const double& smoothing, bool fixVoltage, OptVariablesBlock* pgk, OptVariablesBlock* delta);

    //uses 'dict_basecase_vars'
    bool set_warm_start_from_basecase();
    //also looks up 'dict_basecase_vars'
    bool warm_start_variable_from_basecase(OptVariables& v);
  protected:
    std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict_basecase_vars;
  };

} //end namespace
#endif

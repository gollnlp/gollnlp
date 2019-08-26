#ifndef SC_CONTINGENCY_PROBLEM_WITH_FIXING
#define SC_CONTINGENCY_PROBLEM_WITH_FIXING

#include "ContingencyProblem.hpp"
#include <vector>

namespace gollnlp {

  class ContingencyProblemWithFixing : public ContingencyProblem
  {
  public:
    ContingencyProblemWithFixing(SCACOPFData& d_in, int K_idx_, int my_rank)
      : ContingencyProblem(d_in, K_idx_, my_rank) { };
    virtual ~ContingencyProblemWithFixing();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0)
    {
      return ContingencyProblem::default_assembly(vn0, thetan0, bs0, pg0, qg0);
    }
    virtual bool optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);

    bool do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
			    OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
				 OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_AGC(const double& smoothing, bool fixVoltage, OptVariablesBlock* pgk, OptVariablesBlock* delta);


  };

} //end namespace
#endif

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
	dict_basecase_vars(dict_basecase_vars_), solv1_Pg_was_enough(true)
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

    void add_cons_pg_nonanticip_using(OptVariablesBlock* pg0,
				      const std::vector<int>& idxs_pg0_nonparticip, 
				      const std::vector<int>& idxs_pgK_nonparticip);

    // Make up for excess or surplus 'P' in the contingency by ramping up or down AGC generation.
    // Iterates over gens and fix them when the bounds are hit, till 'P' can be apparently made up for
    // or it cannot because there is not enough generation (return false in this case).

    // Needed ramp : delta1 * sum { alpha[i] : i in particip } = P  (needed delta)
    // Blocking  delta
    // delta2 = min { (Pub[i]-pg0[i])/alpha[i] : i in particip }  if P>0 (in which case delta>0)
    //        or
    //        = max { (Plb[i]-pg0[i])/alpha[i] : i in particip }  if P<0 (in which case delta<0)
    // if delta1<=delta2 it appears to be enough generation
    // else 
    //   fix pgK[i] to upper (P>0) or lower bounds (P<0) for those 'i' that block
    //   decrease P accordingly and repeat
    bool push_and_fix_AGCgen(SCACOPFData& dB, const double& P, const double& delta_in,
			     std::vector<int>& idxs_pg0_AGC_particip, std::vector<int>& idxs_pgK_AGC_particip,
			     std::vector<int>& idxs_pg0_nonparticip, std::vector<int>& idxs_pgK_nonparticip,
			     OptVariablesBlock* pg0, OptVariablesBlock* pgK,
			     std::vector<double>& Plb, std::vector<double>& Pub, std::vector<double>& alpha,
			     double& delta_out, double& delta_needed, double& delta_blocking, 
			     double& delta_lb, double& delta_ub,
			     double& residual_Pg);

     bool do_qgen_fixing_for_PVPQ(OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
			    OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
				 OptVariablesBlock* vnk, OptVariablesBlock* qgk);

    bool do_fixing_for_AGC(const double& smoothing, bool fixVoltage, OptVariablesBlock* pgk, OptVariablesBlock* delta);

    //uses 'dict_basecase_vars'
    bool set_warm_start_from_basecase();
    //also looks up 'dict_basecase_vars'
    bool warm_start_variable_from_basecase(OptVariables& v);

    void estimate_active_power_deficit(double& p_plus, double& p_minus, double& p_overall);
  protected:
    std::unordered_map<std::string, gollnlp::OptVariablesBlock*>& dict_basecase_vars;
    std::vector<int> solv1_pg0_partic_idxs, solv1_pgK_partic_idxs, solv1_pgK_nonpartic_idxs, solv1_pg0_nonpartic_idxs;
    double solv1_delta_out, solv1_delta_needed, solv1_delta_blocking, solv1_delta_lb, solv1_delta_ub, solv1_delta_optim;
    bool solv1_Pg_was_enough;
  };

} //end namespace
#endif

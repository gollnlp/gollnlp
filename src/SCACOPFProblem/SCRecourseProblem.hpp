#ifndef SC_RECOURSE_PROBLEM
#define SC_RECOURSE_PROBLEM

#include "SCACOPFProblem.hpp"

#include <vector>

namespace gollnlp {
  class SCRecourseProblem;

  class SCRecourseObjTerm : public OptObjectiveTerm {
  public:
    SCRecourseObjTerm(SCACOPFData& d_in, 
		      OptVariablesBlock* pg0, OptVariablesBlock* vn0, 
		      const std::vector<int>& K_Cont={});
    virtual ~SCRecourseObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);
  protected:
  private:
    std::vector<SCRecourseProblem*> recou_probs;
    SCACOPFData& data_sc;
    OptVariablesBlock *p_g0, *v_n0;
  };

  class SCRecourseProblem : public SCACOPFProblem
  {
  public:
    SCRecourseProblem(SCACOPFData& d_in, int K_idx_);
    virtual ~SCRecourseProblem();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);

  protected: 
    //indexes of non-participating AGC generators in data_K[0] and data_sc, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_nonpartic_idxs, pg0_nonpartic_idxs;
    // set lb = ub =pg0 for p_gK  for non-AGC generators
    void enforce_nonanticip_coupling(OptVariablesBlock* pg0);

    std::vector<int> pgK_partic_idxs, pg0_partic_idxs;
    void add_cons_AGC(OptVariablesBlock* pg0);
    void update_cons_AGC(OptVariablesBlock* pg0);


    void add_cons_PVPQ(OptVariablesBlock* vn0);
    void update_cons_PVPQ(OptVariablesBlock* vn0);

  protected:
    int K_idx;
  };
} //end namespace

#endif

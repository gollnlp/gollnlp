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
    virtual bool eval_f_grad();
  private:
    std::vector<SCRecourseProblem*> recou_probs;
    SCACOPFData& data_sc;
    OptVariablesBlock *p_g0, *v_n0;
    double f;
    double *grad_p_g0, *grad_v_n0;
  };

  class SCRecourseProblem : public SCACOPFProblem
  {
  public:
    SCRecourseProblem(SCACOPFData& d_in, int K_idx_);
    virtual ~SCRecourseProblem();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    // += evaluates recourse for pg0 and vn0 (these are 'in' arguments) and returns
    // the value of the recourse in the 'out' argument 'f' and of the gradient in the
    // 'out' arguments 'grad_' if grad_xxx!=NULL
    virtual bool eval_recourse(OptVariablesBlock* pg0, OptVariablesBlock* vn0,
			       double& f, double* grad_pg0=NULL, double *grad_vn0=NULL);
  protected: 
    //indexes of non-participating AGC generators in data_K[0] and data_sc, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_nonpartic_idxs, pg0_nonpartic_idxs;
    // set lb = ub =pg0 for p_gK  for non-AGC generators
    void bodyof_cons_nonanticip_using(OptVariablesBlock* pg0);
    inline void add_cons_nonanticip_using(OptVariablesBlock* pg0) {
      bodyof_cons_nonanticip_using(pg0);
      printf("SCRecourseProblem K_id %d: AGC: %lu gens NOT participating: fixed all of "
	     "them.\n", K_idx, pg0_nonpartic_idxs.size());
    }
    inline void update_cons_nonanticip_using(OptVariablesBlock* pg0) {
      bodyof_cons_nonanticip_using(pg0);
    }
    void add_grad_pg0_nonanticip_part_to(double* grad);

    std::vector<int> pgK_partic_idxs, pg0_partic_idxs;
    void add_cons_AGC_using(OptVariablesBlock* pg0);
    void update_cons_AGC_using(OptVariablesBlock* pg0);
    void add_grad_pg0_AGC_part_to(double* grad);

    void add_cons_PVPQ_using(OptVariablesBlock* vn0);
    void update_cons_PVPQ_using(OptVariablesBlock* vn0);
    void add_grad_vn0_to(double* grad);
  protected:
    int K_idx;
    bool restart;
    //options
    double relax_factor_nonanticip_fixing;
  };
} //end namespace

#endif

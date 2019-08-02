#ifndef SC_CONTINGENCY_PROBLEM
#define SC_CONTINGENCY_PROBLEM

#include "SCACOPFProblem.hpp"
#include <vector>

namespace gollnlp {

  class ContingencyProblem : public SCACOPFProblem
  {
  public:
    ContingencyProblem(SCACOPFData& d_in, int K_idx_, int my_rank);
    virtual ~ContingencyProblem();

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0);

    //evaluates objective/penalty given pg0 and vn0 (these are 'in' arguments)
    virtual bool eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);
    
    //similar to the above, but may use a different warm-starting procedure
    virtual bool optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f);

    virtual void get_solution_simplicial_vectorized(std::vector<double>& vsln);

    //
    // warm-starting
    //
    // these functions return false whenever there is a discrepancy between 
    // this' and srcProb's variables

    // warm-starts both primal and dual variables 
    //safe to call "reoptimize" with PD warm-start
    bool set_warm_start_from_base_of(SCACOPFProblem& srcProb);
    bool set_warm_start_from_contingency_of(SCACOPFProblem& srcProb);
    bool set_warm_start_from(ContingencyProblem& srcProb)
    {
      assert(false && "not implemented yet");
      return false;
    }

    //
    // non-anticipativity
    //
    // simply set lb = ub =pg0 for p_gK  for non-AGC generators
    void bodyof_cons_nonanticip_using(OptVariablesBlock* pg0);

    inline void add_cons_nonanticip_using(OptVariablesBlock* pg0) {
      bodyof_cons_nonanticip_using(pg0);
#ifdef DEBUG
      printf("ContProb K_id %d on rank %d: "
	     "AGC: %lu gens NOT participating: fixed all of them.\n",
	     K_idx, my_rank, pg0_nonpartic_idxs.size());
#endif
    }
    inline void update_cons_nonanticip_using(OptVariablesBlock* pg0) {
      bodyof_cons_nonanticip_using(pg0);
    }
  protected:
    //
    // AGC
    //

    //indexes of non-participating AGC generators in data_K[0].G_Generator and data_sc.G_Generator, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_nonpartic_idxs, pg0_nonpartic_idxs;
    //indexes of participating AGC generators in data_K[0].G_Generator and data_sc.G_Generator, respectively
    //these indexes exclude 'outidx' when K_idx is a generator contingency
    std::vector<int> pgK_partic_idxs, pg0_partic_idxs;
    //indexes of data_K[0].G_Generator in data_sc.G_Generator
    //these indexes exclude 'outidx' when K_idx is a generator contingency; otherwise Gk=0,1,2,...
    std::vector<int> Gk;

    void add_cons_AGC_using(OptVariablesBlock* pg0);
    void update_cons_AGC_using(OptVariablesBlock* pg0);

    //
    // PVPQ
    //
    void add_const_nonanticip_v_n_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    void add_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    void update_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);

    void add_regularizations();
  public:
    int K_idx;
    int my_rank;

    //regularizations: gamma* || x - xbasecase]]^2
    bool reg_vn, reg_thetan, reg_bs, reg_pg, reg_qg;
  protected:
    OptVariablesBlock *v_n0, *theta_n0, *b_s0, *p_g0, *q_g0;
  };

} //end namespace
#endif

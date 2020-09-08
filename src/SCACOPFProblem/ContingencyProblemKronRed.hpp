#ifndef SC_CONTINGENCY_PROBLEMKRON
#define SC_CONTINGENCY_PROBLEMKRON

#include "ACOPFProblemKronRed.hpp"
#include "SCACOPFProblem.hpp"

#include <vector>
#include "goUtils.hpp"

#ifdef GOLLNLP_FAULT_HANDLING
#include "goSignalHandling.hpp"
extern volatile sig_atomic_t solve_is_alive;
#endif


namespace gollnlp {

  class ContingencyProblemKronRedWithFixingCode1;
  
  class ContingencyProblemKronRed : public ACOPFProblemKronRed
  {
  friend class ContingencyProblemKronRedWithFixingCode1;
  public:
    ContingencyProblemKronRed(SCACOPFData& d_in, int K_idx_, int my_rank);
    virtual ~ContingencyProblemKronRed();

    virtual bool assemble();
    
    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);
    virtual bool default_assembly(OptVariablesBlock* vn0,
				  OptVariablesBlock* thetan0,
				  OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0,
				  OptVariablesBlock* qg0);

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
    bool set_warm_start_from(ContingencyProblemKronRed& srcProb)
    {
      assert(false && "not implemented yet");
      return false;
    }

    //
    // non-anticipativity
    //
    // simply set lb = ub =pg0 for p_gK  for non-AGC generators
    void bodyof_cons_nonanticip_using(OptVariablesBlock* pg0);
    void add_cons_nonanticip_using(OptVariablesBlock* pg0);
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
    std::vector<int> Gk_;

    // commented in cpp void add_cons_AGC_using(OptVariablesBlock* pg0);
    void update_cons_AGC_using(OptVariablesBlock* pg0);

    //
    // PVPQ
    //
    void add_const_nonanticip_v_n_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    // commented in cpp void add_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
    void update_cons_PVPQ_using(OptVariablesBlock* vn0, const std::vector<int>& Gk);
  public:
    //if no regularization term exists in the problem, one is added and 'primal_problem_changed' is called; 
    //otherwise the term is updated
    void regularize_vn(const double& gamma=1e-4);    
    void regularize_thetan(const double& gamma=1e-4);    
    void regularize_bs(const double& gamma=1e-4);    
    void regularize_pg(const double& gamma=1e-4);    
    void regularize_qg(const double& gamma=1e-4);    
  protected:
    //update gamma for all the above regularizations
    //internal "helper" function
    void update_regularizations(const double& gamma=1e-4);
  public:
    int K_idx;
    int my_rank;

  protected:
    OptVariablesBlock *v_n0, *theta_n0, *b_s0, *p_g0, *q_g0;
  protected:
    /* Member to which any callbacks to this class, e.g., @iterate_callback, are carbon-copied. 
     * This class usually does a great deal of work/logic to monitor performance and troubleshoot
     * abnormalities. For example 'XXXWithFixing' classes have a  complicated logic to for premature exit 
     * or deal with convergence issues.*/
    ContingencyProblemKronRedWithFixingCode1* cc_callbacks_;
  public:
    inline void set_cc_callback(ContingencyProblemKronRedWithFixingCode1* p)
    {
      cc_callbacks_ = p;
    }
  public:
    virtual bool iterate_callback(int iter, const double& obj_value,
				  const double* primals,
				  const double& inf_pr, const double& inf_pr_orig_pr, 
				  const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials, OptimizationMode mode,
				  const double* duals_con=NULL,
				  const double* duals_lb=NULL, const double* duals_ub=NULL);
    /** methods from SCACOPFProblem */
    
  public:
    //contingencies' SCACOPFData
    std::vector<SCACOPFData*> data_K;
  };

} //end namespace
#endif

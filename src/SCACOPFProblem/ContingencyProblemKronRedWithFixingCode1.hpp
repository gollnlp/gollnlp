#ifndef SC_CONTINGENCY_PROBLEMKRON_CODE1_WITH_FIXING
#define SC_CONTINGENCY_PROBLEMKRON_CODE1_WITH_FIXING

#include "ContingencyProblemWithFixing.hpp"
#include "ContingencyProblemKronRed.hpp"

namespace gollnlp {
  class ContingencyProblemKronRedWithFixingCode1 : public ContingencyProblemWithFixing
  {
  public:
    ContingencyProblemKronRedWithFixingCode1(SCACOPFData& d_in, int K_idx_in, 
					     int my_rank, int comm_size_,
					     std::unordered_map<std::string, 
					     gollnlp::OptVariablesBlock*>& dict_in,
					     const int& num_K_done_in, const double& time_so_far_in,
					     bool safe_mode=false)
      : ContingencyProblemWithFixing(d_in, K_idx_in, my_rank, comm_size_, dict_in, 
    				     num_K_done_in, time_so_far_in, safe_mode)
    {

      prob_mds_ = new ContingencyProblemKronRed(d_in, K_idx_in, my_rank); 
      p_li10=q_li10=p_li20=q_li20=p_ti10=q_ti10=p_ti20=q_ti20=NULL;
    }

    virtual ~ContingencyProblemKronRedWithFixingCode1()
    {
      delete prob_mds_;
    };

    virtual bool default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0);

    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0)
    {
      theta_n0=thetan0; b_s0=bs0; q_g0=qg0;
      return default_assembly(pg0, vn0);
    }
    
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0,
				  OptVariablesBlock* pli10, OptVariablesBlock* qli10,
				  OptVariablesBlock* pli20, OptVariablesBlock* qli20,
				  OptVariablesBlock* pti10, OptVariablesBlock* qti10,
				  OptVariablesBlock* pti20, OptVariablesBlock* qti20)
    {
      p_li10=pli10; q_li10=qli10; p_li20=pli20; q_li20=qli20;
      p_ti10=pti10; q_ti10=qti10; p_ti20=pti20; q_ti20=qti20;
      return default_assembly(vn0, thetan0, bs0, pg0, qg0);
    }
    
    //evaluates objective/penalty given pg0 and vn0 (these are 'in' arguments)
    virtual bool eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, double* data_for_master);

    virtual void use_nlp_solver(const std::string& name)
    {
      prob_mds_->use_nlp_solver(name);
    }

    virtual bool do_solve1();
    virtual bool do_solve2(bool first_solve_OK);
    
    
    inline void set_no_recourse_action(double* data_for_master, const double& pen=0.)
    {
      data_for_master[0] = pen;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
    }
    bool determine_recourse_action(double* data_for_master);
    //this is for transmission contingencies
    bool recourse_action_from_voltages(int outidx, bool isLine, double* info_out);

    //
    //overwrites 
    //
    bool add_cons_AGC_simplified(SCACOPFData& dB, 
				 const std::vector<int>& idxs_pg0_AGC_particip, 
				 const std::vector<int>& idxs_pgK_AGC_particip,
				 OptVariablesBlock* pg0);
    
    void add_cons_pg_nonanticip_using(OptVariablesBlock* pg0,
				      const std::vector<int>& idxs_pg0_nonparticip, 
				      const std::vector<int>& idxs_pgK_nonparticip);
    
    bool do_qgen_fixing_for_PVPQ(OptVariablesBlock* vnk, OptVariablesBlock* qgk)
    {
      //see ContingencyProblemWithFixing
      assert(false);
      return true;
    }

    void get_objective_penalties(double& pen_p_balance, double& pen_q_balance, 
				 double& pen_line_limits, double& pen_trans_limits);
    void estimate_active_power_deficit(double& p_plus, double& p_minus, double& p_overall);
    void estimate_reactive_power_deficit(double& q_plus, double& q_minus, double& q_overall);
  
  protected:
    OptVariablesBlock *p_li10, *q_li10, *p_li20, *q_li20, *p_ti10, *q_ti10, *p_ti20, *q_ti20;

    //this is the MDS Contigency Problem evaluation to which ops are delegated
    ContingencyProblemKronRed* prob_mds_;
  };

} //end namespace
#endif

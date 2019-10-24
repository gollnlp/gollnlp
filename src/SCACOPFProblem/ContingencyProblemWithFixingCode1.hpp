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
      p_li10=q_li10=p_li20=q_li20=p_ti10=q_ti10=p_ti20=q_ti20=NULL;
    }

    virtual ~ContingencyProblemWithFixingCode1() {};
    virtual bool default_assembly(OptVariablesBlock* vn0, OptVariablesBlock* thetan0, OptVariablesBlock* bs0, 
				  OptVariablesBlock* pg0, OptVariablesBlock* qg0,
				  OptVariablesBlock* pli10, OptVariablesBlock* qli10,
				  OptVariablesBlock* pli20, OptVariablesBlock* qli20,
				  OptVariablesBlock* pti10, OptVariablesBlock* qti10,
				  OptVariablesBlock* pti20, OptVariablesBlock* qti20)
    {
      p_li10=pli10; q_li10=qli10; p_li20=pli20; q_li20=qli20;
      p_ti10=pti10; q_ti10=qti10; p_ti20=pti20; q_ti20=qti20;
      return ContingencyProblem::default_assembly(vn0, thetan0, bs0, pg0, qg0);
    }
    //evaluates objective/penalty given pg0 and vn0 (these are 'in' arguments)
    virtual bool eval_obj(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f, double* data_for_master);

    inline void set_no_recourse_action(double* data_for_master, const double& pen=0.)
    {
      data_for_master[0] = pen;
      data_for_master[1]=data_for_master[2]=data_for_master[3]=data_for_master[4]=0.;
    }
    bool determine_recourse_action(double* data_for_master);
    //this is for transmission contingencies
    bool recourse_action_from_voltages(int outidx, bool isLine, double* info_out);

  protected:
    OptVariablesBlock *p_li10, *q_li10, *p_li20, *q_li20, *p_ti10, *q_ti10, *p_ti20, *q_ti20;
  };

} //end namespace
#endif

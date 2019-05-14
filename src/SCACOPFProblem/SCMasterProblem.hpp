#ifndef SCACOPF_MASTER_PROBLEM
#define SCACOPF_MASTER_PROBLEM

#include "SCACOPFProblem.hpp"
#include "SCRecourseProblem.hpp"

namespace gollnlp {

  class SCMasterProblem : public SCACOPFProblem
  {
  public:
    SCMasterProblem(SCACOPFData& d_in, const std::vector<int> K_Cont) 
      : SCACOPFProblem(d_in), Contingencies(K_Cont), recou_objterm(NULL)
    {}
    SCMasterProblem(SCACOPFData& d_in)  : SCMasterProblem(d_in, d_in.K_Contingency) {}
    virtual ~SCMasterProblem() {}

    virtual bool default_assembly();

    inline OptVariablesBlock* p_g0_vars() {return variable("p_g", data_sc);}
    inline OptVariablesBlock* v_n0_vars() {return variable("v_n", data_sc);}

    //overwrites of OptProblem
    virtual bool iterate_callback(int iter, const double& obj_value, const double* primals,
				  const double& inf_pr, const double& inf_du, 
				  const double& mu, 
				  const double& alpha_du, const double& alpha_pr,
				  int ls_trials);
    inline void append_recourse_obj(SCRecourseObjTerm* obj_term) {
      assert(NULL==recou_objterm && "recourse term already added");
      recou_objterm = obj_term;
      append_objterm(recou_objterm);
    }
  public:
    std::vector<int> Contingencies;
    SCRecourseObjTerm* recou_objterm;
  };

} //end namespace
#endif

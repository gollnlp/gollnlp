#ifndef GOLLNLP_OBJTERMS
#define GOLLNLP_OBJTERMS

#include "OptProblem.hpp"

namespace gollnlp {
  //
  // LinearPenaltyObjTerm
  // add term M*x to the objective, where x is enforced to be nonnegative
  //
  class LinearPenaltyObjTerm : public OptObjectiveTerm
  {
  public:
    LinearPenaltyObjTerm(const std::string& id_, OptVariablesBlock* x_, const double& M_=1.)
      : OptObjectiveTerm(id_), x(x_), M(M_) {}
    virtual ~LinearPenaltyObjTerm()  {}
    
    virtual bool eval_f(const OptVariables& vars, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars, bool new_x, double* grad);
  private:
    OptVariablesBlock* x;
    double M;
  };

} //end of namespace

#endif

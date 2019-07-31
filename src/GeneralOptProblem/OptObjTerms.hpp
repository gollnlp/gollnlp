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

  //
  // Quadratic regularization term
  // \gamma ||a.*x-x_0||^2 = ( sum(a[i]*x[i]-x_0[i])^2 )
  //
  class QuadrRegularizationObjTerm : public OptObjectiveTerm
  {
  public:
    QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
			       const double& gamma, double* a_, double* x0_);

    // a = all ones
    QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
			       const double& gamma, double* x0_);
    // x0 = x0scalar and a = all ones
    QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
			       const double& gamma, const double& x0scalar_);
    virtual ~QuadrRegularizationObjTerm();

    virtual bool eval_f(const OptVariables& vars, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars, bool new_x, double* grad);

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const double& obj_factor,
			       const int& nnz, int* i, int* j, double* M);

    virtual int get_HessLagr_nnz();
    // (i,j) entries in the HessLagr to which this term contributes to
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);
  protected:
    OptVariablesBlock* x;
    double gamma;
    double *x0, *a;
    //keep the index for each nonzero elem in the Hessian that this constraints block contributes to
    int *H_nz_idxs;
  };

  //
  // Quadratic away-from-bounds penalization QPen(x) = \gamma * q(x)
  //
  // Given l <= x <= u and u>l, q(x) is such that
  // q(l) = q(u) = 1 and q((u-l)/2)=0
  // 
  // The exact form: q(x) = 4/[(u-l)^2] * [ x - (l+u)/2 ]^2
  // or, equivalently, in the QuadrRegularizationObjTerm form
  //        ||   2          l+u  ||^2
  // q(x) = || ----- * x - ----- ||
  //        ||  u-l         u-l  ||
  //
  // To do: quartic q4(x) = 16/[(u-l)^4] * [ x - (l+u)/2 ]^4
  class QuadrAwayFromBoundsObjTerm : public QuadrRegularizationObjTerm
  {
  public:
    QuadrAwayFromBoundsObjTerm(const std::string& id_, OptVariablesBlock* x_, 
			       const double& gamma, 
			       const double* lb, const double* ub);
    virtual ~QuadrAwayFromBoundsObjTerm() {};
  };

} //end of namespace

#endif

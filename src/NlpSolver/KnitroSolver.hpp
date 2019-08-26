#ifndef GOLLNLP_KNITROSOLVER
#define GOLLNLP_KNITROSOLVER

#include "OptProblem.hpp"
#include "NlpSolver.hpp"

#include "KTRProblem.h"
#include "KTRSolver.h"

#include "goTimer.hpp"

using namespace Knitro;

#include <iostream>

namespace gollnlp {

class KnitroNlp : public knitro::KTRProblem
{
public:  
  /* constructor */
  KnitroNlp(OptProblem* p) :
    KTRProblem(p->get_num_variables(),
               p->get_num_constraints(),
	       p->get_nnzJaccons(),
	       p->get_nnzHessLagr()),
    prob(p)
  {
    setObjectiveProperties();
    setVariableProperties();
    setConstraintProperties();
    setDerivativeProperties();
    setStartingPoint();
  }
  
  /* destructor */
  virtual ~KnitroNlp()
  {
  }
  
  /* objective and constraints evaluator */
  double evaluateFC(const double* const x, double* const c, double* const objGrad,
                    double* const jac) {
    prob->eval_cons(x, true, c);
    return prob->eval_obj(x, true, obj_value);
  }
  
  /* gradient evaluator */
  int evaluateGA(const double* const x, double* const objGrad, double* const jac) {
    if(!prob->eval_gradobj(x, true, objGrad)) {
      return KTR_RC_CALLBACK_ERR;
    }
    if(!prob->eval_Jaccons(x, true, getNNZJ(), NULL, NULL, jac)) {
      return KTR_RC_CALLBACK_ERR;
    }
    return 0;
  }
  
  /* Lagrangian hessian evaluator */
  int evaluateHess(const double* const x, double objScaler, const double* const lambda,
                   double* const hess) {
    if(!prob->eval_HessLagr(x, true, objScaler, lambda, true, getNNZH(), NULL, NULL, hess)) {
      return KTR_RC_CALLBACK_ERR;
    }
    return 0;
  }
  
private:
  virtual void setObjectiveProperties() {
    setObjType(knitro::KTREnums::ObjectiveType::ObjQuadratic);  // PENDING: maybe ObjLinear useful when using piecewise linear penalties
    setObjGoal(knitro::KTREnums::ObjectiveGoal::Minimize);
  }
  
  virtual void setVariableProperties() {
    std::vector<double> x_bnd(getNumVars());
    prob->fill_vars_lower_bounds(x_bnd);
    setVarLoBnds(x_bnd);
    prob->fill_vars_upper_bounds(x_bnd);
    setVarUpBnds(x_bnd);
  }
  
  virtual void setConstraintProperties() {
    std::vector<double> bnds(getNumCons());
    setConTypes(knitro::KTREnums::ConstraintType::General); // PENDING: most constrains are actually quadratic -- to be revised
    prob->fill_cons_lower_bounds(bnds);
    setConLoBnds(bnds);
    prob->fill_cons_upper_bounds(bnds);
    setConUpBnds(bnds);
  }

  virtual void setDerivativeProperties() {
    
    // jacobian structure
    {
      auto jac_nnz = getNNZJ();
      std::vector<int> jac_iRow(jac_nnz);
      std::vector<int> jac_jCol(jac_nnz);
      prob->eval_Jaccons(NULL, false, jac_nnz, jac_iRow, jac_jCol, NULL);
      setJacIndexCons(jac_iRow);
      setJacIndexVars(jac_jCol);
    }
    
    // hessian structure
    {
      auto hess_nnz = getNNZH();
      std::vector<int> hess_iRow(hess_nnz);
      std::vector<int> hess_jCol(hess_nnz);
      prob->eval_HessLagr(NULL, false, 1.0, getNumCons(), NULL, false,
                          hess_nnz, hess_iRow, hess_jCol, NULL);
      setHessIndexRows(hess_iRow);
      setHessIndexCols(hess_jCol);
    }
    
  }

  virtual void setStartingPoint() {
    
    // get sizes
    auto nvars = getNumVars();
    auto ncons = getNumCons();
    
    // setting primal starting point
    {
      std::vector<double> x0(nvars);
      if(prob->fill_primal_start(x0)) {
        setXInitial(x0);
      }
    }

    // setting dual starting point
    // lambda contains both dual multipliers of constraints and bounds
    // assumed order: (constraints, bounds) see https://www.artelys.com/docs/knitro/3_referenceManual/oldAPI.html?highlight=lambdainitial
    {
      std::vector<double> lambda0(ncons + nvars);
      std::vector<double> lambda0_cons(lambda0.cbegin(), lambda0.cbegin()+ncons);
      std::vector<double> lambda0_bnds(lambda0.cbegin()+ncons, lambda0.cbegin()+ncons+nvars);
      if(!prob->fill_dual_cons_start(lambda0_cons)) {
        return;
      }
      {
        std::vector<double> z_L(nvars);
	std::vector<double> z_U(nvars);
	if(!prob->fill_dual_bounds_start(z_L, z_U)) {
	  return;
        }
        for(int i=0; i<nvars; i++){
          lambda0_bnds[i] = z_L[i] + z_U[i]; // either lower or upper bound is binding, not both at the same time
        }
      }
      setLambdaInitial(lambda0);
    }
    
  }

private:
  OptProblem* prob;

  KnitroNlp();
  KnitroNlp(const KnitroNlp&);
  KnitroNlp& operator=(const KnitroNlp&);

}

class KnitroSolver : public NlpSolver : {
  KnitroSolver(OptProblem* p_) : NlpSolver(p_) {}
  virtual ~KnitroSolver() {}
  
  virtual bool initialize() {
    knitro_nlp_spec = new gollnlp::KnitroNlp(prob);
    app = new knitro::KTRSolver(knitro_nlp_spec);
    return true;
  }
  
  virtual bool optimize() {
    
    // solve optimization problem
    int status = app->asolve();
    if(status == knitro::KN_RC_OPTIMAL_OR_SATISFACTORY ||
       status == knitro::KN_RC_NEAR_OPT) {
    } else {
      printf("Knitro solve FAILED with status %d!!!\nSee https://www.artelys.com/docs/knitro/3_referenceManual/returnCodes.html#returncodes\n", status);
      return false;
    }
    
    // record solution -- knitro does not call a finalize procedure
    OptProblem& p = knitro_nlp_spec->prob;
    p->set_obj_value(app->getObjValue());
    p->set_num_iters(app->getNumberIters());
    {
      std::vector<double>& x_sol = app->getXValues();
      p->set_primal_vars(x_sol);
    }
    {
      auto nvars = app->getNumVars();
      auto ncons = app->getNumCons();
      std::vector<double>& lambda_sol = app->getLambdaValues();
      std::vector<double> lambda_cons(lambda_sol.cbegin(), lambda_sol.cbegin()+ncons);
      std::vector<double> lambda_bnds(lambda_sol.cbegin()+ncons, lambda_sol.cbegin()+ncons+nvars);
      p->set_duals_vars_cons(lambda_cons);
      std::vector<double> z_L(nvars);
      std::vector<double> z_U(nvars);
      for(int i=0; i<nvars; i++){
        if(lambda_bnds[i] > 0) {
          z_L[i] = lambda_bnds[i];
	  z_U[i] = 0.0;
        }else{
          z_L[i] = 0.0;
          z_U[i] = lambda_bnds[i];
        }
      }
      p->set_duals_vars_bounds(z_L, z_U);
    }
    return true;
    
  }
  
  virtual bool set_option(const std::string& name, int value) {
    app->setParam(name, value);
    return true;
  }
  
  virtual bool set_option(const std::string& name, double value) {
    app->setParam(name, value);
    return true;
  };
  
  virtual bool set_option(const std::string& name, const std::string& value) {
    app->setParam(name, value);
    return true;
  };
  
private:
  gollnlp::KnitroNlp* knitro_nlp_spec;
  knitro::KTRSolver* app;
}

} //endnamespace
#endif

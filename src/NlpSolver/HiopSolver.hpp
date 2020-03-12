#ifndef GOLLNLP_HIOPSOLVER
#define GOLLNLP_HIOPSOLVER

#include "OptProblem.hpp"
#include "NlpSolver.hpp"

#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "goTimer.hpp"

//using namespace hiop;

#include <iostream>

namespace gollnlp {

  class HiopNlp : public hiop::hiopInterfaceMDS
{
public:
  /**constructor */
  HiopNlp(OptProblem* p) : prob(p)
  { 
  } 

  /** default destructor */
  virtual ~HiopNlp()
  {
  }

  bool get_prob_sizes(long long& n, long long& m)
  { 
    n = prob->get_num_variables();
    m = prob->get_num_constraints();
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    assert(n == prob->get_num_variables());
    prob->fill_vars_lower_bounds(xlow);
    prob->fill_vars_upper_bounds(xupp);
    for(int i=0; i<n; i++) type[i]=hiopNonlinear;
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == prob->get_num_constraints());
    prob->fill_cons_lower_bounds(clow);
    prob->fill_cons_upper_bounds(cupp);
    for(int i=0; i<m; i++) type[i]=hiopNonlinear;
    return true;
  }

  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
  {
    return true;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    return true;
  }

  bool eval_cons(const long long& n, const long long& m, 
		 const long long& num_cons, const long long* idx_cons,  
		 const double* x, bool new_x, double* cons)
  {
    return true;
  }

  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    return true;
  }

  bool eval_Jac_cons(const long long& n, const long long& m, 
		     const long long& num_cons, const long long* idx_cons,
		     const double* x, bool new_x,
		     const long long& nsparse, const long long& ndense, 
		     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		     double** JacD)
  {
    return true;
  }

  bool eval_Hess_Lagr(const long long& n, const long long& m, 
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const long long& nsparse, const long long& ndense, 
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			      double** HDD,
			      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    return true;
  }

  bool get_starting_point(const long long& global_n, double* x0)
  {
    if(!prob->fill_primal_start(x0)) return false;
    return true;
  }
private:
  OptProblem* prob;

  /**@name Methods to block default compiler methods.
   */
  //@{
  HiopNlp();
  HiopNlp(const HiopNlp&);
  HiopNlp& operator=(const HiopNlp&);
  //@}
};

class HiopSolver : public NlpSolver {
public:
  HiopSolver(OptProblem* p_) : NlpSolver(p_), app_status(Invalid_Option) {}
  virtual ~HiopSolver() {}

  virtual bool set_start_type(OptProblem::RestartType t)
  {
    // if(t==OptProblem::primalDualRestart) {
    //   app->Options()->SetStringValue("warm_start_init_point", "yes");
    // 	ipopt_nlp_spec->set_advanced_primaldual_restart(false);
    // } else {
    //   if(t==OptProblem::primalRestart)
    // 	app->Options()->SetStringValue("warm_start_init_point", "no");
    //   else { //	advancedPrimalDualRestart	
    // 	app->Options()->SetStringValue("warm_start_init_point", "yes");
    // 	app->Options()->SetStringValue("warm_start_entire_iterate", "yes");
    // 	ipopt_nlp_spec->set_advanced_primaldual_restart(true);
    //   }
    // }
    return true;
  }


  virtual bool initialize() {

    hiop_nlp_spec = new gollnlp::HiopNlp(prob);
    return true;
  }

  OptimizationStatus return_code() const {
    return app_status;
  }

  virtual int optimize() {
    hiop::hiopNlpMDS nlp(*hiop_nlp_spec);
    hiop::hiopAlgFilterIPMNewton solver(&nlp);
    hiop::hiopSolveStatus status = solver.run();
    double objective = solver.getObjective();

    // Ask Ipopt to solve the problem
    // app_status = app->OptimizeTNLP(ipopt_nlp_spec);

    // if (app_status == Ipopt::Solve_Succeeded || 
    // 	app_status == Ipopt::Solved_To_Acceptable_Level) {
    //   //|| app_status == Ipopt::User_Requested_Stop) {
    //   return true;
    // }
    // else {
    //   if(app_status != Ipopt::User_Requested_Stop)
    // 	printf("Ipopt solve FAILED with status %d!!!\n", app_status);
    //   return false;
    // }
    return true;
  }

  virtual int reoptimize() {
    assert(false);
    return false;
  }

  virtual bool set_option(const std::string& name, int value)
  {
    //app->Options()->SetIntegerValue(name, value);
    assert(false);return true;
  }
  virtual bool set_option(const std::string& name, double value)
  {
    //app->Options()->SetNumericValue(name, value);
    assert(false);return true;
  };
  virtual bool set_option(const std::string& name, const std::string& value)
  {
    //app->Options()->SetStringValue(name, value);
    assert(false);return true;
  };

private:
  HiopNlp* hiop_nlp_spec;
  OptimizationStatus app_status;
};



} //endnamespace
#endif

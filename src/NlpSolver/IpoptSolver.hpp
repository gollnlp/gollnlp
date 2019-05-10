#ifndef GOLLNLP_IPOPTSOLVER
#define GOLLNLP_IPOPTSOLVER

#include "OptProblem.hpp"
#include "NlpSolver.hpp"

#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"
#include "IpIpoptData.hpp"

#include "goTimer.hpp"

using namespace Ipopt;

#include <iostream>

namespace gollnlp {

class IpoptNlp : public Ipopt::TNLP
{
public:
  /**constructor */
  IpoptNlp(OptProblem* p) : prob(p), have_adv_pd_restart(false) 
  { assert(prob); } 

  /** default destructor */
  virtual ~IpoptNlp() {};

  /**@name Overloaded from TNLP */
  //@{
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                            Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    n = prob->get_num_variables();
    m = prob->get_num_constraints();
    nnz_jac_g = prob->get_nnzJaccons();
    nnz_h_lag = prob->get_nnzHessLagr();
    return true;
  }

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Ipopt::Index n, Number* x_l, Number* x_u,
                               Ipopt::Index m, Number* g_l, Number* g_u)
  {
    assert(n == prob->get_num_variables());
    assert(m == prob->get_num_constraints());
    prob->fill_vars_lower_bounds(x_l);
    prob->fill_vars_upper_bounds(x_u);
    prob->fill_cons_lower_bounds(g_l);
    prob->fill_cons_upper_bounds(g_u);
    return true;
  }

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Number* x,
                                  bool init_z, Number* z_L, Number* z_U,
                                  Ipopt::Index m, bool init_lambda,
                                  Number* lambda)
  {
    std::cout << init_x << " " << init_z << " " << init_lambda << std::endl;
    if(init_x)
      if(!prob->fill_primal_start(x)) return false;
    if(init_z)
      if(!prob->fill_dual_bounds_start(z_L, z_U)) return false;
    if(init_lambda)
      if(!prob->fill_dual_cons_start(lambda)) return false;	
    return true;
  }

  /** Method to return the objective value */
  virtual bool eval_f(Ipopt::Index n, const Number* x, bool new_x, Number& obj_value) {
    assert(prob->get_num_variables() == n);
    return prob->eval_obj(x, new_x, obj_value);
  }

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Ipopt::Index n, const Number* x, bool new_x, Number* grad_f) {
    return prob->eval_gradobj(x, new_x, grad_f);
  }

  /** Method to return the constraint residuals */
  virtual bool eval_g(Ipopt::Index n, const Number* x, bool new_x, Ipopt::Index m, Number* g)
  {
    assert( m == prob->get_num_constraints() );
    return prob->eval_cons(x, new_x, g);
  }

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Ipopt::Index n, const Number* x, bool new_x,
                          Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                          Number* values)
  {
    return prob->eval_Jaccons(x, new_x, nele_jac, iRow, jCol, values);
  }

  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Ipopt::Index n, const Number* x, bool new_x,
                      Number obj_factor, Ipopt::Index m, const Number* lambda,
                      bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                      Ipopt::Index* jCol, Number* values)
  {
    return prob->eval_HessLagr(x, new_x, obj_factor, lambda, new_lambda, nele_hess, iRow, jCol, values);
  }

  //@}

  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Ipopt::Index n, const Number* x, const Number* z_L, const Number* z_U,
                                 Ipopt::Index m, const Number* g, const Number* lambda,
                                 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq)

  {
    //SmartPtr< const IteratesVector > a = 	ip_data->curr () ;
    //IteratesVector b = *a;
    prob->set_obj_value(obj_value);
    prob->set_primal_vars(x);
    prob->set_duals_vars_bounds(z_L, z_U);
    prob->set_duals_vars_cons(lambda);
    iter_vector = ip_data->curr () ;
  }

  
  virtual bool intermediate_callback(AlgorithmMode mode,
				     Ipopt::Index iter, Number obj_value,
				     Number inf_pr, Number inf_du,
				     Number mu, Number d_norm,
				     Number regularization_size,
				     Number alpha_du, Number alpha_pr,
				     Ipopt::Index ls_trials,
				     const IpoptData* ip_data,
				     IpoptCalculatedQuantities* ip_cq)

  {
    return true;
  }

  virtual bool get_warm_start_iterate(IteratesVector& warm_start_iterate)
  {
    // really advanced primal-dual start -> warm_start_entire_iterate
    // https://github.com/coin-or/Bonmin/blob/master/Bonmin/src/Interfaces/BonTMINLP2TNLP.cpp
    // https://list.coin-or.org/pipermail/ipopt/2011-April/002428.html
    if(have_adv_pd_restart) {
	printf("Using advanced pd restart\n");
	goTimer t; t.start();
	warm_start_iterate.Copy(*iter_vector);
	t.stop();
	printf("copy took %g sec\n", t.getElapsedTime());
      return true;
    }
    return false;
  }

  virtual void set_advanced_primaldual_restart(bool v) { have_adv_pd_restart=v; }

  //@}
private:
  OptProblem* prob;
  SmartPtr< const IteratesVector > iter_vector;
  bool have_adv_pd_restart;
  /**@name Methods to block default compiler methods.
   */
  //@{
  IpoptNlp();
  IpoptNlp(const IpoptNlp&);
  IpoptNlp& operator=(const IpoptNlp&);
  //@}
};

class IpoptSolver : public NlpSolver {
public:
  IpoptSolver(OptProblem* p_) : NlpSolver(p_) {}
  virtual ~IpoptSolver() {}

  virtual bool set_start_type(OptProblem::RestartType t)
  {
    if(t==OptProblem::primalDualRestart) {
      app->Options()->SetStringValue("warm_start_init_point", "yes");
	ipopt_nlp_spec->set_advanced_primaldual_restart(false);
    } else {
      if(t==OptProblem::primalRestart)
	app->Options()->SetStringValue("warm_start_init_point", "no");
      else {		
	app->Options()->SetStringValue("warm_start_init_point", "yes");
	app->Options()->SetStringValue("warm_start_entire_iterate", "yes");
	ipopt_nlp_spec->set_advanced_primaldual_restart(true);
      }
    }
    return true;
  }

  virtual bool initialize() {
    app = IpoptApplicationFactory();
    // Intialize the IpoptApplication and process the options
    ApplicationReturnStatus status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded) {
      printf("\n\n*** Error during initialization!\n");
      return false;
    }

    ipopt_nlp_spec = new gollnlp::IpoptNlp(prob);

    return true;
  }

  virtual int optimize() {

    // Ask Ipopt to solve the problem
    ApplicationReturnStatus status = app->OptimizeTNLP(ipopt_nlp_spec);

    if (status == Ipopt::Solve_Succeeded) {
      printf("\n\n*** The problem solved!\n");
      return true;
    }
    else {
      printf("\n\n*** The problem FAILED!\n");
      return false;
    }
  }
  virtual bool set_option(const std::string& name, int value)
  {
    app->Options()->SetIntegerValue(name, value);
    return true;
  }
  virtual bool set_option(const std::string& name, double value)
  {
    app->Options()->SetNumericValue(name, value);
    return true;
  };
  virtual bool set_option(const std::string& name, const std::string& value)
  {
    app->Options()->SetStringValue(name, value);
    return true;
  };

private:
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app;
  Ipopt::SmartPtr<gollnlp::IpoptNlp> ipopt_nlp_spec;
};



} //endnamespace
#endif

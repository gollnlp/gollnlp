#ifndef GOLLNLP_IPOPTSOLVER
#define GOLLNLP_IPOPTSOLVER

#include "OptProblem.hpp"
#include "NlpSolver.hpp"

#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
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
  { 
    assert(prob); 
    _primals=NULL;
    _primals_sz=0;
    _duals_con=NULL;;
    _duals_con_sz=0;
    _duals_lb=_duals_ub=NULL;
    _duals_b_sz=0;
  } 

  /** default destructor */
  virtual ~IpoptNlp()
  {
    _primals_sz = 0;
    delete[] _primals;
    _primals = NULL;

    _duals_con_sz=0;
    delete[] _duals_con;
    _duals_con=NULL;;

    _duals_b_sz=0;
    delete [] _duals_lb;
    delete [] _duals_ub;
    _duals_lb=_duals_ub=NULL;    
  }

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
    index_style = C_STYLE;
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
    //std::cout << init_x << " " << init_z << " " << init_lambda << std::endl;
    if(init_x)
      if(!prob->fill_primal_start(x)) return false;
    if(init_z) {
      bool bret = prob->fill_dual_bounds_start(z_L, z_U);
#ifdef DEBUG
      //for(int i=0; i<n; i++) printf("[outt] %4d %15.8e %15.8e\n", i, z_L[i], z_U[i]);
#endif
      if(!bret) return false;
    }
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
    prob->set_obj_value_barrier(ip_cq->curr_barrier_obj());
    prob->set_num_iters(ip_data->iter_count());
    //assert(false);
    prob->set_primal_vars(x);
    prob->set_duals_vars_bounds(z_L, z_U);
#ifdef DEBUG
    //for(int i=0; i<n; i++) printf("[inn] %4d %15.8e %15.8e\n", i, z_L[i], z_U[i]);
#endif    
    prob->set_duals_vars_cons(lambda);
    //iter_vector = ip_data->curr () ;

    prob->iterate_finalize();
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
    //! this is working code - commented out since it is not needed anymore
    // Ipopt::TNLPAdapter* tnlp_adapter = NULL;
    // if( ip_cq != NULL ) {
    //   bool bret;
    //   Ipopt::OrigIpoptNLP* orignlp;
    //   orignlp = dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
    //   if( orignlp != NULL ) {
    // 	tnlp_adapter = dynamic_cast<TNLPAdapter*>(GetRawPtr(orignlp->nlp()));

    // 	int n = prob->get_num_variables();
    // 	double* primals = new double[n];
    // 	tnlp_adapter->ResortX(*ip_data->curr()->x(), primals);
    // 	bret = prob->iterate_callback(iter, obj_value, primals, inf_pr, inf_du, mu, 
    // 				      alpha_du, alpha_pr, ls_trials);
    // 	delete[] primals;
    // 	return bret;
    //   } else {
    // 	// orignlp == NULL
    // 	//Ipopt in restauration -> call with primals = NULL
    //   }
    // } 
    double inf_pr_orig_problem = inf_pr;
    if(NULL!=ip_cq) {
      inf_pr_orig_problem = ip_cq->curr_nlp_constraint_violation(Ipopt::NORM_MAX);

      //get primal variables
      Ipopt::OrigIpoptNLP* orignlp = dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
      if( orignlp != NULL ) {
     	Ipopt::TNLPAdapter* tnlp_adapter = dynamic_cast<TNLPAdapter*>(GetRawPtr(orignlp->nlp()));
	if(tnlp_adapter) {
	  if(_primals_sz != prob->get_num_variables()) {
	    delete [] _primals;
	    _primals = NULL; 
	    _primals_sz = 0;
	  }
	  if(NULL == _primals) {
	    _primals_sz = prob->get_num_variables();
	    _primals = new double[_primals_sz];
	  }
	  tnlp_adapter->ResortX(*ip_data->curr()->x(), _primals);

	  if(prob->requests_intermediate_duals()) {
	    if(_duals_con_sz != prob->get_num_constraints()) {
	      delete [] _duals_con;
	      _duals_con = NULL;
	      _duals_con_sz = 0;
	    }
	    if(_duals_b_sz != prob->get_num_variables()) {
	      delete [] _duals_lb;
	      delete [] _duals_ub;
	      _duals_lb =  _duals_ub = NULL;
	      _duals_b_sz = 0;
	    }
	    if( NULL==_duals_con ) {
	      _duals_con_sz = prob->get_num_constraints();
	      _duals_con = new double[_duals_con_sz];
	    }
	    if(NULL==_duals_lb) {
	      assert(NULL==_duals_ub);
	      _duals_b_sz = prob->get_num_variables();
	      _duals_lb = new double[_duals_b_sz];
	      _duals_ub = new double[_duals_b_sz];
	    }
	    tnlp_adapter->ResortG(*ip_data->curr()->y_c(), *ip_data->curr()->y_d(), _duals_con);
	    tnlp_adapter->ResortBnds(*ip_data->curr()->z_L(), _duals_lb,
				     *ip_data->curr()->z_U(), _duals_ub);
	  }

	  return prob->iterate_callback(iter, obj_value, _primals, inf_pr, inf_pr_orig_problem, inf_du, mu, 
					alpha_du, alpha_pr, ls_trials, mode,
					_duals_con, _duals_lb, _duals_ub);
	  
	}
      }
    }

    // algorithm in restoration or some other abnormal Ipopt situation -> primals=NULL
    // Also, passing primals=NULL under most normal/common circumstances since there is no need for them
    return prob->iterate_callback(iter, obj_value, NULL, inf_pr, inf_pr_orig_problem, inf_du, mu, 
				  alpha_du, alpha_pr, ls_trials, mode);
  }

  virtual bool get_warm_start_iterate(IteratesVector& warm_start_iterate)
  {
    // really advanced primal-dual start -> warm_start_entire_iterate
    // https://github.com/coin-or/Bonmin/blob/master/Bonmin/src/Interfaces/BonTMINLP2TNLP.cpp
    // https://list.coin-or.org/pipermail/ipopt/2011-April/002428.html
    if(have_adv_pd_restart) {
	printf("Using advanced pd restart\n");
	goTimer t; t.start();
	assert(false && "do not use -- needs work");
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

  double* _primals;
  int _primals_sz;

  double* _duals_con;
  int _duals_con_sz;

  double *_duals_lb, *_duals_ub;
  int _duals_b_sz;
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
  IpoptSolver(OptProblem* p_) : NlpSolver(p_), app_status(Invalid_Option) {}
  virtual ~IpoptSolver() {}

  virtual bool set_start_type(OptProblem::RestartType t)
  {
    if(t==OptProblem::primalDualRestart) {
      app->Options()->SetStringValue("warm_start_init_point", "yes");
	ipopt_nlp_spec->set_advanced_primaldual_restart(false);
    } else {
      if(t==OptProblem::primalRestart)
	app->Options()->SetStringValue("warm_start_init_point", "no");
      else { //	advancedPrimalDualRestart	
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
    app_status = app->Initialize();
    if (app_status != Ipopt::Solve_Succeeded) {
      printf("\n\n*** Error during initialization!\n");
      return false;
    }
    ipopt_nlp_spec = new gollnlp::IpoptNlp(prob);
    return true;
  }

  OptimizationStatus return_code() const {
    return app_status;
  }

  virtual int optimize() {

    // Ask Ipopt to solve the problem
    app_status = app->OptimizeTNLP(ipopt_nlp_spec);

    if (app_status == Ipopt::Solve_Succeeded || 
	app_status == Ipopt::Solved_To_Acceptable_Level) {
      //|| app_status == Ipopt::User_Requested_Stop) {
      return true;
    }
    else {
      if(app_status != Ipopt::User_Requested_Stop)
	printf("Ipopt solve FAILED with status %d!!!\n", app_status);
      return false;
    }
  }

  virtual int reoptimize() {
    // Ask Ipopt to solve the problem
    app_status = app->ReOptimizeTNLP(ipopt_nlp_spec);

    if (app_status == Ipopt::Solve_Succeeded || 
	app_status == Ipopt::Solved_To_Acceptable_Level) {
      //|| app_status == Ipopt::User_Requested_Stop) {
      return true;
    }
    else {
      //if(status != Ipopt::User_Requested_Stop)
      printf("Ipopt resolve FAILED with status %d!!!\n", app_status);
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
  OptimizationStatus app_status;
};



} //endnamespace
#endif

#ifndef GOLLNLP_HIOPSOLVER_SPARSE
#define GOLLNLP_HIOPSOLVER_SPARSE

#include "OptProblem.hpp"
#include "NlpSolver.hpp"

#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "goTimer.hpp"

#include "IpoptSolver.hpp" 

#include "IpoptAdapter.hpp"

#include <iostream>

namespace gollnlp {

  class HiopNlp : public hiop::hiopInterfaceSparse
  {
  public:
    /**constructor */
    HiopNlp(OptProblem* p)
      : prob(p)
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

    bool get_sparse_blocks_info(int& nx,
                                int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
                                int& nnz_sparse_Hess_Lagr)
    {
      nx = prob->get_num_variables();
      int m = prob->get_num_constraints();

      int nnz_Jac = prob->compute_nnzJaccons();
      double clow[m];
      double cupp[m];
      prob->fill_cons_lower_bounds(clow);
      prob->fill_cons_upper_bounds(cupp);
      
      prob->compute_nnzJac_eq_ineq(nx, m, clow, cupp, nnz_sparse_Jaceq, nnz_sparse_Jacineq);
      assert(nnz_Jac == nnz_sparse_Jace+nnz_sparse_Jaci);
      
      nnz_sparse_Hess_Lagr = prob->compute_nnzHessLagr();
      
      printf("n=%d  m=%d  nnz_Jaceq=%d nnz_Jacineq=%d  nnz_Hess=%d\n",
             nx, m, nnz_sparse_Jaceq, nnz_sparse_Jacineq, nnz_sparse_Hess_Lagr);
      fflush(stdout);
      
      return true;
    }

    bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
    {
      assert(prob->get_num_variables() == n);
      return prob->eval_obj(x, new_x, obj_value);
    }

    bool eval_cons(const long long& n, const long long& m, 
		   const long long& num_cons, const long long* idx_cons,  
		   const double* x, bool new_x, double* cons)
    {
      //we work with the one-call / full-body constraints
      return false;
    }
    bool eval_cons(const long long& n, const long long& m, 
		   const double* x, bool new_x, 
		   double* cons)
    {
      return prob->eval_cons(x, new_x, cons);
    }

    bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
    {
      return prob->eval_gradobj(x, new_x, gradf);
      return true;
    }

    bool eval_Jac_cons(const long long& n, const long long& m,
                       const long long& num_cons, const long long* idx_cons,
                       const double* x, bool new_x,
                       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
    {
      //force hiop to use the one-call method below
      return false;
    }
    
    bool eval_Jac_cons(const long long& n, const long long& m,
                       const double* x, bool new_x,
                       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
    {
      return prob->eval_Jaccons(x, new_x, nnzJacS, iJacS, jJacS, MJacS);
    }
    
    bool eval_Hess_Lagr(const long long& n, const long long& m,
                        const double* x, bool new_x, const double& obj_factor,
                        const double* lambda, bool new_lambda,
                        const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
    {
      return prob->eval_HessLagr(x, new_x, obj_factor, lambda, new_lambda, nnzHSS, iHSS, jHSS, MHSS);
    }

    bool get_starting_point(const long long& global_n, double* x0)
    {
      if(!prob->fill_primal_start(x0)) return false;
      return true;
    }

    /**
     * Method provides a primal or a primal-dual primal-dual starting point. This point is subject 
     * to internal adjustments in HiOp.
     *
     * If the user (implementer of this method) has good estimates only of the primal variables,
     * the method should populates 'x0' with these values and return true. The 'duals_avail' 
     * should be set to false; internally, HiOp will not access 'z_bndL0', 'z_bndU0', and 
     * 'lambda0'.
     *
     * If the user (implementer of this method) has good estimates of the duals of bound constraints 
     * and of inequality and equality constraints, 'duals_avail' boolean argument should 
     * be set to true and the respective duals should be provided (in 'z_bndL0' and 'z_bndU0' and 
     * 'lambda0', respectively). In this case, the user should also set 'x0' to his/her estimate 
     * of primal variables and return 'true'.
     *
     * If user does not have high-quality (primal or primal-dual) starting points, the method should 
     * return false (see note below).
     *
     * Note: when this method returns false, HiOp will call the overload
     * @get_starting_point(long long&, double*)
     * This behaviour is for backward compatibility and will be removed in a future release.
     * 
     */
    virtual bool get_starting_point(const long long& n, const long long& m,
				    double* x0,
				    bool& duals_avail,
				    double* z_bndL0, double* z_bndU0,
				    double* lambda0)
    {
      if(warm_start_type_ == OptProblem::primalRestart) {
	duals_avail = false;
	return prob->fill_primal_start(x0);
      } else {
	duals_avail = true;
	assert(false && "not yet implemented");
	return false;
      }
      return true;
    }

    void solution_callback(hiop::hiopSolveStatus status,
			   int n, const double* x,
			   const double* z_L,
			   const double* z_U,
			   int m, const double* g,
			   const double* lambda,
			   double obj_value)
    {
      prob->set_obj_value(obj_value);
      prob->set_obj_value_barrier(1e+20);//
      prob->set_num_iters(-1);//ip_data->iter_count());

      prob->set_primal_vars(x);
      prob->set_duals_vars_bounds(z_L, z_U);

      prob->set_duals_vars_cons(lambda);
      
      prob->iterate_finalize();
    }

    /** Methods not part of hiopInterfaceSparse */
    void set_warm_start_type(OptProblem::RestartType t)
    {
      warm_start_type_ = t;
    }
  private:
    OptProblem* prob;
    OptProblem::RestartType warm_start_type_;
    /** Block default compiler methods.
     */
    HiopNlp();
    HiopNlp(const HiopNlp&);
    HiopNlp& operator=(const HiopNlp&);
  };

  /** 
   *Wrapper class to feed OptProblem into a hiopInterfaceSparse nlp object
   */
  
  class HiopSolverSparse : public NlpSolver {
  public:
    HiopSolverSparse(OptProblem* p_)
      : NlpSolver(p_), hiop_nlp_spec(NULL), app_status(OptProblem::Invalid_Option)
    {
    }
  public:
    virtual ~HiopSolverSparse()
    {
      delete hiop_nlp_spec;
    }

    virtual bool set_start_type(OptProblem::RestartType t)
    {
      warm_start_type_ = t;
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

      hiop_nlp_spec = new gollnlp::HiopNlp(this->prob);
      return true;
    }

    OptimizationStatus return_code() const {
      return app_status;
    }

    virtual int optimize() {
      hiop::hiopNlpSparse nlp(*hiop_nlp_spec);
      hiop::hiopAlgFilterIPMNewton solver(&nlp);

      printf("[55555] hiop optimize\n");
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
      if(warm_start_type_==OptProblem::primalRestart) {

      } else if(warm_start_type_==OptProblem::primalRestart) {
	
      } else {
	assert(false && "warm-start type not supported");
	warm_start_type_=OptProblem::primalRestart;
      }
      return true;
    }

    virtual bool set_option(const std::string& name, int value)
    {
      //app->Options()->SetIntegerValue(name, value);
      //assert(false);
      return true;
    }
    virtual bool set_option(const std::string& name, double value)
    {
      //app->Options()->SetNumericValue(name, value);
      //assert(false);
      return true;
    };
    virtual bool set_option(const std::string& name, const std::string& value)
    {
      //app->Options()->SetStringValue(name, value);
      //assert(false);
      return true;
    };

  private:
    gollnlp::HiopNlp* hiop_nlp_spec;
    OptimizationStatus app_status;

    /**
     * Internal member that keeps track of calls to @set_start_type by the user of this glue code
     * and decide the behavior in @get_starting_point overloads
     *
     * Accepted values: 
     * OptProblem::primalRestart and OptProblem::primalDualRestart
     * Currently, OptProblem::advancedPrimalDualRestart is not suported
     */
    gollnlp::OptProblem::RestartType warm_start_type_;
  };


  class IpoptSolver_HiopSparse : public IpoptSolver {
  public:
    IpoptSolver_HiopSparse(OptProblem* p_) : IpoptSolver(p_), hiop_nlp_spec(NULL) {}
    virtual ~IpoptSolver_HiopSparse()
    {
      delete hiop_nlp_spec;
    }

    virtual bool initialize()
    {
      app = IpoptApplicationFactory();
      // Intialize the IpoptApplication and process the options
      app_status = app->Initialize();
      if (app_status != Ipopt::Solve_Succeeded) {
	printf("\n\n*** Error during initialization!\n");
	return false;
      }
      //first allocate the Hiop specification class

      hiop_nlp_spec = new gollnlp::HiopNlp(this->prob);

      assert(false);
      //ipopt_nlp_spec = new hiop::hiopMDS2IpoptTNLP(hiop_nlp_spec);
      return true;
    }
  protected:
    gollnlp::HiopNlp* hiop_nlp_spec;
  };
} //end namespace
#endif

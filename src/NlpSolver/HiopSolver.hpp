#ifndef GOLLNLP_HIOPSOLVER
#define GOLLNLP_HIOPSOLVER

#include "OptProblemMDS.hpp"
#include "NlpSolver.hpp"

#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "goTimer.hpp"

#include "IpoptSolver.hpp" //for IpoptSolver_HiopMDS

#include "IpoptAdapter.hpp" //for hiopMDS2Ipopt, part of Hiop

#include <iostream>

namespace gollnlp {

  class HiopNlpMDS : public hiop::hiopInterfaceMDS
  {
  public:
    /**constructor */
    HiopNlpMDS(OptProblemMDS* p) : prob(p)
    { 
    } 

    /** default destructor */
    virtual ~HiopNlpMDS()
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
      int n = prob->get_num_variables();
      int m = prob->get_num_constraints();
    
      prob->compute_num_variables_dense_sparse(nx_dense, nx_sparse);
      assert(n == nx_dense+nx_sparse);
    
      nnz_sparse_Jace = prob->compute_nnzJac_eq();
      printf("nx_sparse=%d  nx_dense=%d\n", nx_sparse, nx_dense);
      fflush(stdout);

      nnz_sparse_Jaci = prob->compute_nnzJac_ineq();
      nnz_sparse_Hess_Lagr_SS = prob->compute_nnzHessLagr_SSblock();
      nnz_sparse_Hess_Lagr_SD = 0;
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
      if(num_cons==0) return true;
      assert(num_cons==m);
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
		       const long long& nsparse, const long long& ndense, 
		       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		       double** JacD)
    {
      if(num_cons==0) return true;
      assert(num_cons==m);
      return prob->eval_Jaccons_eq(x, new_x, nsparse, ndense,
				   nnzJacS, iJacS, jJacS, MJacS,
				   JacD);
    }

    bool eval_Hess_Lagr(const long long& n, const long long& m, 
			const double* x, bool new_x, const double& obj_factor,
			const double* lambda, bool new_lambda,
			const long long& nsparse, const long long& ndense, 
			const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			double** HDD,
			int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
    {
      return prob->eval_HessLagr(x, new_x, obj_factor, lambda, new_lambda,
				 nsparse, ndense,
				 nnzHSS, iHSS, jHSS, MHSS,
				 HDD,
				 nnzHSD, iHSD, jHSD, MHSD);
    }

    bool get_starting_point(const long long& global_n, double* x0)
    {
      if(!prob->fill_primal_start(x0)) return false;
      return true;
    }
  private:
    OptProblemMDS* prob;

    /** Block default compiler methods.
     */
    HiopNlpMDS();
    HiopNlpMDS(const HiopNlpMDS&);
    HiopNlpMDS& operator=(const HiopNlpMDS&);
  };

  class HiopSolverMDS : public NlpSolver {
  public:
    HiopSolverMDS(OptProblemMDS* p_)
      : NlpSolver(p_), hiop_nlp_spec(NULL), app_status(OptProblem::Invalid_Option)
    {
    }
  private:
    HiopSolverMDS(OptProblem* p_)
      : NlpSolver(p_), hiop_nlp_spec(NULL), app_status(OptProblem::Invalid_Option)
    {
    }
  public:
    virtual ~HiopSolverMDS()
    {
      delete hiop_nlp_spec;
    }

    virtual bool set_start_type(OptProblem::RestartType t)
    {
      assert(false && "not yet implemented");
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

      //this->prob is inherited from base NlpSolver and has the base type, OptProblem
      OptProblemMDS* probMDS = dynamic_cast<OptProblemMDS*>(this->prob);
      if(probMDS==NULL) {
	printf("HiopSolver did not received an MDS OptProblem and cannot initialize\n");
	return false;
      }
    
      hiop_nlp_spec = new gollnlp::HiopNlpMDS(probMDS);
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
      assert(false && "not yet implemented");
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
    gollnlp::HiopNlpMDS* hiop_nlp_spec;
    OptimizationStatus app_status;
  };

  /** IpoptSolver_HiopMDS class to be used for testing only since it is not optimized
   * for performance (and really Ipopt cannot solve MDS/Kron reduction problems 
   * very efficiently).
   *
   *  Takes a OptProblemMDS as input, constructs the HiopNlpMDS, and then uses 
   *  Hiop's Ipopt MDS adapter to solve the input MDS problem.
   */

  class IpoptSolver_HiopMDS : public IpoptSolver {
  public:
    IpoptSolver_HiopMDS(OptProblem* p_) : IpoptSolver(p_), hiop_nlp_spec(NULL) {}
    virtual ~IpoptSolver_HiopMDS()
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
      
      //this->prob is inherited from base NlpSolver and has the base type, OptProblem
      OptProblemMDS* probMDS = dynamic_cast<OptProblemMDS*>(this->prob);
      if(probMDS==NULL) {
	printf("IpoptSolver_HiopMDS did not received an MDS OptProblem and cannot initialize\n");
	return false;
      }
      hiop_nlp_spec = new gollnlp::HiopNlpMDS(probMDS);
      
      ipopt_nlp_spec = new hiop::hiopMDS2IpoptTNLP(hiop_nlp_spec);
      return true;
    }
  protected:
    gollnlp::HiopNlpMDS* hiop_nlp_spec;
  };
} //end namespace
#endif

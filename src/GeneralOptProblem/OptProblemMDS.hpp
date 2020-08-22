#ifndef GOLLNLP_OPTPROB_MDS
#define GOLLNLP_OPTPROB_MDS

#include "OptProblem.hpp"
#include "goUtils.hpp"

class HiopSolver;

namespace gollnlp {

  /*
   * The MDS classes serve as templates to
   *  1. Specify the problem (and derivative) in the so-called mixed dense-sparse form
   *  2. Specify the constraints as eq and ineq separately
   */
  
  class OptObjectiveTermMDS : public OptObjectiveTerm {
  public:
    OptObjectiveTermMDS(const std::string& id_) : OptObjectiveTerm(id_) {};
    virtual ~OptObjectiveTermMDS() {};
  public:
    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
    {
      return false;
    }
    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const double& obj_factor,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD) = 0;

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return false; }
    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij) = 0;
    //TODO virtual bool get_HessLagr_SDblock_ij(std::vector<OptSparseEntry>& vij) = 0;
    
    virtual int get_HessLagr_SSblock_nnz() = 0;
    //TODO int get_HessLagr_SDblock_nnz() = 0;
  };

  class OptConstraintsBlockMDS : public OptConstraintsBlock {
  public:
    OptConstraintsBlockMDS(const std::string& id_, int num)
      : OptConstraintsBlock(id_, num)
    {
    }
    virtual ~OptConstraintsBlockMDS()
    {
    }
    //
    // Jacobian
    //
    virtual bool eval_Jac(const OptVariables& x, bool new_x, 
			  const int& nnz, int* i, int* j, double* M)
    {
      return false;
    }

    virtual bool eval_Jac_eq(const OptVariables& x, bool new_x, 
			     const int& nxsparse, const int& nxdense,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			     double** JacD) = 0;
    virtual bool eval_Jac_ineq(const OptVariables& x, bool new_x, 
			       const int& nxsparse, const int& nxdense,
			       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			       double** JacD) = 0;
    //only of the sparse part
    virtual int get_spJacob_eq_nnz() = 0;
    virtual int get_spJacob_ineq_nnz() = 0;
    virtual int get_Jacob_nnz()
    {
      assert(false && "should not be used, use _eq and _ineq methods instead");
      return -1;
    }
    
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij) = 0;
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij) = 0;
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
    {
      assert(false && "should not be used, use _eq and _ineq methods instead");
      return false;
    }
    
    //
    // Hessian
    //
    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nnz, int* i, int* j, double* M) 
    {
      return false;
    }

    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD) = 0;
    
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return false; }
    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij) = 0;
    //TODO virtual bool get_HessLagr_SDblock_ij(std::vector<OptSparseEntry>& vij) = 0;
    
    virtual int get_HessLagr_SSblock_nnz() = 0;
    //TODO int get_HessLagr_SDblock_nnz() = 0;
  };

  
  class OptProblemMDS : public OptProblem {
  public:
    OptProblemMDS()
      : OptProblem()
    {
      nnz_Jac_eq = nnz_Jac_ineq = -1;
      nnz_HessLagr_SSblock = -1;
    }
    virtual ~OptProblemMDS()
    {
    }

    virtual void primal_problem_changed()
    {
      nnz_Jac_eq = nnz_Jac_ineq = -1;
      nnz_HessLagr_SSblock = -1;
      ij_Jac_eq.clear();
      ij_HessLagr_SSblock.clear();
      
      OptProblem::primal_problem_changed();
    }
    virtual void dual_problem_changed()
    {
      nnz_Jac_eq = nnz_Jac_ineq = -1;
      nnz_HessLagr_SSblock = -1;
      ij_Jac_eq.clear();
      ij_HessLagr_SSblock.clear();
      OptProblem::dual_problem_changed();
    }
    
    bool eval_Jaccons_eq(const double* x, bool new_x, 
			 const int& nxsparse, const int& nxdense,
			 const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			 double** JacD);
    bool eval_Jaccons_ineq(const double* x, bool new_x, 
			   const int& nxsparse, const int& nxdense,
			   const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			   double** JacD)
    {
      assert(false && "not yet supported");
      return false;
    }

    bool eval_HessLagr(const double* x, bool new_x, 
		       const double& obj_factor, 
		       const double* lambda, bool new_lambda, 
		       const int& nxsparse, const int& nxdense, 
		       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
		       double** HDD,
		       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);

    int compute_num_variables_sparse() const;
    int compute_num_variables_dense() const;
    bool compute_num_variables_dense_sparse(int& ndense, int& nsparse) const;
    
    int compute_nnzJac_eq();
    int compute_nnzJac_ineq() { return 0; }
    int compute_nnzHessLagr_SSblock();
    
    void use_nlp_solver(const std::string& name);
  private:
    int nnz_Jac_eq, nnz_Jac_ineq;
    std::vector<OptSparseEntry> ij_Jac_eq, ij_Jac_ineq;

    int nnz_HessLagr_SSblock;
    std::vector<OptSparseEntry> ij_HessLagr_SSblock;
  };

} //end namespace

#endif

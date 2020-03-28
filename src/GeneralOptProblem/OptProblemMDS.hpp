#ifndef GOLLNLP_OPTPROB_MDS
#define GOLLNLP_OPTPROB_MDS

#include "OptProblem.hpp"
#include "goUtils.hpp"

class HiopSolver;

namespace gollnlp {

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
    }
    virtual ~OptProblemMDS()
    {
    }
    
    bool eval_Jaccons(const double* x, bool new_x, 
		      const int& nxsparse, const int& nxdense,
		      const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		      double** JacD)
    {
      if(new_x) {
	new_x_fgradf=true;
	vars_primal->attach_to(x);
      }
      if(MJacS==NULL) {
	for(auto& con_gen: cons->vblocks) {

	  OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	  assert(NULL!=con);
	  if(!con) continue;
	  
	  if(!con->eval_Jac_eq(*vars_primal, new_x, nxsparse, nxdense, nnzJacS, iJacS, jJacS, MJacS, JacD)) {
	    assert(false && "eval_Jaccons should be called after get_nnzJaccons");
	  }
	}
	return true;
      }
      
      //M!=NULL > just fill in the values
      for(int i=0; i<nnzJacS; i++) MJacS[i]=0.;
      for(auto& con_gen: cons->vblocks) {

	OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	assert(NULL!=con);
	if(!con) continue;
	
	if(!con->eval_Jac_eq(*vars_primal, new_x, nxsparse, nxdense, nnzJacS, iJacS, jJacS, MJacS, JacD))
	  return false;
      }
      return true;
    }

    
    bool eval_HessLagr(const double* x, bool new_x, 
		       const double& obj_factor, 
		       const double* lambda, bool new_lambda, 
		       const int& nxsparse, const int& nxdense, 
		       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
		       double** HDD,
		       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
    {
      if(new_x) {
	new_x_fgradf=true; 
	vars_primal->attach_to(x);
      }
      if(new_lambda) vars_duals_cons->attach_to(lambda);
      
      if(NULL==MHSS) {
	for(auto& ot_gen: obj->vterms) {

	  OptObjectiveTermMDS* ot = dynamic_cast<OptObjectiveTermMDS*>(ot_gen);
	  if(NULL==ot) {
	    assert(false && "check this");
	    continue;
	  }
	  
	  if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, 
				nxsparse, nxdense,
				nnzHSS, iHSS, jHSS, MHSS,
				HDD,
				nnzHSD, iHSD, jHSD, MHSD)) {
	    assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
	  }
	}
	for(auto& con_gen: cons->vblocks) {
	  OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	  if(NULL==con) {
	    assert(false && "check this");
	    continue;
	  }
	  if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, 
				 nxsparse, nxdense,
				 nnzHSS, iHSS, jHSS, MHSS,
				 HDD,
				 nnzHSD, iHSD, jHSD, MHSD)) {
	    assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
	  }
	}
      } else {
	// case of M!=NULL > just fill in the values
	for(int it=0; it<nnzHSS; it++) MHSS[it]=0.;
	assert(nnzHSD==0);
	
	for(auto& ot_gen: obj->vterms) {
	  
	  OptObjectiveTermMDS* ot = dynamic_cast<OptObjectiveTermMDS*>(ot_gen);
	  if(NULL==ot) {
	    assert(false && "check this");
	    continue;
	  }
	  
	  if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, 
				nxsparse, nxdense,
				nnzHSS, iHSS, jHSS, MHSS,
				HDD,
				nnzHSD, iHSD, jHSD, MHSD))
	    return false;
	}
	
	for(auto& con_gen: cons->vblocks) {
	  OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	  if(NULL==con) {
	    assert(false && "check this");
	    continue;
	  }

	  if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, 
				 nxsparse, nxdense,
				 nnzHSS, iHSS, jHSS, MHSS,
				 HDD,
				 nnzHSD, iHSD, jHSD, MHSD))
	    return false;
	}
      }
      return true;
    } // end of eval_HessLagr

    int get_num_variables_sparse() const
    {
      int nsparse=0;
      for(auto& var_block: vars_primal->vblocks)
	if(var_block->areVarsSparse)
	  nsparse += var_block->n;
      return nsparse;
    }
    int get_num_variables_dense() const
    {
      int ndense=0;
      for(auto& var_block: vars_primal->vblocks)
	if(false==var_block->areVarsSparse)
	  ndense += var_block->n;
      return ndense;
    }
    bool get_num_variables_dense_sparse(int& ndense, int& nsparse) const
    {
      for(auto& var_block: vars_primal->vblocks)
	if(var_block->areVarsSparse)
	  nsparse += var_block->n;
	else
	  ndense += var_block->n;
      return true;
    }
    int get_nnzJac_eq();
    int get_nnzHessLagr_SSblock();
    
    void use_nlp_solver(const std::string& name);
  };

} //end namespace

#endif

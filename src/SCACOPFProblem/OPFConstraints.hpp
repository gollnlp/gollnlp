#ifndef OPF_VARIABLES
#define OPF_VARIABLES

#include "OptProblem.hpp"

namespace gollnlp {

  // pq == A*vi^2 + B*vi*vj*cos(thetai - thetaj + Theta) + C*vi*vj*sin(thetai - thetaj + Theta)
  class PFConRectangular : public OptConstraintsBlock
  {
  public:
    PFConRectangular(const std::string& id_,
		     OptVariablesBlock* pq_, 
		     OptVariablesBlock* v_n_, 
		     OptVariablesBlock* theta_n_, 
		     const SCACOPFData& d_)
      : OptConstraintsBlock(id_, 0), pq(pq_), v_n(v_n_), theta_n(theta_n_), d(d_)
    {
      H_nz_idxs = J_nz_idxs = NULL;
    }
    virtual ~PFConRectangular() {};

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
    {
      
      return true;
    }
    
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M)
    {
      if(NULL==M) {
      } else {
      }
      return true;
    }
    
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* i, int* j, double* M)
    {
      const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
      //assert(lambda!=NULL);
      //assert(lambda->n==1);
      
      if(NULL==M) {
      } else {
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return 0; }
    virtual int get_Jacob_nnz(){ return 0; }
    
    // (i,j) entries in the HessLagr to which the implementer's contributes to
    // this is only called once
    // push_back in vij 
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    {
      int n=0, i;
      if(n==0) return true;
      
      if(NULL==H_nz_idxs) {
	H_nz_idxs = new int[n];
      }

      //for(int it=0; it<n; it++) {
      //i = x->index+it;
      //vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
      //}
      return true;
    }

    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
    {
      int i;
      if(n<=0) return true;
      
      //vij.push_back(OptSparseEntry(this->index, x->index, &J_nz_start));

      //for(int it=1; it<x->n; it++) {
      //vij.push_back(OptSparseEntry(this->index, x->index+it, NULL));
      //}
      return true;
  } 
  protected:
    const OptVariablesBlock *pq, *v_n, *theta_n;
    const SCACOPFData& d;
    int* J_nz_idxs;
    int* H_nz_idxs;
  };

}

#endif

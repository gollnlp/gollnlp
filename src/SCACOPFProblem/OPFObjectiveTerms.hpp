#ifndef OPF_OBJTERMS
#define OPF_OBJTERMS

#include "OptProblem.hpp"

#include <string>
#include <cassert>
#include <vector>

#include <cmath>

namespace gollnlp {

//for 0.5||x||^2 -> to be used in testing
class DummySingleVarQuadrObjTerm : public OptObjectiveTerm {
public: 
  DummySingleVarQuadrObjTerm(const std::string& id, OptVariablesBlock* x_) 
    : OptObjectiveTerm(id), x(x_), H_nz_idxs(NULL)
  {};

  virtual ~DummySingleVarQuadrObjTerm() 
  {
    if(H_nz_idxs) 
      delete[] H_nz_idxs;
  }

  virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
  {
    int nvars = x->n; double aux;
    for(int it=0; it<nvars; it++) {
      aux = x->xref[it] - 1.;
      obj_val += aux * aux * 0.5;
    }
    return true;
  }
  virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
  {
    double* g = grad + x->index;
    for(int it=0; it<x->n; it++) 
      g[it] += x->xref[it] - 1.;
    return true;
  }
  virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
  {
    if(NULL==M) {
      int idx, row;
      for(int it=0; it<x->n; it++) {
	idx = H_nz_idxs[it]; 
	if(idx<0) return false;
	i[idx] = j[idx] = x->index+it;
      }
    } else {
      for(int it=0; it<x->n; it++) {
	assert(H_nz_idxs[it]>=0);
	assert(H_nz_idxs[it]<nnz);
	M[H_nz_idxs[it]] += obj_factor;
      }
    }
    return true;
  }

  virtual int get_HessLagr_nnz() { return x->n; }
  // (i,j) entries in the HessLagr to which this term contributes to
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  { 
    int nvars = x->n, i;
    if(NULL==H_nz_idxs)
      H_nz_idxs = new int[nvars];

    for(int it=0; it < nvars; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
    }
    return true; 
  }

private:
  OptVariablesBlock* x;
  //keep the index for each nonzero elem in the Hessian that this constraints block contributes to
  int *H_nz_idxs;
};

} //end namespace

#endif
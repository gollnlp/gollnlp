#ifndef OPF_OBJTERMS
#define OPF_OBJTERMS

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include <string>
#include <cassert>
#include <vector>

#include <cmath>

namespace gollnlp {

  //////////////////////////////////////////////////////////////////////////////
  // Production cost piecewise linear objective
  // min sum_g( sum_h CostCi[g][h]^T t[g][h])
  // constraints (handled outside) are
  //   t>=0, sum_h t[g][h]=1
  //   p_g[g] - sum_h CostPi[g][h]*t[g][h] =0
  //////////////////////////////////////////////////////////////////////////////
  class PFProdCostPcLinObjTerm : public OptObjectiveTerm {
  public: 
    //Gidx contains the indexes (in d.G_Generator) of the generator participating
    PFProdCostPcLinObjTerm(const std::string& id_, OptVariablesBlock* t_h_, 
			   const std::vector<int>& Gidx_,
			   const SCACOPFData& d_);
    virtual ~PFProdCostPcLinObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);
    //Hessian is all zero

  private:
    friend class PFProdCostAffineCons;
    std::string id;
    OptVariablesBlock* t_h;
    int* Gidx;
    double *CostCi;
    const SCACOPFData& d;
    int ngen;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Slack penalty piecewise linear objective
  // min sum_i( sum_h P[i][h] sigma_h[i][h])
  // constraints (handled outside) are
  //   0<= sigma[i][h] <= Pseg_h, 
  //   slacks[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
  //////////////////////////////////////////////////////////////////////////////
  class PFPenaltyPcLinObjTerm : public OptObjectiveTerm {
  public: 
    //Gidx contains the indexes (in d.G_Generator) of the generator participating
    PFPenaltyPcLinObjTerm(const std::string& id_, 
			  OptVariablesBlock* sigma_,
			  const std::vector<double>& pen_coeff,
			  const double& obj_weight,
			  const SCACOPFData& d_);
    virtual ~PFPenaltyPcLinObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);
    //Hessian is all zero

  private:
    std::string id;
    OptVariablesBlock* sigma;
    double weight;
    const SCACOPFData& d;
    double P1, P2, P3;
  };


  //for 0.5||x||^2 -> to be used in testing
  class DummySingleVarQuadrObjTerm : public OptObjectiveTerm {
  public: 
    DummySingleVarQuadrObjTerm(const std::string& id, OptVariablesBlock* x_) 
      : OptObjectiveTerm(id), x(x_), H_nz_idxs(NULL)
    {assert(false);};

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

#include "OptObjTerms.hpp"


#include <cstring>

namespace gollnlp {

bool LinearPenaltyObjTerm::eval_f(const OptVariables& vars, bool new_x, double& obj_val)
{
  if(x->n==0) return true;
  assert(x->n>=1);
  double aux = x->xref[0];
  for(int i=1; i<x->n; i++) 
    aux += x->xref[i];

  obj_val += M*aux;
  return true;
}

bool LinearPenaltyObjTerm::eval_grad(const OptVariables& vars, bool new_x, double* grad)
{
  assert(x->n>=0);
  double *g=grad+x->index;

  for(int i=0; i<x->n; i++) 
    g[i] += M;

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Quadratic regularization term
// \gamma ||x-x_0||^2
//////////////////////////////////////////////////////////////////////////////////////////////////
QuadrRegularizationObjTerm::QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
						       const double& gamma_, double* a_, double* x0_)
  : OptObjectiveTerm(id_), x(x_), gamma(gamma_), H_nz_idxs(NULL)
{
  x0 = new double[x->n];
  memcpy(x0, x0_, x->n*sizeof(double));
  a = new double[x->n];
  memcpy(a, a_, x->n*sizeof(double));
}

QuadrRegularizationObjTerm::QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
						       const double& gamma_, double* x0_)
  : OptObjectiveTerm(id_), x(x_), gamma(gamma_), H_nz_idxs(NULL)
{
  x0 = new double[x->n];
  memcpy(x0, x0_, x->n*sizeof(double));
  a = new double[x->n];
  for(int i=0; i<x->n; i++) a[i]=1.;
}
QuadrRegularizationObjTerm::QuadrRegularizationObjTerm(const std::string id_, OptVariablesBlock* x_, 
						       const double& gamma_, const double& x0scalar_)
  : OptObjectiveTerm(id_), x(x_), gamma(gamma_), H_nz_idxs(NULL)
{
  x0 = new double[x->n];
  for(int i=0; i<x->n; i++) x0[i]=x0scalar_;
  a = new double[x->n];
  for(int i=0; i<x->n; i++) a[i]=1.;
}

QuadrRegularizationObjTerm::~QuadrRegularizationObjTerm()
{
  delete [] x0;
  delete [] a;
  delete [] H_nz_idxs;
}

bool QuadrRegularizationObjTerm::eval_f(const OptVariables& vars, bool new_x, double& obj_val)
{
  if(x->n==0) return true;
  assert(x->n>=1);
  double sum = a[0]*x->xref[0]-x0[0], aux; sum *= sum;
  for(int i=1; i<x->n; i++) {
    aux = a[i]*x->xref[i]-x0[i];
    sum += aux*aux;
  }
  obj_val += gamma*sum;
  return true;
}

bool QuadrRegularizationObjTerm::eval_grad(const OptVariables& vars, bool new_x, double* grad)
{
  assert(x->n>=0);
  double *g=grad+x->index;

  double gamma2 = 2*gamma;
  for(int i=0; i<x->n; i++) {
    g[i] += gamma2*(a[i]*x->xref[i]-x0[i]);
  }
  return true;
}
bool QuadrRegularizationObjTerm::
eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
	      const double& obj_factor,
	      const int& nnz, int* i, int* j, double* M)
{
  if(NULL==M) {
    int idx;
    for(int it=0; it<x->n; it++) {
      idx = H_nz_idxs[it]; 
      assert(idx>=0);
      //if(idx<0) return false;
      i[idx] = j[idx] = x->index+it;
    }
  } else {
    double aux = 2*gamma*obj_factor;
    for(int it=0; it<x->n; it++) {
      assert(H_nz_idxs[it]>=0);
      assert(H_nz_idxs[it]<nnz);
      M[H_nz_idxs[it]] += a[it]*aux;
    }
  }
  return true;
}

int QuadrRegularizationObjTerm::get_HessLagr_nnz() { return x->n; }

// (i,j) entries in the HessLagr to which this term contributes to
bool QuadrRegularizationObjTerm::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)
{
  if(NULL==H_nz_idxs)
    H_nz_idxs = new int[x->n];
  
  int i;
  for(int it=0; it < x->n; it++) {
    i = x->index+it;
    vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
  }
  return true;
}

//
// Quadratic away-from-bounds penalization QPen(x) = \gamma * q(x)
//
// Given l <= x <= u and u>l, q(x) is such that
// q(l) = q(u) = 1 and q((u-l)/2)=0
// 
// The exact form: q(x) = 4/[(u-l)^2] * [ x - (l+u)/2 ]^2
QuadrAwayFromBoundsObjTerm::
QuadrAwayFromBoundsObjTerm(const std::string& id_, OptVariablesBlock* x_, 
			   const double& gamma_, 
			   const double* lb, const double* ub)
  : QuadrRegularizationObjTerm(id_, x_, gamma_, 0.)
{
  //update a and x_0 based on lb and ub
  double aux;
  for(int i=0; i<x->n; i++) {
    aux = ub[i]-lb[i];
    
    if(aux<1e-10) {
      assert(aux> -1e-12);
      a[i] = 1.;
      x0[i] = 0.5*(lb[i]+ub[i]);
    } else {
      a[i]  = 2./aux;
      x0[i] = (ub[i]+lb[i])/aux;
    }
  }
}

} //end of namespace

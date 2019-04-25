#include "OPFObjectiveTerms.hpp"

#include "blasdefs.hpp"

#include "string.h"

namespace gollnlp {

//////////////////////////////////////////////////////////////////////////////
// Production cost piecewise linear objective
//////////////////////////////////////////////////////////////////////////////

PFProdCostPcLinObjTerm::
PFProdCostPcLinObjTerm(const std::string& id_, OptVariablesBlock* t_h_, 
		       const std::vector<int>& Gidx_,
		       const std::vector<std::vector<double> >& G_CostCi)
  : OptObjectiveTerm(id_), t_h(t_h_)
{
  ngen = Gidx_.size();
  Gidx = new int[ngen];
  memcpy(Gidx, Gidx_.data(), ngen*sizeof(int));

  CostCi = new double[t_h->n]; 
  double*it = CostCi; int sz;
  for(int i=0; i<ngen; i++) {
    //int aaa = Gidx[i];
    sz = G_CostCi[Gidx[i]].size();
    memcpy(it, G_CostCi[Gidx[i]].data(), sz*sizeof(double));
    it += sz;
  }
  assert(CostCi+t_h->n == it);
}

PFProdCostPcLinObjTerm::~PFProdCostPcLinObjTerm()
{
  delete[] Gidx;
  delete[] CostCi;
}
bool PFProdCostPcLinObjTerm::eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
{
  obj_val += DDOT(&(t_h->n), const_cast<double*>(t_h->xref), &ione, CostCi, &ione);
  return true;
}
bool PFProdCostPcLinObjTerm::eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
{
  //printf("Pen2: grad %p\n", grad);
  //printf("t_h index %d\n", t_h->index);
  DAXPY(&(t_h->n), &done, CostCi, &ione, grad+t_h->index, &ione);
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Slack penalty piecewise linear objective
// min sum_i( sum_h P[i][h] sigma_h[i][h])
//////////////////////////////////////////////////////////////////////////////
PFPenaltyPcLinObjTerm::
PFPenaltyPcLinObjTerm(const std::string& id_, 
		      OptVariablesBlock* sigma_,
		      const std::vector<double>& pen_coeff,
		      const double& obj_weight,
		      const double& slacks_rescale)
  : OptObjectiveTerm(id_), sigma(sigma_), weight(obj_weight)
{
  assert(pen_coeff.size()==3); 
  assert(weight>=0 && weight<=1);

  if(weight>0) {
    P1 = pen_coeff[0]*weight; P2 = pen_coeff[1]*weight; P3 = pen_coeff[2]*weight; 
  } else {
    P1 = pen_coeff[0]; P2 = pen_coeff[1]; P3 = pen_coeff[2]; 
  }
  P1 /= slacks_rescale; P2 /= slacks_rescale; P3 /= slacks_rescale; 
}

PFPenaltyPcLinObjTerm::~PFPenaltyPcLinObjTerm()
{ }

bool PFPenaltyPcLinObjTerm::eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
{  
  if(weight>0) {
    int i; 
    for(i=0; i<sigma->n; i+=3) {
      //printf("pslack_n sigma l=%d   [%g, %g, %g] \n  %g %g %g\n", 
      //	   i/3, sigma->xref[i], sigma->xref[i+1], sigma->xref[i+2], P1, P2, P3);
      obj_val += sigma->xref[i]*P1 + sigma->xref[i+1]*P2 + sigma->xref[i+2]*P3;
    }
    assert(i==sigma->n);
  }
  return true;
}
bool PFPenaltyPcLinObjTerm::eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
{
  //printf("Pen1: grad %p\n", grad);
  if(weight>0) {
    double* g = grad + sigma->index;
    for(int i=0; i<sigma->n; i+=3) {
      *g++ += P1; *g++ += P2; *g++ += P3;
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Slack penalty quadratic objective
// q(0)=0
// q'(0) = P1 (slope of the piecewise linear penalty at 0)
// q(s1+s2) = piecewise_linear_penalty(s1+s2) (=P1*s1+P2*s2)
// 
// q(x) = a*x^2 + b*x, where
// b = P1, a=(P2-P1)*S2/(s1+s2)^2
//
// Assumed is that the piecewise linear penalty is defined over 3 segments
// [0, s1], [s1, s1+s2], [s1+s2, s1+s2+s3] with slopes P1, P2, P3
//
PFPenaltyQuadrApproxObjTerm::
PFPenaltyQuadrApproxObjTerm(const std::string& id_, 
			    OptVariablesBlock* slacks_,
			    const std::vector<double>& pen_coeff,
			    const std::vector<double>& pen_segm,
			    const double& obj_weight,
			    const double& slacks_rescale)
  : OptObjectiveTerm(id_), x(slacks_), weight(obj_weight), H_nz_idxs(NULL), aux(0)
{
  assert(pen_coeff.size()==3);
  assert(pen_segm.size()==3);
  double P1, P2, S1, S2;
  assert(weight>=0 && weight<=1);
  if(weight>0) {
    P1 = pen_coeff[0]*weight; P2 = pen_coeff[1]*weight; //P3 = pen_coeff[2]*weight; 
  } else {
    P1 = pen_coeff[0]; P2 = pen_coeff[1]; //P3 = pen_coeff[2]; 
  }
  S1=pen_segm[0]; S2=pen_segm[1]; //S3=pen_segm[2];

  assert(slacks_rescale>0);

  //f is usually 1/256 or 1/512
  f = slacks_rescale>0 ? 1/slacks_rescale : 1.;
  //double rescale = slacks_rescale>0 ? slacks_rescale : 1.;

  //a = (P2-P1)*S2/((S1+S2)*(S1+S2));
  double aux = S1+S2; 
  a = ((((P2-P1)*f)/aux)*f)/aux;
  b = P1*f;
}
PFPenaltyQuadrApproxObjTerm::~PFPenaltyQuadrApproxObjTerm() {}

//a*x^2+b*x
bool PFPenaltyQuadrApproxObjTerm::eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
{
  aux = 0.;
  for(int i=0; i<x->n; i++) 
    aux += x->xref[i] * x->xref[i];
  obj_val += a*aux;

  aux = 0.; 
  for(int i=0; i<x->n; i++)
    aux += x->xref[i];
  obj_val += b*aux;

  return true;
}

bool PFPenaltyQuadrApproxObjTerm::eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
{
  double* g = grad+x->index;
  // += 2*a*x
  aux = 2*a;
  assert(aux>0);
  DAXPY(&(x->n), &aux, const_cast<double*>(x->xref), &ione, g, &ione);

  for(int i=0; i<x->n; i++)
    g[i] += b;
    
  return true;
}

bool PFPenaltyQuadrApproxObjTerm::
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
       aux = 2*a*obj_factor;
       for(int it=0; it<x->n; it++) {
	 assert(H_nz_idxs[it]>=0);
	 assert(H_nz_idxs[it]<nnz);
	 M[H_nz_idxs[it]] += aux;
       }
  }
  return true;
}

int PFPenaltyQuadrApproxObjTerm::get_HessLagr_nnz() { return x->n; }

// (i,j) entries in the HessLagr to which this term contributes to
bool PFPenaltyQuadrApproxObjTerm::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)
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
} //end namespace

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
		       const SCACOPFData& d_)
  : OptObjectiveTerm(id_), t_h(t_h_), d(d_)
{
  ngen = Gidx_.size();
  Gidx = new int[ngen];
  memcpy(Gidx, Gidx_.data(), ngen*sizeof(int));

  CostCi = new double[t_h->n]; 
  double*it = CostCi; int sz;
  for(int i=0; i<ngen; i++) {
    //int aaa = Gidx[i];
    sz = d.G_CostCi[Gidx[i]].size();
    memcpy(it, d.G_CostCi[Gidx[i]].data(), sz*sizeof(double));
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

  //for(int i=0;i<t_h->n; i++)
  //  obj_val += 0.5*t_h->xref[i]*(1.5*CostCi[i]);

  //t_h->print();

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
		      const SCACOPFData& d_)
  : OptObjectiveTerm(id_), sigma(sigma_), weight(obj_weight), d(d_)
{
  assert(pen_coeff.size()==3); 
  assert(weight>=0 && weight<=1);

  if(weight>0) {
    P1 = pen_coeff[0]*weight; P2 = pen_coeff[1]*weight; P3 = pen_coeff[2]*weight; 
  } else {
    P1 = pen_coeff[0]; P2 = pen_coeff[1]; P3 = pen_coeff[2]; 
  }
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

} //end namespace

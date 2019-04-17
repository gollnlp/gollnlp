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
    int aaa = Gidx[i];
    sz = d.G_CostCi[Gidx[i]].size();
    memcpy(it, d.G_CostCi[Gidx[i]].data(), sz*sizeof(double));
    it += sz;
  }
  assert(CostCi+t_h->n == it);
}

PFProdCostPcLinObjTerm::~PFProdCostPcLinObjTerm()
{
  delete[] Gidx;
  //delete[] CostPi;
  delete[] CostCi;
}
bool PFProdCostPcLinObjTerm::eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
{
  obj_val += DDOT(&(t_h->n), const_cast<double*>(t_h->xref), &ione, CostCi, &ione);
  return true;
}
bool PFProdCostPcLinObjTerm::eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
{
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
		      const SCACOPFData& d_)
  : OptObjectiveTerm(id_), sigma(sigma_), d(d_)
{
  assert(pen_coeff.size()==3); 
  P1 = pen_coeff[0]; P2 = pen_coeff[1]; P3 = pen_coeff[2]; 
}

PFPenaltyPcLinObjTerm::~PFPenaltyPcLinObjTerm()
{ }

bool PFPenaltyPcLinObjTerm::eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
{  
  int i;
  for(i=0; i<sigma->n; i+=3) 
    obj_val += sigma->xref[i]*P1 + sigma->xref[i+1]*P2 + sigma->xref[i+2]*P3;
  assert(i==sigma->n);
  return true;
}
bool PFPenaltyPcLinObjTerm::eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
{
  grad += sigma->index;
  for(int i=0; i<sigma->n; i+=3) {
    *grad++ += P1; *grad++ += P2; *grad++ += P3;
  }
  return true;
}

} //end namespace

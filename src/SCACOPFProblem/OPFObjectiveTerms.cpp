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
} //end namespace

#include "OptObjTerms.hpp"

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
} //end of namespace

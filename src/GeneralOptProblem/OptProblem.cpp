#include "OptProblem.hpp"

#include <iostream>

#include "blasdefs.hpp"

using namespace std;

namespace gollnlp {

OptProblem::OptProblem(OptVariables* vars)
  : m_vars(vars)
{
}

OptProblem::~OptProblem()
{
  for(auto& t: m_objterms) delete t;
  for(auto& c: m_conblocks) delete c;
  //delete m_vars; 
}

bool OptProblem::eval_obj(double* x, double& obj_val)
{
  m_vars->attach_to(x);
  for(auto& obj: m_objterms) 
    if (!obj->eval_body(*m_vars, &obj_val) )
      return false;
  return true;
}
bool OptProblem::eval_cons    (double* x, double* cons)
{
  //! we need to accumulate in 'cons' in eval_body
  m_vars->attach_to(x);
  for(auto& con: m_conblocks)
    if(!con->eval_body(*m_vars, cons))
      return false;
  return true;
}
bool OptProblem::eval_gradobj (double* x, double* grad)
{
  m_vars->attach_to(x);
  for(auto& obj: m_objterms)
    if(!obj->eval_deriv(*m_vars, grad))
      return false;
  return true;
}
bool OptProblem::eval_Jaccons (double* x, const int& nnz, int* i, int* j, double* M)
{
  m_vars->attach_to(x);
  //! here we need to call eval_deriv with correct jumps in i,j, and M
  for(auto& con: m_conblocks)
    if(!con->eval_deriv(*m_vars, nnz, i,j,M))
      return false;
  return true;
}
//! mulipliers
bool OptProblem::eval_HessLagr(double* x, const int& nnz, int* i, int* j, double* M)
{
  //!
  m_vars->attach_to(x);
  for(auto& con: m_conblocks)
    if(!con->eval_Hess (*m_vars, nnz, i,j,M))
      return false;
  for(auto& obj: m_objterms) 
    if (!obj->eval_Hess(*m_vars, nnz, i,j,M))
      return false;
  return true;
}

OptVariables::OptVariables()
  : n(0) {}


OptVariables::~OptVariables()
{
  for(auto b: m_vblocks)
    delete b;
}
bool OptVariables::append_varsblock(OptVarsBlock* b)
{
  if(b) {
    if(m_mblocks.find(b->id)!= m_mblocks.end()) {
      cerr << "appendVarsBlock: cannot add block since its identifier has been already used" << endl;
      assert(false);
      return false;
    }
    m_vblocks.push_back(b);
    m_mblocks[b->id] = b;
    b->index=this->n;
    this->n += b->n;
  }
}

void OptVariables::attach_to(double *x)
{
  for(auto b: m_vblocks) b->x = x + b->index;
}

OptVariables::OptVarsBlock::OptVarsBlock(const int& n_, const std::string& id_)
  : n(n_), id(id_), index(-1), x(NULL)
{
  assert(n>=0);
  int i;
  lb = new double[n];
  for(i=0; i<n; i++) lb[i] = -1e+20;

  ub = new double[n];
  for(i=0; i<n; i++) ub[i] = +1e+20;
}

OptVariables::OptVarsBlock::OptVarsBlock(const int& n_, const std::string& id_, double* lb_, double* ub_)
  : n(n_), id(id_), index(-1), x(NULL)
{
  assert(n>=0);
  int i, one=1;
  lb = new double[n];
  if(lb_)
    DCOPY(&n, lb, &one, lb_, &one);
  else
    for(i=0; i<n; i++) lb[i] = -1e+20;

  ub = new double[n];
  if(ub_)
    DCOPY(&n, ub, &one, ub_, &one);
  else
    for(i=0; i<n; i++) ub[i] = +1e+20;
}

OptVariables::OptVarsBlock::~OptVarsBlock()
{
  delete[] lb;
  delete[] ub;
}

} //end of namespace

#include "OptProblem.hpp"

namespace gollnlp {

OptProblem::OptProblem(OptVariables* vars)
  : m_vars(vars)
{
}

OptProblem::~OptProblem()
{
  for(auto& t: m_objterms) delete t;
  for(auto& c: m_conblocks) delete c;
  delete m_vars; 
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
bool OptProblem::eval_Jaccons (double* x, const size_t& nnz, int* i, int* j, double* M)
{
  m_vars->attach_to(x);
  //! here we need to call eval_deriv with correct jumps in i,j, and M
  for(auto& con: m_conblocks)
    if(!con->eval_deriv(*m_vars, nnz, i,j,M))
      return false;
  return true;
}
//! mulipliers
bool OptProblem::eval_HessLagr(double* x, const size_t& nnz, int* i, int* j, double* M)
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

}

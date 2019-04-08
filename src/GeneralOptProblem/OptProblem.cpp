#include "OptProblem.hpp"

#include <iostream>

#include "blasdefs.hpp"

using namespace std;

namespace gollnlp {

OptProblem::OptProblem()
{
  vars_primal = new OptVariables();
  cons = new OptConstraints();
  obj = new OptObjective();

  vars_duals_bounds = NULL;
  vars_duals_cons = NULL;
}


OptProblem::~OptProblem()
{
  delete obj;
  delete cons;
  delete vars_duals_bounds;
  delete vars_duals_cons;
  delete vars_primal;
}

bool OptProblem::eval_obj(const double* x, bool new_x, double& obj_val)
{
  vars_primal->attach_to(x);
  for(auto& ot: obj->vterms) 
    if (!ot->eval_body(*vars_primal, &obj_val) )
       return false;
  return true;
}
bool OptProblem::eval_cons    (const double* x, bool new_x, double* cons)
{
  // //! we need to accumulate in 'cons' in eval_body
  // m_vars->attach_to(x);
  // for(auto& con: m_conblocks)
  //   if(!con->eval_body(*m_vars, cons))
  //     return false;
  return true;
}
bool OptProblem::eval_gradobj (const double* x, bool new_x, double* grad)
{
  // m_vars->attach_to(x);
  // for(auto& obj: m_objterms)
  //   if(!obj->eval_deriv(*m_vars, grad))
  //     return false;
  return true;
}
bool OptProblem::eval_Jaccons (const double* x, bool new_x, const int& nnz, int* i, int* j, double* M)
{
  // m_vars->attach_to(x);
  // //! here we need to call eval_deriv with correct jumps in i,j, and M
  // for(auto& con: m_conblocks)
  //   if(!con->eval_deriv(*m_vars, nnz, i,j,M))
  //     return false;
  return true;
}
//! mulipliers
bool OptProblem::eval_HessLagr(const double* x, bool new_x, const int& nnz, int* i, int* j, double* M)
{
  //!
  // m_vars->attach_to(x);
  // for(auto& con: m_conblocks)
  //   if(!con->eval_Hess (*m_vars, nnz, i,j,M))
  //     return false;
  // for(auto& obj: m_objterms) 
  //   if (!obj->eval_Hess(*m_vars, nnz, i,j,M))
  //     return false;
  return true;
}

OptVariables*  OptProblem::new_duals_vec_cons()
{
  OptVariables* duals = new OptVariables();
  for(auto b: cons->vblocks) {
    duals->append_varsblock(new OptVariablesBlock(b->n, string("duals_") + b->id));
  }
  return duals;
}
OptVariables*  OptProblem::new_duals_vec_bounds()
{
  OptVariables* duals = new OptVariables();
  for(auto b: vars_primal->vblocks) {
    duals->append_varsblock(new OptVariablesBlock(b->n, string("duals_bnd_") + b->id));
  }
  return duals;
}

bool OptProblem::use_nlp_solver(const std::string& nlpsolver)
{
  assert(false);
  return true;
}
//these setters return false if the option is not recognized by the NLP solver
bool OptProblem::set_solver_option(const std::string& name, int value)
{
  assert(false);
  return true;
}
bool OptProblem::set_solver_option(const std::string& name, double value)
{
  assert(false);
  return true;
}
bool OptProblem::set_solver_option(const std::string& name, const std::string& value)
{
  assert(false);
  return true;
}
bool OptProblem::optimize(const std::string& nlpsolver)
{
  assert(false);
  return true;
}

/////////////////////////////////////////
// OptVariables
/////////////////////////////////////////

OptVariables::OptVariables()
{
}

OptVariables::~OptVariables()
{
  for(auto b: vblocks)
    delete b;
}
bool OptVariables::append_varsblock(OptVariablesBlock* b)
{
  if(b) {
    if(mblocks.find(b->id)!= mblocks.end()) {
      cerr << "appendVarsBlock:  block (name) already exists" << endl;
      assert(false);
      return false;
    }
    b->index=this->n();
    vblocks.push_back(b);
    mblocks[b->id] = b;
  }
  return true;
}

void OptVariables::attach_to(const double *x)
{
  for(auto b: vblocks) b->xref = x + b->index;
}

OptVariablesBlock::OptVariablesBlock(const int& n_, const std::string& id_)
  : n(n_), id(id_), index(-1)
{
  assert(n>=0);
  int i;

  x = new double[n];

  lb = new double[n];
  for(i=0; i<n; i++) lb[i] = -1e+20;

  ub = new double[n];
  for(i=0; i<n; i++) ub[i] = +1e+20;

  assert(false);
}

OptVariablesBlock::OptVariablesBlock(const int& n_, const std::string& id_, double* lb_, double* ub_)
  : n(n_), id(id_), index(-1)
{
  assert(n>=0);

  x = new double[n];

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
OptVariablesBlock::OptVariablesBlock(const int& n_, const std::string& id_, double lb_, double ub_)
  : n(n_), id(id_), index(-1)
{
  assert(n>=0);

  x = new double[n];

  int i, one=1;
  lb = new double[n];
  for(i=0; i<n; i++) lb[i] = lb_;

  ub = new double[n];
  for(i=0; i<n; i++) ub[i] = ub_;
}

OptVariablesBlock::~OptVariablesBlock()
{
  delete[] x;
  delete[] lb;
  delete[] ub;
}

/////////////////////////////////////////
// OptConstraints
/////////////////////////////////////////
OptConstraints::OptConstraints()
{
}
OptConstraints::~OptConstraints()
{
  for(auto b: vblocks)
    delete b;
}

OptConstraintsBlock* OptConstraints::get_block(const std::string& id)
{
  auto it = mblocks.find(id);
  if(it!=mblocks.end()) {
    return it->second;
  } else {
    cerr << "constraints block " << id << " was not found" << endl;
    return NULL;
  }
}

bool OptConstraints::append_consblock(OptConstraintsBlock* b)
{
  if(b) {
    if(mblocks.find(b->id)!= mblocks.end()) {
      cerr << "append_consblock:  block " << b->id << "already exists." << endl;
      assert(false);
      return false;
    }
    b->index=this->m();
    vblocks.push_back(b);
    mblocks[b->id] = b;
  }
  return true;
}

//////////////////////////////////////////////////////
// OptObjective
//////////////////////////////////////////////////////
OptObjective::~OptObjective()
{
  for(auto t: vterms)
    delete t;
}
bool OptObjective::append_objterm(OptObjectiveTerm* term)
{
  if(term) {
    if(mterms.find(term->id) != mterms.end()) {
      cerr << "append_objterm:  term " << term->id << " already exists." << endl;
      assert(false);
      return false;
    }
    vterms.push_back(term);
    mterms[term->id] = term;
  }
  return true;
}

} //end of namespace

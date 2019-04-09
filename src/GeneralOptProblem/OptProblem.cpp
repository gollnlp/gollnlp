#include "OptProblem.hpp"
#include "IpoptSolver.hpp"
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

  nnz_Jac = nnz_Hess = -1;
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
  
  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms) 
    if (!ot->eval_f(*vars_primal, new_x, obj_val) )
       return false;
  return true;
}
bool OptProblem::eval_cons    (const double* x, bool new_x, double* g)
{
  if(new_x) vars_primal->attach_to(x);
  for(auto& con: cons->vblocks)
    if(!con->eval_body(*vars_primal, new_x, g))
      return false;
  return true;
}
bool OptProblem::eval_gradobj (const double* x, bool new_x, double* grad)
{
  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms)
    if(!ot->eval_grad(*vars_primal, new_x, grad))
      return false;
  return true;
}
bool OptProblem::eval_Jaccons (const double* x, bool new_x, const int& nnz, int* i, int* j, double* M)
{
  if(new_x) vars_primal->attach_to(x);
  for(auto& con: cons->vblocks)
    if(!con->eval_Jac(*vars_primal, new_x, nnz, i,j,M))
      return false;
  return true;
}
bool OptProblem::
eval_HessLagr(const double* x, bool new_x, 
	      const double& obj_factor, 
	      const double* lambda, bool new_lambda,
	      const int& nnz, int* i, int* j, double* M)
{
  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms)
    if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, nnz,i,j,M))
      return false;

  if(new_lambda) vars_duals_cons->attach_to(lambda);

  for(auto& con: cons->vblocks)
    if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, nnz,i,j,M))
      return false;

  return true;
}

static int inline uniquely_indexise(vector<OptSparseEntry>& ij)
{
  int nnz = ij.size();
  if(nnz==0) return 0;

  std::sort(ij.begin(), ij.end());
  int nnz_unique = 0;
  for(int nz=1; nz<nnz; nz++) {
    if( ij[nz-1] < ij[nz] ) nnz_unique++;
    else assert(ij[nz-1].i == ij[nz].i && ij[nz-1].j == ij[nz].j);
    if(ij[nz].idx) *(ij[nz].idx) = nnz_unique;	
  }
  assert(nnz_unique<nnz);
  return nnz_unique+1;
}
int OptProblem::get_nnzJaccons()
{
  if(nnz_Jac<0) {
    vector<OptSparseEntry> ij;
    for(auto& con: cons->vblocks)
      con->get_Jacob_ij(ij);
    nnz_Jac = uniquely_indexise(ij);
  }
  return nnz_Jac;
}
int OptProblem::get_nnzHessLagr()
{
  if(nnz_Hess<0) {
    vector<OptSparseEntry> ij;

    for(auto& ot: obj->vterms) 
      ot->get_HessLagr_ij(ij);
    for(auto& con: cons->vblocks)
      con->get_HessLagr_ij(ij);
    nnz_Hess = uniquely_indexise(ij);
  }
  return nnz_Hess;
}

void OptProblem::fill_primal_vars(double* x) 
{
  int one=1;
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->x, &one, x + b->index, &one);
  }
}
void OptProblem::fill_vars_lower_bounds(double* lb)
{
  int one=1;
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->lb, &one, lb + b->index, &one);
  }
}
void OptProblem::fill_vars_upper_bounds(double* ub)
{
  int one=1;
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->ub, &one, ub + b->index, &one);
  }
}
void OptProblem::fill_cons_lower_bounds(double* lb)
{
  int one=1;
  for(auto b: cons->vblocks) {
    cout << b->id << endl;
    assert(b->n>=0);
    assert(b->index>=0);
    assert(b->lb!=NULL && b->n>0);
    DCOPY(&(b->n), b->lb, &one, lb + b->index, &one);
  }
}
void OptProblem::fill_cons_upper_bounds(double* ub)
{
  int one=1;
  for(auto b: cons->vblocks) {
    assert(b->ub!=NULL && b->n>0);
    DCOPY(&(b->n), b->ub, &one, ub + b->index, &one);
  }
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
  IpoptSolver solver(this);
  solver.initialize();

  solver.optimize();

  solver.finalize();
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
  : n(n_), id(id_), index(-1), xref(NULL)
{
  assert(n>=0);
  int i;

  x = new double[n];

  lb = new double[n];
  for(i=0; i<n; i++) lb[i] = -1e+20;

  ub = new double[n];
  for(i=0; i<n; i++) ub[i] = +1e+20;
}

OptVariablesBlock::OptVariablesBlock(const int& n_, const std::string& id_, double* lb_, double* ub_)
  : n(n_), id(id_), index(-1), xref(NULL)
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
  : n(n_), id(id_), index(-1), xref(NULL)
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

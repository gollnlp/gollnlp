#include "OptProblem.hpp"
#include "IpoptSolver.hpp"
#include <iostream>

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <string.h> //for memcpy

using namespace std;

namespace gollnlp {

static int one=1;

static void printnnz(int nnz, int* i, int*j, double* M=NULL)
{
  std::cout << "matrix with " << nnz << " nnz." << std::endl;
  for(int it=0; it<nnz; it++) {
    std::cout << i[it] << " " << j[it];
    if(M) std::cout << M[it];
    std::cout << endl;
  }
}

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
  obj_val=0.;
  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms) 
    if (!ot->eval_f(*vars_primal, new_x, obj_val) )
       return false;
  return true;
}
bool OptProblem::eval_cons    (const double* x, bool new_x, double* g)
{
  for(int i=0; i<cons->m(); i++) g[i]=0.;

  if(new_x) vars_primal->attach_to(x);
  for(auto& con: cons->vblocks)
    if(!con->eval_body(*vars_primal, new_x, g))
      return false;
  return true;
}
bool OptProblem::eval_gradobj (const double* x, bool new_x, double* grad)
{
  for(int i=0; i<vars_primal->n(); i++) grad[i]=0.;

  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms)
    if(!ot->eval_grad(*vars_primal, new_x, grad))
      return false;
  return true;
}

static void uniqueize(vector<OptSparseEntry>& ij, const int& nnz_unique, int* i, int* j) 
{
  //ij is sorted at this point
  int nnz = ij.size();
  if(nnz==0) return;

  i[0]=ij[0].i;
  j[0]=ij[0].j;

  int it_nz_unique = 1;
  for(int nz=1; nz<nnz; nz++) {
    if( ij[nz-1] < ij[nz] ) {
        i[it_nz_unique]=ij[nz].i;
	j[it_nz_unique]=ij[nz].j;
	it_nz_unique++;
    } else {
      assert(ij[nz-1].i == ij[nz].i && ij[nz-1].j == ij[nz].j);
    }
  }
  assert(nnz_unique<=nnz);
  assert(it_nz_unique==nnz_unique);
}
bool OptProblem::eval_Jaccons (const double* x, bool new_x, const int& nnz, int* i, int* j, double* M)
{
  if(M==NULL) {
    //fill in i and j based on ij_Hess

    //in some case, e.g. derivative checking, eval_Jaccons is called more than one time with M==NULL
    //re-uniqueize and display a warning
    if(nnz>ij_Jac.size()) {
      cout << "[Warning] Request for i and j for for Jacobian for " << nnz << " nnz, rebuilding ij_Jac." << endl;
      cout << "[Warning] If such requests happen repeatedly, performance/speed will be seriously affected." << endl;
      nnz_Jac=-1;
      nnz_Jac = get_nnzJaccons();
      assert(nnz == nnz_Jac && "structure of Jacobian changed");
    }
    uniqueize(ij_Jac, nnz, i, j);
    hardclear(ij_Jac);
    return true;
  }

  // case of M!=NULL > just fill in the values
  for(int i=0; i<nnz; i++) M[i]=0.;

  if(new_x) vars_primal->attach_to(x);
  for(auto& con: cons->vblocks)
    if(!con->eval_Jac(*vars_primal, new_x, nnz, i,j,M))
      return false;
  return true;
}

bool OptProblem::eval_HessLagr(const double* x, bool new_x, 
			       const double& obj_factor, 
			       const double* lambda, bool new_lambda,
			       const int& nnz, int* i, int* j, double* M)
{
  if(M==NULL) {
    //fill in i and j based on ij_Hess

    //in some case, e.g. derivative checking, eval_Jaccons is called more than one time with M==NULL
    //re-uniqueize and display a warning
    if(nnz>ij_Hess.size()) {
      cout << "[Warning] Request for i and j for for Hessian for " << nnz << " nnz, rebuilding ij_Hess." << endl;
      cout << "[Warning] If such requests happen repeatedly, performance/speed will be seriously affected." << endl;
      nnz_Hess=-1;
      nnz_Jac = get_nnzHessLagr();
      assert(nnz == nnz_Hess && "structure of Jacobian changed");
    }
    assert(nnz<=ij_Hess.size() && "ij_Hess was not yet built or has been cleared");
    uniqueize(ij_Hess, nnz, i, j);
    hardclear(ij_Hess);
    //printnnz(nnz,i,j,M);
    return true;
  }

  // case of M!=NULL > just fill in the values
  for(int i=0; i<nnz; i++) M[i]=0.;

  if(new_x) vars_primal->attach_to(x);
  for(auto& ot: obj->vterms)
    if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, nnz,i,j,M))
      return false;


  if(new_lambda) vars_duals_cons->attach_to(lambda);

  for(auto& con: cons->vblocks)
    if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, nnz,i,j,M))
      return false;

  //printnnz(nnz,i,j,M);


  return true;
}

static int inline uniquely_indexise(vector<OptSparseEntry>& ij)
{
  int nnz = ij.size();
  if(nnz==0) return 0;

  std::sort(ij.begin(), ij.end());

  int nnz_unique = 0;
  if(ij[0].idx) *(ij[0].idx)=0;

  for(int nz=1; nz<nnz; nz++) {
    if( ij[nz-1] < ij[nz] ) 
      nnz_unique++;
    else { 
      assert(ij[nz-1].i == ij[nz].i && ij[nz-1].j == ij[nz].j);
    }

    if(ij[nz].idx) *(ij[nz].idx) = nnz_unique;
  }
  assert(nnz_unique<nnz);
  return nnz_unique+1;
}
int OptProblem::get_nnzJaccons()
{
  if(nnz_Jac<0) {
    for(auto& con: cons->vblocks)
      con->get_Jacob_ij(ij_Jac);
    nnz_Jac = uniquely_indexise(ij_Jac);
  }
  return nnz_Jac;
}

#ifdef DEBUG
static bool check_is_upper(const vector<OptSparseEntry>& ij)
{
  for(auto& e: ij) if(e.j<e.i) return false;
  return true;
}
#endif
  
int OptProblem::get_nnzHessLagr()
{
  if(nnz_Hess<0) {
    for(auto& ot: obj->vterms) {
      ot->get_HessLagr_ij(ij_Hess);
#ifdef DEBUG
      if(false==check_is_upper(ij_Hess)) {
	printf("[Warning] Objective term %s returned nonzero elements in the lower triangular part of the Hessian.", ot->id.c_str());
	//assert(false);
      }
#endif
    }
    for(auto& con: cons->vblocks) {
      con->get_HessLagr_ij(ij_Hess);
#ifdef DEBUG
      if(false==check_is_upper(ij_Hess)) {
	printf("[Warning] Constraint term %s returned nonzero elements in the lower triangular part of the Hessian.", con->id.c_str());
	//assert(false);
      }
#endif      
    }
    nnz_Hess = uniquely_indexise(ij_Hess);
  }
  return nnz_Hess;
}

void OptProblem::fill_primal_vars(double* x) 
{
  
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->x, &one, x + b->index, &one);
  }
}
void OptProblem::fill_vars_lower_bounds(double* lb)
{
  
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->lb, &one, lb + b->index, &one);
  }
}
void OptProblem::fill_vars_upper_bounds(double* ub)
{
  
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->ub, &one, ub + b->index, &one);
  }
}
void OptProblem::fill_cons_lower_bounds(double* lb)
{
  
  for(auto b: cons->vblocks) {
    assert(b->n>=0);
    assert(b->index>=0);
    assert(b->lb!=NULL && b->n>0);
    DCOPY(&(b->n), b->lb, &one, lb + b->index, &one);
  }
}
void OptProblem::fill_cons_upper_bounds(double* ub)
{
  
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

bool OptProblem::fill_primal_start(double* x)
{
  for(auto b: vars_primal->vblocks) {
    if(b->providesStartingPoint)
      DCOPY(&(b->n), b->x, &one, x + b->index, &one);
    else for(int i=0; i<b->n; i++) x[i]=0.; 
  }
  return true;
}
bool OptProblem::fill_dual_bounds_start(double* z_L, double* z_U)
{
  assert(false);
  return true;
}
bool OptProblem::fill_dual_cons_start(double* lambda)
{
  for(auto b: vars_duals_cons->vblocks) {
    if(b->providesStartingPoint)
      DCOPY(&(b->n), b->x, &one, lambda + b->index, &one);
    else for(int i=0; i<b->n; i++) lambda[i]=0.; 
  }
  return true;
}

bool OptProblem::optimize(const std::string& nlpsolver)
{
  if(vars_duals_bounds) delete vars_duals_bounds;
  if(vars_duals_cons) delete vars_duals_cons;
  vars_duals_bounds = new_duals_vec_bounds();
  vars_duals_cons = new_duals_vec_cons();

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
  : n(n_), id(id_), index(-1), xref(NULL), providesStartingPoint(false)
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
  : n(n_), id(id_), index(-1), xref(NULL), providesStartingPoint(false)
{
  assert(n>=0);

  x = new double[n];

  int i;
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
  : n(n_), id(id_), index(-1), xref(NULL), providesStartingPoint(false)
{
  assert(n>=0);

  x = new double[n];

  int i;
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

void OptVariablesBlock::set_start_to(const double& scalar)
{
  providesStartingPoint = true;
  for(int i=0; i<n; i++) x[i]=scalar;
}
void OptVariablesBlock::set_start_to(const double* values)
{
  providesStartingPoint = true;
  //DCOPY(&n, this->x, &one, values, &one);

  //use memcpy since DCOPY does not take const double*
  memcpy(this->x, values, this->n * sizeof(double));
}
void OptVariablesBlock::set_start_to(const OptVariablesBlock& block)
{
  assert(block.n == this->n);
  set_start_to(block.x);
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

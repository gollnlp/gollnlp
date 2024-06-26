#include "OptProblem.hpp"
#include "IpoptSolver.hpp"
#include <iostream>

#include "blasdefs.hpp"
#include "goUtils.hpp"
#include "goTimer.hpp"

#include <string.h> //for memcpy

using namespace std;

namespace gollnlp {

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
  obj_value   = +1e+20;
  obj_barrier = +1e+20;
  num_iter = -1;
  
  vars_duals_bounds_L = vars_duals_bounds_U = NULL;
  vars_duals_cons = NULL;

  nnz_Jac = nnz_Hess = -1;
  nlp_solver = NULL;

  new_x_fgradf = true;

  need_intermediate_duals=false;
}

OptProblem::~OptProblem()
{
  if(nlp_solver)
    nlp_solver->finalize();
  delete obj;
  delete cons;
  delete vars_duals_bounds_L;
  delete vars_duals_bounds_U;
  delete vars_duals_cons;
  delete vars_primal;
  delete nlp_solver;
}

bool OptProblem::eval_obj(const double* x, bool new_x, double& obj_val)
{
  obj_val=0.;
  if(new_x) {
    vars_primal->attach_to(x);
  } else {
    if(new_x_fgradf) { new_x=true; new_x_fgradf=false; }
  }
  for(auto& ot: obj->vterms) {
    if (!ot->eval_f(*vars_primal, new_x, obj_val) )
       return false;
  }
  return true;
}

void OptProblem::print_objterms_evals()
{
  double total = 0., objterm; bool new_x=false; 
  printf("Objective breakdown:\n");
  for(auto& ot: obj->vterms) {
    objterm = 0.;
    if (!ot->eval_f(*vars_primal, new_x, objterm) )
      printf("  objterm '%s' -> error evaluating\n", ot->id.c_str());
    else 
      printf("  objterm '%s' -> %15.8e\n", ot->id.c_str(), objterm);
    total += objterm;
  }
  printf("Objective total -> %15.8e\n", total);
}

bool OptProblem::eval_cons(const double* x, bool new_x, double* g)
{
  for(int i=0; i<cons->m(); i++) g[i]=0.;

  if(new_x) {
    new_x_fgradf = true;
    vars_primal->attach_to(x);
  }
  for(auto& con: cons->vblocks) {
    if(!con->eval_body(*vars_primal, new_x, g))
      return false;
  }
  return true;
}
bool OptProblem::eval_gradobj (const double* x, bool new_x, double* grad)
{
  for(int i=0; i<vars_primal->n(); i++) grad[i]=0.;

  if(new_x) {
    vars_primal->attach_to(x);
  } else {
    if(new_x_fgradf) { new_x=true; new_x_fgradf=false; }
  }
  for(auto& ot: obj->vterms) {
    if(!ot->eval_grad(*vars_primal, new_x, grad))
      return false;
#ifdef DEBUG
    //int nn=vars_primal->n();
    //double nrm=DNRM2(&nn, grad, &ione);
    //printf("Norm of grad: %g after eval grad of %s.\n", nrm, ot->id.c_str());
#endif    
  }
  return true;
}

// static void uniqueize(vector<OptSparseEntry>& ij, const int& nnz_unique, int* i, int* j) 
// {
//   //ij is sorted at this point
//   int nnz = ij.size();
//   if(nnz==0) return;

//   i[0]=ij[0].i;
//   j[0]=ij[0].j;

//   int it_nz_unique = 1;
//   for(int nz=1; nz<nnz; nz++) {
//     if( ij[nz-1] < ij[nz] ) {
//         i[it_nz_unique]=ij[nz].i;
// 	j[it_nz_unique]=ij[nz].j;
// 	it_nz_unique++;
//     } else {
//       assert(ij[nz-1].i == ij[nz].i && ij[nz-1].j == ij[nz].j);
//     }
//   }
//   assert(nnz_unique<=nnz);
//   assert(it_nz_unique==nnz_unique);
// }
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
    //goTimer tm; tm.start();

    for(auto& con: cons->vblocks)
      con->get_Jacob_ij(ij_Jac);

    nnz_Jac = uniquely_indexise(ij_Jac);

    //tm.stop();
    //printf("Jacobian structure took %g sec\n", tm.getElapsedTime());
  }
  return nnz_Jac;
}

  // we assume that eval_Jaccons is called after get_nnzJaccons
bool OptProblem::eval_Jaccons (const double* x, bool new_x, const int& nnz, int* i, int* j, double* M)
{
  if(new_x) {
    new_x_fgradf=true;
    vars_primal->attach_to(x);
  }
  if(M==NULL) {


    for(auto& con: cons->vblocks) {
      if(!con->eval_Jac(*vars_primal, new_x, nnz, i,j,M)) {
	assert(false && "eval_Jaccons should be called after get_nnzJaccons");
      }
    }
    return true;
  }

  // case of M!=NULL > just fill in the values
  for(int i=0; i<nnz; i++) M[i]=0.;

  for(auto& con: cons->vblocks) {
    if(!con->eval_Jac(*vars_primal, new_x, nnz, i,j,M))
      return false;
#ifdef DEBUG
    //double maxmag=0.;
    //for(int ii=0; ii<nnz; ii++) if(fabs(M[ii]) > maxmag) maxmag=fabs(M[ii]);
    //printf("Jacob entries max mag is %g after evaluating constraints %s.\n", maxmag, con->id.c_str());
#endif    
  }
  return true;
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
    //goTimer tm; tm.start();

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

    //tm.stop();
    //printf("Hessian structure %g sec\n", tm.getElapsedTime());
  }
  return nnz_Hess;
}


bool OptProblem::eval_HessLagr(const double* x, bool new_x, 
			       const double& obj_factor, 
			       const double* lambda, bool new_lambda,
			       const int& nnz, int* i, int* j, double* M)
{
  if(new_x) {
    new_x_fgradf=true; 
    vars_primal->attach_to(x);
  }
  if(new_lambda) vars_duals_cons->attach_to(lambda);
  if(M==NULL) {
    for(auto& ot: obj->vterms) {
      if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, nnz,i,j,M)) {
	assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
      }
    }
    for(auto& con: cons->vblocks) {
      if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, nnz,i,j,M)) {
	assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
      }
    }
    //printnnz(nnz,i,j,M);
  } else {

    // case of M!=NULL > just fill in the values
    for(int it=0; it<nnz; it++) M[it]=0.;
    
    for(auto& ot: obj->vterms)
      if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, nnz,i,j,M))
	return false;
    
    for(auto& con: cons->vblocks)
      if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, nnz,i,j,M))
	return false;
    
    //printnnz(nnz,i,j,M);
  }
  return true;
}

void OptProblem::fill_primal_vars(double* x) 
{ 
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->x, &ione, x + b->index, &ione);
  }
}


void OptProblem::set_obj_value(const double& f)
{
  obj_value = f;
}
void OptProblem::set_obj_value_barrier(const double& f)
{
  obj_barrier = f;
}

void OptProblem::set_num_iters(int n_iter)
{
  num_iter = n_iter;
}

void OptProblem::set_primal_vars(const double* x)
{
  for(auto b: vars_primal->vblocks) {
    memcpy(b->x, x+b->index, b->n*sizeof(double));
    //DCOPY(&(b->n), x, &one, x + b->index, &one);
  }
}


void OptProblem::fill_vars_lower_bounds(double* lb)
{
  
  for(auto b: vars_primal->vblocks) {
    //cout << "----fill_vars_lower_bounds: " << b->id << endl;
    // if(b->id == "pslack_n_p_balance_1") {
    //   //printf("\n");
    //   //for(int i=0; i<b->n; i++) {printf("%12.5e ", b->x[i]); b->x[i]=0.;}
    //   //printf("\n");

    //   int size=b->n;
    //   double buf1[size], buf2[size];
    //   for(int i=0; i<size; i++) buf1[i]=10.;
      
    //   DCOPY(&size, buf1, &ione, buf2, &ione);

    //   printf("---%g %d\n", buf2[size/2], b->n);
    // }
    DCOPY(&(b->n), b->lb, &ione, lb + b->index, &ione);
  }
}
void OptProblem::fill_vars_upper_bounds(double* ub)
{
  
  for(auto b: vars_primal->vblocks) {
    DCOPY(&(b->n), b->ub, &ione, ub + b->index, &ione);
  }
}
void OptProblem::fill_cons_lower_bounds(double* lb)
{
  
  for(auto b: cons->vblocks) {
    assert(b->n>=0);
    assert(b->index>=0);
    DCOPY(&(b->n), b->lb, &ione, lb + b->index, &ione);
  }
}
void OptProblem::fill_cons_upper_bounds(double* ub)
{
  for(auto b: cons->vblocks) {
    DCOPY(&(b->n), b->ub, &ione, ub + b->index, &ione);
  }
}

void OptProblem::set_duals_vars_bounds(const double* zL, const double* zU)
{
  for(auto b: vars_primal->vblocks) {
    auto* bdualL = vars_duals_bounds_L->get_block(string("duals_bndL_") + b->id);
    auto* bdualU = vars_duals_bounds_U->get_block(string("duals_bndU_") + b->id);
    assert(bdualL->n == b->n);
    assert(bdualU->n == b->n);
    memcpy(bdualL->x, zL + b->index, b->n*sizeof(double));
    memcpy(bdualU->x, zU + b->index, b->n*sizeof(double));
  }
}

void OptProblem::fill_dual_vars_bounds(double* zL, double* zU)
{
  for(auto b: vars_primal->vblocks) {
    auto* bdualL = vars_duals_bounds_L->get_block(string("duals_bndL_") + b->id);
    auto* bdualU = vars_duals_bounds_U->get_block(string("duals_bndU_") + b->id);
    DCOPY(&(b->n), bdualL->x, &ione, zL + b->index, &ione);
    DCOPY(&(b->n), bdualU->x, &ione, zU + b->index, &ione);
  }
}

bool OptProblem::fill_dual_bounds_start(double* zL, double* zU)
{
  for(auto b: vars_primal->vblocks) {
    auto* bdualL = vars_duals_bounds_L->get_block(string("duals_bndL_") + b->id);
    auto* bdualU = vars_duals_bounds_U->get_block(string("duals_bndU_") + b->id);

    if(!bdualL->providesStartingPoint) {
      assert(false == bdualU->providesStartingPoint);
      for(int i=0; i<b->n; i++) {
	zL[b->index+i] = zU[b->index+i] = 0.;
      }
      continue;
    }

    DCOPY(&(b->n), bdualL->x, &ione, zL + b->index, &ione);
    DCOPY(&(b->n), bdualU->x, &ione, zU + b->index, &ione);
  }  
  return true;
}

OptVariables* OptProblem::new_duals_cons()
{
  OptVariables* duals = new OptVariables();
  for(auto b: cons->vblocks) {
    duals->append_varsblock(new OptVariablesBlock(b->n, string("duals_") + b->id));
  }
  return duals;
}
OptVariables* OptProblem::new_duals_lower_bounds()
{
  OptVariables* duals = new OptVariables();
  for(auto b: vars_primal->vblocks) {
    duals->append_varsblock(new OptVariablesBlock(b->n, string("duals_bndL_") + b->id));
  }
  return duals;
}
OptVariables* OptProblem::new_duals_upper_bounds()
{
  OptVariables* duals = new OptVariables();
  for(auto b: vars_primal->vblocks) {
    duals->append_varsblock(new OptVariablesBlock(b->n, string("duals_bndU_") + b->id));
  }
  return duals;
}

//these setters return false if the option is not recognized by the NLP solver
bool OptProblem::set_solver_option(const std::string& name, int value)
{
  return nlp_solver->set_option(name,value);
}
bool OptProblem::set_solver_option(const std::string& name, double value)
{
  return nlp_solver->set_option(name,value);
}
bool OptProblem::set_solver_option(const std::string& name, const std::string& value)
{
  return nlp_solver->set_option(name,value);
}

bool OptProblem::fill_primal_start(double* x)
{
  for(auto b: vars_primal->vblocks) {
    if(b->providesStartingPoint)
      DCOPY(&(b->n), b->x, &ione, x + b->index, &ione);
    else for(int i=b->index; i<b->n+b->index; i++) x[i]=0.; 
  }
  return true;
}

bool OptProblem::fill_dual_cons_start(double* lambda)
{
  for(auto b: vars_duals_cons->vblocks) {
    if(b->providesStartingPoint)
      DCOPY(&(b->n), b->x, &ione, lambda + b->index, &ione);
    else for(int i=0; i<b->n; i++) lambda[i]=0.; 
  }
  return true;
}

void OptProblem::set_duals_vars_cons(const double* lambda)
{
  for(auto b: vars_duals_cons->vblocks) {
    memcpy(b->x, lambda + b->index, b->n*sizeof(double));
  }
}

void OptProblem::use_nlp_solver(const std::string& name)
{
  if(NULL == nlp_solver) {
    nlp_solver = new IpoptSolver(this);
    nlp_solver->initialize();
  }
}

void OptProblem::primal_problem_changed()
{
  nnz_Jac = nnz_Hess = -1;
  ij_Jac.clear();
  ij_Hess.clear();
}

void OptProblem::dual_problem_changed()
{
  nnz_Jac = nnz_Hess = -1;
  ij_Jac.clear();
  ij_Hess.clear();

  if(vars_duals_bounds_L) delete vars_duals_bounds_L;
  if(vars_duals_bounds_U) delete vars_duals_bounds_U;
  if(vars_duals_cons) delete vars_duals_cons;
  vars_duals_bounds_L = new_duals_lower_bounds();
  vars_duals_bounds_U = new_duals_upper_bounds();
  vars_duals_cons = new_duals_cons();
}

void OptProblem::reallocate_nlp_solver()
{
  if(nlp_solver) {
    nlp_solver->finalize();
    delete nlp_solver;
    nlp_solver=NULL;
  }
  use_nlp_solver("ipopt");
}

OptimizationStatus OptProblem::optimization_status() const
{
  if(nlp_solver)
    return nlp_solver->return_code();
  else 
    return Invalid_Option;
}

bool OptProblem::optimize(const std::string& solver_name)
{

  if(vars_duals_bounds_L) delete vars_duals_bounds_L;
  if(vars_duals_bounds_U) delete vars_duals_bounds_U;
  if(vars_duals_cons) delete vars_duals_cons;
  vars_duals_bounds_L = new_duals_lower_bounds();
  vars_duals_bounds_U = new_duals_upper_bounds();
  vars_duals_cons = new_duals_cons();

  if(!nlp_solver) {
    cout << "call 'use_nlp_solver' first\n";
    return false;
  }

  if(true==nlp_solver->optimize()) {
    this->set_have_start();
  } else {
    return false;
  }

  return true;
}

bool OptProblem::reoptimize(RestartType t)
{
  assert(vars_duals_bounds_L && "first call optimize instead");
  assert(vars_duals_bounds_U && "first call optimize instead");
  assert(vars_duals_cons && "call optimize instead");

  nlp_solver->set_start_type(t);

  if(true==nlp_solver->optimize()) {
    this->set_have_start();
  } else {
    return false;
  }

  return true;
}

void OptProblem::set_have_start()
{
  //var_primals were updated with the values from the solver's finalize method
  for(auto b: vars_primal->vblocks) b->providesStartingPoint=true;
  for(auto b: vars_duals_bounds_L->vblocks) b->providesStartingPoint=true;
  for(auto b: vars_duals_bounds_U->vblocks) b->providesStartingPoint=true;
  for(auto b: vars_duals_cons->vblocks) b->providesStartingPoint=true;
}



void  OptProblem::print_summary() const
{
  printf("\n*************************************************************************\n");
  printf("Problem with %d variables and %d constraints\n", get_num_variables(), get_num_constraints());
  printf("Variables blocks: \n");
  for(auto it : vars_primal->vblocks) 
    printf("\t'%s' size %d  startsAt %d   providesStPoint %d\n",
	   it->id.c_str(), it->n, it->index, it->providesStartingPoint);

  printf("Constraints blocks: \n");
  for(auto it : cons->vblocks) 
    printf("\t'%s' size %d  startsAt %d\n", it->id.c_str(), it->n, it->index);
  
  printf("Objective terms: \n");
  for(auto it: obj->vterms) printf("\t'%s' \n", it->id.c_str());

  if(vars_duals_bounds_L) {
    printf("Duals variables lower bounds: \n");
    for(auto it: vars_duals_bounds_L->vblocks) 
      printf("\t'%s' size %d  startsAt %d   providesStPoint %d\n",
	     it->id.c_str(), it->n, it->index, it->providesStartingPoint);
  }
  if(vars_duals_bounds_U) {
    printf("Duals variables upper bounds: \n");
    for(auto it: vars_duals_bounds_U->vblocks) 
      printf("\t'%s' size %d  startsAt %d   providesStPoint %d\n",
	     it->id.c_str(), it->n, it->index, it->providesStartingPoint);
  }
  if(vars_duals_cons) {
    printf("Duals variables constraints: \n");
    for(auto it: vars_duals_cons->vblocks) 
      printf("\t'%s' size %d  startsAt %d   providesStPoint %d\n",
	     it->id.c_str(), it->n, it->index, it->providesStartingPoint);
  }
}

///////////////////////////////////////////////////////////////////
// OptVariables
///////////////////////////////////////////////////////////////////

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
      cerr << "appendVarsBlock:  block " << b->id << " already exists." << endl;
      assert(false);
      return false;
    }
    b->index=this->n();
    vblocks.push_back(b);
    mblocks[b->id] = b;
  }
  return true;
}

void OptVariables::print_summary(const std::string var_name) const
{
  printf("Optimization variable %s\n", var_name.c_str());
  for(auto& b: vblocks)
    printf("    '%s' size %d startsAt %d providesStPoint %d\n", 
	   b->id.c_str(), b->n, b->index, b->providesStartingPoint);
  
}


void OptVariables::print(const std::string var_name) const
{
  printf("--------------------\n");
  printf("Optimization variable %s of size %d\n", var_name.c_str(), n());
  for(auto b: vblocks) {
    b->print();
  }
  printf("--------------------\n");
}


bool OptVariables::append_varsblocks(std::vector<OptVariablesBlock*> v)
{
  for(auto& b: v) if(!append_varsblock(b)) return false;
  return true;
}

void OptVariables::attach_to(const double *x)
{
  for(auto b: vblocks) b->xref = x + b->index;
}

int OptVariables::MPI_Bcast_x(int root,
			      MPI_Comm comm,
			      int my_rank, double* buffer)
{
  int dealloc=false;
  if(NULL == buffer) {
    buffer = new double[this->n()];
    dealloc=true;
  }
  //pack
  if(my_rank==root) {
    for(auto b: vblocks) {
      assert(b->index+b->n <=this->n());
      memcpy(buffer+b->index, b->x, b->n*sizeof(double));
    }
  }

  int ierr = MPI_Bcast(buffer, this->n(), MPI_DOUBLE, root, comm);
  assert(MPI_SUCCESS==ierr);
  
  //unpack
  if(my_rank!=root) 
  {
    for(auto b: vblocks) {
      assert(b->index+b->n <=this->n());
      memcpy(b->x, buffer+b->index, b->n*sizeof(double));
    }
  }
  
  if(dealloc) {
    delete[] buffer;
    buffer = NULL;
  }
  return ierr;
}

bool OptVariables::provides_start()
{
  for(auto& b: vblocks)
    if(!b->providesStartingPoint)
      return false;
  return true;
}

bool OptVariables::set_start_to(const OptVariables& src)
{
  for(auto b: this->vblocks) {
    auto bsrc = src.get_block(b->id);
    //assert(bsrc!=NULL);
    if(bsrc==NULL) {
#ifdef DEBUG
      printf("warning: set_start_to could not get block [%s] from src\n", b->id.c_str());
      src.print_summary("source");
      print_summary("destination");
#endif      
      return false;
    }
    assert(b->n == bsrc->n);
    if(b->n != bsrc->n) return false;
    b->set_start_to(*bsrc);
  }
  return true;
}

void OptVariables::copy_to(double* a)
{
  for(auto b: this->vblocks) {
    memcpy(a+b->index, b->x, b->n * sizeof(double));
  }
}
void OptVariables::copy_to(std::vector<double>& v)
{
  if(v.size()<this->n())
    v = vector<double>(this->n());
  this->copy_to(v.data());
}
void OptVariables::copy_from(const double* v)
{
  for(auto b: this->vblocks) {
    b->set_start_to(v + b->index);
  }
}

void OptVariables::set_inactive_duals_lb_to(double ct, const OptVariables& primals)
{
  OptVariablesBlock *bdual=NULL;
  const OptVariablesBlock *bprimal=NULL;
  if(vblocks.size()!=primals.vblocks.size()) { assert(false); return; }
  for(int b=0; b<vblocks.size(); b++) {

    bdual = vblocks[b];
    bprimal = primals.vblocks[b];

    if(bdual->n != bprimal->n) { assert(false); return; }
    if(bdual->index != bprimal->index) { assert(false); return; }

    for(int idx=0; idx<bdual->n; idx++) {
      if(bprimal->lb[idx]<=-1e+20) {
	//printf("  \t wrote for idx=%d\n", idx);
	bdual->x[idx] = ct;
      }
    }
  }
}
void OptVariables::set_inactive_duals_ub_to(double ct, const OptVariables& primals)
{
  OptVariablesBlock *bdual=NULL;
  const OptVariablesBlock *bprimal=NULL;
  if(vblocks.size()!=primals.vblocks.size()) { assert(false); return; }
  for(int b=0; b<vblocks.size(); b++) {

    bdual = vblocks[b];
    bprimal = primals.vblocks[b];

    if(bdual->n != bprimal->n) { assert(false); return; }
    if(bdual->index != bprimal->index) { assert(false); return; }

    for(int idx=0; idx<bdual->n; idx++) {
      if(bprimal->ub[idx]>=+1e+20) {
	//printf("  \t wrote for idx=%d\n", idx);
	bdual->x[idx] = ct;
      }
    }
  }
}

void OptVariables::delete_block(const std::string& id)
{
  OptVariablesBlock* block = NULL;
  auto it = mblocks.find(id);
  if(it!=mblocks.end()) {
    block =  it->second; 
    mblocks.erase(it);
  }

  auto vit = find(vblocks.begin(), vblocks.end(), block);
  if(block) { assert(vit!=vblocks.end()); }

  if(vit != vblocks.end()) vblocks.erase(vit);

  if(block) delete block;
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
    DCOPY(&n, lb_, &ione, lb, &ione);
  else
    for(i=0; i<n; i++) lb[i] = -1e+20;

  ub = new double[n];
  if(ub_)
    DCOPY(&n, ub_, &ione, ub, &ione);
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
void OptVariablesBlock::print() const
{
  printf("vector '%s' of size %d providesStPt %d\n", id.c_str(), n, providesStartingPoint);
  for(int i=0; i<n; i++) printf("%12.5e ", x[i]);
  printf("\n");
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
    //cerr << "constraints block " << id << " was not found" << endl;
    return NULL;
  }
}

void OptConstraints::delete_block(const std::string& id)
{
  OptConstraintsBlock* block = NULL;
  auto it = mblocks.find(id);
  if(it!=mblocks.end()) {
    block =  it->second; 
    mblocks.erase(it);
  }

  auto vit = find(vblocks.begin(), vblocks.end(), block);
  if(block) { assert(vit!=vblocks.end()); }

  if(vit != vblocks.end()) vblocks.erase(vit);

  if(block) delete block;
}

bool OptConstraints::append_consblock(OptConstraintsBlock* b)
{
  if(b) {
    if(mblocks.find(b->id)!= mblocks.end()) {
      cerr << "append_consblock:  block " << b->id << " already exists." << endl;
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

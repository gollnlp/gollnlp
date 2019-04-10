#ifndef GOLLNLP_EX1OPTOBJ
#define GOLLNLP_EX1OPTOBJ

#include "OptProblem.hpp"

//provides optimization objects for
// min 0.5||x-1||^2 + 0.5*||y-z||^2 + 0.5*||s||^2
//s.t. sum(x^2) <=10
//     sin(x_i) + z_i + s_i = cos(y_i), for i=1,...,size(y)
//     x free, s>=0, z<=5, 0<=y<=10
// the slacks and the last objective term are created by the second constraint block
//
// also the optimization objects here are used to compose the related problem
// min 0.5||x||^2 + 0.5*||y-z||^2
//s.t. sum(x^2) <=10
//     sin(x_i) + z_i <= cos(y_i), for i=1,...,size(y)
//     x free, z<=5, 0<=y<=10

using namespace gollnlp;
using namespace std;

#include <cmath>

//for 0.5||x||^2
class Ex1SingleVarQuadrObjTerm : public OptObjectiveTerm {
public: 
  Ex1SingleVarQuadrObjTerm(const std::string& id, OptVariablesBlock* x_) 
    : OptObjectiveTerm(id), x(x_), H_nz_idxs(NULL)
  {};

  virtual ~Ex1SingleVarQuadrObjTerm() 
  {
    if(H_nz_idxs) 
      delete[] H_nz_idxs;
  }

  virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
  {
    int nvars = x->n; double aux;
    assert(x == vars_primal.get_block("x"));
    for(int it=0; it<nvars; it++) {
      aux = x->xref[it] - 1.;
      obj_val += aux * aux * 0.5;
    }
    return true;
  }
  virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
  {
    assert(x == vars_primal.get_block("x"));
    double* g = grad + x->index;
    for(int it=0; it<x->n; it++) 
      g[it] += x->xref[it] - 1.;
    return true;
  }
  virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
  {
    assert(x == vars_primal.get_block("x"));

    for(int it=0; it<x->n; it++) {
      assert(H_nz_idxs[it]>=0);
      assert(H_nz_idxs[it]<nnz);
      M[H_nz_idxs[it]] += obj_factor;
    }
    return true;
  }

  virtual int get_HessLagr_nnz() { return x->n; }
  // (i,j) entries in the HessLagr to which this term contributes to
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  { 
    int nvars = x->n, i;
    if(NULL==H_nz_idxs)
      H_nz_idxs = new int[nvars];

    for(int it=0; it < nvars; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
    }
    return true; 
  }

private:
  OptVariablesBlock* x;
  //keep the index for each nonzero elem in the Hessian that this constraints block contributes to
  int *H_nz_idxs;
};

// for computing 0.5*||a-b||^2
class Ex1TwoVarsQuadrObjTerm : public OptObjectiveTerm {
public:
  Ex1TwoVarsQuadrObjTerm(const std::string& id, OptVariablesBlock* a_, OptVariablesBlock* b_)
    : OptObjectiveTerm(id), a(a_), b(b_), H_nz_idxs(NULL)
  {};
  virtual ~Ex1TwoVarsQuadrObjTerm()
  {
    if(H_nz_idxs)
      delete[] H_nz_idxs;
  }
  virtual bool eval_f(const OptVariables& x, bool new_x, double& obj_val)
  {
    assert(a->n == b->n); 
    double aux;
    for(int i=0; i<a->n; i++) {
      aux = a->xref[i] - b->xref[i];
      obj_val += 0.5 * aux*aux;
    }
    return true;
  }
  virtual bool eval_grad(const OptVariables& x, bool new_x, double* grad)
  {
    assert(a->n == b->n); 
    double aux;
    for(int i=0; i<a->n; i++) {
      aux = a->xref[i] - b->xref[i];
      grad[i+a->index] =  aux;
      grad[i+b->index] = -aux;
    }
    return true;
  }
  virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
  {
    int itaux;
    for(int it=0; it<2*a->n; ) {
      assert(H_nz_idxs[it]>=0);
      assert(H_nz_idxs[it]<nnz);
      
      M[H_nz_idxs[it]] += obj_factor; // with respect to a_i, a_i
      it++;
      M[H_nz_idxs[it]] -= obj_factor; // with respect to a_i, b_i
      it++;
    }
    for(int it=2*a->n; it<2*a->n+b->n; it++) {
      assert(H_nz_idxs[it]>=0);
      assert(H_nz_idxs[it]<nnz);
      M[H_nz_idxs[it]] += obj_factor; // with respect to b_i, b_i
    }

    return true;
  }
  virtual int get_HessLagr_nnz() { return 2*a->n + b->n; }
  // (i,j) entries in the HessLagr to which this term contributes to
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  { 
    assert(a->n == b->n); 
    int nnz = this->get_HessLagr_nnz();
    if(NULL==H_nz_idxs)
      H_nz_idxs = new int[nnz];

    int i;
    for(int it=0; it < a->n; it++) {
      i = a->index+it;
      vij.push_back(OptSparseEntry(i,i, H_nz_idxs + 2*it));
      vij.push_back(OptSparseEntry(i,b->index+it, H_nz_idxs + 2*it+1));
    }
    for(int it=0; it < b->n; it++) {
      i = b->index+it;
      vij.push_back(OptSparseEntry(i,i, H_nz_idxs + 2*a->n + it));
    }
    return true; 
  }
private:
  OptVariablesBlock *a, *b;
  //upper triangular indexes
  //first 2n entries are for Hessian w.r.t. (ai,ai) and (ai,bi)
  //last  n  entries are for Hessian w.r.t. (bi,bi)
  int* H_nz_idxs;
};

//for sum(x^2) <=10
class Ex1SumOfSquaresConstraints : public OptConstraintsBlock {
public:
  Ex1SumOfSquaresConstraints(const std::string& id_, OptVariablesBlock* x_) 
    : OptConstraintsBlock(id_,1), x(x_), H_nz_idxs(NULL), J_nz_start(-1)
  { 
    lb[0]=-1e+20; 
    ub[0]= 10.;
  };
  virtual ~Ex1SumOfSquaresConstraints() 
  {
    if(H_nz_idxs)
      delete[] H_nz_idxs;
  };

  // all these functions 
  //  - should add their contribution to the output
  //  - return false if an error occurs in the evaluation
  virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
  {
    assert(vars_primal.get_block("x")==x);

    for(int it=0; it<x->n; it++) body[this->index] += x->xref[it] * x->xref[it];
    return true;
  };
  virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			const int& nnz, int* i, int* j, double* M)
  {
    assert(M);
    // if(NULL==M) {
    //   //we have (i0,j0), (i0,j1), ..., (i0, jn), where 
    //   // i0 = this->index
    //   // j0 = x->index
    //   // n  = x->n
    //   //we put these starting at this->J_nz_start
    //   int *ip = i+this->J_nz_start, *jp = j+this->J_nz_start, it;
    //   for(it=0; it<x->n; it++) *(ip++) = this->index;
    //   for(it=0; it<x->n; it++) *(jp++) = x->index+it;      
    // } else {

    //starting at this->J_nz_start
    double *Mp = M+this->J_nz_start; 
    for(int it=0; it<x->n; it++) {
      *(Mp++) +=  2*x->xref[it];
      //printf("[%d] = %g x=%g  start=%d\n", it, M[this->J_nz_start+it], x->xref[it], this->J_nz_start);
    }
    
    return true;
  };
  virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			     const OptVariables& lambda_vars, bool new_lambda,
			     const int& nnz, int* i, int* j, double* M)
  {
    const OptVariablesBlock* lambda = lambda_vars.get_block(string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==1);

    for(int it=0; it < x->n; it++)
      M[H_nz_idxs[it]] += 2. * lambda->xref[0];
    return true;
  };

  // methods that need to be implemented to specify the sparsity pattern of the 
  // implementer's contribution to the sparse derivatives
  virtual int get_HessLagr_nnz() { return x->n; }
  virtual int get_Jacob_nnz(){ return x->n; }

  // (i,j) entries in the HessLagr to which the implementer's contributes to
  // this is only called once
  // push_back in vij 
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  {
    int n=x->n, i;
    if(n<=0) return true;
    
    if(NULL==H_nz_idxs) {
      H_nz_idxs = new int[n];
    }

    for(int it=0; it<n; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
    }
    return true;
  }

  virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
  {
    int i;
    if(n<=0) return true;

    vij.push_back(OptSparseEntry(this->index, x->index, &J_nz_start));

    for(int it=1; it<x->n; it++) {
      vij.push_back(OptSparseEntry(this->index, x->index+it, NULL));
    }
    return true;
  } 

private:
  OptVariablesBlock* x; 
  int *H_nz_idxs;
  //we only expect the global index in the Jacob for the first entry; all the 
  //other entries coming from this block will be consecutive to that.
  int J_nz_start;
};


//for sin(x_i) + z_i - cos(y_i) <=0 or sin(x_i) + z_i - cos(y_i) + s_i = 0, i=1,..,size(y)
class Ex1Constraint2 : public OptConstraintsBlock {
public: 
  Ex1Constraint2(const std::string& id_, 
		 OptVariablesBlock* x_, OptVariablesBlock *y_, OptVariablesBlock* z_,
		 bool useSlacks=false) 
    : OptConstraintsBlock(id_, y_->n), 
      x(x_), y(y_), z(z_), s(NULL), 
      use_slacks(useSlacks), H_nz_idxs(NULL), J_nz_idxs(NULL)
  {
    for(int i=0; i<n; i++) ub[i] = 0.;
    if(use_slacks)
      for(int i=0; i<n; i++) lb[i] = 0.;
    else
      for(int i=0; i<n; i++) lb[i] = -1e+20;
  };
  // one can also receives the entire set of variables and initialize variables
  // block used by this constraint block by looking up x, y, z in 'vars'
  Ex1Constraint2(const std::string& id_, OptVariables* vars, bool useSlacks=false) 
    : OptConstraintsBlock(id_,0), s(NULL), 
      use_slacks(useSlacks), H_nz_idxs(NULL), J_nz_idxs(NULL)
  {
    x = vars->get_block("x"); assert(x);
    y = vars->get_block("y"); assert(y);
    z = vars->get_block("z"); assert(z);
    this->n = y->n;

    for(int i=0; i<n; i++) ub[i] = 0.;
    if(use_slacks)
      for(int i=0; i<n; i++) lb[i] = 0.;
    else
      for(int i=0; i<n; i++) lb[i] = -1e+20;
  };
   
  virtual ~Ex1Constraint2()
  {
    if(H_nz_idxs)
      delete[] H_nz_idxs;
    if(J_nz_idxs)
      delete[] J_nz_idxs;
  };

  virtual OptVariablesBlock* create_varsblock() { 
    if(!use_slacks) return NULL; 
    
    assert(s==NULL);
    s = new OptVariablesBlock(y->n, "s", 0, 1e+20);
    return s;
  };

  virtual OptObjectiveTerm* create_objterm() { 
    if(!use_slacks) return NULL; 
    assert(s);
    return new Ex1SingleVarQuadrObjTerm("quadr_pen_s", s);
  };


  // all these functions 
  //  - should add their contribution to the output
  //  - return false if an error occurs in the evaluation
  virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
  {
    assert(vars_primal.get_block("x")==x);

    for(int it=0; it<this->n; it++){
      body[it+this->index] += sin(x->xref[it]) + z->xref[it] - cos(y->xref[it]);
    }
    if(use_slacks) {
      for(int it=0; it<this->n; it++){
	body[it+this->index] += s->xref[it];
      }
    }
    return true;
  };
  virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			const int& nnz, int* i, int* j, double* M)
  {
    //for sin(x_i) + z_i - cos(y_i) <=0 or sin(x_i) + z_i - cos(y_i) + s_i = 0, i=1,..,size(y)
    //3 or 4 (when use_slacks) per row

    assert(M);
    int itnz=0;
    for(int i=0; i<n; i++) {
      M[J_nz_idxs[itnz]] += cos(x->xref[i]); itnz++; //w.r.t. x_i
      M[J_nz_idxs[itnz]] += sin(y->xref[i]); itnz++; //w.r.t. y_i
      M[J_nz_idxs[itnz]] += 1.;              itnz++; //w.r.t. z_i
    
      if(use_slacks) {
	M[J_nz_idxs[itnz]] += 1.;             itnz++; //w.r.t. s_i
      }
      assert(itnz<nnz);
	
    }
    assert(itnz==get_Jacob_nnz());
    return true;
  };
  virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			     const OptVariables& lambda_vars, bool new_lambda,
			     const int& nnz, int* i, int* j, double* M)
  {
    const OptVariablesBlock* lambda = lambda_vars.get_block(string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==n);

    for(int it=0; it<n; it++)
      M[H_nz_idxs[it]] += -sin(x->xref[it]) * lambda->xref[it];

    int* H_nz_head=H_nz_idxs+n;
    for(int it=0; it<n; it++)
      M[H_nz_head[it]] += cos(y->xref[it]) * lambda->xref[it];

    return true;
  };

  // methods that need to be implemented to specify the sparsity pattern of the 
  // implementer's contribution to the sparse derivatives
  virtual int get_Jacob_nnz() 
  { 
    if(use_slacks) return 4*n;
    else           return 3*n; 
  }
  virtual int get_HessLagr_nnz() 
  { 
    return 2*n;
  }

  // (i,j) entries in the HessLagr to which the implementer's contributes to
  // this is only called once
  // push_back in vij 
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  {
    int nnz = get_HessLagr_nnz();
    if(NULL==H_nz_idxs) {
      H_nz_idxs = new int[nnz];
    }
    int i;
    for(int it=0; it<n; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
    }
    int* H_nz_head = H_nz_idxs + n;
    for(int it=0; it<n; it++) {
      i = y->index+it;
      vij.push_back(OptSparseEntry(i,i,H_nz_head + it));
    }
    return true;
  }

  virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
  {
    int i, itnz=0, nnz = get_Jacob_nnz();

    if(NULL==J_nz_idxs)
      J_nz_idxs = new int[nnz];
 
    //3 or 4 (when use_slacks) per row
    for(int row=0; itnz<nnz; row++) {
      i = this->index+row;
      vij.push_back(OptSparseEntry(i, x->index+row, J_nz_idxs+itnz));
      itnz++;

      vij.push_back(OptSparseEntry(i, y->index+row, J_nz_idxs+itnz));
      itnz++;

      vij.push_back(OptSparseEntry(i, z->index+row, J_nz_idxs+itnz));
      itnz++;

      if(use_slacks) {
	vij.push_back(OptSparseEntry(i, s->index+row, J_nz_idxs+itnz));
	itnz++;
      }
    }
    return true;
  } 

private:
  //s is null when use_slacks==false
  const OptVariablesBlock *x, *y, *z;
  OptVariablesBlock *s;
  bool use_slacks;

  int* J_nz_idxs;
  int* H_nz_idxs;
};


#endif

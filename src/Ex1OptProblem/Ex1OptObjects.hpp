#ifndef GOLLNLP_EX1OPTOBJ
#define GOLLNLP_EX1OPTOBJ

#include "OptProblem.hpp"

//provides optimization objects for
// min 0.5||x||^2 + 0.5*||y-z||^2 + 0.5*||s||^2
//s.t. sum(x^2) <=10
//     sin(x) + z + s = cos(y)
//     x free, s>=0, z<=5, 0<=y<=10
// the slacks and the last objective term are created by the second constraint block
//
// also the optimization objects here are used to compose the related problem
// min 0.5||x||^2 + 0.5*||y-z||^2
//s.t. sum(x^2) <=10
//     sin(x) + z <= cos(y)
//     x free, z<=5, 0<=y<=10

using namespace gollnlp;
using namespace std;

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
    int nvars = x->n;
    assert(x == vars_primal.get_block("x"));
    for(int it=0; it<nvars; it++) 
      obj_val += x->x[it] * x->x[it] * 0.5;
    return true;
  }
  virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
  {
    assert(x == vars_primal.get_block("x"));
    double* g = grad + x->index;
    for(int it=0; it<x->n; it++) 
      g[it] += x->x[it];
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
      M[H_nz_idxs[it]] += 1.;
    }
    return true;
  }

  virtual int get_HessLagr_nnz() { return x->n; }
  // (i,j) entries in the HessLagr to which this term contributes to
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
  { 
    assert(H_nz_idxs==NULL);
    int nvars = x->n, i;
    H_nz_idxs = new int[nvars];
#ifdef DEBUG
    for(i=0; i<nvars; i++) H_nz_idxs[i]=-1;
#endif

    for(int it=0; it < nvars; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
    }
    return true; 
  }

private:
  OptVariablesBlock* x;
  int *H_nz_idxs;
};

// for computing 0.5*||a-b||^2
class Ex1TwoVarsQuadrObjTerm : public OptObjectiveTerm {
public:
  Ex1TwoVarsQuadrObjTerm(const std::string& id, OptVariablesBlock* a_, OptVariablesBlock* b_)
    : OptObjectiveTerm(id), a(a_), b(b_) 
  {};
  virtual bool eval_f(const OptVariables& x, bool new_x, double& obj_val)
  {
    assert(false);
    return true;
  }
  virtual bool eval_grad(const OptVariables& x, bool new_x, double* grad)
  {
    assert(false);
    return true;
  }
  virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
  {
    assert(false);
    return true;
  }

private:
  OptVariablesBlock *a, *b;
};

//for sum(x^2) <=10
class Ex1SumOfSquaresConstraints : public OptConstraintsBlock {
public:
  Ex1SumOfSquaresConstraints(const std::string& id_, OptVariablesBlock* x_) 
    : OptConstraintsBlock(id_), x(x_), H_nz_idxs(NULL), J_nz_start(-1)
  {};
  virtual ~Ex1SumOfSquaresConstraints() 
  {
    if(H_nz_idxs)
      delete[] H_nz_idxs;
  };

  // all these functions 
  //  - should add their contribution to the output
  //  - return false if an error occurs in the evaluation
  virtual bool eval_body (const OptVariables& x, bool new_x, double* body)
  {
    assert(false);
    return true;
  };
  virtual bool eval_Jac(const OptVariables& x, bool new_x, 
			const int& nnz, int* i, int* j, double* M)
  {
    assert(false);
    return true;
  };
  virtual bool eval_HessLagr(const OptVariables& x_in, bool new_x, 
			     const OptVariables& lambda_vars, bool new_lambda,
			     const int& nnz, int* i, int* j, double* M)
  {
    const OptVariablesBlock* lambda = lambda_vars.get_block(string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==1);

    for(int it=0; it < x->n; it++)
      M[H_nz_idxs[it]] = 2. * lambda->xref[0];
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
    assert(NULL==H_nz_idxs);
    int n=x->n, i;
    if(n<=0) return true;
    
    H_nz_idxs = new int[n];
    for(int it=0; it<n; it++) {
      i = x->index+it;
      vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
    }
    return true;
  }

  virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
  {
    assert(-1 == J_nz_start);
    int n=x->n, i;
    if(n<=0) return true;

    vij.push_back(OptSparseEntry(this->index, x->index, &J_nz_start));

    for(int it=1; it<n; it++) {
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


//for sin(x) + z <= cos(y) or sin(x) + z + s = cos(y)
class Ex1Constraint2 : public OptConstraintsBlock {
public: 
  Ex1Constraint2(const std::string& id_, 
		 OptVariablesBlock* x_, OptVariablesBlock *z_, OptVariablesBlock* y_,
		 bool useSlacks=false) 
    : OptConstraintsBlock(id), x(x_), y(y_), z(z_)
  {};
  // one can also receives the entire set of variables and initialize variables
  // block used by this constraint block by looking up x, y, z in 'vars'
  Ex1Constraint2(const std::string& id, OptVariables* vars, bool useSlacks=false) 
    : OptConstraintsBlock(id), use_slacks(useSlacks) 
  {
    x = vars->get_block("x"); assert(x);
    y = vars->get_block("y"); assert(y);
    z = vars->get_block("z"); assert(z);
  };

   
  virtual ~Ex1Constraint2(){};

  virtual OptVariablesBlock* create_varsblock() { 
    if(!use_slacks) return NULL; 
    else return new OptVariablesBlock(x->n, "s", 0, 1e+20);
  };

  virtual OptObjectiveTerm* create_objterm() { 
    if(!use_slacks) return NULL; 
    assert(s);
    return new Ex1SingleVarQuadrObjTerm("quadr_pen_s", s);
  };

private:
  //s is null when use_slacks==false
  const OptVariablesBlock *x, *y, *z;
  OptVariablesBlock *s;
  bool use_slacks;
};


#endif

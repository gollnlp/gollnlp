#ifndef GOLLNLP_OPTPROB
#define GOLLNLP_OPTPROB

#include <string>
#include <cassert>
#include <vector>
namespace gollnlp {

class OptVariables {
public:
  OptVariables(int num, double lb=-1e+20, double ub=+1e+20);
  OptVariables(int num, double* lb=NULL, double* ub=NULL);
  virtual ~OptVariables();
  
  virtual void attach_to(double* x) = 0;
public:
  double *lb, *ub;
  size_t size;
private:
  OptVariables() : size(-1), lb(NULL), ub(NULL) {};
};
  
class OptDerivativeEval {
public:
  virtual bool eval_body (const OptVariables& x, double* body) = 0;
  virtual bool eval_deriv(const OptVariables& x, double* grad) = 0;
  virtual bool eval_deriv(const OptVariables& x, const size_t& nnz, 
			  int* i, int* j, double* M) = 0;
  virtual bool eval_Hess(const OptVariables& x, const size_t& nnz, 
			 int* i, int* j, double* M) = 0;
};

class OptConstraintsBlock : public OptDerivativeEval {
public:
  OptConstraintsBlock();
  virtual ~OptConstraintsBlock();
  virtual bool eval_body (const OptVariables& x, double* body);
  virtual bool eval_deriv(const OptVariables& x, double* grad)
  {
    assert(false && "constraints cannot provide dense derivatives");
    return false;
  }
  virtual bool eval_deriv(const OptVariables& x, const size_t& nnz, int* i, int* j, double* M);
  virtual bool eval_Hess(const OptVariables& x, const size_t& nnz,  int* i, int* j, double* M);
public: 
  size_t offset, size;
  int id;
};

class OptObjectiveTerm : public OptDerivativeEval {
public:
  OptObjectiveTerm() : sense(1) {};
  // +1 min, -1 max
  int sense; 
  int id;
  virtual ~OptObjectiveTerm();
  virtual bool eval_body (const OptVariables& x, double* body);
  virtual bool eval_deriv(const OptVariables& x, double* grad);
  virtual bool eval_deriv(const OptVariables& x, const size_t& nnz, int* i, int* j, double* M)
  {
    assert(false && "objective cannot provide Jacobians");
    return false;
  }
  virtual bool eval_Hess(const OptVariables& x, const size_t& nnz, int* i, int* j, double* M);
};

class OptProblem {
public:
  OptProblem(OptVariables* vars);
  virtual ~OptProblem();

protected:
  OptVariables*  m_vars;
  std::vector<OptConstraintsBlock*> m_conblocks;
  std::vector<OptObjectiveTerm*> m_objterms;

public:
  virtual bool eval_obj     (double* x, double& obj);
  virtual bool eval_cons    (double* x, double* cons);
  virtual bool eval_gradobj (double* x, double* grad);
  virtual bool eval_Jaccons (double* x, const size_t& nnz, int* i, int* j, double* M);
  //! mulipliers
  virtual bool eval_HessLagr(double* x, const size_t& nnz, int* i, int* j, double* M);

  inline void add_conblock(OptConstraintsBlock* con) { 
    if(con) {
      m_conblocks.push_back(con); 
      m_conblocks.back()->id = m_conblocks.size();
    } else assert(con);
  }
  inline void add_objterm(OptObjectiveTerm* obj) { 
    if(obj) {
      m_objterms.push_back(obj);
      m_objterms.back()->id = m_objterms.size();
    } else assert(obj);
  }
  
private:
  OptProblem() : m_vars(NULL) {};
  };
  
} //end namespace

#endif

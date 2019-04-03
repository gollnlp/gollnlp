#ifndef GOLLNLP_OPTPROB
#define GOLLNLP_OPTPROB

#include <string>
#include <cassert>
#include <vector>
#include <map>
namespace gollnlp {

class OptVariables {
public:
  OptVariables();
  //create on OptVarsBlock with no bounds with id "main"
  //OptVariables(const int& n);

  virtual ~OptVariables();

  struct OptVarsBlock {
    OptVarsBlock(const int& n_, const std::string& id_);
    OptVarsBlock(const int& n_, const std::string& id_, double* lb, double* ub=NULL);
    virtual ~OptVarsBlock();
    //number of vars in the block
    int n; 
    // index at which the block starts within OptVariables
    int index;
    // identifier, unique within OptVariables; is maintained by OptVariables
    std::string id; 
    //pointer/reference to the first elem in "x" that corresponds to this block
    double* x;
    //lower and upper vector bounds -> these are allocated and deallocated by this class
    double *lb, *ub;
  };
  // appends b to list of blocks; updates this->n and b->index
  bool append_varsblock(OptVarsBlock* b);

  virtual void attach_to(double* x);
public:
  int n;
protected:
  // "list" of pointers to blocks
  std::vector<OptVarsBlock*> m_vblocks;
  // "dict" with the pointers for quick lookup
  std::map<std::string, OptVarsBlock*> m_mblocks;
private:

};
  
class OptDerivativeEval {
public:
  virtual bool eval_body (const OptVariables& x, double* body) = 0;
  virtual bool eval_deriv(const OptVariables& x, double* grad) = 0;
  virtual bool eval_deriv(const OptVariables& x, const int& nnz, 
			  int* i, int* j, double* M) = 0;
  virtual bool eval_Hess(const OptVariables& x, const int& nnz, 
			 int* i, int* j, double* M) = 0;
};

// Holds objective terms to be minimized
class OptObjectiveTerm : public OptDerivativeEval {
public:
  OptObjectiveTerm() : id(0) {};

  int id;
  virtual ~OptObjectiveTerm();
  virtual bool eval_body (const OptVariables& x, double* body);
  virtual bool eval_deriv(const OptVariables& x, double* grad);
  virtual bool eval_deriv(const OptVariables& x, const int& nnz, int* i, int* j, double* M)
  {
    assert(false && "objective cannot provide Jacobians");
    return false;
  }
  virtual bool eval_Hess(const OptVariables& x, const int& nnz, int* i, int* j, double* M);
};

class OptConstraintsBlock : public OptDerivativeEval {
public:
  OptConstraintsBlock();
  virtual ~OptConstraintsBlock();

  // some constraints needs to create additional variables (e.g., slacks)
  // this method is called by OptProblem to get the constraints' varblock whenever
  // the constraints are added to the OptProblem (via append_constraints method)
  // NULL should be returned when the constraints need not create a vars block
  virtual OptVariables::OptVarsBlock* create_varsblock() { return NULL; }

  //same as above. OptProblem calls this when its append_objterm is called
  virtual OptObjectiveTerm* create_objterm() { return NULL; }

  virtual bool eval_body (const OptVariables& x, double* body);
  virtual bool eval_deriv(const OptVariables& x, double* grad)
  {
    assert(false && "constraints cannot provide dense derivatives");
    return false;
  }
  virtual bool eval_deriv(const OptVariables& x, const int& nnz, int* i, int* j, double* M);
  virtual bool eval_Hess(const OptVariables& x, const int& nnz,  int* i, int* j, double* M);
public: 
  // number of constraints
  int n;
  //index where the block starts in the constraints
  int index;
  // identifier - the block number, starting at 1
  int id;
};

class OptProblem {
public:
  OptProblem(OptVariables* vars);
  virtual ~OptProblem();

  inline int get_num_constraints() const 
  {
    return m_conblocks.size()>0 ? m_conblocks.back()->index + m_conblocks.back()->n : 0;
  }

  inline int get_num_variables() const { return m_vars->n; }

protected:
  OptVariables*  m_vars;
  std::vector<OptConstraintsBlock*> m_conblocks;
  std::vector<OptObjectiveTerm*> m_objterms;

public:
  virtual bool eval_obj     (double* x, double& obj);
  virtual bool eval_cons    (double* x, double* cons);
  virtual bool eval_gradobj (double* x, double* grad);
  virtual bool eval_Jaccons (double* x, const int& nnz, int* i, int* j, double* M);
  //! multipliers
  virtual bool eval_HessLagr(double* x, const int& nnz, int* i, int* j, double* M);

  inline void append_constraints(OptConstraintsBlock* con) { 
    if(con) {
      con->index = this->get_num_constraints();
      m_conblocks.push_back(con); 
      m_conblocks.back()->id = m_conblocks.size();
      
      m_vars->append_varsblock(con->create_varsblock());
      m_objterms.push_back(con->create_objterm());
    } else assert(con);
  }
  inline void append_objterm(OptObjectiveTerm* obj) { 
    if(obj) {
      m_objterms.push_back(obj);
      m_objterms.back()->id = m_objterms.size();
    } 
  }
  
private:
  OptProblem() : m_vars(NULL) {};
  };
  
} //end namespace

#endif

#ifndef GOLLNLP_OPTPROB
#define GOLLNLP_OPTPROB

#include "NlpSolver.hpp"

#include <string>
#include <cassert>
#include <vector>
#include <map>
namespace gollnlp {



class OptVariablesBlock {
public:
  OptVariablesBlock(const int& n_, const std::string& id_);
  OptVariablesBlock(const int& n_, const std::string& id_, double* lb, double* ub=NULL);
  //all lb and ub are set to lb_ and ub_
  OptVariablesBlock(const int& n_, const std::string& id_, double lb_, double ub_);
  virtual ~OptVariablesBlock();
  //number of vars in the block
  int n; 
  // index at which the block starts within OptVariables
  int index;
  // identifier, unique within OptVariables; is maintained by OptVariables
  std::string id; 
  //array that holds the solution; maintained by this class
  double* x;
  //lower and upper vector bounds; maintained by this class
  double *lb, *ub;
  //pointer/reference to the first elem in NLP solver's "x" that corresponds to this block
  //handled by attach_to
  const double* xref;
};
  
 

// a collection of blocks of variables
// This class needs NOT be specialized/derived.
class OptVariables {
public:
  OptVariables();
  ~OptVariables();

  OptVariablesBlock* get_block(const std::string& id);
  //total number of vars
  int n();

  friend class OptProblem;
private:
  // "list" of pointers to blocks
  std::vector<OptVariablesBlock*> m_vblocks;
  // "dict" with the pointers for quick lookups by name
  std::map<std::string, OptVariablesBlock*> m_mblocks;

  // appends b to list of blocks; updates this->n and b->index
  bool append_varsblock(OptVariablesBlock* b);
  virtual void attach_to(const double* xfromsolver);
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

// holds one objective terms to be minimized
class OptObjectiveTerm : public OptDerivativeEval {
public:
  OptObjectiveTerm(const std::string& id_) : id(id_) {};
  virtual ~OptObjectiveTerm();
  int index;
  std::string id;
  
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

  // Some constraints create additional variables (e.g., slacks).
  // This method is called by OptProblem (in 'append_constraints') to get and add
  // the additional variables block that OptConstraintsBlock may need to add.
  // NULL should be returned when the OptConstraintsBlock need not create a vars block
  virtual OptVariablesBlock* create_varsblock() { return NULL; }

  //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
  //term (e.g., penalization) that OptConstraintsBlock may need
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
  // identifier - 
  std::string id;

  double *lb, *ub;
};

// A collection of (summable) OptObjTerms
// This class needs NOT be specialized/derived.
class OptObjective {
  OptObjective() {};
  ~OptObjective();

  OptObjectiveTerm* get_objterm(const std::string& id);

  friend class OptProblem;
private:
  // appends a new obj term and sets his 'index'
  bool append_objterm(OptObjectiveTerm* term);
  // "list" of pointers to terms
  std::vector<OptObjectiveTerm*> vterms;
  // "dict" with the pointers for quick lookups by name
  std::map<std::string, OptObjectiveTerm*> mblocks;
};


// A collection of constraints blocks
// This class should NOT be specialized/derived.
class OptConstraints 
{
  OptConstraints();
  ~OptConstraints();

  OptConstraintsBlock* get_block(const std::string& id);
  inline int m() {
    return  vblocks.size()>0 ? vblocks.back()->index + vblocks.back()->n : 0;
  }

  friend class OptProblem;
private:
  // appends a new constraints block and sets his 'index'
  bool append_consblock(OptConstraintsBlock* b);
  // "list" of pointers to blocks
  std::vector<OptConstraintsBlock*> vblocks;
  // "dict" with the pointers for quick lookups by id
  std::map<std::string, OptConstraintsBlock*> mblocks;
};




class OptProblem {
public:
  OptProblem(OptVariables* vars);
  virtual ~OptProblem();

  inline int get_num_constraints() const { return cons->m(); }
  inline int get_num_variables() const { return vars_primal->n(); }

protected:
  OptVariables*    vars_primal;
  OptConstraints*  cons;
  OptObjective*    obj;

  OptVariables*    vars_duals_bounds;
  OptVariables*    vars_duals_cons;

public:

  inline void append_constraints(OptConstraintsBlock* con) { 
    if(con) {
      cons->append_consblock(con);
      
      vars_primal->append_varsblock(con->create_varsblock());
      obj->append_objterm(con->create_objterm());
    } else assert(con);
  }
  inline void append_objterm(OptObjectiveTerm* objterm) { 
    if(obj) {
      obj->append_objterm(objterm);
    } 
  }

  virtual OptVariables* new_duals_vec_cons();
  virtual OptVariables* new_duals_vec_bounds();

  //
  // optimization and NLP solver related stuff
  //
  virtual bool use_nlp_solver(const std::string& nlpsolver);
  //these setters return false if the option is not recognized by the NLP solver
  virtual bool set_solver_option(const std::string& name, int value);
  virtual bool set_solver_option(const std::string& name, double value);
  virtual bool set_solver_option(const std::string& name, const std::string& value);

  virtual void optimize(const std::string& nlpsolver);

  //getters -> copy to x; x should be allocated
  void fill_primal_vars(double* x);
  void fill_vars_lower_bounds(double* lb);
  void fill_vars_upper_bounds(double* ub);
  void fill_cons_lower_bounds(double* lb);
  void fill_cons_upper_bounds(double* ub);

  //DUALS -> for constraints
  void fill_dual_vars_con(double*x);
  //for bounds
  void fill_dual_vars_bounds(double*x);

  // Callbacks
  // This method is called by NlpSolver instance after each iteration (if supported by the solver)
  // Derive a class from OptProblem to hook your code
  virtual bool iterate_callback() {return true;}
public:
  //
  // internal NLP functions evaluators fed to the NLP solver
  // these are not to be used by the user and not to be overriden
  //
  bool eval_obj     (const double* x, bool new_x, double& obj);
  bool eval_cons    (const double* x, bool new_x, double* cons);
  bool eval_gradobj (const double* x, bool new_x, double* grad);
  bool eval_Jaccons (const double* x, bool new_x, const int& nnz, int* i, int* j, double* M);
  //! multipliers
  virtual bool eval_HessLagr(const double* x, bool new_x, const int& nnz, int* i, int* j, double* M);

  int get_nnzJaccons();
  int get_nnzHessLagr();


private:
  OptProblem() {};
};
  
} //end namespace

#endif

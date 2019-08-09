#ifndef GOLLNLP_OPTPROB
#define GOLLNLP_OPTPROB

#include <string>
#include <cassert>
#include <vector>
#include <map>

#include "mpi.h"
namespace gollnlp {

class NlpSolver;

class OptVariablesBlock {
public:
  OptVariablesBlock(const int& n_, const std::string& id_);
  OptVariablesBlock(const int& n_, const std::string& id_, /*const*/ double* lb, /*const*/double* ub=NULL);
  //all lb and ub are set to lb_ and ub_
  OptVariablesBlock(const int& n_, const std::string& id_, double lb_, double ub_);
  virtual ~OptVariablesBlock();

  void set_start_to(const double& scalar);
  void set_start_to(const double* values);
  void set_start_to(const OptVariablesBlock& block);

  void inline set_xref_to_x() { xref=x; }
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
  bool providesStartingPoint;
  
  void print() const;
};
  
////////////////////////////////////////////////////////////
// OptVariables
//
// a collection of blocks of variables
// This class needs NOT be specialized/derived.
////////////////////////////////////////////////////////////
class OptVariables {
public:
  OptVariables();
  ~OptVariables();

  const OptVariablesBlock* get_block(const std::string& id) const
  {
    auto it = mblocks.find(id);
    if(it != mblocks.end())
      return it->second;
    return NULL;
  }

  //non-const version of the above accepting a name of the variables (used
  //for debugging purposes)
  OptVariablesBlock* vars_block(const std::string& id, const std::string& var_name="") 
  {
    auto it = mblocks.find(id);
    if(it != mblocks.end())
      return it->second;
#ifdef DEBUG
    printf("Warning: block id '%s' was not found in optimiz variables '%s'\n", 
	   id.c_str(), var_name==""?"name_not_passed":var_name.c_str());
#endif
    return NULL;
  }

  bool provides_start();

  inline void set_xref_to_x() 
  {
    for(auto& b : vblocks) 
      b->set_xref_to_x();
  }

  //total number of vars
  inline int n()  const
  {
    return  vblocks.size()>0 ? vblocks.back()->index + vblocks.back()->n : 0;
  }

  bool set_start_to(const OptVariables& src);

  void copy_to(double* a);
  void copy_to(std::vector<double>& v);
  void copy_from(const std::vector<double>& v);


  void print_summary(const std::string var_name="") const;
  void print(const std::string var_name="") const;

public:
  // "list" of pointers to blocks
  std::vector<OptVariablesBlock*> vblocks;
  // "dict" with the pointers for quick lookups by name
  std::map<std::string, OptVariablesBlock*> mblocks;
friend class OptProblem;

public: //MPI_Helpers
  //broadcasts 'this'; of non-null, 'buffer' will be used to pack/unpack variables blocks
  int MPI_Bcast_x(int root, MPI_Comm comm, int my_rank, double* buffer=NULL);
private:
  // appends b to list of blocks; updates this->n and b->index
  bool append_varsblock(OptVariablesBlock* b);
  bool append_varsblocks(std::vector<OptVariablesBlock*> vVarBlocks);

  virtual void attach_to(const double* xfromsolver);
};

struct OptSparseEntry
{
  OptSparseEntry(int i_, int j_, int* p) : i(i_), j(j_), idx(p) { };
  int i,j;
  //Memory address where the index of (i,j) in the FINAL Hessian/Jacobian should be put by OptProblem
  //Only OptProblem and its subsidiaries write in this address.
  //
  //idx==NULL means that the implementer (Objective or Constraints evaluator) does not need the nz 
  //of (i,j). This is the case when the implementer can compute it cheaply on the fly.
  int* idx; 

  inline bool operator<(const OptSparseEntry& b) const
  {
    if(i < b.i) return true;
    if(i > b.i) return false;
    return j<b.j;
  }
private:
  OptSparseEntry() {};
};

class OptObjectiveEvaluator {
public:
  // all these functions 
  //  - should add their contribution to the result (obj_val, grad, or M)
  //  - return false if an error occurs in the evaluation
  virtual bool eval_f(const OptVariables& x, bool new_x, double& obj_val) = 0;
  virtual bool eval_grad(const OptVariables& x, bool new_x, double* grad) = 0;
  virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const double& obj_factor,
			     const int& nnz, int* i, int* j, double* M)
  { return true; }

  // methods that need to be implemented to specify the sparsity pattern of the 
  // implementer's contribution to the sparse derivatives
  virtual int get_HessLagr_nnz() { return 0; }

  // (i,j) entries in the HessLagr to which the implementer's contributes to
  // this is only called once
  // push_back in vij 
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return true; }
};

class OptConstraintsEvaluator {
public:
  // all these functions 
  //  - should add their contribution to the output
  //  - return false if an error occurs in the evaluation
  virtual bool eval_body (const OptVariables& x, bool new_x, double* body) = 0;
  virtual bool eval_Jac(const OptVariables& x, bool new_x, 
			const int& nnz, int* i, int* j, double* M) = 0;
  virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const OptVariables& lambda, bool new_lambda,
			     const int& nnz, int* i, int* j, double* M) 
  { return true; }

  // methods that need to be implemented to specify the sparsity pattern of the 
  // implementer's contribution to the sparse derivatives
  virtual int get_HessLagr_nnz() { return 0; }
  virtual int get_Jacob_nnz() = 0; 

  // (i,j) entries in the HessLagr to which the implementer's contributes to
  // this is only called once
  // push_back in vij 
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return true; }
  virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij) = 0; 
};

// holds one objective terms to be minimized
class OptObjectiveTerm : public OptObjectiveEvaluator {
public:
  OptObjectiveTerm(const std::string& id_) : id(id_) {};
  virtual ~OptObjectiveTerm() {};
  int index;
  std::string id;
};

///////////////////////////////////////////////////////////////
//OptConstraintsBlock
///////////////////////////////////////////////////////////////
class OptConstraintsBlock : public OptConstraintsEvaluator {
public:
  OptConstraintsBlock(const std::string& id_, int num) 
    : n(num), index(-1), id(id_), lb(NULL), ub(NULL)
  {
    if(n>0) {
      lb = new double[n];
      ub = new double[n];
    }
  }
  virtual ~OptConstraintsBlock() 
  {
    delete[] lb;
    delete[] ub;
  };

  // Some constraints create additional variables (e.g., slacks).
  // This method is called by OptProblem (in 'append_constraints') to get and add
  // the additional variables block that OptConstraintsBlock may need to add.
  // NULL should be returned when the OptConstraintsBlock need not create a vars block
  virtual OptVariablesBlock* create_varsblock() { return NULL; }
  // Use the following method when the constraints blocks needs to create multiple 
  // variables (e.g., introduce multiple and/or different types of slacks).
  // Both create_XXX_varsblock(s) methods can be implemented to return 
  // non-null/empty variable blocks.
  virtual std::vector<OptVariablesBlock*> create_multiple_varsblocks() { return std::vector<OptVariablesBlock*>(); }

  //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
  //term (e.g., penalization) that OptConstraintsBlock may need
  virtual OptObjectiveTerm* create_objterm() { return NULL; }

public: 
  // number of constraints
  int n;
  //index where the block starts in the constraints
  int index;
  // identifier - 
  std::string id;

  double *lb, *ub;
};

//////////////////////////////////////////////////////
// OptObjective
//
// A collection of (summable) OptObjTerms
// This class needs NOT be specialized/derived.
//////////////////////////////////////////////////////
class OptObjective {
  OptObjective() {};
  ~OptObjective();

  //OptObjectiveTerm* get_objterm(const std::string& id);
  inline OptObjectiveTerm* objterm(const std::string& id) 
  {
    auto it = mterms.find(id);
    if(it != mterms.end())
      return it->second;
    return NULL;
  }
  friend class OptProblem;
private:
  // appends a new obj term and sets his 'index'
  bool append_objterm(OptObjectiveTerm* term);
  // "list" of pointers to terms
  std::vector<OptObjectiveTerm*> vterms;
  // "dict" with the pointers for quick lookups by name
  std::map<std::string, OptObjectiveTerm*> mterms;
};

//////////////////////////////////////////////////////////
// OptConstraints
//
// A collection of constraints blocks
// This class needs NOT be specialized/derived.
/////////////////////////////////////////////////////////
class OptConstraints 
{
  OptConstraints();
  ~OptConstraints();

  OptConstraintsBlock* get_block(const std::string& id);
  inline int m() 
  {
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


//////////////////////////////////////////////////////////
// OptProblem
/////////////////////////////////////////////////////////
class OptProblem {
public:
  OptProblem();
  virtual ~OptProblem();

  inline int get_num_constraints() const { return cons->m(); }
  inline int get_num_variables() const { return vars_primal->n(); }

public:
  inline OptVariables* primal_variables() { return vars_primal; }
  inline OptVariables* duals_bounds_lower() { return vars_duals_bounds_L; }
  inline OptVariables* duals_bounds_upper() { return vars_duals_bounds_U; }
  inline OptVariables* duals_constraints() { return vars_duals_cons; }
  
  inline void append_variables(OptVariablesBlock* vars)
  {
    vars_primal->append_varsblock(vars);
  }
  inline void append_constraints(OptConstraintsBlock* con) 
  { 
    if(con) {
      cons->append_consblock(con);
      
      vars_primal->append_varsblock (con->create_varsblock());
      vars_primal->append_varsblocks(con->create_multiple_varsblocks());

      obj->append_objterm(con->create_objterm());
    } else assert(con);
  }
  inline void append_objterm(OptObjectiveTerm* objterm) 
  { 
    obj->append_objterm(objterm);
  }

  inline OptConstraintsBlock* constraints_block(const std::string& id)
  {
    return cons->get_block(id);
  }
  inline OptVariablesBlock* vars_block(const std::string& id)
  {
//     auto it = vars_primal->mblocks.find(id);
//     if(it != vars_primal->mblocks.end())
//       return it->second;
// #ifdef DEBUG
//     printf("inquiry for vars_block '%s' failed.\n", id.c_str());
//     assert(false);
// #endif    
//     return NULL;
    return vars_primal->vars_block(id, "vars_primal");
  }
  inline OptVariablesBlock* vars_block_duals_bounds_lower(const std::string& id)
  {
    return vars_duals_bounds_L->vars_block(id, "vars_duals_bounds_L");
  }
  inline OptVariablesBlock* vars_block_duals_bounds_upper(const std::string& id)
  {
    return vars_duals_bounds_U->vars_block(id, "vars_duals_bounds_U");
  }
  inline OptVariablesBlock* vars_block_duals_cons(const std::string& id)
  {
    return vars_duals_cons->vars_block(id, "vars_cons");
  }
  inline OptObjectiveTerm* objterm(const std::string& id)
  {
    return obj->objterm(id);
  }

  //
  // optimization and NLP solver related stuff
  //
  virtual void use_nlp_solver(const std::string& name);
  //these setters return false if the option is not recognized by the NLP solver
  virtual bool set_solver_option(const std::string& name, int value);
  virtual bool set_solver_option(const std::string& name, double value);
  virtual bool set_solver_option(const std::string& name, const std::string& value);


  //Starting points - 'vars_primals' and 'vars_duals_xxx' are used when optimize and reoptimize
  //More specifically, OptProblem will check if any of the blocks 'b' in the primal and dual
  //variables has b.providesStartingPoint==true and will use exactly b.x; otherwise will use zeros

  //
  // NLP optimization and reoptimization
  //
  enum RestartType{primalRestart, primalDualRestart, advancedPrimalDualRestart};
  
  // method should be called before 'optimize' whenever the size or the sparsity 
  // pattern of the derivatives changes
  virtual void problem_changed();

  virtual bool optimize(const std::string& nlpsolver);
  virtual bool reoptimize(RestartType t=primalRestart);

  inline double objective_value() const { return obj_value; }
  inline double objective_value_barrier() const { return obj_barrier; }
  inline int number_of_iterations() const { return num_iter; }
  //
  // Callbacks
  //
  // This method is called by NlpSolver instance after each iteration (if supported by the solver)
  // Derive a class from OptProblem to hook your code
  virtual bool iterate_callback(int iter, const double& obj_value,
				const double* primals,
				const double& inf_pr, const double& inf_du, 
				const double& mu, 
				const double& alpha_du, const double& alpha_pr,
				int ls_trials) 
  { return true; }

  virtual bool iterate_finalize()
  {
    vars_primal->set_xref_to_x();
    vars_duals_bounds_L->set_xref_to_x();
    vars_duals_bounds_U->set_xref_to_x();
    vars_duals_cons->set_xref_to_x();
    return true;
  }
public:
  //
  // internal NLP functions evaluators fed to the NLP solver
  // these are not to be used by the user and not to be overriden
  //
  bool eval_obj     (const double* x, bool new_x, double& obj);
  bool eval_cons    (const double* x, bool new_x, double* cons);
  bool eval_gradobj (const double* x, bool new_x, double* grad);
  bool eval_Jaccons (const double* x, bool new_x, 
		     const int& nnz, int* i, int* j, double* M);
  bool eval_HessLagr(const double* x, bool new_x, 
		     const double& obj_factor, 
		     const double* lambda, bool new_lambda,
		     const int& nnz, int* i, int* j, double* M);
  //getters -> copy to x; x is expected to be allocated
  void fill_primal_vars(double* x);
  void fill_vars_lower_bounds(double* lb);
  void fill_vars_upper_bounds(double* ub);
  void fill_cons_lower_bounds(double* lb);
  void fill_cons_upper_bounds(double* ub);

  //DUALS -> for constraints
  void fill_dual_vars_con(double*x);
  //for bounds
  void fill_dual_vars_bounds(double*zL, double* zU);

  // other internal NLP-related methods
  bool fill_primal_start(double* x);
  bool fill_dual_bounds_start(double* z_L, double* z_U);
  bool fill_dual_cons_start(double* lambda);

  //setters
  void set_obj_value(const double& f);
  void set_obj_value_barrier(const double& flogbar);
  void set_num_iters(int n_iter);
  void set_primal_vars(const double* x);
  void set_duals_vars_bounds(const double* zL, const double* zU);
  void set_duals_vars_cons(const double* lambda);

  void set_have_start();

  //other internal methods
  virtual OptVariables* new_duals_cons();
  virtual OptVariables* new_duals_lower_bounds();
  virtual OptVariables* new_duals_upper_bounds();

  //OptVariables* new_copy_of_primal_vars();
  
  int get_nnzJaccons();
  int get_nnzHessLagr();


public:
  //utilities
  void print_summary() const;
  void print_objterms_evals();
protected:
  OptVariables*    vars_primal;
  OptConstraints*  cons;
  OptObjective*    obj;
  double obj_value, obj_barrier;
  int num_iter;

  OptVariables *vars_duals_bounds_L, *vars_duals_bounds_U;
  OptVariables *vars_duals_cons;

  int nnz_Jac, nnz_Hess;

  NlpSolver* nlp_solver;

  //these two vectors have limited storage lifetime
  std::vector<OptSparseEntry> ij_Jac, ij_Hess;

  bool new_x_fgradf;
};
  
} //end namespace

#endif

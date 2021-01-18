#ifndef GOLLNLP_OPTPROB
#define GOLLNLP_OPTPROB

#include <string>
#include <cassert>
#include <vector>
#include <map>

#include "mpi.h"

//this should not be included here -> instead OptProblem should have its own error structures
//quick hack - will be revisited
#include "IpIpoptApplication.hpp"

namespace gollnlp {

//quick hack to avoid poluting gollnlp code with Ipopt::
typedef Ipopt::ApplicationReturnStatus OptimizationStatus;
typedef Ipopt::AlgorithmMode OptimizationMode;

class NlpSolver;

class OptProblem;
class OptVariables;
 
class OptVariablesBlock {
public:
  OptVariablesBlock(const int& n_, const std::string& id_,
		    OptVariables* owner=NULL);
  OptVariablesBlock(const int& n_, const std::string& id_,
		    const double* lb, const double* ub=NULL,
		    OptVariables* owner=NULL);
  //all lb and ub are set to lb_ and ub_
  OptVariablesBlock(const int& n_, const std::string& id_,
		    double lb_, double ub_,
		    OptVariables* owner=NULL);
  virtual ~OptVariablesBlock();

  void set_start_to(const double& scalar);
  void set_start_to(const double* values);
  void set_start_to(const OptVariablesBlock& block);

  void inline set_xref_to_x() { xref=x; }
  //number of vars in the block
  int n; 
  //index at which the block starts within OptVariables
  int index;
  //identifier, unique within OptVariables; is maintained by OptVariables
  std::string id; 
  //array that holds the solution; memory maintained by this class
  double* x;
  //lower and upper vector bounds; memory maintained by this class
  double *lb, *ub;
  //pointer/reference to the first elem in NLP solver's "x" that corresponds to this block
  //handled by attach_to
  const double* xref;
  //starting point provided?
  bool providesStartingPoint;
  
  /** 
   * Flag indicating whether the block is of dense or sparse variables. All variables in a block have 
   * the same type. Default value: true
   */
  bool sparseBlock;
  //index at which the sparse block start within the sparse variables within OptVariables container
  //if this is a dense block, the "sparse" index is negative, indicating the last sparse index of the
  //previously sparse block within OptVariables container
  int indexSparse;

  inline int compute_indexDense() const
  {
    assert(!sparseBlock);
    assert(indexSparse<=0);
    assert(index>=0);
    return index+indexSparse;
  }

  /** Grow the variables array/vector 
   * Reallocates 'x', 'lb', 'ub' therefore use carefully.
   * Flags 'sparseBlock' and 'providesStartingPoint' do not change.
   * Automatically reuses starting point values of the existing variables in the block
   *
   * NULL values passed as arguments signifies default values: 
   *   this->lb = -1e+20  this->ub= 1e+20  this->x=0.
   */
  void append_variables(const int& how_many, const double* lb, const double* ub, const double* x0=NULL);
  
  inline OptVariablesBlock* new_copy() 
  {
    auto b = new OptVariablesBlock(n, id, lb, ub, owner_vars_);
    b->set_start_to(*this);
    return b;
  }

  void print() const;
  
  friend class OptVariables;
private:
  OptVariables* owner_vars_;
};
  
////////////////////////////////////////////////////////////
// OptVariables
//
// a collection of blocks of variables
// This class needs NOT be specialized/derived.
////////////////////////////////////////////////////////////
class OptVariables {
public:
  OptVariables(OptProblem* prob=NULL);
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
    //#ifdef DEBUG
    //printf("Warning: block id '%s' was not found in optimiz variables '%s'\n", 
    //	   id.c_str(), var_name==""?"name_not_passed":var_name.c_str());
    //#endif
    return NULL;
  }

  void delete_block(const std::string& id);

  bool provides_start();

  inline void set_xref_to_x() 
  {
    for(auto& b : vblocks) 
      b->set_xref_to_x();
  }

  //total number of vars
  inline int n() const
  {
    return vblocks.size()>0 ? vblocks.back()->index + vblocks.back()->n : 0;
  }

  bool set_start_to(const OptVariables& src);
  void set_start_to(const double& scalar)
  {
    for(auto& b : vblocks)
      if(b) b->set_start_to(scalar);      
  }
  void copy_to(double* a);
  void copy_to(std::vector<double>& v);
  inline void copy_from(const std::vector<double>& v) { copy_from(v.data()); }
  void copy_from(const double* v);

  //
  //'this' is a dual (lb or ub) corresponding to 'primals'
  //
  //the method sets all the inactive duals (primals.lb=-1e+20) to 'ct'
  void set_inactive_duals_lb_to(double ct, const OptVariables& primals);
  //the method sets all the inactive duals (primals.ub=+1e+20) to 'ct'
  void set_inactive_duals_ub_to(double ct, const OptVariables& primals);

  OptVariables* new_copy() 
  {
    OptVariables* new_vars = new OptVariables();
    for(auto b : this->vblocks) {
      new_vars->append_varsblock(b->new_copy());
    }
    return new_vars;
  }

  void print_summary(const std::string var_name="") const;
  void print(const std::string var_name="") const;

public:
  // "list" of pointers to blocks
  std::vector<OptVariablesBlock*> vblocks;
  // "dict" with the pointers for quick lookups by name
  std::map<std::string, OptVariablesBlock*> mblocks;
  friend class OptProblem;
  friend class OptProblemMDS;

public: //MPI_Helpers
  //broadcasts 'this'; if non-null, 'buffer' will be used to pack/unpack variables blocks
  int MPI_Bcast_x(int root, MPI_Comm comm, int my_rank, double* buffer=NULL);
public:
  //grows the vars block specified by 'id'
  void append_vars_to_varsblock(const std::string& id,
  				int num_vars_to_add,
  				const double* lb,
  				const double* ub,
  				const double* start);
protected:
  // appends b to list of blocks; updates this->n and b->index
  bool append_varsblock(OptVariablesBlock* b);
  bool append_varsblocks(std::vector<OptVariablesBlock*> vVarBlocks);
  
  virtual void attach_to(const double* xfromsolver);
protected:
  OptProblem* owner_prob_;
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

// holds one objective term to be minimized
class OptObjectiveTerm {
public:
  OptObjectiveTerm(const std::string& id_) : id(id_) {};
  virtual ~OptObjectiveTerm() {};
  int index;
  std::string id;

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
  // this is only called once and is supposed to push_back in vij
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return true; }

  /*
   * Objective terms keep various pointers and links to the primal problem. These may
   * need to be invalidated (and later reconstructed) when the primal problem changes.
   */
  virtual void primal_problem_changed()
  {
  }

  /*
   * Objective terms keep various pointers and links to the dual problem. These may
   * need to be invalidated (and later reconstructed) when the dual problem changes.
   */
  virtual void dual_problem_changed()
  {
  }

};

///////////////////////////////////////////////////////////////
//OptConstraintsBlock
///////////////////////////////////////////////////////////////
class OptConstraints;
class OptConstraintsBlock {
public:
  OptConstraintsBlock(const std::string& id_, int num, OptConstraints* owner=NULL) 
    : n(num), index(-1), id(id_), lb(NULL), ub(NULL),
      owner_cons_(owner)
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
  virtual std::vector<OptVariablesBlock*> create_multiple_varsblocks()
  {
    return std::vector<OptVariablesBlock*>();
  }

  //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
  //term (e.g., penalization) that OptConstraintsBlock may need
  virtual OptObjectiveTerm* create_objterm() { return NULL; }

  /*
   * Constraint blocks keeps various pointers and links to the primal problem. These may
   * need to be invalidated (and later reconstructed) when the primal problem changes.
   */
  virtual void primal_problem_changed()=0;
  //{
  //}

  /*
   * Constraint blocks keeps various pointers and links to the dual problem. These may
   * need to be invalidated (and later reconstructed) when the dual problem changes.
   */
  virtual void dual_problem_changed()
  {
  }

  
public: 
  // number of constraints
  int n;
  //index where the block starts in the constraints
  int index;
  // identifier - 
  std::string id;

  double *lb, *ub;

public:
  // all these functions 
  //  - should add their contribution to the output
  //  - return false if an error occurs in the evaluation
  //Notes 
  // 1. eval_Jac and eval_Hess_Lagr should be called after get_Jacob_nnz and
  // get_HessLagr_nnz

  virtual bool eval_body (const OptVariables& x, bool new_x, double* body) = 0;
  virtual bool eval_Jac(const OptVariables& x, bool new_x, 
			const int& nnz, int* i, int* j, double* M) = 0;

  virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			     const OptVariables& lambda, bool new_lambda,
			     const int& nnz, int* i, int* j, double* M) 
  { return true; }


  // methods that need to be implemented to specify the sparsity pattern of the 
  // implementer's contribution to the sparse derivatives
  // Note 1: for MDS problems, these methods are only called for the sparse part of 
  // the derivative
  // Note 2: for MDS problems, 'get_HessLagr_nnz' is called for the sparse (1,1)
  // block. TODO: addtl methods may be needed for the other sparse block, i.e. (2,1)
  virtual int get_HessLagr_nnz() { return 0; }
  virtual int get_Jacob_nnz() = 0; 

  // (i,j) entries in the sparse HessLagr to which the implementer's contributes to.
  // These methods are only called called once
  // Internally, they 'push_back' in vij stored for (sparse parts of) Hess of 
  // the Lagr or for the Jacob
  // Note 1: For MDS problems, these methods should contain only entries in the 
  // sparse part(s) of the of the derivatives
  virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return true; }
  virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij) = 0;

  friend class OptConstraints;
protected:
  OptConstraints* owner_cons_;
};

//////////////////////////////////////////////////////
// OptObjective
//
// A collection of (summable) OptObjTerms
//////////////////////////////////////////////////////
class OptObjective {
public:
  OptObjective() {};
  virtual ~OptObjective();

  //OptObjectiveTerm* get_objterm(const std::string& id);
  inline OptObjectiveTerm* objterm(const std::string& id) 
  {
    auto it = mterms.find(id);
    if(it != mterms.end())
      return it->second;
    return NULL;
  }

  friend class OptProblem;
  friend class OptProblemMDS;
protected:
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
  OptConstraints(OptProblem* prob=NULL);
  ~OptConstraints();

  OptConstraintsBlock* get_block(const std::string& id);

  void delete_block(const std::string& id);

    inline int m() 
  {
    return  vblocks.size()>0 ? vblocks.back()->index + vblocks.back()->n : 0;
  }

  friend class OptProblem;
  friend class OptProblemMDS;
protected:
  // appends a new constraints block and sets his 'index'
  bool append_consblock(OptConstraintsBlock* b);
  // "list" of pointers to blocks
  std::vector<OptConstraintsBlock*> vblocks;
  // "dict" with the pointers for quick lookups by id
  std::map<std::string, OptConstraintsBlock*> mblocks;

public:
  OptProblem* opt_problem() { return owner_prob_; }
protected:
  OptProblem* owner_prob_;
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
  
  virtual int compute_nnzJaccons();
  
  virtual void compute_nnzJac_eq_ineq(const int& n, const int& m,
                                      const double* clow, const double* cupp,
                                      int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq);
  
  virtual int compute_nnzHessLagr();
public:
  inline OptVariables* primal_variables() { return vars_primal; }
  inline OptVariables* duals_bounds_lower() { return vars_duals_bounds_L; }
  inline OptVariables* duals_bounds_upper() { return vars_duals_bounds_U; }
  inline OptVariables* duals_constraints() { return vars_duals_cons; }

  //inline void append_vars_to_varsblock(const std::string& id_varsblock,
  //				       int num_vars_to_add,
  //				       const double* lb,
  //				       const double* ub,
  //				       const double* start)
  //{
  //  vars_primal->append_vars_to_varsblock(id_varsblock, num_vars_to_add, lb, ub, start);
  //}
  
  inline void append_varsblock(OptVariablesBlock* vars)
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
  inline void append_duals_constraint(const std::string& constraint_id)
  {
    OptConstraintsBlock* con_block = constraints_block(constraint_id); assert(con_block);
    if(con_block) {
      std::string strVarName = std::string("duals_") + con_block->id;
      vars_duals_cons->append_varsblock(new OptVariablesBlock(con_block->n, strVarName));
    }
  }

  inline OptConstraintsBlock* constraints_block(const std::string& id)
  {
    return cons->get_block(id);
  }
  inline OptVariablesBlock* vars_block(const std::string& id)
  {
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

  inline void delete_constraint_block(const std::string& id)
  {
    cons->delete_block(id);
  }

  inline void delete_duals_constraint(const std::string& constraint_id)
  {
    vars_duals_cons->delete_block("duals_" + constraint_id);
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
  
  // method should be called before 'optimize' or 'reoptimize' whenever 
  // i. the number of the optimization variables 
  // or 
  // ii. the sparsity pattern of the derivatives changes
  virtual void primal_problem_changed();

  //method to be called before 'reoptimize' whenever the number of constraints changes
  //no need to call this before 'optimize'
  virtual void dual_problem_changed();

  virtual bool optimize(const std::string& nlpsolver);
  virtual bool reoptimize(RestartType t=primalRestart);

  virtual OptimizationStatus optimization_status() const;

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
				const double& inf_pr, const double& inf_pr_orig_problem, 
				const double& inf_du, 
				const double& mu, 
				const double& alpha_du, const double& alpha_pr,
				int ls_trials, OptimizationMode mode,
				const double* duals_con=NULL,
				const double* duals_lb=NULL, const double* duals_ub=NULL) 
  { return true; }
  inline void enable_intermediate_duals() { need_intermediate_duals=true; }
  inline bool requests_intermediate_duals() const { return need_intermediate_duals; }
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
  bool eval_Jaccons (const double* x, bool new_x, 
		     const int& nxsparse, const int& nxdense,
		     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		     double** JacD);

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
public:
  //utilities
  void print_summary() const;
  void print_objterms_evals();

  //returns new nnz, sorts ij, and updates ij[k].idx with the idx of k-th entry
  //in the (virtual) sorted vector of (i,j)s 
  static int uniquely_indexise_spTripletIdxs(std::vector<OptSparseEntry>& ij);

  bool check_is_upper(const std::vector<OptSparseEntry>& ij);
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

  void reallocate_nlp_solver();

  // Temporary storage for the entries of the sparse parts of the derivatives
  // these two vectors have limited storage lifetime
  std::vector<OptSparseEntry> ij_Jac, ij_Hess;

  bool new_x_fgradf;

  bool need_intermediate_duals;
public:
  //quick hack - will be revisited
  static const OptimizationStatus Solve_Succeeded = Ipopt::Solve_Succeeded;
  static const OptimizationStatus Solved_To_Acceptable_Level=Ipopt::Solved_To_Acceptable_Level;
  static const OptimizationStatus Infeasible_Problem_Detected=Ipopt::Infeasible_Problem_Detected;
  static const OptimizationStatus Search_Direction_Becomes_Too_Small=Ipopt::Search_Direction_Becomes_Too_Small;
  static const OptimizationStatus Diverging_Iterates=Ipopt::Diverging_Iterates;
  static const OptimizationStatus User_Requested_Stop=Ipopt::User_Requested_Stop;
  static const OptimizationStatus Feasible_Point_Found=Ipopt::Feasible_Point_Found;
  static const OptimizationStatus Maximum_Iterations_Exceeded=Ipopt::Maximum_Iterations_Exceeded;
  static const OptimizationStatus Restoration_Failed=Ipopt::Restoration_Failed;
  static const OptimizationStatus Error_In_Step_Computation=Ipopt::Error_In_Step_Computation;
  static const OptimizationStatus Maximum_CpuTime_Exceeded=Ipopt::Maximum_CpuTime_Exceeded;
  static const OptimizationStatus Not_Enough_Degrees_Of_Freedom=Ipopt::Not_Enough_Degrees_Of_Freedom;
  static const OptimizationStatus Invalid_Problem_Definition=Ipopt::Invalid_Problem_Definition;
  static const OptimizationStatus Invalid_Option=Ipopt::Invalid_Option;
  static const OptimizationStatus Invalid_Number_Detected=Ipopt::Invalid_Number_Detected;
  static const OptimizationStatus Unrecoverable_Exception=Ipopt::Unrecoverable_Exception;
  static const OptimizationStatus NonIpopt_Exception_Thrown=Ipopt::NonIpopt_Exception_Thrown;
  static const OptimizationStatus Insufficient_Memory=Ipopt::Insufficient_Memory;
  static const OptimizationStatus Internal_Error=Ipopt::Internal_Error;
  
  static const OptimizationMode RegularMode = Ipopt::RegularMode;
  static const OptimizationMode RestorationPhaseMode = Ipopt::RestorationPhaseMode;
};
  
} //end namespace

#endif

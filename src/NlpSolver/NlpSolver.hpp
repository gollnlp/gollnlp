#ifndef GOLLNLP_NLPSOLVER
#define GOLLNLP_NLPSOLVER

#include "OptProblem.hpp"

#include <vector>

namespace gollnlp {

class NlpSolver {
public:
  NlpSolver(OptProblem* p_);
  virtual ~NlpSolver();

  //any NlpSolver specific init and finalize
  virtual bool initialize() {};
  virtual bool finalize() {};

  virtual bool set_starting_point(OptVariables* v) = 0;


  // solves the problem and return success (0) or error codes (tbd)
  virtual int optimize() = 0;
  virtual int reoptimize() { return optimize(); }

  /*  //geters
  double optimal_obj() = 0;

  //vector with optimal primal values
  //for OptVariables
  std::vector<double> optimal_primals()=0;
  //for a subblock
  std::vector<double> optimal_primals(OptVariables::OptVarsBlock* b)=0;
  //for a subblock with identifier id
  std::vector<double> optimal_primals(const std::string& blId)=0;

  //dual variables 
  //-> constraints
  std::vector<double> optimal_duals_cons()=0;
  //for constraint subblock
  std::vector<double> optimal_duals_cons(OptConstraintsBlock* b)=0;
  std::vector<double> optimal_duals_cons(const std::string& consId)=0;
  // -> bounds
  std::vector<double> optimal_duals_bounds()=0;
  std::vector<double> optimal_duals_bounds(OptVariables::OptVarsBlock* b)=0;
  //for a subblock with identifier id
  std::vector<double> optimal_duals_bounds(const std::string& blId)=0;
  */
protected:
  //reference
  OptProblem* prob;
} //end namespace

#endif

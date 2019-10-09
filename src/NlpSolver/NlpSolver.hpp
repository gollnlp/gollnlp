#ifndef GOLLNLP_NLPSOLVER
#define GOLLNLP_NLPSOLVER

#include "OptProblem.hpp"

namespace gollnlp {

class OptVariables;
class OptProblem;

class NlpSolver {
public:
  NlpSolver(OptProblem* p_) : prob(p_) {};
  virtual ~NlpSolver() {};

  //any NlpSolver specific init and finalize
  virtual bool initialize() {return true;}
  virtual bool finalize() {return true;}

  virtual bool set_start_type(OptProblem::RestartType t)=0;

  virtual bool set_option(const std::string& name, int value) = 0;
  virtual bool set_option(const std::string& name, double value) = 0;
  virtual bool set_option(const std::string& name, const std::string& value) = 0;

  // solves the problem and return success (0) or error codes (tbd)
  virtual int optimize() = 0;
  virtual int reoptimize() = 0;

  virtual OptimizationStatus return_code() const = 0;
protected:
  //reference
  OptProblem* prob;
};
} //end namespace

#endif

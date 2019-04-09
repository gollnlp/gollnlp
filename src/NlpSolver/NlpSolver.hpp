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
  virtual bool initialize() {};
  virtual bool finalize() {};

  virtual bool set_starting_point(OptVariables* v) = 0;


  // solves the problem and return success (0) or error codes (tbd)
  virtual int optimize() = 0;
  virtual int reoptimize() { return optimize(); }
protected:
  //reference
  OptProblem* prob;
};
} //end namespace

#endif

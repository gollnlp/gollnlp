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

//for 0.5||x||^2
class Ex1SingleVarQuadrObjTerm : public OptObjectiveTerm {
public: 
  Ex1SingleVarQuadrObjTerm(const std::string& id, OptVariablesBlock* x_) 
    : OptObjectiveTerm(id), x(x_) 
  {};

private:
  OptVariablesBlock* x;
};

// for computing 0.5*||a-b||^2
class Ex1TwoVarsQuadrObjTerm : public OptObjectiveTerm {
public:
  Ex1TwoVarsQuadrObjTerm(const std::string& id, OptVariablesBlock* a_, OptVariablesBlock* b_)
    : OptObjectiveTerm(id), a(a_), b(b_) 
  {};

private:
  OptVariablesBlock *a, *b;
};

//for sum(x^2) <=10
class Ex1SumOfSquaresConstraints : public OptConstraintsBlock {
public:
  Ex1SumOfSquaresConstraints(const std::string& id_, OptVariablesBlock* x_) 
    : OptConstraintsBlock(id_), x(x_) 
  {};
  virtual ~Ex1SumOfSquaresConstraints() {};
private:
  OptVariablesBlock* x; 
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
  OptVariablesBlock *x, *y, *z, *s;
  bool use_slacks;
};


#endif

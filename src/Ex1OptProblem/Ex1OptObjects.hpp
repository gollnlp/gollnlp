#ifndef GOLLNLP_EX1OPTOBJ
#define GOLLNLP_EX1OPTOBJ

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

//for 0.5||x||^2
class Ex1SingleVarQuadrObjTerm : public OptObjectiveTerm {
public: 
  Ex1SingleVarQuadrObjTerm(OptVariables::OptVarsBlock* x_) : x(x_) {}

private:
  OptVariables::OptVarsBlock* x;
};

// for computing 0.5*||a-b||^2
class Ex1TwoVarsQuadrObjTerm : public OptObjectiveTerm {
public:
  Ex1TwoVarsQuadrObjTerm(OptVariables::OptVarsBlock* a_, OptVariables::OptVarsBlock* b_)
    : a(a_), b(b_) {}

private:
  OptVariables::OptVarsBlock *a, *b;
};

//for sum(x^2) <=10
class Ex1SumOfSquaresConstraints : public OptConstraintsBlock {
public:
  Ex1SumOfSquaresConstraints(OptVariables::OptVarsBlock* x_) : x(x_) {}

private:
  OptVariables::OptVarsBlock* x; 
};


//for sin(x) + z <= cos(y) or sin(x) + z + s = cos(y)
class Ex1Constraint2 : public OptConstraintsBlock {
public: 
  Ex1Constraint2(OptVariables* vars, bool useSlacks) 
    : m_vars(vars), use_slacks(useSlacks) {}
  virtual ~Ex1Constraint2();

  virtual OptVariables::OptVarsBlock* create_varsblock() { 
    if(!use_slacks) return NULL; 
    else return new OptVariables::OptVarsBlock(m_vars->get_block("x")->n, "s", 0, 1e+20);
  };

  virtual OptObjectiveTerm* create_objterm() { 
    if(!use_slacks) return NULL; 
    else return new Ex1SingleVarQuadrObjTerm(m_vars->get_block("s"));
  };

private:
  OptVariables* m_vars;
  bool use_slacks;
};


#endif

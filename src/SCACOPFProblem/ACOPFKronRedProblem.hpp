#ifndef ACOPF_KRON_PROBLEM
#define ACOPF_KRON_PROBLEM

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include <cstring>
#include "blasdefs.hpp"
#include "goTimer.hpp"

namespace gollnlp {

  /* ACOPF with Kron reduction */
  class ACOPFKronRedProblem : public OptProblem
  {
  public:
    ACOPFKronRedProblem(SCACOPFData& d_in) 
      : data_sc(d_in)
    {
    }
    virtual ~ACOPFKronRedProblem();
    
    /* initialization method: performs Kron reduction and builds the OptProblem */
    virtual bool assembly();
    
  protected: 
    void add_variables();
    void add_cons_pf();
    void add_obj_prod_cost();
  protected:
    //utilities
  protected: 
    //members
    SCACOPFData& data_sc;
  };

} //end namespace
#endif

#ifndef ACOPF_KRON_PROBLEM
#define ACOPF_KRON_PROBLEM

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include "goTimer.hpp"

#include "hiopKronReduction.hpp"

#include <vector>
#include <cstring>

namespace gollnlp {

  /* ACOPF with Kron reduction */
  class ACOPFKronRedProblem : public OptProblem
  {
  public:
    ACOPFKronRedProblem(SCACOPFData& d_in) 
      : data_sc_(d_in)
    {
    }
    virtual ~ACOPFKronRedProblem();
    
    /* initialization method: performs Kron reduction and builds the OptProblem */
    virtual bool assemble();
    
  protected: 
    void add_variables();
    void add_cons_pf();
    void add_obj_prod_cost();

    hiop::hiopMatrixComplexSparseTriplet* construct_YBus_matrix();
    void construct_buses_idxs(std::vector<int>& idxs_nonaux, std::vector<int>& idxs_aux);
  protected:
    //utilities
  protected: 
    //members
    SCACOPFData& data_sc_;
  };

} //end namespace
#endif

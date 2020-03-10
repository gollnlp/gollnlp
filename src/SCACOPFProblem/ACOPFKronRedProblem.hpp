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
      : data_sc(d_in), Ybus_red(NULL)
    {
    }
    virtual ~ACOPFKronRedProblem();
    
    /* initialization method: performs Kron reduction and builds the OptProblem */
    virtual bool assemble();
    
  protected: 
    void add_variables(SCACOPFData& dB, bool SysCond_BaseCase = true);
    void add_cons_pf(SCACOPFData& d);
    void add_obj_prod_cost(SCACOPFData& d);

    hiop::hiopMatrixComplexSparseTriplet* construct_YBus_matrix();
    void construct_buses_idxs(std::vector<int>& idxs_nonaux, std::vector<int>& idxs_aux);
  protected:
    //utilities
  protected: 
    //members
    SCACOPFData& data_sc;
    std::vector<int> idxs_buses_nonaux, idxs_buses_aux;
    hiop::hiopMatrixComplexDense* Ybus_red;
  };

} //end namespace
#endif

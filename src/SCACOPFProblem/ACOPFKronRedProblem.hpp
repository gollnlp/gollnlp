#ifndef ACOPF_KRON_PROBLEM
#define ACOPF_KRON_PROBLEM

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include <cstring>
#include "blasdefs.hpp"
#include "goTimer.hpp"

#include "MatrixSparseTripletStorage.hpp"
typedef std::complex<double> dcomplex;
typedef MatrixSparseTripletStorage<int, dcomplex > MatrixSpTComplex;

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

    MatrixSpTComplex* construct_YBus_matrix();
  protected:
    //utilities
  protected: 
    //members
    SCACOPFData& data_sc_;
    MatrixSpTComplex* YBus_;
  };

} //end namespace
#endif

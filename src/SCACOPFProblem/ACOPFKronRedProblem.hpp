#ifndef ACOPF_KRON_PROBLEM
#define ACOPF_KRON_PROBLEM

#include "OptProblemMDS.hpp"
#include "SCACOPFData.hpp"

#include "goTimer.hpp"

#include "hiopKronReduction.hpp"

#include <vector>
#include <cstring>

namespace gollnlp {

  /* ACOPF with Kron reduction */
  class ACOPFKronRedProblem : public OptProblemMDS
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

    /** Computes v_aux via Kron reduction and then returns all (nonaux and aux) voltages 
     */
    void compute_voltages_nonaux(const OptVariablesBlock* v_nonaux,
				 const OptVariablesBlock* theta_nonaux,
				 std::vector<std::complex<double> >& v_complex_all);
    
    /** Computes power flows over all branches and returns them for lines and transformers 
     * 'pli' and 'pti' are two dimensional vectors (each of the two rows represents the powers
     * at the terminals).
     */
    void compute_power_flows(const std::vector<std::complex<double> >& v_complex_all,
			     std::vector<std::vector<std::complex<double> > >& pli,
			     std::vector<std::vector<std::complex<double> > >& pti);
    
    /** Finds indexes of the aux buses (in the entire, aux+nonaux set of buses) at which the
     * voltages are violated.
     */
    void find_voltage_viol_busidxs(const std::vector<std::complex<double> >& v_complex_all,
				   std::vector<int>& Nidx_voltoutofbnds);

    /** Finds indexes in lines/transformers and in to/from arrays corresponding to lines/transformers
    * that are overloaded
    */
    void find_power_viol_LTidxs(const std::vector<std::complex<double> >& v_complex_all,
				const std::vector<std::vector<std::complex<double> > >& pli,
				const std::vector<std::vector<std::complex<double> > >& pti,
				std::vector<int>& Lidx_overload,
				std::vector<int>& Lin_overload,
				std::vector<int>& Tidx_overload,
				std::vector<int>& Tin_overload);
				  
  protected: 
    //members
    SCACOPFData& data_sc;
    std::vector<int> idxs_buses_nonaux, idxs_buses_aux;
    hiop::hiopKronReduction reduction_;
    hiop::hiopMatrixComplexDense* Ybus_red;
  };

} //end namespace
#endif

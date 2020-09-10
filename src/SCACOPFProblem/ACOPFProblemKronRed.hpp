#ifndef ACOPF_KRON_PROBLEM
#define ACOPF_KRON_PROBLEM

#include "OptProblemMDS.hpp"
#include "SCACOPFData.hpp"

#include "SCACOPFUtils.hpp"
#include "goTimer.hpp"

#include "hiopKronReduction.hpp"

#include <vector>
#include <cstring>

namespace gollnlp {

  /* ACOPF with Kron reduction */
  class ACOPFProblemKronRed : public OptProblemMDS
  {
  public:
    ACOPFProblemKronRed(SCACOPFData& d_in);
    virtual ~ACOPFProblemKronRed();

    
    /* initialization method: performs Kron reduction and prepares everything needed to 
     * assemble the problem.
     */
    bool initialize(bool SysCond_BaseCase = true);

    /*builds the OptProblem if requested. In some cases, the calling code assemble the problems, and, 
    * as a result, the function is not called.
    */
    virtual bool assemble();

    /** Override of the parent @optimize method that performs the solve of the Kron
     * problem in a loop by adding binding transmission constraints and voltages
     * bounds for non-auxiliary buses.
     * 
     * Internally, calls @OptProblemMDS::optimize and @OptProblemMDS::reoptimize
     * inside the solve loop.
     */
    virtual bool optimize(const std::string& nlpsolver);
    
    /** See @optimize above */
    virtual bool reoptimize(RestartType t=primalRestart);

    /** Sets the voltages (both aux and nonaux) to their values in the full-space (fs) 
     * voltages variable 'v_fs'
     */
    virtual bool v_and_theta_start_from_fs(SCACOPFData& d,
					   OptVariablesBlock* v_n,
					   OptVariablesBlock* theta_n,
					   OptVariablesBlock* v_aux_n,
					   OptVariablesBlock* theta_aux_n,
					   const OptVariablesBlock& v_fs,
					   const OptVariablesBlock& th_fs)
    {
      assert(th_fs.n == v_fs.n);
      assert(v_fs.n == idxs_buses_nonaux.size() + idxs_buses_aux.size());
      assert(map_idxbuses_idxsoptimiz_.size() == v_fs.n);

      //auto v_n     = vars_block(var_name("v_n", d));
      //auto v_aux_n = vars_block(var_name("v_aux_n", d));
      //auto theta_n     = vars_block(var_name("theta_n", d));
      //auto theta_aux_n = vars_block(var_name("theta_aux_n", d));
      
      assert(v_n);
      assert(theta_n);
      
      for(int i=0; i<map_idxbuses_idxsoptimiz_.size(); i++) {

	assert(i<v_fs.n);

	//not part of the optimization
	if(map_idxbuses_idxsoptimiz_[i]==-1) continue;
	
	if(map_idxbuses_idxsoptimiz_[i]<0) {
	  const int idx_aux = -map_idxbuses_idxsoptimiz_[i]-2;
	  assert(idx_aux>=0);
	  assert(idx_aux<idxs_buses_aux.size());

	  assert(v_aux_n);
	  assert(v_aux_n->n == idxs_buses_aux.size());

	  assert(theta_aux_n);
	  assert(theta_aux_n->n == idxs_buses_aux.size());

	  v_aux_n->x[idx_aux] = v_fs.x[i];
	  theta_aux_n->x[idx_aux] = th_fs.x[i];
	  
	} else {
	  const int idx_nonaux = map_idxbuses_idxsoptimiz_[i];
	  assert(idx_nonaux>=0); assert(idx_nonaux<idxs_buses_nonaux.size());

	  v_n->x[idx_nonaux] = v_fs.x[i];
	  theta_n->x[idx_nonaux] = th_fs.x[i];
	}
      }

      v_n->providesStartingPoint = true;
      theta_n->providesStartingPoint = true; 
      
      return true;
    }
    
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
    
    //indexes in N_Bus
    std::vector<int> idxs_buses_nonaux, idxs_buses_aux;
    
    hiop::hiopKronReduction reduction_;
    hiop::hiopMatrixComplexDense* Ybus_red_;

    std::vector<int> N_idx_voutofobounds_;

    std::vector<int> Lidx_overload_;
    std::vector<int> Lin_overload_;
    std::vector<int> Tidx_overload_;
    std::vector<int> Tin_overload_;


    /* Mapping of bus N indexes into 
     *  - the index inside of the optimization variable v_n and theta_n for nonaux buses
     *
     *  - the index 'i' inside v_aux_n and theta_aux_n for aux buses included in the optimization 
     * (as the result of voltage bounds and thermal limit violations)
     * !!!  for these buses, we store -i-2, where i is the index in v_aux_n and theta_aux_n
     *
     *  - -1 for aux buses not included in the optimization
     */
    std::vector<int> map_idxbuses_idxsoptimiz_;

    
  /** ACOPF interface - similar to SCACOPFProblem
   */
  protected:

    bool useQPen;

  protected:
    // returns the idxs of PVPQ gens and corresponding buses
    // generators at the same PVPQ bus are aggregated
    //
    // Gk are the indexes of all gens other than the outgen (for generator contingencies) 
    // in data_sc.G_Generator
    void get_idxs_PVPQ(SCACOPFData& dB, const std::vector<int>& Gk,
		       std::vector<std::vector<int> >& idxs_gen_agg, std::vector<int>& idxs_bus_pvpq,
		       std::vector<double>& Qlb, std::vector<double>& Qub,
		       int& nPVPQGens, int &num_qgens_fixed, 
		       int& num_N_PVPQ, int& num_buses_all_qgen_fixed);
  public:
    inline OptVariablesBlock* variable(const std::string& prefix, const SCACOPFData& d) { 
      return vars_block(var_name(prefix, d));
    }
    inline OptVariablesBlock* variable(const std::string& prefix, int Kid) { 
      return vars_block(var_name(prefix, Kid));
    }
    inline OptConstraintsBlock* constraint(const std::string& prefix, const SCACOPFData& d) { 
      return constraints_block(con_name(prefix, d));
    }
    inline OptConstraintsBlock* constraint(const std::string& prefix, int Kid) {
      return constraints_block(con_name(prefix, Kid));
    }
    
    inline OptVariablesBlock* variable_duals_lower(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_bounds_lower(prefix+"_"+std::to_string(d.id));
    }
    inline OptVariablesBlock* variable_duals_upper(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_bounds_upper(prefix+"_"+std::to_string(d.id));
    }
    inline OptVariablesBlock* variable_duals_cons(const std::string& prefix, const SCACOPFData& d) {
      return vars_block_duals_cons(prefix+"_"+std::to_string(d.id));
    }
  };

} //end namespace
#endif

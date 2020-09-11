#ifndef GO_OPF_CONSTRAINTS_KRON
#define GO_OPF_CONSTRAINTS_KRON

#include "OptProblemMDS.hpp"
#include "SCACOPFData.hpp"

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <cstring>
#include <cmath>

//#include "OPFObjectiveTerms.hpp"

#include "hiopMatrixComplexDense.hpp"

namespace gollnlp {
  ///////////////////////////////////////////////////////////////////////////////
  // Active Balance constraints
  //
  // for i=1:length(nonaux)
  //
  // sum(p_g[g] for g=Gn[nonaux[i]]) 
  // - sum(v_n[i]*v_n[j]*
  //      ( Gred[i,j]*cos(theta_n[i]-theta_n[j]) 
  //      + Bred[i,j]*sin(theta_n[i]-theta_n[j])) for j=1:length(nonaux))
  // - pslackp_n[i] + r*pslackm_n[i]
  // ==  N[:Pd][nonaux[i]]
  ///////////////////////////////////////////////////////////////////////////////
  class PFActiveBalanceKron : public OptConstraintsBlockMDS
  {
  public:
    PFActiveBalanceKron(const std::string& id_, int numcons,
			OptVariablesBlock* p_g_, 
			OptVariablesBlock* v_n_,
			OptVariablesBlock* theta_n_,
			const std::vector<int>& bus_nonaux_idxs_,
			const std::vector<std::vector<int> >& Gn_full_space_,
			const hiop::hiopMatrixComplexDense& Ybus_red_,
			const std::vector<double>& N_Pd_full_space_);
    virtual ~PFActiveBalanceKron();

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);

    virtual bool eval_Jac_eq(const OptVariables& x, bool new_x, 
			     const int& nxsparse, const int& nxdense,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			     double** JacD);

    virtual bool eval_Jac_ineq(const OptVariables& x, bool new_x, 
			       const int& nxsparse, const int& nxdense,
			       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			       double** JacD)
    {
      return true;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);
    //nnz for sparse part (all zeros)
    virtual int get_HessLagr_SSblock_nnz() { return 0; }

    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij) { return true; }

    virtual void primal_problem_changed()
    {
      if(J_nz_idxs)
	delete[] J_nz_idxs;
      J_nz_idxs = NULL;
      
      if(H_nz_idxs)
	delete[] H_nz_idxs;
      H_nz_idxs = NULL;
    }
    // 
    virtual OptVariablesBlock* create_varsblock() 
    {
      assert(pslack_n_==NULL);
      pslack_n_ = new OptVariablesBlock(2*n, "pslack_n_"+id, 0, 1e+20);
      return pslack_n_; 
    }
    inline OptVariablesBlock* slacks() { return pslack_n_; }
  protected:
    OptVariablesBlock *p_g, *v_n, *theta_n;
    const std::vector<int> &bus_nonaux_idxs;
    std::vector<double> N_Pd;//in nonaux
    const std::vector<std::vector<int> > &Gn_fs; //full space
    const hiop::hiopMatrixComplexDense& Ybus_red;

    int* J_nz_idxs;
    int* H_nz_idxs;

    OptVariablesBlock *pslack_n_; //2*n -> containss pslackp_n, pslackm_n;
  };


  /**
   * Reactive Balance constraints
   *
   * for i=1:length(nonaux)
   *
   * sum(q_g[g] for g=Gn[nonaux[i]]) - N[:Qd][nonaux[i]] +
   * v_n[i]^2*sum(b_s[s] for s=SShn[nonaux[i]]) ==
   *   sum(v_n[i]*v_n[j]*(Gred[i,j]*sin(theta_n[i]-theta_n[j]) -
   *   Bred[i,j]*cos(theta_n[i]-theta_n[j])) for j=1:length(nonaux)))
   */
  class PFReactiveBalanceKron : public OptConstraintsBlockMDS
  {
  public:
    PFReactiveBalanceKron(const std::string& id_, int numcons,
			  OptVariablesBlock* q_g_, 
			  OptVariablesBlock* v_n_,
			  OptVariablesBlock* theta_n_,
			  OptVariablesBlock* b_s_,
			  const std::vector<int>& bus_nonaux_idxs_,
			  const std::vector<std::vector<int> >& Gn_full_space_,
			  const std::vector<std::vector<int> >& SShn_full_space_in,
			  const hiop::hiopMatrixComplexDense& Ybus_red_,
			  const std::vector<double>& N_Qd_full_space_);
    virtual ~PFReactiveBalanceKron();

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);

    virtual bool eval_Jac_eq(const OptVariables& x, bool new_x, 
			     const int& nxsparse, const int& nxdense,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			     double** JacD);

    virtual bool eval_Jac_ineq(const OptVariables& x, bool new_x, 
			       const int& nxsparse, const int& nxdense,
			       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			       double** JacD)
    {
      return true;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);
    //nnz for sparse part (all zeros)
    virtual int get_HessLagr_SSblock_nnz() { return 0; }

    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij) { return true; }

    virtual void primal_problem_changed()
    {
      if(J_nz_idxs)
	delete[] J_nz_idxs;
      J_nz_idxs = NULL;
      
      if(H_nz_idxs)
	delete[] H_nz_idxs;
      H_nz_idxs = NULL;
    }
    virtual OptVariablesBlock* create_varsblock() 
    { 
      assert(qslack_n_==NULL);
      qslack_n_ = new OptVariablesBlock(2*n, "qslack_n_"+id, 0, 1e+20);
      return qslack_n_; 
    }
    inline OptVariablesBlock* slacks() { return qslack_n_; }
  protected:
    OptVariablesBlock *q_g, *v_n, *theta_n, *b_s;
    const std::vector<int> &bus_nonaux_idxs;
    std::vector<double> N_Qd;//in nonaux
    const std::vector<std::vector<int> > &Gn_fs; //full space
    const std::vector<std::vector<int> >& SShn_fs; //full space
    const hiop::hiopMatrixComplexDense& Ybus_red;

    int* J_nz_idxs;
    int* H_nz_idxs;

    OptVariablesBlock *qslack_n_; //2*n -> containss pslackp_n, pslackm_n;
  };

  /*********************************************************************************************
   * Voltage violations constraints at auxiliary buses
   *
   * for nix = idxs_busviol_in_aux_buses
   *    vaux_n[nix]*cos(thetaaux_n[nix]) ==
   *	sum((re_ynix[i]*v_n[i]*cos(theta_n[i]) - im_ynix[i]*v_n[i]*sin(theta_n[i])) for i=1:length(nonaux)))
   *
   *    vaux_n[nix]*sin(thetaaux_n[nix]) ==
   *    sum((re_ynix[i]*v_n[i]*sin(theta_n[i]) + im_ynix[i]*v_n[i]*cos(theta_n[i])) for i=1:length(nonaux)))
   *
   * where re_ynix=Re(vmap[nix,:]) and im_ynix=Im(vmap[nix,:])
   *********************************************************************************************/
  class VoltageConsAuxBuses : public OptConstraintsBlockMDS
  {
  public:
    VoltageConsAuxBuses(const std::string& id_in, int numcons,
			OptVariablesBlock* v_n_in,
			OptVariablesBlock* theta_n_in,
			OptVariablesBlock* v_aux_n_in,
			OptVariablesBlock* theta_aux_n_in,
			const std::vector<int>& vtheta_aux_idxs_in,
			const hiop::hiopMatrixComplexDense& vmap_in);
    virtual ~VoltageConsAuxBuses();

    //add (append to existing) voltage constraints for v and theta variables specified by the
    //indexes passed in 'vtheta_aux_idxs_new'
    void append_constraints(const std::vector<int>& vtheta_aux_idxs_new);

    //
    // OptProblem ConstraintsBlock interface methods
    //
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);

    virtual bool eval_Jac_eq(const OptVariables& x, bool new_x, 
			     const int& nxsparse, const int& nxdense,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			     double** JacD);

    virtual bool eval_Jac_ineq(const OptVariables& x, bool new_x, 
			       const int& nxsparse, const int& nxdense,
			       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			       double** JacD)
    {
      return true;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);
    //nnz for sparse part (all zeros)
    virtual int get_HessLagr_SSblock_nnz() { return 0; }

    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij) { return true; }

    virtual void primal_problem_changed() { }
  protected:
    OptVariablesBlock *v_n_, *theta_n_, *v_aux_n_, *theta_aux_n_;
    std::vector<int> idxs_busviol_in_aux_buses_;
    const hiop::hiopMatrixComplexDense& vmap_;
  };

  /*********************************************************************************************
   * Constraints for enforcing line thermal limits violations
   * 
   * for lix=1:length(Lidx_overload_pass)
   *   l = Lidx_overload_pass[lix]
   *   i = Lin_overload_pass[lix]
   *   vi = vall_n[L_Nidx[l,i]]
   *   thetai = thetaall_n[L_Nidx[l,i]]
   *   vj = vall_n[L_Nidx[l,3-i]]
   *   thetaj = thetaall_n[L_Nidx[l,3-i]]
   *   slack_li >=0 starts at max(0, abs(s_li[l,i]) - abs(v_n_all_complex[L_Nidx[l,i]])*L[:RateBase][l])
   *
   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - L[:RateBase][l]^2*vi^2 - slack_li <=0
   * ((the last line was originally:  - (L[:RateBase][l]*vi + slack_li)^2 <=0))
   *
   *********************************************************************************************/

  class LineThermalViolCons : public OptConstraintsBlockMDS
  {
  public:
    LineThermalViolCons(const std::string& id_in,
			int numcons,
			const std::vector<int>& Lidx_overload_in,
			const std::vector<int>& Lin_overload_in,
			/*idxs of L_From and L_To in N_Bus (in L_Nidx[0] and L_Nidx[1])*/
			const std::vector<std::vector<int> >& L_Nidx_in,
			const std::vector<double>& L_Rate_in,
			const std::vector<double>& L_G_in,
			const std::vector<double>& L_B_in,
			const std::vector<double>& L_Bch_in,
			/*from N idxs to idxs in aux and nonaux optimiz vars*/
			const std::vector<int>& map_idxbuses_idxsoptimiz_in, 
			OptVariablesBlock* v_n_in,
			OptVariablesBlock* theta_n_in,
			OptVariablesBlock* v_aux_n_in,
			OptVariablesBlock* theta_aux_n_in);
    virtual ~LineThermalViolCons();
    
    //add (append to existing block of) line thermal violations
    //indexes passed in 'vtheta_aux_idxs_new'
    void append_constraints(const std::vector<int>& Lidx_overload_pass_in,
			    const std::vector<int>& Lin_overload_pass_in);

    //
    // OptProblem ConstraintsBlock interface methods
    //
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);

    virtual bool eval_Jac_eq(const OptVariables& x, bool new_x, 
			     const int& nxsparse, const int& nxdense,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			     double** JacD);

    virtual bool eval_Jac_ineq(const OptVariables& x, bool new_x, 
			       const int& nxsparse, const int& nxdense,
			       const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			       double** JacD);

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz();
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij);

    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);
    //nnz for sparse part (all zeros)
    virtual int get_HessLagr_SSblock_nnz();

    virtual bool get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij);

    virtual OptVariablesBlock* create_varsblock() 
    {
      assert(slacks_==NULL);
      slacks_ = new OptVariablesBlock(n, "slacks_"+id, 0, 1e+20);
      slacks_->set_start_to(0.);
      return slacks_; 
    }

    virtual void primal_problem_changed()
    {
      if(J_nz_idxs_)
	delete[] J_nz_idxs_;
      J_nz_idxs_ = NULL;
    }
  protected:
    /* Get the values from v_n or v_aux_n and theta_n or theta_aux_n at bus 'idx_bus'. 
     * 'map_idxbuses_idxsoptimiz_[idx_bus]' decides whether the value is from v_n and theta_n
     * (nonaux bus) or from v_aux_n and theta_aux_n (aux bus)
     */
    //void get_v_theta_vals(const int& idx_bus, double& vval, double& thetaval);

    /* Same as above but also returns the "dense" indexes of v and theta */
    void get_v_theta_vals_and_idxs(const int& idx_bus,
				   double& vval, double& thetaval,
				   int& v_idx_out, int& theta_idx_out);				  
  protected:
    OptVariablesBlock *v_n_, *theta_n_, *v_aux_n_, *theta_aux_n_;
    OptVariablesBlock *slacks_; 
    std::vector<int> Lidx_overload_;
    std::vector<int> Lin_overload_;
    
    /* indexes of L_From and L_To in N_Bus (stored in L_Nidx[0] and L_Nidx[1])*/
    const std::vector<std::vector<int> >& L_Nidx_;

    const std::vector<double>& L_Rate_, L_G_, L_B_, L_Bch_;
    
    /* from N idxs to idxs in aux and nonaux optimiz vars*/
    const std::vector<int>& map_idxbuses_idxsoptimiz_;

    int* J_nz_idxs_;
  };


} //end namespace

#endif

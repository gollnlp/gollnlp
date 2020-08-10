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
      return false;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      assert(false);
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return false;
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
  protected:
    OptVariablesBlock *p_g, *v_n, *theta_n;
    const std::vector<int> &bus_nonaux_idxs;
    std::vector<double> N_Pd;//in nonaux
    const std::vector<std::vector<int> > &Gn_fs; //full space
    const hiop::hiopMatrixComplexDense& Ybus_red;

    int* J_nz_idxs;
    int* H_nz_idxs;
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
      return false;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      assert(false);
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return false;
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
  protected:
    OptVariablesBlock *q_g, *v_n, *theta_n, *b_s;
    const std::vector<int> &bus_nonaux_idxs;
    std::vector<double> N_Qd;//in nonaux
    const std::vector<std::vector<int> > &Gn_fs; //full space
    const std::vector<std::vector<int> >& SShn_fs; //full space
    const hiop::hiopMatrixComplexDense& Ybus_red;

    int* J_nz_idxs;
    int* H_nz_idxs;
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
      return false;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      assert(false);
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return false;
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


   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - (L[:RateBase][l]*vi + slack_li)^2 <=0
   *
   *
   *********************************************************************************************/

  class LineThermalViolCons : public OptConstraintsBlockMDS
  {
  public:
    LineThermalViolCons(const std::string& id_in, int numcons,
			OptVariablesBlock* v_n_in,
			OptVariablesBlock* theta_n_in,
			OptVariablesBlock* v_aux_n_in,
			OptVariablesBlock* theta_aux_n_in,
			const std::vector<int>& vtheta_aux_idxs_in,
			const hiop::hiopMatrixComplexDense& vmap_in);
    virtual ~LineThermalViolCons();
    
    //add (append to existing block of) line thermal violations
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
      return false;
    }

    virtual int get_spJacob_eq_nnz();
    virtual int get_spJacob_ineq_nnz()
    {
      assert(false);
      return 0;
    }
    virtual bool get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij);
    virtual bool get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
    {
      return false;
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
  protected:
    OptVariablesBlock *v_n_, *theta_n_, *v_aux_n_, *theta_aux_n_;
    std::vector<int> vtheta_aux_idxs_;
    const hiop::hiopMatrixComplexDense& vmap_;
  };


} //end namespace

#endif

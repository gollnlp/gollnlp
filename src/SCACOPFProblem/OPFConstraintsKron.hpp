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
  
} //end namespace

#endif

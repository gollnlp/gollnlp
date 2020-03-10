#ifndef GO_OPF_CONSTRAINTS_KRON
#define GO_OPF_CONSTRAINTS_KRON

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <cstring>
#include <cmath>

#include "OPFObjectiveTerms.hpp"

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
  class PFActiveBalanceKron : public OptConstraintsBlock
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

    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M) 
    {
      return false;
    }

    virtual bool eval_Jac(const OptVariables& x, bool new_x, 
			  const int& nxsparse, const int& nxdense,
			  const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
			  double** JacD);

    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M)
    {
      return false;
    }
    virtual bool eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       int& nnzHSD, int* iHSD, int* jHSD, double* MHSD);
    //nnz for sparse part (all zeros)
    virtual int get_HessLagr_nnz() { return 0; }

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) { return true; }
  protected:
    OptVariablesBlock *p_g, *v_n, *theta_n;
    const std::vector<int> &bus_nonaux_idxs;
    std::vector<double> N_Pd;//in nonaux
    const std::vector<std::vector<int> > &Gn_fs; //full space
    const hiop::hiopMatrixComplexDense& Ybus_red;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

} //end namespace

#endif

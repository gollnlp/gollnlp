#ifndef GO_OPF_CONSTRAINTS
#define GO_OPF_CONSTRAINTS

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <cstring>
#include <cmath>


#include "OPFObjectiveTerms.hpp"

namespace gollnlp {
  //////////////////////////////////////////////////////////////////
  // Power flows in rectangular form
  // pq == A*vi^2 + B*vi*vj*cos(thetai - thetaj + Theta) + 
  //       C*vi*vj*sin(thetai - thetaj + Theta)
  //////////////////////////////////////////////////////////////////
  class PFConRectangular : public OptConstraintsBlock
  {
  public:
    PFConRectangular(const std::string& id_, int numcons,
		     OptVariablesBlock* pq_, 
		     OptVariablesBlock* v_n_, 
		     OptVariablesBlock* theta_n_,
		     const std::vector<int>& Nidx1, //T_Nidx or L_Nidx indexes
		     const std::vector<int>& Nidx2);
    //normally we would also have these arguments in the constructor, but want to avoid 
    //excessive copying and the caller needs to update this directly
    //const std::vector<double>& A,
    //const std::vector<double>& B,
    //const std::vector<double>& C,
    //const std::vector<double>& T, //Theta
    //  const SCACOPFData& d_)
    virtual ~PFConRectangular();

    //accessers
    inline double* get_A() { return A;}
    inline double* get_B() { return B;}
    inline double* get_C() { return C;}
    inline double* get_T() { return T;}

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz(){ return 5*n; }
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M);

    virtual int get_HessLagr_nnz() { return 9*n; }

    // (i,j) entries in the HessLagr to which the implementer's contributes to
    // this is only called once
    // push_back in vij 
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);

    // computes pq given vi,vj, thetai,thetaj
    // updates p_or_q->x  using v_n->x and theta_n->x
    // this is used in computing the starting points for example
    virtual void compute_power(OptVariablesBlock* p_or_q);
  protected:
    const OptVariablesBlock *pq, *v_n, *theta_n;
    int* J_nz_idxs;
    int* H_nz_idxs;

    double *A, *B, *C, *T;
    int *E_Nidx1, *E_Nidx2;
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Active Balance constraints
  //
  // sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
  // sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
  // sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - 
  // r*pslackp_n[n] + r*pslackm_n[n]    =   N[:Pd][n])
  //
  // r=1/'slacks_rescale' 
  ///////////////////////////////////////////////////////////////////////////////
  class PFActiveBalance : public OptConstraintsBlock
  {
  public:
    PFActiveBalance(const std::string& id_, int numcons,
		    OptVariablesBlock* p_g_, 
		    OptVariablesBlock* v_n_,
		    OptVariablesBlock* p_li1_,
		    OptVariablesBlock* p_li2_,
		    OptVariablesBlock* p_ti1_,
		    OptVariablesBlock* p_ti2_,
		    const std::vector<double>& N_Gsh_,
		    const std::vector<double>& N_Pd_,
		    const std::vector<std::vector<int> >& Gn_,
		    const std::vector<std::vector<int> >& Lidxn1_,
		    const std::vector<std::vector<int> >& Lidxn2_,
		    const std::vector<std::vector<int> >& Tidxn1_,
		    const std::vector<std::vector<int> >& Tidxn2_,
		    const double& slacks_rescale=1.);
    virtual ~PFActiveBalance();

    OptVariablesBlock* slacks() { return pslack_n; }
    void compute_slacks(OptVariablesBlock* slacks) const;
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);

    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M);

    virtual int get_HessLagr_nnz() { return n; }

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);

    // Some constraints create additional variables (e.g., slacks).
    // This method is called by OptProblem (in 'append_constraints') to get and add
    // the additional variables block that OptConstraintsBlock may need to add.
    // NULL should be returned when the OptConstraintsBlock need not create a vars block
    virtual OptVariablesBlock* create_varsblock() 
    { 
      assert(pslack_n==NULL);
      pslack_n = new OptVariablesBlock(2*n, "pslack_n_"+id, 0, 1e+20);
      return pslack_n; 
    }
    
    //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
    //term (e.g., penalization) that OptConstraintsBlock may need
    virtual OptObjectiveTerm* create_objterm() 
    { 
      //this is done externally
      return NULL;//new DummySingleVarQuadrObjTerm("pen_pslack_n", pslack_n); 
    }
  protected:
    OptVariablesBlock *p_g, *v_n, *p_li1, *p_li2, *p_ti1, *p_ti2;
    double r;
    OptVariablesBlock *pslack_n; //2*n -> containss pslackp_n, pslackm_n;

    const std::vector<double> &N_Gsh, &N_Pd;
    const std::vector<std::vector<int> > &Gn, &Lidxn1, &Lidxn2, &Tidxn1, &Tidxn2;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Reactive Balance constraints
  //
  //sum(q_g[g] for g=Gn[n]) - 
  // (-N[:Bsh][n] - sum(b_s[s] for s=SShn[n]))*v_n[n]^2 -
  // sum(q_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
  // sum(q_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - 
  // qslackp_n[n] + qslackm_n[n])  ==  N[:Qd][n]
  //
  // r=1/'slacks_rescale' 
  ////////////////////////////////////////////////////////////////////////////////////////////
  class PFReactiveBalance  : public OptConstraintsBlock
  {
  public:
    PFReactiveBalance(const std::string& id_, int numcons,
		      OptVariablesBlock* q_g_, 
		      OptVariablesBlock* v_n_,
		      OptVariablesBlock* q_li1_,
		      OptVariablesBlock* q_li2_,
		      OptVariablesBlock* q_ti1_,
		      OptVariablesBlock* q_ti2_,
		      OptVariablesBlock* b_s_,
		      const std::vector<double>& N_Bsh_,
		      const std::vector<double>& N_Qd_,	
		      const std::vector<std::vector<int> >& Gn_,
		      const std::vector<std::vector<int> >& SShn_,
		      const std::vector<std::vector<int> >& Lidxn1_,
		      const std::vector<std::vector<int> >& Lidxn2_,
		      const std::vector<std::vector<int> >& Tidxn1_,
		      const std::vector<std::vector<int> >& Tidxn2_,
		      const double& slacks_rescale=1.);
    virtual ~PFReactiveBalance();

    OptVariablesBlock* slacks() { return qslack_n; }
    void compute_slacks(OptVariablesBlock* qslacks_n);
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    
    // 
    // Jacobian (nonlinear parts)
    //   2*( N[:Bsh][n] + sum(b_s[s] for s=SShn[n]) ) * v_n[n]   - w.r.t. v_n
    //   v_n[n]^2   - w.r.t. to b_s[s]  for all a=SShn[n]
    //
    // Hessian
    // 2*( N[:Bsh][n] + sum(b_s[s] for s=SShn[n]) )  - w.r.t. v_n,v_n
    // 2*v_n                                         - w.r.t. v_n,b_s[s] for s=SShn[n]
    // total nnz = n + sum( cardinal(SShn[i])  )  for i=1,...,n
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M);

    virtual int get_HessLagr_nnz() ;

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);
    virtual OptVariablesBlock* create_varsblock() 
    { 
      assert(qslack_n==NULL);
      qslack_n = new OptVariablesBlock(2*n, "qslack_n_"+id, 0, 1e+20);
      return qslack_n; 
    }
    virtual OptObjectiveTerm* create_objterm() 
    { 
      //this is done externally
      return NULL;//new DummySingleVarQuadrObjTerm("pen_qslack_n", qslack_n); 
    }
  protected:
    OptVariablesBlock *q_g, *v_n, *q_li1, *q_li2, *q_ti1, *q_ti2, *b_s;
    double r;
    OptVariablesBlock *qslack_n; //2*n -> containss pslackp_n, pslackm_n;

    const std::vector<double>& N_Bsh;
    const std::vector<double>& N_Qd;	
    const std::vector<std::vector<int> >& Gn;
    const std::vector<std::vector<int> >& SShn;
    const std::vector<std::vector<int> >& Lidxn1;
    const std::vector<std::vector<int> >& Lidxn2;
    const std::vector<std::vector<int> >& Tidxn1;
    const std::vector<std::vector<int> >& Tidxn2;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  //////////////////////////////////////////////////////////////////
  // PFLineLimits - Line thermal limits
  //
  // p_li[l,i]^2 + q_li[l,i]^2 <= (L[RateSymb][l]*v_n[L_Nidx[l,i]] + 
  //                               r*sslack_li[l,i])^2
  // r = 1/slacks_rescale
  //////////////////////////////////////////////////////////////////
  class PFLineLimits  : public OptConstraintsBlock
  {
  public:
    PFLineLimits(const std::string& id_, int numcons,
		 OptVariablesBlock* p_li_, 
		 OptVariablesBlock* q_li_,
		 OptVariablesBlock* v_n_,
		 const std::vector<int>& L_Nidx_,
		 const std::vector<double>& L_Rate_,
		 const double& slacks_rescale=1);
    virtual ~PFLineLimits();

    OptVariablesBlock* slacks() { return sslack_li; }
    virtual void compute_slacks(OptVariablesBlock* sslacks_li);
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M);
    virtual int get_HessLagr_nnz();
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);

    virtual OptVariablesBlock* create_varsblock();
    virtual OptObjectiveTerm* create_objterm();
  protected:
    OptVariablesBlock *p_li, *q_li, *v_n;
    const std::vector<int> &Nidx;
    //const std::vector<double> &L_Rate;
    std::vector<double> L_Rate;
    double r;
    OptVariablesBlock *sslack_li; // sslackp_li1 or sslackm_li2;
    
    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  ////////////////////////////////////////////////////////////////////
  // PFTransfLimits - Transformer thermal limits
  //
  // p_ti[t,i]^2 + q_ti[t,i]^2 <= (T[RateSymb][t] + r*sslack_ti[t,i])^2
  //
  // r = 1/slacks_rescale
  ////////////////////////////////////////////////////////////////////
  class PFTransfLimits  : public OptConstraintsBlock
  {
  public:
    PFTransfLimits(const std::string& id_, int numcons,
		   OptVariablesBlock* p_ti_, 
		   OptVariablesBlock* q_ti_,
		   const std::vector<double>& T_Rate_,
		   const double& slacks_rescale=1.);
    virtual ~PFTransfLimits();
    OptVariablesBlock* slacks() { return sslack_ti; }
    void compute_slacks(OptVariablesBlock* sslack_ti);
    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M);
    virtual int get_HessLagr_nnz();
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);

    virtual OptVariablesBlock* create_varsblock();
    virtual OptObjectiveTerm* create_objterm();
  protected:
    OptVariablesBlock *p_ti, *q_ti;
    std::vector<double> T_Rate;
    double r;
    OptVariablesBlock *sslack_ti; // sslackp_ti1 or sslackm_ti2;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  ////////////////////////////////////////////////////////////////////////////
  // PFProdCostAffineCons - constraints needed by the piecewise linear 
  // production cost function 
  // min sum_g( sum_h CostCi[g][h]^T t[g][h])
  // constraints (handled outside) are
  //   t>=0, sum_h t[g][h]=1
  //   p_g[g] = sum_h CostPi[g][h]*t[g][h] 
  ///////////////////////////////////////////////////////////////////////////
  class PFProdCostAffineCons  : public OptConstraintsBlock
  {
  public:
    PFProdCostAffineCons(const std::string& id_, int numcons,
			 OptVariablesBlock* p_g_, 
			 const std::vector<int>& G_Nidx_,
			 const std::vector<std::vector<double> >& G_CostCi,
			 const std::vector<std::vector<double> >& G_CostPi);

    virtual ~PFProdCostAffineCons();

    OptVariablesBlock* get_t_h() { return t_h; }
    void compute_t_h(OptVariablesBlock* th);

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);

    virtual OptVariablesBlock* create_varsblock() { return t_h; }
    virtual OptObjectiveTerm* create_objterm() { return obj_term; }
  protected:
    OptVariablesBlock *p_g, *t_h;
    PFProdCostPcLinObjTerm* obj_term;
    std::vector<std::vector<double> > G_CostPi;
    int* J_nz_idxs; //only in size of generators
  };

  //////////////////////////////////////////////////////////////////////////////
  // Slack penalty constraints block
  // min sum_i( sum_h P[i][h] sigma_h[i][h])
  // constraints (handled outside) are
  //   0<= sigma[i][h] <= Pseg_h, 
  //   slacks[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
  //////////////////////////////////////////////////////////////////////////////
  class PFPenaltyAffineCons  : public OptConstraintsBlock
  {
  public:
    PFPenaltyAffineCons(const std::string& id_, int numcons,
			OptVariablesBlock* slack_, 
			const std::vector<double>& pen_coeff,
			const std::vector<double>& pen_segm,
			const double& obj_weight, // DELTA or (1-DELTA)/NumK
			const double& slacks_rescale=1.);
    virtual ~PFPenaltyAffineCons();

    OptVariablesBlock* get_sigma() { return sigma; }
    virtual void compute_sigma(OptVariablesBlock *slackv);

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    
    virtual OptObjectiveTerm* create_objterm() { return obj_term; }
    virtual OptVariablesBlock* create_varsblock() { return sigma; }
  protected:
    OptVariablesBlock *slack, *sigma;
    PFPenaltyPcLinObjTerm* obj_term;
    double f; //factor used in rescaling slacks, 
    int* J_nz_idxs; //only in size of slack
    double P1, P2, P3;
    double S1, S2, S3;
  };
  //////////////////////////////////////////////////////////////////////////////
  // (double) Slacks penalty constraints block
  // min sum_i( sum_h P[i][h] sigma_h[i][h])
  // constraints (handled outside) are
  //   0<= sigma[i][h] <= Pseg_h, 
  //   slacksp[i] + slacksm[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
  //
  // the two slacks are kept vectorized: slack_ = [slacksp; slackm]
  //////////////////////////////////////////////////////////////////////////////
  class PFPenaltyAffineConsTwoSlacks  : public PFPenaltyAffineCons
  {
  public:
    PFPenaltyAffineConsTwoSlacks(const std::string& id_, int numcons,
				 OptVariablesBlock* slack_, 
				 const std::vector<double>& pen_coeff,
				 const std::vector<double>& pen_segm,
				 const double& obj_weight, // DELTA or (1-DELTA)/NumK
				 const double& slacks_rescale=1.)
      : PFPenaltyAffineCons(id_, numcons, slack_, pen_coeff, pen_segm, obj_weight, slacks_rescale) {};
    virtual ~PFPenaltyAffineConsTwoSlacks() {};

    virtual void compute_sigma(OptVariablesBlock *slackv);

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
  };

  //////////////////////////////////////////////////////////////////////////////
  // AGC reserve constraints 
  // i. loss reserve
  //   a. max loss:  sum(Pub[i]-pg[i]: i in AGC) - f*max_loss + splus >=0
  //   b. Kgen loss: sum(Pub[i]-pg[i]: i in AGC) - f*pg[Kgen] + splus >=0
  //
  // ii. gain reserve
  //   a. max gain:  sum(pg[i]-Plb[i]: i in AGC) - f*max_gain + sminus >=0 
  //   b. Kgen gain: sum(pg[i]-Plb[i]: i in AGC) + f*pg[Kgen] + sminus >=0 
  //
  // 'f' is the percentage of the loss/gain that should be covered by the AGC gens
  // usually 1. or closely to the left of 1. (0.95 or 0.99)
  class AGCReservesCons  : public OptConstraintsBlock
  {
  public:
    AGCReservesCons(const std::string& id_, OptVariablesBlock* p_g_);
    virtual ~AGCReservesCons();

    //indexes are in G_Generator; Plb and Pub have the size(G_Generator)

    void add_max_loss_reserve(const std::vector<int>& idxs_agc, 
			      const double& max_loss, const double& f,
			      const std::vector<double>& Pub);
    void add_Kgen_loss_reserve(const std::vector<int>& idxs_agc, 
			       const int& idx_Kgen, const double& f,
			       const std::vector<double>& Pub);

    void add_max_gain_reserve(const std::vector<int>& idxs_agc, 
			      const double& max_gain, const double& f,
			      const std::vector<double>& Plb);
    void add_Kgen_gain_reserve(const std::vector<int>& idxs_agc, 
			       const int& idx_Kgen, const double& f,
			       const std::vector<double>& Plb);

    void add_penalty_objterm(const std::vector<double>& P_pen,
			     const std::vector<double>& P_qua,
			     const double& obj_weight,
			     const double& slacks_scale=1.0);

    void finalize_setup();
    OptVariablesBlock* get_slacks() { return slacks; }

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);

    virtual OptVariablesBlock* create_varsblock() { return slacks; }
    virtual OptObjectiveTerm* create_objterm() { return obj_penalty; }
  protected:
    OptVariablesBlock *p_g;
    OptVariablesBlock *slacks; 
    PFPenaltyQuadrApproxObjTerm *obj_penalty;
    int* J_nz_idxs; //only in size of generators

    std::vector<std::vector<int> >    idxs_;
    std::vector<std::vector<double> > coeff_;
    std::vector<double>              lb_, ub_;
#ifdef DEBUG
    bool isAssembled;
#endif
  };



  //  or 
  //  sum(pg[i]-Plb[i]: i in AGC) - c*pgK + d*max_gain + sminus >=0 
  //
  // the general form is
  // sum(
  //////////////////////////////////////////////////////////////////////////////
}

#endif

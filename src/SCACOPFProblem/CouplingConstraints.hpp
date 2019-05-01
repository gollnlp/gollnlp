#ifndef GO_COUPLING_CONS
#define GO_COUPLING_CONS

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <cstring>
#include <cmath>

#include "OPFObjectiveTerms.hpp"

namespace gollnlp {
  /////////////////////////////////////////////////////////////////////////////////
  // non-anticipativity-like constraints 
  /////////////////////////////////////////////////////////////////////////////////
  class NonAnticipCons : public OptConstraintsBlock
  {
  public:
    NonAnticipCons(const std::string& id_, int numcons,
		   OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, 
		   const std::vector<int>& idx0_, const std::vector<int>& idxK_);
    virtual ~NonAnticipCons();

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    
  protected:
    OptVariablesBlock *pg0, *pgK;
    int *idx0, *idxK;
    int* J_nz_idxs; 
  };

  /////////////////////////////////////////////////////////////////////////////////
  // AGC smoothing using complementarity function
  // 
  // p0 + alpha*deltak - pk - gb * rhop + gb * rhom = 0
  // rhop, rhom >=0
  // -r <= (pk-Pub)/gb * rhop <= r
  // -r <= (pk-Plb)/gb * rhom <= r
  //
  // when r=0, the last two constraints are enforced as equalities == 0
  // scaling parameter gb = generation band = Pub-Plb
  //
  // also, a penalty objective term can be added
  // min M * [ rhop*(Pub-pk)/gb + rhom*(pk-Plb)/gb ];
  /////////////////////////////////////////////////////////////////////////////////
  class AGCComplementarityCons : public OptConstraintsBlock
  {
  public:
    AGCComplementarityCons(const std::string& id_, int numcons,
			   OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, OptVariablesBlock* deltaK_,
			   const std::vector<int>& idx0_, const std::vector<int>& idxK_,
			   const std::vector<double>& Plb, const std::vector<double>& Pub, 
			   const std::vector<double>& G_alpha_,
			   const double& r_,
			   bool add_penalty_obj=false, const double& bigM=0);
    virtual ~AGCComplementarityCons();

    OptVariablesBlock* get_rhop() { return rhop; }
    OptVariablesBlock* get_rhom() { return rhom; }
    void compute_rhos(OptVariablesBlock* rp, OptVariablesBlock* rm);


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
    virtual std::vector<OptVariablesBlock*> create_multiple_varsblocks();
    virtual OptObjectiveTerm* create_objterm();
    double *gb; //size n/3 = nGenAGC
  protected:
    OptVariablesBlock *p0, *pk, *deltak, *rhop, *rhom;
    int *idx0, *idxk; //size n/3 = nGenAGC
    double r;
    
    double *Plb, *Pub; //size n/3 = nGenAGC
    const double* G_alpha; //size ngen base case, accessed via idx0
    int* J_nz_idxs; 
    int* H_nz_idxs;
  };

  /////////////////////////////////////////////////////////////////////////////////
  // AGC smoothing using complementarity function
  // 
  // v - vk + alpha*deltak -  nup + num = 0
  // nup, num >=0
  // -r <= (qk-Qub)/gb * nup <= r
  // -r <= (qk-Qlb)/gb * num <= r
  //
  // when r=0, the last two constraints are enforced as equalities == 0
  // scaling parameter gb = generation band = Qub-Qlb
  //
  // also, a big-M penalty objective term can be added penalize 
  // -(qk-Qub)/gb * nup and  (qk-Qlb)/gb * num
  /////////////////////////////////////////////////////////////////////////////////
  class PVPQComplementarityCons : public OptConstraintsBlock
  {
  public:
    PVPQComplementarityCons(const std::string& id_, int numcons,
			    OptVariablesBlock* v0_, OptVariablesBlock* vK_, OptVariablesBlock* qK_,
			    const std::vector<int>& idx0_, const std::vector<int>& idxK_,
			    const std::vector<double>& Qlb, const std::vector<double>& Qub, 
			    const double& r_,
			    bool add_penalty_obj=false, const double& bigM=0);
    virtual ~PVPQComplementarityCons();

    OptVariablesBlock* get_nup() { return nup; }
    OptVariablesBlock* get_num() { return num; }
    void compute_nus(OptVariablesBlock* np, OptVariablesBlock* nm);


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
    virtual std::vector<OptVariablesBlock*> create_multiple_varsblocks();
    virtual OptObjectiveTerm* create_objterm();
    double *gb; //size n/3 = nGenAGC
  protected:
    OptVariablesBlock *v0, *vk, *qk, *nup, *num;
    int *idx0, *idxk; //size n/3 = nGenAGC
    double r;
    
    double *Qlb, *Qub; //size n/3 = nGenAGC
    int* J_nz_idxs; 
    int* H_nz_idxs;
  };

}//end namespace
#endif

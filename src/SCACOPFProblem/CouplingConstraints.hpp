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
  // simplified AGC constraints
  // p0 + alpha*deltak - pk = 0 
  // these constraints are enforced only for AGC generators
  //  - idx0 are the indexes of AGC generators in p0
  //  - idxk are the indexes of AGC generators in pk
  /////////////////////////////////////////////////////////////////////////////////
  class AGCSimpleCons : public OptConstraintsBlock
  {
  public:
    AGCSimpleCons(const std::string& id_, int numcons,
		  OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, OptVariablesBlock* deltaK_,
		  const std::vector<int>& idx0_, const std::vector<int>& idxK_,
		  const std::vector<double>& G_alpha_);

    virtual ~AGCSimpleCons();

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body);
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
    
  protected:
    OptVariablesBlock *pg0, *pgK, *deltaK;
    int *idx0, *idxK;
    const double* G_alpha; //size ngen base case, accessed via idx0
    int* J_nz_idxs; 
  };

  //this time with pg0 fixed
  class AGCSimpleCons_pg0Fixed : public AGCSimpleCons
  {
  public:
    AGCSimpleCons_pg0Fixed(const std::string& id_, int numcons,
		  OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, OptVariablesBlock* deltaK_,
		  const std::vector<int>& idx0_, const std::vector<int>& idxK_,
		  const std::vector<double>& G_alpha_)
      : AGCSimpleCons(id_, numcons, pg0_, pgK_, deltaK_, idx0_, idxK_, G_alpha_)
    { }

    virtual ~AGCSimpleCons_pg0Fixed() {};

    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M);
    virtual int get_Jacob_nnz();
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij);
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
			   bool add_penalty_obj=false, const double& bigM=0,
			   bool fix_p_g0=false);
    virtual ~AGCComplementarityCons();

    OptVariablesBlock* get_p_g0() { return p0; }
    OptVariablesBlock* get_rhop() { return rhop; }
    OptVariablesBlock* get_rhom() { return rhom; }
    void compute_rhos(OptVariablesBlock* rp, OptVariablesBlock* rm);

    void update_smoothing(const double& val);

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
    bool fixed_p_g0;
  };

  ///////////////////////////////////////////////////////////////////////////////////
  // PVPQ smoothing using complementarity function
  // 
  // v[n] - vk[n] + sum(qk[g]:g=idxs_gen[n]) - nup[n]+num[n] = 0, for all n=idxs_bus
  // nup, num >=0
  // -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
  // -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
  //
  // when r=0, the last two constraints are enforced as equalities == 0
  // scaling parameter gb[n] = generation band at the bus = Qub[n]-Qlb[n]
  //
  // also, a big-M penalty objective term can be added to penalize 
  // -(qk-Qub)/gb * nup and  (qk-Qlb)/gb * num
  ///////////////////////////////////////////////////////////////////////////////////
  class PVPQComplementarityCons : public OptConstraintsBlock
  {
  public:
    PVPQComplementarityCons(const std::string& id_, int numcons,
			    OptVariablesBlock* v0_, OptVariablesBlock* vK_, 
			    OptVariablesBlock* qK_,
			    const std::vector<int>& idxs_bus_,
			    const std::vector<std::vector<int> >& idxs_gen_, 
			    const std::vector<double>& Qlb, const std::vector<double>& Qub, 
			    const double& r_,
			    bool add_penalty_obj=false, const double& bigM=0,
			    bool fix_vn0=false);
    virtual ~PVPQComplementarityCons();

    OptVariablesBlock* get_nup() { return nup; }
    OptVariablesBlock* get_num() { return num; }
    void compute_nus(OptVariablesBlock* np, OptVariablesBlock* nm);

    void update_smoothing(const double& val);

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
    int *idxs_bus; // size n/3 = nPVPQBuses
    // size n/3, each elem has at least one elem
    std::vector<std::vector<int> > idxs_gen; 
    double r;
    bool fixed_vn0;

    double *Qlb, *Qub; //size n/3 
    int* J_nz_idxs; 
    int* H_nz_idxs;
  };

}//end namespace
#endif

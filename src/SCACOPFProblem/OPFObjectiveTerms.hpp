#ifndef OPF_OBJTERMS
#define OPF_OBJTERMS

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include <string>
#include <cassert>
#include <vector>

#include <cmath>

namespace gollnlp {

  //////////////////////////////////////////////////////////////////////////////
  // Production cost piecewise linear objective
  // min sum_g( sum_h CostCi[g][h]^T t[g][h])
  // constraints (handled outside) are
  //   t>=0, sum_h t[g][h]=1
  //   p_g[g] - sum_h CostPi[g][h]*t[g][h] =0
  //////////////////////////////////////////////////////////////////////////////
  class PFProdCostPcLinObjTerm : public OptObjectiveTerm {
  public: 
    //Gidx contains the indexes (in d.G_Generator) of the generator participating
    PFProdCostPcLinObjTerm(const std::string& id_, OptVariablesBlock* t_h_, 
			   const std::vector<int>& Gidx_,
			   const std::vector<std::vector<double> >& G_CostCi);
    virtual ~PFProdCostPcLinObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);
    //Hessian is all zero

  private:
    friend class PFProdCostAffineCons;
    std::string id;
    OptVariablesBlock* t_h;
    int* Gidx;
    double *CostCi;
    int ngen;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Slack penalty piecewise linear objective
  // min sum_i( sum_h P[i][h] sigma_h[i][h])
  // constraints (handled outside) are
  //   0<= sigma[i][h] <= Pseg_h, 
  //   slacks[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
  //////////////////////////////////////////////////////////////////////////////
  class PFPenaltyPcLinObjTerm : public OptObjectiveTerm {
  public: 
    //Gidx contains the indexes (in d.G_Generator) of the generator participating
    PFPenaltyPcLinObjTerm(const std::string& id_, 
			  OptVariablesBlock* sigma_,
			  const std::vector<double>& pen_coeff,
			  const double& obj_weight,
			  const double& slacks_rescale=1.);
    virtual ~PFPenaltyPcLinObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);
    //Hessian is all zero

  private:
    std::string id;
    OptVariablesBlock* sigma;
    double weight;
    double P1, P2, P3;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Slack penalty quadratic objective
  // q(0)=0
  // q'(0) = P1 (slope of the piecewise linear penalty at 0)
  // q(s1+s2) = piecewise_linear_penalty(s1+s2) (=P1*s1+P2*s2)
  // 
  // q(x) = a*x^2 + b*x, where
  // b = P1, a=(P2-P1)/(s1+s2)^2
  //
  // Assumed is that the piecewise linear penalty is defined over 3 segments
  // [0, s1], [s1, s1+s2], [s1+s2, s1+s2+s3] with slopes P1, P2, P3
  //
  // An objective weight is applied and slacks are subject to rescaling
  class PFPenaltyQuadrApproxObjTerm : public OptObjectiveTerm {
  public: 
    //Gidx contains the indexes (in d.G_Generator) of the generator participating
    PFPenaltyQuadrApproxObjTerm(const std::string& id_, 
				OptVariablesBlock* slacks_,
				const std::vector<double>& pen_coeff,
				const std::vector<double>& pen_segm,
				const double& obj_weight,
				const double& slacks_rescale=1.);
    virtual ~PFPenaltyQuadrApproxObjTerm();
    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val);
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad);

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const double& obj_factor,
			       const int& nnz, int* i, int* j, double* M);

    virtual int get_HessLagr_nnz();
    // (i,j) entries in the HessLagr to which this term contributes to
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij);
  private:
    std::string id;
    OptVariablesBlock* x;
    double a,b;
    double weight, f;
    //keep the index for each nonzero elem in the Hessian that this constraints block contributes to
    int *H_nz_idxs;
    double aux;
  };

  //for 0.5||x||^2 -> to be used in testing
  class DummySingleVarQuadrObjTerm : public OptObjectiveTerm {
  public: 
    DummySingleVarQuadrObjTerm(const std::string& id, OptVariablesBlock* x_) 
      : OptObjectiveTerm(id), x(x_), H_nz_idxs(NULL)
    {};

    virtual ~DummySingleVarQuadrObjTerm() 
    {
      if(H_nz_idxs) 
	delete[] H_nz_idxs;
    }

    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
    {
      int nvars = x->n; double aux;
      for(int it=0; it<nvars; it++) {
	aux = x->xref[it]-1;
	obj_val += aux * aux * 0.5;
      }
      return true;
    }
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
    {
      double* g = grad + x->index;
      for(int it=0; it<x->n; it++) 
	g[it] += x->xref[it]-1;
      return true;
    }
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const double& obj_factor,
			       const int& nnz, int* i, int* j, double* M)
    {
      if(NULL==M) {
	int idx, row;
	for(int it=0; it<x->n; it++) {
	  idx = H_nz_idxs[it]; 
	  if(idx<0) return false;
	  i[idx] = j[idx] = x->index+it;
	}
      } else {
	for(int it=0; it<x->n; it++) {
	  assert(H_nz_idxs[it]>=0);
	  assert(H_nz_idxs[it]<nnz);
	  M[H_nz_idxs[it]] += obj_factor;
	}
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return x->n; }
    // (i,j) entries in the HessLagr to which this term contributes to
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    { 
      int nvars = x->n, i;
      if(NULL==H_nz_idxs)
	H_nz_idxs = new int[nvars];

      for(int it=0; it < nvars; it++) {
	i = x->index+it;
	vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
      }
      return true; 
    }

  private:
    OptVariablesBlock* x;
    //keep the index for each nonzero elem in the Hessian that this constraints block contributes to
    int *H_nz_idxs;
  };

  //
  // Penalty for contingency generators approximating a known positive (large) penalty 'f'>0 occured 
  // in the respective contingency with the generator in the basecase at 'p0'
  // 
  // Supports quadratic and linear penalty functions
  //  - quadratic penalty q(p): q(p0)=f and q(a)=0
  //    i. if p0 is negative:  a=min(a,gen_upper_bound)
  //   ii. if p0 is positive:  a=max(0,gen_lower_bound)
  // That is: q(p) = f / [(p0-a)^2] * (x-a)^2= c * (x-a)^2, where
  //   c = f/[(p0-a)^2]
  //
  //  - linear penalty    l(p):  l(p0)=f and l(a)=0
  //    i. if p0 is negative: a=min(a,gen_upper_bound)
  //   ii. if p0 is positive: a=max(0,gen_lower_bound)
  // That is: l(p) = f * (x-a) / (p0-a) = c*x + d, where
  //   c = f / (p0-a) and d = - f*a / (p0-a)

  class GenerKPenaltyObjTerm : public OptObjectiveTerm {
  public:
    GenerKPenaltyObjTerm(const std::string& id, OptVariablesBlock* x_) 
      : OptObjectiveTerm(id), x(x_), H_nnz(0), H_nz_idxs(NULL)
    {}
    virtual ~GenerKPenaltyObjTerm()
    {
      delete[] H_nz_idxs;
    }

    void add_linear_penalty(const int& idx_gen, const double& p0, const double& f_pen, 
			    const double& lb, const double& ub)
    {
      assert(f_pen>1e-2); assert(p0<=ub && p0>=lb); assert(fabs(p0)>1e-6); assert(idx_gen>=0 && idx_gen<x->n);
      
      if(f_pen<=1e-6) return;

      double a = std::max(0., lb);
      if(p0<0) a = std::min(0., ub);
     
      double aux = p0-a;   assert(fabs(aux)>1e-6);
      aux = f_pen/aux;
      ci_lin.push_back(CoeffIdxLin(aux, -a*aux, idx_gen));
    }
    void add_quadr_penalty(const int& idx_gen, const double& p0, const double& f_pen, 
			   const double& lb, const double& ub)
    {
#ifdef DEBUG
      if(p0>ub || p0<lb) {
	printf(" !!!!!!!!!!!  %.5e < %.5e < %.5e \n", lb, p0, ub);
	//p0 = std::min(p0,ub);
	//p0 = std::max(p0,lb);
      }
#endif
      assert(f_pen>1e-2); assert(p0<=ub && p0>=lb); 
      //assert(fabs(p0)>1e-6); 
      assert(idx_gen>=0 && idx_gen<x->n);
      
      if(f_pen<=1e-2) return;

      double a = std::max(0., lb);
      if(p0<0) a = std::min(0., ub);

      double aux = p0-a;   

      if(fabs(aux)<=1e-6) {
	printf("[warning] idx_gen=%d cannot be penalized effectively: lb=%g ub=%g p0=%g f_pen=%g\n", 
	       idx_gen, lb, ub, p0, f_pen);
	return;
      }

      assert(fabs(aux)>1e-6);
      double c = f_pen/(aux*aux); 

      ci_qua.push_back(CoeffIdxQua(c, a, idx_gen));
    }
    void remove_penalty(const int& idx_gen)
    {
      while(true) {
	auto it = ci_lin.begin();
	for(; it!=ci_lin.end(); ++it) if(it->idx==idx_gen) break;
	if(it!=ci_lin.end()) 
	  ci_lin.erase(it);
	else 
	  break;
      }
      while(true) {
	auto it = ci_qua.begin();
	for(; it!=ci_qua.end(); ++it) if(it->idx==idx_gen) break;
	if(it!=ci_qua.end()) 
	  ci_qua.erase(it);
	else 
	  break;
      }
    }

    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
    {  
      // c*x + d
      for(CoeffIdxLin& e: ci_lin) obj_val += (x->xref[e.idx] * e.c + e.d);

      //c * (x-a)^2
      double aux;
      for(CoeffIdxQua& e: ci_qua) {
	aux = x->xref[e.idx]-e.a; 
	obj_val += e.c * aux * aux;
      }
      return true;
    }
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
    {
      double* g = grad + x->index;
      for(CoeffIdxLin& e: ci_lin) g[e.idx] += e.c;
      for(CoeffIdxQua& e: ci_qua) g[e.idx] += 2. * e.c * (x->xref[e.idx]-e.a);
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const double& obj_factor,
			       const int& nnz, int* i, int* j, double* M)
    {
      assert(ci_qua.size() == H_nnz);
      if(NULL==M) {
	int idx, row;
	for(int it=0; it<ci_qua.size(); it++) {
	  idx = H_nz_idxs[it]; 
	  if(idx<0) {assert(false); return false; }
	  i[idx] = j[idx] = x->index + ci_qua[it].idx;
	}
      } else {
	for(int it=0; it<ci_qua.size(); it++) {
	  assert(H_nz_idxs[it]>=0);
	  assert(H_nz_idxs[it]<nnz);
	  M[H_nz_idxs[it]] += obj_factor * 2.* ci_qua[it].c;
	}
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return ci_qua.size(); }
    // (i,j) entries in the HessLagr to which this term contributes to
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    { 
      int nnz = ci_qua.size(), i;
      
      if(nnz!=H_nnz) {
	delete [] H_nz_idxs;
	H_nnz = nnz;
	H_nz_idxs = new int[H_nnz];
      }

      int it=0;
      for(CoeffIdxQua& e: ci_qua) {
	i = x->index + e.idx;
	vij.push_back(OptSparseEntry(i,i, H_nz_idxs+it));
	assert(it<H_nnz);
	it++;
      }
      assert(it==H_nnz);
      return true; 
    }

  protected:
    OptVariablesBlock* x;
    struct CoeffIdxLin 
    { 
      CoeffIdxLin(const double& c_, const double& d_, const int& i) : c(c_), d(d_), idx(i) {};
      double c, d; int idx;  
    };
    struct CoeffIdxQua
    { 
      CoeffIdxQua(const double& c_, const double& a_, const int& i) : c(c_), a(a_), idx(i) {};
      double c, a; int idx;  
    };

    std::vector<CoeffIdxLin> ci_lin;
    std::vector<CoeffIdxQua> ci_qua;
    int H_nnz;
    int *H_nz_idxs;
  };

  class TransmKPenaltyObjTerm : public OptObjectiveTerm {
   public:
     TransmKPenaltyObjTerm(const std::string& id, 
			   OptVariablesBlock* p1_, OptVariablesBlock* q1_, 
			   OptVariablesBlock* p2_, OptVariablesBlock* q2_) 
      : OptObjectiveTerm(id)
    {
      pen = new GenerKPenaltyObjTerm*[4];
      pen[p1] = new GenerKPenaltyObjTerm(id+"_pli1", p1_);
      pen[q1] = new GenerKPenaltyObjTerm(id+"_qli1", q1_);
      pen[p2] = new GenerKPenaltyObjTerm(id+"_pli2", p2_);
      pen[q2] = new GenerKPenaltyObjTerm(id+"_qli2", q2_);
    }
    virtual ~TransmKPenaltyObjTerm()
    {
      for(int i=0; i<4; i++) delete pen[i];
      delete [] pen;
    }
    
    void add_penalty(const int& idx_line, 
		     const double& pli10, const double& qli10, 
		     const double& pli20, const double& qli20, 
		     const double& f_pen) 
    {
      double pq1_0 = pli10*pli10 + qli10*qli10;
      pen[p1]->add_quadr_penalty(idx_line, pq1_0, f_pen, 0., 1e+10);
      pen[q1]->add_quadr_penalty(idx_line, pq1_0, f_pen, 0., 1e+10);

      double pq2_0 = pli20*pli20 + qli20*qli20;
      pen[p2]->add_quadr_penalty(idx_line, pq2_0, f_pen, 0., 1e+10);
      pen[q2]->add_quadr_penalty(idx_line, pq2_0, f_pen, 0., 1e+10);
    }
    void remove_penalty(const int& idx_line)
    {
      for(int i=0; i<4; i++) pen[i]->remove_penalty(idx_line);
    }

    virtual bool eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
    { 
      for(int i=0; i<4; i++) pen[i]->eval_f(vars_primal, new_x, obj_val);
      return true;
    }
    virtual bool eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
    {
      for(int i=0; i<4; i++) pen[i]->eval_grad(vars_primal, new_x, grad);
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const double& obj_factor,
			       const int& nnz, int* ii, int* jj, double* M)
    {
      
      for(int i=0; i<4; i++) 
	pen[i]->eval_HessLagr(vars_primal, new_x, obj_factor, nnz, ii, jj, M);
      return true;
    }

    virtual int get_HessLagr_nnz() { 
      int nnz=pen[0]->get_HessLagr_nnz();
      for(int i=1; i<4; i++)  nnz += pen[i]->get_HessLagr_nnz();
      return nnz; 
    }
    // (i,j) entries in the HessLagr to which this term contributes to
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    { 
      for(int i=0; i<4; i++) pen[i]->get_HessLagr_ij(vij);
      return true; 
    }
  protected:
    enum {p1=0, q1, p2, q2};
    GenerKPenaltyObjTerm** pen;
  };
} //end namespace

#endif

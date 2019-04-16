#ifndef OPF_VARIABLES
#define OPF_VARIABLES

#include "OptProblem.hpp"
#include "SCACOPFData.hpp"

#include "blasdefs.hpp"
#include "goUtils.hpp"

#include <cstring>
#include <cmath>


#include "OPFObjectiveTerms.hpp"

namespace gollnlp {

  // pq == A*vi^2 + B*vi*vj*cos(thetai - thetaj + Theta) + C*vi*vj*sin(thetai - thetaj + Theta)
  class PFConRectangular : public OptConstraintsBlock
  {
  public:
    PFConRectangular(const std::string& id_, int numcons,
		     OptVariablesBlock* pq_, 
		     OptVariablesBlock* v_n_, 
		     OptVariablesBlock* theta_n_,
		     const std::vector<int>& Nidx1, //T_Nidx or L_Nidx indexes
		     const std::vector<int>& Nidx2)
    //normally we would also have these arguments in the constructor, but want to avoid 
    //excessive copying and the caller needs to update this directly
    //const std::vector<double>& A,
    //const std::vector<double>& B,
    //const std::vector<double>& C,
    //const std::vector<double>& T, //Theta
    //  const SCACOPFData& d_)
      : OptConstraintsBlock(id_, numcons), pq(pq_), v_n(v_n_), theta_n(theta_n_)
    {
      H_nz_idxs = J_nz_idxs = NULL;
      assert(n==Nidx1.size());
      assert(n==Nidx2.size());

      A = new double[n];
      B = new double[n];
      C = new double[n];
      T = new double[n];
      E_Nidx1 = new int[n];
      memcpy(E_Nidx1, Nidx1.data(), n*sizeof(int));

      E_Nidx2 = new int[n];
      memcpy(E_Nidx2, Nidx2.data(), n*sizeof(int));

      //rhs
      for(int i=0; i<n; i++) lb[i]=0.;
      DCOPY(&n, lb, &ione, ub, &ione);
    }
    virtual ~PFConRectangular() 
    {
      delete[] A; 
      delete[] B;
      delete[] C;
      delete[] T;
      delete[] E_Nidx1;
      delete[] E_Nidx2;
      delete[] J_nz_idxs;
      delete[] H_nz_idxs;
    };

    //accessers
    inline double* get_A() { return A;}
    inline double* get_B() { return B;}
    inline double* get_C() { return C;}
    inline double* get_T() { return T;}


    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
    {
      // - pq[i] + A[i] * v[Nidx1[i]]^2 
      //         + B[i] * v[Nidx1[i]] * v[Nidx2[i]] * cos(theta[Nidx1[i]] - theta[Nidx2[i]] + Theta[i]) +
      //         + C[i] * v[Nidx1[i]] * v[Nidx2[i]] * sin(theta[Nidx1[i]] - theta[Nidx2[i]] + Theta[i])
      double v1, v1v2, ththT, *itbody=body+this->index;
      for(int i=0; i<n; i++) {
	ththT = theta_n->xref[E_Nidx1[i]] - theta_n->xref[E_Nidx2[i]] + T[i];
	v1 = v_n->xref[E_Nidx1[i]];
	v1v2 = v1 * v_n->xref[E_Nidx2[i]];
	*itbody++ +=  A[i]*v1*v1 + B[i]*v1v2*cos(ththT) + C[i]*v1v2*sin(ththT) - pq->xref[i];
      }
      assert(body+this->index+n == itbody);
      return true;
    }
    
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M)
    {
      int row, *itnz=J_nz_idxs;
      if(NULL==M) {
	for(int it=0; it<n; it++) {
	  if(itnz[0]<0 || itnz[1]<0 || itnz[2]<0 || itnz[3]<0 || itnz[4]<0) return false;
	  row = this->index+it;

	  i[*itnz]=row; j[*itnz]=pq->index+it; itnz++;
	  i[*itnz]=row; j[*itnz]=v_n->index+E_Nidx1[it]; itnz++;
	  i[*itnz]=row; j[*itnz]=v_n->index+E_Nidx2[it]; itnz++;
	  i[*itnz]=row; j[*itnz]=theta_n->index+E_Nidx1[it]; itnz++;
	  i[*itnz]=row; j[*itnz]=theta_n->index+E_Nidx2[it]; itnz++;
	}
	assert(J_nz_idxs + 5*n == itnz);
      } else {
	//values
	double v1, v2, v1v2, aux1, sinval, cosval, Bv1v2sin, Cv1v2cos;
	for(int it=0; it<n; it++) {
	  M[*itnz] += -1; itnz++; //w.r.t. pq[i]
	  
	  aux1=theta_n->xref[E_Nidx1[it]]-theta_n->xref[E_Nidx2[it]]+T[it];
	  sinval = sin(aux1); cosval=cos(aux1);
	  v1 = v_n->xref[E_Nidx1[it]];
	  v2 = v_n->xref[E_Nidx2[it]];
	  v1v2=v1*v2;
	  Bv1v2sin = B[it]*v1v2*sinval;
	  Cv1v2cos = C[it]*v1v2*cosval;

	  aux1 = B[it]*cosval + C[it]*sinval;

	  //w.r.t. v1 := v[Nidx1[i]]
	  // 2*A*v1 + B*v2*cos + C*v2*sin 	  
	  M[*itnz] += 2*A[it]*v1 + v2*aux1; //M[*itnz] += 2*A[it]*v1 + B[it]*v2*cosval + C[it]*v2*sinval;
	  itnz++;
 
	  //w.r.t. v2 := v[Nidx2[i]]
	  // B*v1*cos + C*v1*sin
	  M[*itnz] += v1*aux1; //M[*itnz] += B[it]*v1*cosval + C[it]*v1*sinval;
	  itnz++;

	  aux1 = Bv1v2sin - Cv1v2cos;
	  //w.r.t. theta1 = theta[Nidx1[i]]
	  // -B*v1*v2*sin + C*v1*v2*cos
	  M[*itnz] += -aux1; itnz++;

	  //w.r.t theta2 = theta[Nidx2[i]]
	  // B*v1*v2*sin - C*v1*v2*cos
	  M[*itnz] +=  aux1; itnz++;
	}
	assert(J_nz_idxs + 5*n == itnz);
      }
      return true;
    }
    virtual int get_Jacob_nnz(){ return 5*n; }
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
    {
      if(n<=0) return true;

      if(!J_nz_idxs) 
	J_nz_idxs = new int[get_Jacob_nnz()];

      int row, *itnz=J_nz_idxs;
      for(int it=0; it<n; it++) {
	row=this->index+it;
	vij.push_back(OptSparseEntry(row, pq->index+it, itnz++)); // pq[i]
	vij.push_back(OptSparseEntry(row, v_n->index+E_Nidx1[it], itnz++)); //v[Nidx1[i]]
	vij.push_back(OptSparseEntry(row, v_n->index+E_Nidx2[it], itnz++)); //v[Nidx2[i]]
	vij.push_back(OptSparseEntry(row, theta_n->index+E_Nidx1[it], itnz++)); //theta[Nidx1[i]]
	vij.push_back(OptSparseEntry(row, theta_n->index+E_Nidx2[it], itnz++)); //theta[Nidx2[i]]
      }
      assert(J_nz_idxs + 5*n == itnz);
      return true;
    } 
    
    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M)
    {
      if(NULL==M) {
	int row, i,j, aux, *itnz=H_nz_idxs;
	for(int it=0; it<n; it++) {
	  // --- v1
	  row=v_n->index+E_Nidx1[it];
	  ia[*itnz] = ja[*itnz] = row; itnz++; //(v1,v1)

	  i = uppertr_swap(row,j=v_n->index+E_Nidx2[it],aux);  //(v1,v2)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;
	
	  i = uppertr_swap(row,j=theta_n->index+E_Nidx1[it],aux); //(v1,theta1)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;

	  i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux); //(v1,theta2)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;

	  // --- v2
	  row=v_n->index+E_Nidx2[it];
	
	  i = uppertr_swap(row,j=theta_n->index+E_Nidx1[it],aux); //(v2,theta1)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;
	  
	  i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux);  //(v2,theta2)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;

	  // --- theta1
	  row=theta_n->index+E_Nidx1[it]; //(theta1,theta1)
	  ia[*itnz]=ja[*itnz]=row; itnz++;

	  i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux); //(theta1,theta2)
	  ia[*itnz]=i; ja[*itnz]=j; itnz++;
	
	  // --- theta2
	  row=theta_n->index+E_Nidx2[it]; //(theta1,theta1)
	  ia[*itnz]=ja[*itnz]=row; itnz++;
	}
	assert(H_nz_idxs + 9*n == itnz);

      } else {
	const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
	assert(lambda!=NULL);
	assert(lambda->n==n);
	int row, i,j, *itnz=H_nz_idxs;
	double v1, v2, aux2, aux3, Bsin, Bcos, Csin, Ccos, lam;
	for(int it=0; it<n; it++) {
	  lam = lambda->xref[it];
	  aux3=theta_n->xref[E_Nidx1[it]]-theta_n->xref[E_Nidx2[it]]+T[it];
	  v1 = v_n->xref[E_Nidx1[it]];
	  v2 = v_n->xref[E_Nidx2[it]];
	  //sinval = sin(aux1); cosval=cos(aux1);
	  aux2 = sin(aux3); aux3=cos(aux3);
	  Bsin = B[it]*aux2; Bcos=B[it]*aux3; Csin=C[it]*aux2; Ccos=C[it]*aux3;

	  //Jac w.r.t. v1 := v[Nidx1[i]]  -> 2*A*v1 + B*v2*cos(theta1-theta2+T) + C*v2*sin(theta1-theta2+T) 	     
	  //Hess w.r.t. 
	  // (v1,v1) : 2*A   
	  M[*itnz] += 2*A[it]*lam; itnz++;
	  // (v1,v2) : B*cos + C*sin
	  M[*itnz] += (Bcos + Csin)*lam; itnz++;
	  // (v1, theta1) : -B*v2*sin + C*v2*cos
	  aux3=v2*(Bsin-Ccos)*lam;
	  M[*itnz] -= aux3; itnz++;
	  // (v1,theta2) : B*v2*sin - C*v2*cos
	  M[*itnz] += aux3; itnz++;
	  
	  //Jac w.r.t. v2 := v[Nidx2[i]] -> B*v1*cos + C*v1*sin
	  //Hess w.r.t.  (v2, v2)=0  
	  // (v2,theta1) : -B*v1*sin + C*v1*cos 
	  aux3 = lam*v1*(Bsin - Ccos);
	  M[*itnz] -= aux3; itnz++;
	  // (v2,theta2) : B*v1*sin - C*v1*cos
	  M[*itnz] += aux3; itnz++;
	  
	  //Jac w.r.t. theta1 = theta[Nidx1[i]] ->  -B*v1*v2*sin + C*v1*v2*cos
	  //Hess w.r.t. 
	  // (theta1, theta1) : -B*v1*v2*cos - C*v1*v2*sin
	  aux3 = lam*v1*v2*(Bcos+Csin);
	  M[*itnz] -= aux3; itnz++;
	  // (theta1, theta2) :  B*v1*v2*cos + C*v1*v2*sin
	  M[*itnz] += aux3; itnz++;
	  
	  //Jac w.r.t theta2 = theta[Nidx2[i]] -> B*v1*v2*sin - C*v1*v2*cos
	  //Hess w.r.t. 
	  // (theta2,theta2): -B*v1*v2*cos - C*v1*v2*sin
	  M[*itnz] -= aux3; itnz++;
	}
	assert(H_nz_idxs + 9*n == itnz);
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return 9*n; }

    // (i,j) entries in the HessLagr to which the implementer's contributes to
    // this is only called once
    // push_back in vij 
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    {
      if(n==0) return true;
      
      if(NULL==H_nz_idxs) {
	H_nz_idxs = new int[get_HessLagr_nnz()];
      }

      int row, i,j, aux, *itnz=H_nz_idxs;
      for(int it=0; it<n; it++) {
	// --- v1
	row=v_n->index+E_Nidx1[it];
	vij.push_back(OptSparseEntry(row, row, itnz++)); //(v1,v1)

	i = uppertr_swap(row,j=v_n->index+E_Nidx2[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(v1,v2)

	i = uppertr_swap(row,j=theta_n->index+E_Nidx1[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(v1,theta1)

	i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(v1,theta2)

	// --- v2
	row=v_n->index+E_Nidx2[it];
	i = uppertr_swap(row,j=theta_n->index+E_Nidx1[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(v2,theta1)

	i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(v2,theta2)

	// --- theta1
	row=theta_n->index+E_Nidx1[it];
	vij.push_back(OptSparseEntry(row, row, itnz++)); //(theta1,theta1)

	i = uppertr_swap(row,j=theta_n->index+E_Nidx2[it],aux); 
	vij.push_back(OptSparseEntry(i, j, itnz++)); //(theta1,theta2)
	
	// --- theta2
	row=theta_n->index+E_Nidx2[it];
	vij.push_back(OptSparseEntry(row, row, itnz++)); //(theta2,theta2)
      }						
      assert(H_nz_idxs + 9*n == itnz);
      return true;
    }
  protected:
    const OptVariablesBlock *pq, *v_n, *theta_n;
    int* J_nz_idxs;
    int* H_nz_idxs;

    double *A, *B, *C, *T;
    int *E_Nidx1, *E_Nidx2;
  };


  // sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
  // sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
  // sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - pslackp_n[n] + pslackm_n[n] = N[:Pd][n])
  class PFActiveBalance  : public OptConstraintsBlock
  {
  public:
    PFActiveBalance(const std::string& id_, int numcons,
		    OptVariablesBlock* p_g_, 
		    OptVariablesBlock* v_n_,
		    OptVariablesBlock* p_li1_,
		    OptVariablesBlock* p_li2_,
		    OptVariablesBlock* p_ti1_,
		    OptVariablesBlock* p_ti2_,
		    const SCACOPFData& d_)
      : OptConstraintsBlock(id_, numcons), 
	p_g(p_g_), v_n(v_n_), p_li1(p_li1_), p_li2(p_li2_), p_ti1(p_ti1_), p_ti2(p_ti2_), d(d_), pslack_n(NULL)//, pslackm_n(NULL)
    {
      assert(d.N_Pd.size()==n);
      //rhs
      memcpy(lb, d.N_Pd.data(), n*sizeof(double));
      DCOPY(&n, lb, &ione, ub, &ione);
      J_nz_idxs = NULL;
      H_nz_idxs = NULL;
    }
    virtual ~PFActiveBalance() 
    {
      delete[] J_nz_idxs;
      delete[] H_nz_idxs;
    };

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
    {
      assert(pslack_n); //assert(pslackm_n);
      body += this->index;
      double* slacks = const_cast<double*>(pslack_n->xref);
      DAXPY(&n, &dminusone, slacks,   &ione, body, &ione);
      DAXPY(&n, &done,      slacks+n, &ione, body, &ione);

      const double *NGsh=d.N_Gsh.data();
      for(int i=0; i<n; i++) {
	*body -= NGsh[i] * v_n->xref[i] * v_n->xref[i];
	body++;
      }
      body -= n;
      const int *Gn; int nGn;
      for(int i=0; i<n; i++) {
	nGn = d.Gn[i].size(); Gn = d.Gn[i].data();
	for(int ig=0; ig<nGn; ig++) 
	  *body += p_g->xref[Gn[ig]];
	body++;
      }
      body -= n;
      {
	// - sum(p_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
	const int *Lidxn; int nLidx;
	for(int i=0; i<n; i++) {
	  Lidxn=d.Lidxn1[i].data(); nLidx = d.Lidxn1[i].size();
	  for(int ilix=0; ilix<nLidx; ilix++)
	    *body -= p_li1->xref[Lidxn[ilix]];
	  body++;
	}
	body -= n;
	// - sum(p_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
	for(int i=0; i<n; i++) {
	  Lidxn=d.Lidxn2[i].data(); nLidx = d.Lidxn2[i].size();
	  for(int ilix=0; ilix<nLidx; ilix++)
	    *body -= p_li2->xref[Lidxn[ilix]];
	  body++;
	}
	body -= n;
      }
      {
	const int *Tidxn; int nTidx;
	// - sum(p_ti1[Tidxn1[n][tix]] for tix=1:length(Tidxn1[n]))
	for(int i=0; i<n; i++) {
	  Tidxn=d.Tidxn1[i].data(); nTidx = d.Tidxn1[i].size();
	  for(int itix=0; itix<nTidx; itix++)
	    *body -= p_ti1->xref[Tidxn[itix]];
	  body++;
	}
	body -= n;
	// - sum(p_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
	for(int i=0; i<n; i++) {
	  Tidxn=d.Tidxn2[i].data(); nTidx = d.Tidxn2[i].size();
	  for(int itix=0; itix<nTidx; itix++)
	    *body -= p_ti2->xref[Tidxn[itix]];
	  body++;
	}
      }
      return true;
    }
    // sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
    // sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
    // sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - pslackp_n[n] + pslackm_n[n] = N[:Pd][n])
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M)
    {
#ifdef DEBUG
      int nnz_loc=get_Jacob_nnz();
#endif
      int row, *itnz=J_nz_idxs;
      if(NULL==M) {
	for(int it=0; it<n; it++) {
	  row = this->index+it;
	  //v_n
	  i[*itnz]=row; j[*itnz]=v_n->index+it; itnz++;
	  //p_g
	  for(auto g: d.Gn[it]) { i[*itnz]=row; j[*itnz]=p_g->index+g; itnz++; }
	  //p_li1 and p_l2
	  for(auto l: d.Lidxn1[it]) { i[*itnz]=row; j[*itnz]=p_li1->index+l; itnz++; }	  
	  for(auto l: d.Lidxn2[it]) { i[*itnz]=row; j[*itnz]=p_li2->index+l; itnz++; }
	  //p_ti1 and p_ti2
	  for(auto t: d.Tidxn1[it]) { i[*itnz]=row; j[*itnz]=p_ti1->index+t; itnz++; }	  
	  for(auto t: d.Tidxn2[it]) { i[*itnz]=row; j[*itnz]=p_ti2->index+t; itnz++; }
 
	  //slacks
	  i[*itnz]=row; j[*itnz]=pslack_n->index+it;   itnz++;
	  i[*itnz]=row; j[*itnz]=pslack_n->index+it+n; itnz++;
	}
#ifdef DEBUG
	assert(J_nz_idxs + nnz_loc == itnz);
#endif
      } else {
	const double* Gsh=d.N_Gsh.data(); int sz;
	for(int it=0; it<n; it++) {
	  row = this->index+it;
	  M[*itnz] -= 2*v_n->xref[it]*Gsh[it];  itnz++; //vn
	  
	  sz = d.Gn[it].size();
	  for(int ig=0; ig<sz; ig++) { M[*itnz] += 1; itnz++; } //p_g

	  sz=d.Lidxn1[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_li_1
	  sz=d.Lidxn2[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_l2_1

	  sz=d.Tidxn1[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_ti_1
	  sz=d.Tidxn2[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_t2_1

	  //slacks
	  M[*itnz] -= 1; itnz++;
	  M[*itnz] += 1; itnz++;
	}
      }
      return true;
    }
    virtual int get_Jacob_nnz(){ 
      int nnz = 3*n; //slacks and v_n
      for(int i=0; i<n; i++) 
	nnz += d.Gn[i].size() + d.Lidxn1[i].size() + d.Lidxn2[i].size() + d.Tidxn1[i].size() + d.Tidxn2[i].size();
      return nnz; 
    }
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
    {
      if(n<=0) return true;

      int nnz = get_Jacob_nnz();
      if(!J_nz_idxs) 
	J_nz_idxs = new int[nnz];
#ifdef DEBUG
      int n_vij_in = vij.size();
#endif

      int row, *itnz=J_nz_idxs;
      for(int it=0; it<n; it++) {
	row = this->index+it;
	//v_n
	vij.push_back(OptSparseEntry(row, v_n->index+it, itnz++));

	//p_g
	for(auto g: d.Gn[it]) 
	  vij.push_back(OptSparseEntry(row, p_g->index+g, itnz++));

	//p_li1
	for(auto l: d.Lidxn1[it])
	  vij.push_back(OptSparseEntry(row, p_li1->index+l, itnz++));
	//p_li1
	for(auto l: d.Lidxn2[it])
	  vij.push_back(OptSparseEntry(row, p_li2->index+l, itnz++));
			
	//p_ti1
	for(auto t: d.Tidxn1[it])
	  vij.push_back(OptSparseEntry(row, p_ti1->index+t, itnz++));
	//p_ti1
	for(auto t: d.Tidxn2[it])
	  vij.push_back(OptSparseEntry(row, p_ti2->index+t, itnz++));
			
	//slacks
	vij.push_back(OptSparseEntry(row, pslack_n->index+it, itnz++));
	vij.push_back(OptSparseEntry(row, pslack_n->index+it+n, itnz++));
      }
      printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
      assert(nnz+n_vij_in==vij.size());
#endif
      assert(J_nz_idxs+nnz == itnz);
      return true;
    }

    virtual bool eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M)
    {

      int *itnz=H_nz_idxs, nend=v_n->index+n;
      if(NULL==M) {
	for(int it=v_n->index; it<nend; it++) {
	  ia[*itnz] = ja[*itnz] = it; itnz++;
	}
      } else {
	const double* NGsh = d.N_Gsh.data();
	for(int it=0; it<n; it++) {
	  M[*itnz++] -= 2*NGsh[it];
	}
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return n; }

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    {
      if(n==0) return true;
      
      if(NULL==H_nz_idxs) {
	H_nz_idxs = new int[get_HessLagr_nnz()];
      }

      int *itnz=H_nz_idxs, nend=v_n->index+n;
      for(int it=v_n->index; it<nend; it++) vij.push_back(OptSparseEntry(it,it, itnz++));

      return true;
    }

    // Some constraints create additional variables (e.g., slacks).
    // This method is called by OptProblem (in 'append_constraints') to get and add
    // the additional variables block that OptConstraintsBlock may need to add.
    // NULL should be returned when the OptConstraintsBlock need not create a vars block
    virtual OptVariablesBlock* create_varsblock() 
    { 
      assert(pslack_n==NULL);
      pslack_n = new OptVariablesBlock(2*n, "pslack_n", 0, 1e+20);
      return pslack_n; 
    }
    
    //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
    //term (e.g., penalization) that OptConstraintsBlock may need
    virtual OptObjectiveTerm* create_objterm() 
    { 
      return new DummySingleVarQuadrObjTerm("pen_pslack_n", pslack_n); 
    }
  protected:
    OptVariablesBlock *p_g, *v_n, *p_li1, *p_li2, *p_ti1, *p_ti2;
    const SCACOPFData& d;
    OptVariablesBlock *pslack_n; //2*n -> containss pslackp_n, pslackm_n;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  //sum(q_g[g] for g=Gn[n]) - 
  // (-N[:Bsh][n] - sum(b_s[s] for s=SShn[n]))*v_n[n]^2 -
  // sum(q_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
  // sum(q_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - qslackp_n[n] + qslackm_n[n]) == N[:Qd][n]
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
		      const SCACOPFData& d_)
      : OptConstraintsBlock(id_, numcons), 
	q_g(q_g_), v_n(v_n_), q_li1(q_li1_), q_li2(q_li2_), q_ti1(q_ti1_), q_ti2(q_ti2_), b_s(b_s_), d(d_), qslack_n(NULL)//, pslackm_n(NULL)
    {
      assert(d.N_Pd.size()==n);
      //rhs
      memcpy(lb, d.N_Qd.data(), n*sizeof(double));
      DCOPY(&n, lb, &ione, ub, &ione);
      J_nz_idxs = NULL;
      H_nz_idxs = NULL;
    }
    virtual ~PFReactiveBalance() 
    {
      delete[] J_nz_idxs;
      delete[] H_nz_idxs;
    };

    virtual bool eval_body (const OptVariables& vars_primal, bool new_x, double* body)
    {
      assert(qslack_n); 
      body += this->index;
      double* slacks = const_cast<double*>(qslack_n->xref);
      DAXPY(&n, &dminusone, slacks,   &ione, body, &ione);
      DAXPY(&n, &done,      slacks+n, &ione, body, &ione);

      const int *Gn; int nGn;
      for(int i=0; i<n; i++) {
	nGn = d.Gn[i].size(); Gn = d.Gn[i].data();
	for(int ig=0; ig<nGn; ig++) 
	  *body += q_g->xref[Gn[ig]];
	body++;
      }
      body -= n;
      {
	//(N[:Bsh][n]  + sum(b_s[s] for s=SShn[n]))*v_n[n]^2 
	const double *NBsh=d.N_Bsh.data(); const int *SShn; int sz; double aux;
	for(int i=0; i<n; i++) {
	  aux = NBsh[i];
	  SShn = d.SShn[i].data(); sz = d.SShn[i].size();
	  for(int is=0; is<sz; is++) aux += b_s->xref[SShn[is]];
	  *body++ += aux*v_n->xref[i]*v_n->xref[i];
	}
      }
      body -= n;
      {
	// - sum(q_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
	const int *Lidxn; int nLidx;
	for(int i=0; i<n; i++) {
	  Lidxn=d.Lidxn1[i].data(); nLidx = d.Lidxn1[i].size();
	  for(int ilix=0; ilix<nLidx; ilix++)
	    *body -= q_li1->xref[Lidxn[ilix]];
	  body++;
	}
	body -= n;
	// - sum(q_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
	for(int i=0; i<n; i++) {
	  Lidxn=d.Lidxn2[i].data(); nLidx = d.Lidxn2[i].size();
	  for(int ilix=0; ilix<nLidx; ilix++)
	    *body -= q_li2->xref[Lidxn[ilix]];
	  body++;
	}
	body -= n;
      }
      {
	const int *Tidxn; int nTidx;
	// - sum(q_ti1[Tidxn1[n][tix]] for tix=1:length(Tidxn1[n]))
	for(int i=0; i<n; i++) {
	  Tidxn=d.Tidxn1[i].data(); nTidx = d.Tidxn1[i].size();
	  for(int itix=0; itix<nTidx; itix++)
	    *body -= q_ti1->xref[Tidxn[itix]];
	  body++;
	}
	body -= n;
	// - sum(q_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
	for(int i=0; i<n; i++) {
	  Tidxn=d.Tidxn2[i].data(); nTidx = d.Tidxn2[i].size();
	  for(int itix=0; itix<nTidx; itix++)
	    *body -= q_ti2->xref[Tidxn[itix]];
	  body++;
	}
      }
      return true;
    }
    //sum(q_g[g] for g=Gn[n]) - 
    // (-N[:Bsh][n] - sum(b_s[s] for s=SShn[n]))*v_n[n]^2 -
    // sum(q_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
    // sum(q_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - qslackp_n[n] + qslackm_n[n]) == N[:Qd][n]
    virtual bool eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* i, int* j, double* M)
    {
#ifdef DEBUG
      int nnz_loc=get_Jacob_nnz();
#endif
      int row, *itnz=J_nz_idxs;
      if(NULL==M) {
	for(int it=0; it<n; it++) {
	  row = this->index+it;
	  //v_n
	  i[*itnz]=row; j[*itnz]=v_n->index+it; itnz++;
	  //q_g
	  for(auto g: d.Gn[it]) { i[*itnz]=row; j[*itnz]=q_g->index+g; itnz++; }
	  //q_li1 and p_l2
	  for(auto l: d.Lidxn1[it]) { i[*itnz]=row; j[*itnz]=q_li1->index+l; itnz++; }	  
	  for(auto l: d.Lidxn2[it]) { i[*itnz]=row; j[*itnz]=q_li2->index+l; itnz++; }
	  //q_ti1 and q_ti2
	  for(auto t: d.Tidxn1[it]) { i[*itnz]=row; j[*itnz]=q_ti1->index+t; itnz++; }	  
	  for(auto t: d.Tidxn2[it]) { i[*itnz]=row; j[*itnz]=q_ti2->index+t; itnz++; }
	  
	  //b_s
	  for(auto ssh: d.SShn[it])  { i[*itnz]=row; j[*itnz]=b_s->index+ssh; itnz++; }

	  //slacks
	  i[*itnz]=row; j[*itnz]=qslack_n->index+it;   itnz++;
	  i[*itnz]=row; j[*itnz]=qslack_n->index+it+n; itnz++;
	}
#ifdef DEBUG
	assert(J_nz_idxs + nnz_loc == itnz);
#endif
      } else {
	const double* Bsh=d.N_Bsh.data(); int sz, szsshn; const int *sshn; double aux;
	for(int it=0; it<n; it++) {
	  row = this->index+it;

	  aux=Bsh[it];
	  szsshn = d.SShn[it].size(); sshn = d.SShn[it].data();
	  for(int s=0; s<szsshn; s++) aux += b_s->xref[sshn[s]];
	  M[*itnz] += 2*v_n->xref[it]*aux;  itnz++; //vn
	  
	  sz = d.Gn[it].size();
	  for(int ig=0; ig<sz; ig++) { M[*itnz] += 1; itnz++; } //q_g

	  sz=d.Lidxn1[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //q_li_1
	  sz=d.Lidxn2[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_l2_1

	  sz=d.Tidxn1[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //q_ti_1
	  sz=d.Tidxn2[it].size();
	  for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_t2_1

	  aux = v_n->xref[it]*v_n->xref[it];
	  for(int s=0; s<szsshn; s++) {  M[*itnz] += aux; itnz++;} //b_s

	  //slacks
	  M[*itnz] -= 1; itnz++;
	  M[*itnz] += 1; itnz++;
	}
      }
      return true;
    }
    virtual int get_Jacob_nnz(){ 
      int nnz = 2*n; //slacks
      for(int i=0; i<n; i++) 
	nnz += d.Gn[i].size() + (1+d.SShn[i].size()) + d.Lidxn1[i].size() + d.Lidxn2[i].size() + d.Tidxn1[i].size() + d.Tidxn2[i].size();
      return nnz; 
    }
    virtual bool get_Jacob_ij(std::vector<OptSparseEntry>& vij)
    {
      if(n<=0) return true;

      int nnz = get_Jacob_nnz();
      if(!J_nz_idxs) 
	J_nz_idxs = new int[nnz];
#ifdef DEBUG
      int n_vij_in = vij.size();
#endif

      int row, *itnz=J_nz_idxs;
      for(int it=0; it<n; it++) {
	row = this->index+it;
	//v_n
	vij.push_back(OptSparseEntry(row, v_n->index+it, itnz++));

	//q_g
	for(auto g: d.Gn[it]) 
	  vij.push_back(OptSparseEntry(row, q_g->index+g, itnz++));

	//q_li1
	for(auto l: d.Lidxn1[it])
	  vij.push_back(OptSparseEntry(row, q_li1->index+l, itnz++));
	//q_li1
	for(auto l: d.Lidxn2[it])
	  vij.push_back(OptSparseEntry(row, q_li2->index+l, itnz++));
			
	//q_ti1
	for(auto t: d.Tidxn1[it])
	  vij.push_back(OptSparseEntry(row, q_ti1->index+t, itnz++));
	//q_ti1
	for(auto t: d.Tidxn2[it])
	  vij.push_back(OptSparseEntry(row, q_ti2->index+t, itnz++));

	//b_s
	for(auto is: d.SShn[it])
	  vij.push_back(OptSparseEntry(row, b_s->index+is, itnz++));
		
	//slacks
	vij.push_back(OptSparseEntry(row, qslack_n->index+it, itnz++));
	vij.push_back(OptSparseEntry(row, qslack_n->index+it+n, itnz++));
      }
      printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
      assert(nnz+n_vij_in==vij.size());
#endif
      assert(J_nz_idxs+nnz == itnz);
      return true;
    }
    
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
			       const int& nnz, int* ia, int* ja, double* M)
    {
#ifdef DEBUG
      int nnz_loc = get_HessLagr_nnz();
#endif
      int *itnz=H_nz_idxs;
      if(NULL==M) {
	int i, j, row, aux;
	for(int it=0; it<n; it++) {
	  row = v_n->index+it;
	  ia[*itnz] = ja[*itnz] = row; itnz++; // w.r.t. v_n,v_n
	  for(auto is: d.SShn[it]) {
	    i = uppertr_swap(row,j=b_s->index+is, aux); 
	    ia[*itnz] = i; ja[*itnz] = j; itnz++; // w.r.t. v_n,b_s[s])
	  }
	}
      } else {
	const double* NBsh = d.N_Bsh.data(); const int* ssh; int sz; double aux;
	for(int it=0; it<n; it++) {
	  aux = NBsh[it];

	  sz=d.SShn[it].size(); ssh = d.SShn[it].data();
	  for(int is=0; is<sz; is++) {
	    aux += b_s->xref[is];
	  }
	  M[*itnz] += 2*aux; itnz++; //w.r.t. (v_n, v_n)

	  for(int is=0; is<sz; is++) {
	    M[*itnz] += 2*v_n->xref[it]; itnz++; //w.r.t. (v_n, v_n)
	  }
	}
      }
#ifdef DEBUG
	assert(H_nz_idxs+nnz_loc==itnz);
#endif
      return true;
    }

    virtual int get_HessLagr_nnz() 
    { 
      int nnz=n; //v_n
      for(int i=0; i<n; i++) nnz += d.SShn[i].size();
      return nnz; 
    }

    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    {
      if(n==0) return true;
      
      if(NULL==H_nz_idxs) {
	H_nz_idxs = new int[get_HessLagr_nnz()];
      }

      int *itnz=H_nz_idxs, i, j, row, aux;
      for(int it=0; it<n; it++) {
	row = v_n->index+it;
	vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t. v_n,v_n
	
	for(auto is: d.SShn[it]) {
	  i = uppertr_swap(row,j=b_s->index+is, aux); 
	  vij.push_back(OptSparseEntry(i, j, itnz++)); // w.r.t. v_n,b_s[s])
	}
      }
      return true;
    }

    // Some constraints create additional variables (e.g., slacks).
    // This method is called by OptProblem (in 'append_constraints') to get and add
    // the additional variables block that OptConstraintsBlock may need to add.
    // NULL should be returned when the OptConstraintsBlock need not create a vars block
    virtual OptVariablesBlock* create_varsblock() 
    { 
      assert(qslack_n==NULL);
      qslack_n = new OptVariablesBlock(2*n, "qslack_n", 0, 1e+20);
      return qslack_n; 
    }
    
    //same as above. OptProblem calls this (in 'append_constraints') to add an objective 
    //term (e.g., penalization) that OptConstraintsBlock may need
    virtual OptObjectiveTerm* create_objterm() 
    { 
      return new DummySingleVarQuadrObjTerm("pen_qslack_n", qslack_n); 
    }
  protected:
    OptVariablesBlock *q_g, *v_n, *q_li1, *q_li2, *q_ti1, *q_ti2, *b_s;
    const SCACOPFData& d;
    OptVariablesBlock *qslack_n; //2*n -> containss pslackp_n, pslackm_n;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };

  class PFLineLimits  : public OptConstraintsBlock
  {
  public:
    PFLineLimits(const std::string& id_, int numcons,
		 OptVariablesBlock* p_li_, 
		 OptVariablesBlock* q_li_,
		 OptVariablesBlock* v_n_,
		 const std::vector<int>& L_Nidx_,
		 const std::vector<double>& L_Rate_,
		 const SCACOPFData& d_);
    virtual ~PFLineLimits();

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
    const std::vector<double> &L_Rate;
    const SCACOPFData& d;
    OptVariablesBlock *sslack_li; // sslackp_li1 or sslackm_li2;

    int* J_nz_idxs;
    int* H_nz_idxs;
  };
}

#endif

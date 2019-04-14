#ifndef OPF_VARIABLES
#define OPF_VARIABLES

#include "OptProblem.hpp"

#include "blasdefs.hpp"
#include <cstring>
#include <cmath>


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
      DCOPY(&n, lb, &one, ub, &one);
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
      double v1, v1v2, ththT;
      for(int i=0; i<n; i++) {
	ththT = theta_n->xref[E_Nidx1[i]] - theta_n->xref[E_Nidx2[i]] + T[i];
	v1 = v_n->xref[E_Nidx1[i]];
	v1v2 = v1 * v_n->xref[E_Nidx2[i]];
	body[i] +=  A[i]*v1*v1 + B[i]*v1v2*cos(ththT) + C[i]*v1v2*sin(ththT) - pq->xref[i];
      }
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
			       const int& nnz, int* i, int* j, double* M)
    {
      const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
      //assert(lambda!=NULL);
      //assert(lambda->n==1);
      
      if(NULL==M) {
      } else {
      }
      return true;
    }

    virtual int get_HessLagr_nnz() { return 0; }
    
    // (i,j) entries in the HessLagr to which the implementer's contributes to
    // this is only called once
    // push_back in vij 
    virtual bool get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
    {
      int n=0, i;
      if(n==0) return true;
      
      if(NULL==H_nz_idxs) {
	H_nz_idxs = new int[n];
      }

      //for(int it=0; it<n; it++) {
      //i = x->index+it;
      //vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
      //}
      return true;
    }


  protected:
    const OptVariablesBlock *pq, *v_n, *theta_n;
    //const SCACOPFData& d;
    int* J_nz_idxs;
    int* H_nz_idxs;

    double *A, *B, *C, *T;
    int *E_Nidx1, *E_Nidx2;
    static int one;
  };
  int PFConRectangular::one=1;

}

#endif

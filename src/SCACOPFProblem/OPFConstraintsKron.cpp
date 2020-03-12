#include "OPFConstraintsKron.hpp"

using namespace std;
using namespace hiop;

namespace gollnlp {
  PFActiveBalanceKron::PFActiveBalanceKron(const std::string& id_, int numcons,
					   OptVariablesBlock* p_g_, 
					   OptVariablesBlock* v_n_,
					   OptVariablesBlock* theta_n_,
					   const std::vector<int>& bus_nonaux_idxs_,
					   const std::vector<std::vector<int> >& Gn_full_space_,
					   const hiop::hiopMatrixComplexDense& Ybus_red_,
					   const std::vector<double>& N_Pd_full_space_)
    : OptConstraintsBlock(id_, numcons), p_g(p_g_), v_n(v_n_), theta_n(theta_n_),
      bus_nonaux_idxs(bus_nonaux_idxs_), Gn_fs(Gn_full_space_), Ybus_red(Ybus_red), 
      J_nz_idxs(NULL), H_nz_idxs(NULL)
  {
    assert(numcons==bus_nonaux_idxs_.size());

    selectfrom(N_Pd_full_space_, bus_nonaux_idxs, N_Pd);

    //rhs
    //!memcpy(lb, d.N_Pd.data(), n*sizeof(double));
    for(int i=0; i<n; i++) lb[i]=0.;
    DCOPY(&n, lb, &ione, ub, &ione);
  }
  PFActiveBalanceKron::~PFActiveBalanceKron()
  {
    delete[] J_nz_idxs;
    delete[] H_nz_idxs;
  }

  // sum(p_g[g] for g=Gn[nonaux[i]]) 
  // - sum(v_n[i]*v_n[j]*
  //      ( Gred[i,j]*cos(theta_n[i]-theta_n[j]) 
  //      + Bred[i,j]*sin(theta_n[i]-theta_n[j])) for j=1:length(nonaux))
  // - N[:Pd][nonaux[i]] == 0
  ///////////////////////////////////////////////////////////////////////////////
  bool PFActiveBalanceKron::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
  {
    double* body = body_ + this->index;
    assert(n==N_Pd.size());
    assert(n==bus_nonaux_idxs.size());

    double *NPd=N_Pd.data();
    //for(int i=0; i<n; i++) {
    //  *body -= NPd[i];
    //  body++;
    //}
    //body -= n;
    DAXPY(&n, &dminusone, NPd, &ione, body, &ione);

    const int *Gnv; int nGn;
    //for(int i=0; i<n; i++) {
    for(auto& i : bus_nonaux_idxs) {
      nGn = Gn_fs[i].size(); 
      Gnv = Gn_fs[i].data();
      for(int ig=0; ig<nGn; ig++) 
	*body += p_g->xref[Gnv[ig]];
      body++;
    }
    body -= n;

    {
      // - sum(v_n[i]*v_n[j]*
      //      ( Gred[i,j]*cos(theta_n[i]-theta_n[j]) 
      //      + Bred[i,j]*sin(theta_n[i]-theta_n[j])) for j=1:length(nonaux))
      std::complex<double>** YredM = Ybus_red.local_data();
      double aux;

      for(int i=0; i<n; i++) {
	for(int j=0; j<n; j++) {
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  body[i] -= v_n->xref[i]*v_n->xref[j] *
	    ( YredM[i][j].real()*cos(aux) + YredM[i][j].imag()*sin(aux) );
	}
      }
    }
#ifdef DEBUG
#endif
    
    return true;
  }

  // Eqn
  //sum(p_g[g] for g=Gn[nonaux[i]]) 
  // - sum(v_n[i]*v_n[j]*
  //      ( Gred[i,j]*cos(theta_n[i]-theta_n[j]) 
  //      + Bred[i,j]*sin(theta_n[i]-theta_n[j])) for j=1:length(nonaux))
  // - N[:Pd][nonaux[i]] = a_i(p_g, v_n, theta_n), i=1:length(nonaux)
  //
  // Sparse part of the Jacobian w.r.t. p_g, q_g, and b_s is easy -> only
  // ones for each generator at nonaux bus, for all such buses
  //
  // Dense part of the Jacobian w.r.t. v(=v_n) and theta(=theta_n)
  // d a_i
  // ----- = - v_i * (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))  [[[ i!=k ]]]
  // d v_k
  //
  // d a_i        n
  // ----- = -   sum    v_j* (Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
  // d v_i     j=1, j!=i
  //
  //         - 2*v_i*Gred[i,i]
  //
  //   d a_i
  // --------- = - v_i*v_k *(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))  [[[ i!=k ]]]
  // d theta_k
  //
  //  d a_i            n
  // --------- = -    sum    v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
  // d theta_i     j=1, j!=i
  //
  //             
  //
  // Note: cos and sin can be computed only once for d a_i/d v_k and d a_k/d_v_i (same for 
  //partials w.r.t. theta
  bool PFActiveBalanceKron::eval_Jac(const OptVariables& x, bool new_x, 
				     const int& nxsparse, const int& nxdense,
				     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
				     double** JacD)
  {
    //
    // sparse part
    //
    assert(nxsparse+nxdense == n);
#ifdef DEBUG
    int nnz_loc=get_Jacob_nnz();
#endif
    int row, *itnz=J_nz_idxs;
    if(NULL==MJacS) {
      for(int it=0; it<n; it++) {
	row = this->index+it;
	//p_g
	for(auto g: Gn_fs[bus_nonaux_idxs[it]]) { 
	  iJacS[*itnz]=row; 
	  jJacS[*itnz]=p_g->index+g; 
	  itnz++; 
	}
      }
#ifdef DEBUG
      assert(J_nz_idxs + nnz_loc == itnz);
#endif
    } else {
      
      for(int it=0; it<n; it++) {
	//p_g 
	const double sz = Gn_fs[bus_nonaux_idxs[it]].size();
	for(int ig=0; ig<sz; ig++) { 
	  MJacS[*itnz] += 1; 
	  itnz++; 
	}
      }
    }
    //
    // dense part
    //
    if(NULL==MJacS) {

    } else {
      //values
      std::complex<double>** YredM = Ybus_red.local_data();
      double aux, aux_sin, aux_cos, aux_G_cos, aux_B_sin, aux_G_cos_B_sin, aux_G_sin, aux_B_cos, vivj, vivjGcos_Bsin;
      for(int i=0; i<n; i++) {
	//
	//partials w.r.t. to v_i and theta_i
	//
	assert(i<nxdense);
	assert(i+v_n->index<nxdense);

	//one term of da_i/v_i
	JacD[i][i]            = - 2.0*v_n->xref[i]*YredM[i][i].real();
	//here we'll accumulate da_i/theta_i
	JacD[i][v_n->index+i] = 0.;
	
	for(int j=0; j<i; j++) {
	  //for v_i:     -     v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  //for theta_i: - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  //! Yred only stores upper triangle
	  aux_G_cos = YredM[j][i].real()*aux_cos;
	  aux_B_sin = YredM[j][i].imag()*aux_sin;
	  aux_G_cos_B_sin = aux_G_cos + aux_B_sin;

	  aux_G_sin = YredM[j][i].real()*aux_sin;
	  aux_B_cos = YredM[j][i].imag()*aux_cos;
	  vivjGcos_Bsin = v_n->xref[j]*v_n->xref[i]*(aux_G_sin-aux_B_cos);

	  //for da_i/dv_k =   -   v_i * (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))
	  JacD[i][j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/v_i:     -   v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  JacD[i][i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  //for da_i/dtheta_k = - v_i*v_k *(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  JacD[i][j+v_n->index] = -vivj*(aux_G_sin-aux_B_cos);

	  //for da_i/theta_i:   - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i][i+v_n->index] -= vivjGcos_Bsin;
	}
	for(int j=i+1; j<n; j++) {
	  //for v_i:     -     v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  //for theta_i: - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  //! Yred only stores upper triangle
	  aux_G_cos = YredM[i][j].real()*aux_cos;
	  aux_B_sin = YredM[i][j].imag()*aux_sin;
	  aux_G_cos_B_sin = aux_G_cos + aux_B_sin;

	  aux_G_sin = YredM[i][j].real()*aux_sin;
	  aux_B_cos = YredM[i][j].imag()*aux_cos;
	  vivjGcos_Bsin = v_n->xref[j]*v_n->xref[i]*(aux_G_sin-aux_B_cos);

	  //for da_i/v_i:      - v_j*     (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  JacD[i][i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  //for da_i/dv_k =    - v_i *    (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))
	  JacD[i][j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/theta_i:  - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i][i+v_n->index] -= vivjGcos_Bsin;

	  //for da_i/dtheta_k = - v_i*v_k*(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  JacD[i][j+v_n->index] = -vivj*(aux_G_sin-aux_B_cos);
	}
      }
    }
    return true;
  }

  int PFActiveBalanceKron::get_Jacob_nnz(){ 
    int nnz = 0; 
    for(auto& i : bus_nonaux_idxs)
      nnz += Gn_fs[i].size();
    return nnz; 
  }
  
  bool PFActiveBalanceKron::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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
      
      //p_g
      for(auto g: Gn_fs[bus_nonaux_idxs[it]]) 
	vij.push_back(OptSparseEntry(row, p_g->index+g, itnz++));
    }
    //printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
    assert(nnz+n_vij_in==vij.size());
#endif
    assert(J_nz_idxs+nnz == itnz);
    return true;
  }

  // Eqn
  //sum(p_g[g] for g=Gn[nonaux[i]]) 
  // - sum(v_n[i]*v_n[j]*
  //      ( Gred[i,j]*cos(theta_n[i]-theta_n[j]) 
  //      + Bred[i,j]*sin(theta_n[i]-theta_n[j])) for j=1:length(nonaux))
  // - N[:Pd][nonaux[i]] = a_i(p_g, v_n, theta_n), i=1:length(nonaux)
  //
  // Sparse part of the Jacobian w.r.t. p_g, q_g, and b_s is easy -> only
  // ones for each generator at nonaux bus, for all such buses
  //
  // Dense part of the Jacobian w.r.t. v(=v_n) and theta(=theta_n)
  // d a_i
  // ----- = - v_i * (Gred[i,k]*cos(theta_i-theta_k) + Bred[i,k]*sin(theta_i-theta_k))  [[[ i!=k ]]]
  // d v_k
  //
  // d a_i        n
  // ----- = -   sum    v_j* (Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
  // d v_i     j=1, j!=i
  //
  //         - 2*v_i*Gred[i,i]
  // ****************************************************
  // Dense part of Hessian of a_i w.r.t v_k,v_j  (j>=k)
  // k=1,...,i-1 all zeros except
  // d2 a_i/dv_k dv_i = - [Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k)], 
  //
  // k=i 
  // d2 a_i/dv_i dv_i = -2*Gred[i,i]
  // d2 a_i/dv_i dv_j = -[Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)] (j>i)
  //
  // k>i all d2 a_i/dv_k dv_j = 0 (j>=k)
  // ****************************************************
  // ****************************************************
  // Dense part of the Hessian of a_i w.r.t. v_k, theta_j
  // k=1,...,i-1 all zeros except
  // d2 a_i/dv_k dtheta_k = - v_i*[ Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  // d2 a_i/dv_k dtheta_i = - v_i*[-Gred[i,k]*sin(theta_i-theta_k) + Bred[i,k]*cos(theta_i-theta_k)]
  //
  // k=i 
  //                             n
  // d2 a_i/dv_i dtheta_i = -   sum    v_j*[-Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j)]
  //                         j=1, j!=i
  // d2 a_i/dv_i dtheta_j = -  v_j*[Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j)]
  //
  // k>i
  // d2 a_i/dv_k dtheta_k = - v_i*[Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  // d2 a_i/dv_k dtheta_j = 0 (j>k)
  // ****************************************************
  //   d a_i
  // --------- = - v_i*v_k *(Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k))  [[[ i!=k ]]]
  // d theta_k
  //
  //  d a_i            n
  // --------- = -    sum    v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j))
  // d theta_i     j=1, j!=i
  // ****************************************************
  // Dense part of the Hessian of a_i w.r.t theta_k, theta_j
  // k<i
  // d2 a_i / dtheta_k dtheta_k = - v_i*v_k*[-Gred[i,k]*cos(theta_i-theta_k) - Bred[i,k]*sin(theta_i-theta_k)]
  // d2 a_i / dtheta_k dtheta_i = - v_i*v_k*[ Gred[i,k]*cos(theta_i-theta_k) + Bred[i,k]*sin(theta_i-theta_k)]
  //
  // k=i
  // d2 a_i / dtheta_i dtheta_i = - sum{j!=i} v_i*v_j*[-Gred[i,j]*cos(theta_i-theta_j)-Bred[i,j]*sin(theta_i-theta_j)]
  // d2 a_i / dtheta_i dtheta_j = - v_i*v_j*[ Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)]
  //
  // k>i
  // d2 a_i / dtheta_k dtheta_k = - v_i*v_k*[-Gred[i,k]*cos(theta_i-theta_k) - Bred[i,k]*sin(theta_i-theta_k)]
  // d2 a_i / dtheta_k dtheta_j = 0 (j>k)
  // ****************************************************
  // ****************************************************
  // Dense part of the Hessian of a_i w.r.t theta_k, v_j (just to check)
  // k<i
  // d2 a_i / dtheta_k dv_k = - v_i*[Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  // d2 a_i / dtheta_k dv_i = - v_k*[Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  //
  // k=i
  // d2 a_i / dtheta_i dv_i = - sum v_j*[- Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j)]
  // d2 a_i / dtheta_i dv_j = - v_i*[- Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j)]
  //
  // k>i
  // d2 a_i / dtheta_k dv_k = - v_i*[Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  // d2 a_i / dtheta_k dv_j = 0
  bool PFActiveBalanceKron::eval_HessLagr(const OptVariables& x, bool new_x, 
					  const OptVariables& lambda_vars, bool new_lambda,
					  const int& nxsparse, const int& nxdense, 
					  const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
					  double** HDD,
					  int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //
    // sparse part is empty
    //

    //
    // dense part
    //
    if(NULL==MHSS) {
    } else {
      const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
      assert(lambda!=NULL);
      assert(lambda->n==n);
      std::complex<double>** YredM = Ybus_red.local_data();
      double theta_diff, aux_G_sin, aux_B_cos, res;

      //loop over constraints/lambdas
      for(int i=0; i<n; i++) {
	const double& lambda_i = lambda->xref[i];
	int k,j;
	//
	//Dense part of Hessian of a_i w.r.t v_k,v_j  (j>=k)
	//
	
	//! YredM is upper triangle

	//---> k=1,...,i-1 all zeros except d2 a_i/dv_k dv_i
	for(k=0; k<i; k++) {
	  //v_k has index k in dense part of Hessian; v_i has index i
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  HDD[k][i] -= lambda_i * (YredM[k][i].real()*cos(theta_diff) + YredM[k][i].imag()*sin(theta_diff));
	}
	//---> k=i
	k=i;
	// d2 a_i/dv_i dv_i and d2 a_i/dv_i dv_j (j>i)
	HDD[i][i] -= lambda_i * 2 * YredM[i][i].real();
	for(j=k+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  HDD[i][i] -= lambda_i * (YredM[k][i].real()*cos(theta_diff) + YredM[k][i].imag()*sin(theta_diff));
	}
	//---> k>i all zeros

	//
	// Dense part of the Hessian of a_i w.r.t. v_k, theta_j
	//
	// ----> k=1,...,i-1 all zeros except d2 a_i/dv_k dtheta_k  and  d2 a_i/dv_k dtheta_i

	// index of theta_k in dense part of the Hessian is v_n->n+k	

	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  aux_G_sin = YredM[k][i].real()*sin(theta_diff);
	  aux_B_cos = YredM[k][i].imag()*cos(theta_diff);
	  res = lambda_i * v_n->xref[i] * (aux_G_sin - aux_B_cos);
	  //d2 a_i/dv_k dtheta_k
	  assert(k<v_n->n+k);
	  HDD[k][v_n->n+k] -= res;
	  
	  //d2 a_i/dv_k dtheta_i
	  assert(k<v_n->n+i);
	  HDD[k][v_n->n+i] += res;
	}
	// ---> k=i
	k=i;
	//                             n
	// d2 a_i/dv_i dtheta_i = -   sum    v_j*[-Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j)]
	//                         j=1, j!=i
	double sum=0.; 
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(-YredM[j][i].real()*sin(theta_diff) + YredM[j][i].imag()*cos(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(-YredM[i][j].real()*sin(theta_diff) + YredM[i][j].imag()*cos(theta_diff));
	}
	HDD[i][v_n->n+i] -= lambda_i*sum;

	//TODO merge the loop above with the one below maybe
	// d2 a_i/dv_i dtheta_j = -  v_j*[Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j)]
	for(int j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  HDD[i][v_n->n+j] -= lambda_i*(YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));
	}

	// ---> k=i+1, ..., n -> only d2 a_i/dv_k dtheta_k is nonzero
	HDD[k][v_n->n+k] -= lambda_i*v_n->xref[i]*(YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));

	//
	// Dense part of the Hessian of a_i w.r.t theta_k, theta_j
	//
	// ---> k<i : only d2 a_i / dtheta_k dtheta_k and d2 a_i / dtheta_k dtheta_i are nonzeros
	const int idx_of_theta_i = v_n->n+i;
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  res = v_n->xref[i]*v_n->xref[k]*
	    (YredM[k][i].real()*cos(theta_diff) + YredM[k][i].imag()*sin(theta_diff));
	  const int idx_of_theta_k = v_n->n+k;
	  HDD[idx_of_theta_k][idx_of_theta_k] += lambda_i * res;
	  HDD[idx_of_theta_k][idx_of_theta_i] -= lambda_i * res;
	}
	// ---> k=i
	k=i; 
	// d2 a_i / dtheta_i dtheta_i
	sum = 0;
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[j][i].real()*cos(theta_diff) - YredM[j][i].imag()*sin(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[i][j].real()*cos(theta_diff) - YredM[i][j].imag()*sin(theta_diff));
	}
	HDD[idx_of_theta_i][idx_of_theta_i] -= lambda_i*sum;
	// d2 a_i / dtheta_i dtheta_j  (k=i)
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];

	  HDD[idx_of_theta_i][v_n->n+j] -= lambda_i * v_n->xref[i] * v_n->xref[j] *
	    (YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff));
	}

	// ---> k>i : only d2 a_i / dtheta_k dtheta_k is nonzero
	for(k=i+1; k<n; k++) {
	  const int idx_of_theta_k = v_n->n+k;
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  HDD[idx_of_theta_k][idx_of_theta_k] -= lambda_i * v_n->xref[i] * v_n->xref[k] *
	    (-YredM[i][k].real()*cos(theta_diff) - YredM[i][k].imag()*sin(theta_diff));
	}
      } // end of for over i=1,..,n
    }
    return true;
}


  // bool PFActiveBalanceKron::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)  
  // {
  //   if(n==0) return true;
    
  //   if(NULL==H_nz_idxs) {
  //     H_nz_idxs = new int[get_HessLagr_nnz()];
  //   }
    
  //   int *itnz=H_nz_idxs, nend=v_n->index+n;
  //   for(int it=v_n->index; it<nend; it++) vij.push_back(OptSparseEntry(it,it, itnz++));
    
  //   return true;
  // }



} // end of namespace

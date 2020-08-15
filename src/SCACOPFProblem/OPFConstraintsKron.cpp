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
    : OptConstraintsBlockMDS(id_, numcons), p_g(p_g_), v_n(v_n_), theta_n(theta_n_),
      bus_nonaux_idxs(bus_nonaux_idxs_), Gn_fs(Gn_full_space_), Ybus_red(Ybus_red_), 
      J_nz_idxs(NULL), H_nz_idxs(NULL)
  {
    assert(numcons==bus_nonaux_idxs_.size());

    selectfrom(N_Pd_full_space_, bus_nonaux_idxs, N_Pd);

    //rhs
    //!memcpy(lb, d.N_Pd.data(), n*sizeof(double));
    for(int i=0; i<n; i++) lb[i]=0.;
    DCOPY(&n, lb, &ione, ub, &ione);

    assert(p_g->sparseBlock==true);
    assert(v_n->sparseBlock==false);
    assert(theta_n->sparseBlock==false);
    assert(p_g->indexSparse>=0);
    assert(v_n->indexSparse<=0);
    assert(theta_n->indexSparse<=0);

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
      ++body;
    }
    assert(bus_nonaux_idxs.size()==n);
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
  // --------- = -    sum    v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j))
  // d theta_i     j=1, j!=i
  //
  //             
  //
  // Note: cos and sin can be computed only once for d a_i/d v_k and d a_k/d_v_i (same for 
  //partials w.r.t. theta
  bool PFActiveBalanceKron::eval_Jac_eq(const OptVariables& x, bool new_x, 
				     const int& nxsparse, const int& nxdense,
				     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
				     double** JacD)
  {
    //
    // sparse part
    //
    assert(nxsparse+nxdense == x.n());
    int row, *itnz;
 #ifdef DEBUG
    int nnz_loc=get_spJacob_eq_nnz();
#endif
    
    if(iJacS && jJacS) {
      itnz = J_nz_idxs; row=this->index;
      for(int it=0; it<n; it++) {
	const int idxBusNonAux = bus_nonaux_idxs[it];
	const size_t sz = Gn_fs[idxBusNonAux].size();
	//p_g
	for(int ig=0; ig<sz; ++ig) {
	  iJacS[*itnz]=row; 
	  jJacS[*itnz++]=p_g->indexSparse+Gn_fs[idxBusNonAux][ig]; 
	}
	++row;
      }
#ifdef DEBUG
      assert(J_nz_idxs + nnz_loc == itnz);
#endif
      
    }
    if(MJacS) {
      itnz = J_nz_idxs; 
      for(int it=0; it<n; it++) {
	//p_g 
	const size_t sz = Gn_fs[bus_nonaux_idxs[it]].size();
	for(int ig=0; ig<sz; ++ig) { 
	  MJacS[*itnz++] += 1; 
	}
      }
    }

    //
    // dense part
    //
    if(JacD) {
      
      //values
      std::complex<double>** YredM = Ybus_red.local_data();
      double aux, aux_sin, aux_cos, aux_G_cos, aux_B_sin, aux_G_cos_B_sin, aux_G_sin, aux_B_cos;
      double vivj, vivjGsin__Bcos;

      assert(v_n->compute_indexDense() >=0);
      assert(theta_n->compute_indexDense() >= 0);
	
      assert(v_n->compute_indexDense()    +v_n->n     <= nxdense);
      assert(theta_n->compute_indexDense()+theta_n->n <= nxdense);

      assert(v_n->n == n);
      assert(theta_n->n == n);

      const int idx_col_of_v_n     = v_n->compute_indexDense();
      const int idx_col_of_theta_n = theta_n->compute_indexDense();

      //'i2' is the row # in the Jacobian of this set of constraints
      
      for(int i=0, i2=this->index; i<n; i++, i2++) {
	
	//
	//partials w.r.t. to v_i and theta_i
	//

	//first term of da_i/v_i  -> we'll accumulate in here
	JacD[i2][idx_col_of_v_n + i] = - 2.0*v_n->xref[i]*YredM[i][i].real();
	//here we'll accumulate da_i/theta_i
	JacD[i2][idx_col_of_theta_n + i] = 0.;
	
	for(int j=0; j<i; j++) {
	  //for v_i:     -     v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  //for theta_i: - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  vivj = v_n->xref[j]*v_n->xref[i];
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  aux_G_cos = YredM[i][j].real()*aux_cos;
	  aux_B_cos = YredM[i][j].imag()*aux_cos;
	  aux_B_sin = YredM[i][j].imag()*aux_sin;
	  aux_G_sin = YredM[i][j].real()*aux_sin;
	  aux_G_cos_B_sin = aux_G_cos + aux_B_sin;
	  vivjGsin__Bcos = vivj*(aux_G_sin-aux_B_cos);

	  //for da_i/dv_k = - v_i * (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))
	  assert(JacD[i2][idx_col_of_v_n + j]==0. && "should not written previously in this");
	  JacD[i2][idx_col_of_v_n + j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/v_i: add  - v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  JacD[i2][idx_col_of_v_n + i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  
	  //for da_i/dtheta_k = - v_i*v_k *(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  assert(JacD[i2][idx_col_of_theta_n + j]==0. && "should not written previously in this");
	  JacD[i2][idx_col_of_theta_n + j] = - vivjGsin__Bcos;

	  //for da_i/theta_i: add
	  //        - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i2][idx_col_of_theta_n + i] += vivjGsin__Bcos;
	}
	for(int j=i+1; j<n; j++) {
	  //for v_i:     -     v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  //for theta_i: - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  vivj = v_n->xref[j]*v_n->xref[i];
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  aux_G_cos = YredM[i][j].real()*aux_cos;
	  aux_B_cos = YredM[i][j].imag()*aux_cos;
	  aux_B_sin = YredM[i][j].imag()*aux_sin;
	  aux_G_sin = YredM[i][j].real()*aux_sin;
	  aux_G_cos_B_sin = aux_G_cos + aux_B_sin;
	  vivjGsin__Bcos = vivj*(aux_G_sin-aux_B_cos);

	  //for da_i/v_i:  add - v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
	  JacD[i2][idx_col_of_v_n + i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  //for da_i/dv_k =  - v_i * (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))
	  assert(JacD[i2][idx_col_of_v_n + j]==0. && "should not written previously in this");
	  JacD[i2][idx_col_of_v_n + j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/theta_i: add
	  //         - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i2][idx_col_of_theta_n + i] += vivjGsin__Bcos;

	  //for da_i/dtheta_k = - v_i*v_k*(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  JacD[i2][idx_col_of_theta_n + j] = - vivjGsin__Bcos;
	}
      }
    }
    return true;
  }

  int PFActiveBalanceKron::get_spJacob_eq_nnz(){ 
    int nnz = 0; 
    for(auto& i : bus_nonaux_idxs)
      nnz += Gn_fs[i].size();
    return nnz; 
  }
  
  bool PFActiveBalanceKron::get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij)
  {
    if(n<=0) return true;
    
    int nnz = get_spJacob_eq_nnz();
    if(!J_nz_idxs) 
      J_nz_idxs = new int[nnz];
#ifdef DEBUG
    int n_vij_in = vij.size();
#endif
    
    int row=this->index, *itnz=J_nz_idxs;
    for(int it=0; it<n; it++) {      
      //p_g
      for(auto g: Gn_fs[bus_nonaux_idxs[it]]) 
	vij.push_back(OptSparseEntry(row, p_g->indexSparse+g, itnz++));
    
      //if(Gn_fs[bus_nonaux_idxs[it]].size()>0)
      ++row;
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
  // d2 a_i/dv_k dtheta_k = - v_i*[ Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)]
  // d2 a_i/dv_k dtheta_i = - v_i*[-Gred[i,k]*cos(theta_i-theta_k) + Bred[i,k]*sin(theta_i-theta_k)]
  // d2 a_i/dv_k dtheta_j = 0 (j!=k)
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
					  const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //
    // sparse part is empty
    //

    //
    // dense part
    //
    // Note: Only upper triangle part of HDD is updated
    if(NULL!=iHSS && NULL!=jHSS) {
    }

    if(HDD) {
      
      const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
      assert(lambda != NULL);
      assert(lambda->n == n);
      assert(v_n->n == n);
      assert(theta_n->n == n);

      const int idx_col_of_v_n     = v_n->compute_indexDense();
      const int idx_col_of_theta_n = theta_n->compute_indexDense();

      assert(idx_col_of_v_n >= 0 && idx_col_of_v_n+v_n->n <= nxdense);
      assert(idx_col_of_theta_n >= 0 && idx_col_of_theta_n+theta_n->n <= nxdense);
      
      std::complex<double>** YredM = Ybus_red.local_data();
      double theta_diff, aux_G_sin, aux_B_cos, res;

      //loop over constraints/lambdas
      for(int i=0; i<n; i++) {
	
	const double& lambda_i = lambda->xref[i];

	//quick return for when the derivatives are checked
	if(lambda_i==0.) continue;
	
	int k,j;
	const int idx_col_of_v_n_elemi = idx_col_of_v_n+i;
	//********************************************************
	//Dense part of Hessian of a_i w.r.t v_k,v_j  (j>=k)
	//********************************************************

	//---> k=1,...,i-1 all zeros except d2 a_i/dv_k dv_i
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];

	  assert(idx_col_of_v_n+k <= idx_col_of_v_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_v_n_elemi] -= 
	    lambda_i * (YredM[i][k].real()*cos(theta_diff) + YredM[i][k].imag()*sin(theta_diff));
	}
	//---> k=i
	k=i;
	// d2 a_i/dv_i dv_i 
	HDD[idx_col_of_v_n_elemi][idx_col_of_v_n_elemi] -= lambda_i * 2 * YredM[i][i].real();
	// d2 a_i/dv_i dv_j (j>i)
	for(j=k+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  
	  assert(idx_col_of_v_n_elemi <= idx_col_of_v_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_v_n+j] -=
	    lambda_i * (YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff));
	}
	//---> k>i all zeros

	//********************************************************
	// Dense part of the Hessian of a_i w.r.t. v_k, theta_j
	//********************************************************
	
	// ----> k=1,...,i-1 all zeros except d2 a_i/dv_k dtheta_k  and  d2 a_i/dv_k dtheta_i
	// index of theta_k in dense part of the Hessian is idx_col_of_theta_n
	const int idx_col_of_theta_n_elemi = idx_col_of_theta_n+i;
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  aux_G_sin = YredM[i][k].real()*sin(theta_diff);
	  aux_B_cos = YredM[i][k].imag()*cos(theta_diff);
	  res = lambda_i * v_n->xref[i] * (aux_G_sin - aux_B_cos);
	  //d2 a_i/dv_k dtheta_k
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n+k);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n+k] -= res;
	  
	  //d2 a_i/dv_k dtheta_i
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n_elemi] += res;
	}
	// ---> k=i <---
	k=i;
	//                          n
	// d2 a_i/dv_i dtheta_i= - sum  v_j*[-Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j)]
	//                       j=1, j!=i
	double sum=0.; 
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(-YredM[i][j].real()*sin(theta_diff) + YredM[i][j].imag()*cos(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(-YredM[i][j].real()*sin(theta_diff) + YredM[i][j].imag()*cos(theta_diff));
	}
	assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n_elemi);
	HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n_elemi] -= lambda_i*sum;

	// d2 a_i/dv_i dtheta_j = -  v_j*[Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j)]
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];

	  assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n+j] -= lambda_i*v_n->xref[j]*
	    (YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  
	  assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n+j] -= lambda_i*v_n->xref[j]*
	    (YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));
	}

	// ---> k=i+1, ..., n  <---
	for(k=i+1; k<n; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  res = lambda_i*v_n->xref[i]*(YredM[i][k].real()*sin(theta_diff) -
				       YredM[i][k].imag()*cos(theta_diff));
	  //d2 a_i/dv_k dtheta_k is nonzero
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n+k);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n+k] -= res;
	    

	  //d2 a_i/dv_k dtheta_i
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n_elemi] += res;
	}
	
	//********************************************************
	// Dense part of the Hessian of a_i w.r.t theta_k, theta_j
	//********************************************************
	
	// ---> k<i <---
	// only d2 a_i / dtheta_k dtheta_k and d2 a_i / dtheta_k dtheta_i are nonzeros
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  res = v_n->xref[i]*v_n->xref[k]*
	    (YredM[i][k].real()*cos(theta_diff) + YredM[i][k].imag()*sin(theta_diff));
	  const int idx_col_of_theta_n_elemk = idx_col_of_theta_n+k;
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemk] += lambda_i * res;
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemi] -= lambda_i * res;
	}
	// ---> k=i  <---
	k=i; 
	// d2 a_i / dtheta_i dtheta_i
	sum = 0;
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[i][j].real()*cos(theta_diff) - YredM[i][j].imag()*sin(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[i][j].real()*cos(theta_diff) - YredM[i][j].imag()*sin(theta_diff));
	}
	HDD[idx_col_of_theta_n_elemi][idx_col_of_theta_n_elemi] -= lambda_i*sum;
	
	// d2 a_i / dtheta_i dtheta_j  (k=i) for j>i (j<i was updated by symmetry at ---> k<i <---)
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];

	  assert(idx_col_of_theta_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_theta_n_elemi][idx_col_of_theta_n+j] -=
	    lambda_i * v_n->xref[i] * v_n->xref[j] *
	    (YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff));
	}

	// ---> k>i : only d2 a_i / dtheta_k dtheta_k is nonzero
	for(k=i+1; k<n; k++) {
	  const int idx_col_of_theta_n_elemk = idx_col_of_theta_n+k;
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  
	  //assert(idx_col_of_theta_n_elemk <= idx_col_of_theta_n_elemk);
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemk] -=
	    lambda_i * v_n->xref[i] * v_n->xref[k] *
	    (-YredM[i][k].real()*cos(theta_diff) - YredM[i][k].imag()*sin(theta_diff));
	}
	
      } // end of for over i=1,..,n
    }
    return true;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Reactive Balance constraints
  ///////////////////////////////////////////////////////////////////////////////

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
  PFReactiveBalanceKron::PFReactiveBalanceKron(const std::string& id_, int numcons,
					       OptVariablesBlock* q_g_, 
					       OptVariablesBlock* v_n_,
					       OptVariablesBlock* theta_n_,
					       OptVariablesBlock* b_s_,
					       const std::vector<int>& bus_nonaux_idxs_,
					       const std::vector<std::vector<int> >& Gn_full_space_,
					       const std::vector<std::vector<int> >& SShn_full_space_in,
					       const hiop::hiopMatrixComplexDense& Ybus_red_,
					       const std::vector<double>& N_Qd_full_space_)
    : OptConstraintsBlockMDS(id_, numcons),
      q_g(q_g_), v_n(v_n_), theta_n(theta_n_), b_s(b_s_),
      bus_nonaux_idxs(bus_nonaux_idxs_),
      Gn_fs(Gn_full_space_), SShn_fs(SShn_full_space_in),
      Ybus_red(Ybus_red_), 
      J_nz_idxs(NULL), H_nz_idxs(NULL)
  {
    assert(numcons==bus_nonaux_idxs_.size());

    selectfrom(N_Qd_full_space_, bus_nonaux_idxs, N_Qd);

    //rhs
    //!memcpy(lb, d.N_Pd.data(), n*sizeof(double));
    for(int i=0; i<n; i++) lb[i]=0.;
    DCOPY(&n, lb, &ione, ub, &ione);

    assert(q_g->sparseBlock==true);
    assert(v_n->sparseBlock==false);
    assert(theta_n->sparseBlock==false);
    assert(b_s->sparseBlock==false);
    assert(q_g->indexSparse>=0);
    assert(v_n->indexSparse<=0);
    assert(b_s->indexSparse<=0);
    assert(theta_n->indexSparse<=0);

  }
  PFReactiveBalanceKron::~PFReactiveBalanceKron()
  {
    delete[] J_nz_idxs;
    delete[] H_nz_idxs;
  }

  /*
   * sum(q_g[g] for g=Gn[nonaux[i]]) - N[:Qd][nonaux[i]] +
   * v_n[i]^2*sum(b_s[s] for s=SShn[nonaux[i]]) ==
   *   sum(v_n[i]*v_n[j]*(Gred[i,j]*sin(theta_n[i]-theta_n[j]) -
   *   Bred[i,j]*cos(theta_n[i]-theta_n[j])) for j=1:length(nonaux)))
   */
  bool PFReactiveBalanceKron::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
  {
    double* body = body_ + this->index;
    assert(n==N_Qd.size());
    assert(n==bus_nonaux_idxs.size());

    double *NQd=N_Qd.data();
    DAXPY(&n, &dminusone, NQd, &ione, body, &ione);

    const int *Gnv; int nGn;
    //for(int i=0; i<n; i++) {
    for(auto& i : bus_nonaux_idxs) {
      nGn = Gn_fs[i].size(); 
      Gnv = Gn_fs[i].data();
      for(int ig=0; ig<nGn; ig++) 
	*body += q_g->xref[Gnv[ig]];
      ++body;
    }
    assert(bus_nonaux_idxs.size()==n);
    body -= n;

    {
      double sum;
      //v_n[i]^2*sum(b_s[s] for s=SShn[nonaux[i]])
      for(int ii=0; ii<n; ++ii) {  
	sum = 0.;
	for(auto issh : SShn_fs[bus_nonaux_idxs[ii]]) {
	  sum += b_s->xref[issh];
	}
	*body += sum * v_n->xref[ii] * v_n->xref[ii];
	++body;
      }
      body -= n;
    }
    
    {
      std::complex<double>** YredM = Ybus_red.local_data();
      double aux;

      for(int i=0; i<n; i++) {
	for(int j=0; j<n; j++) {
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  body[i] -= v_n->xref[i]*v_n->xref[j] *
	    ( YredM[i][j].real()*sin(aux) - YredM[i][j].imag()*cos(aux) );
	}
      }
    }
    
    return true;
  }

  /**
   * sum(q_g[g] for g=Gn[nonaux[i]]) - N[:Qd][nonaux[i]] +
   * v_n[i]^2*sum(b_s[s] for s=SShn[nonaux[i]]) 
   *   - sum(v_n[i]*v_n[j]*(Gred[i,j]*sin(theta_n[i]-theta_n[j]) 
   *                        - Bred[i,j]*cos(theta_n[i]-theta_n[j])) for j=1:length(nonaux)))   :=   a(i) 
   * -----------------------------------------------------------------------------------------------------
   *
   * Sparse part easy: ones for each gen. at non-aux bus
   *
   * Dense part of the Jacobian w.r.t. v(=v_n) and theta(=theta_n)
   *
   // [[[ i!=k ]]]
   // d a_i
   // ----- =  - v_i * (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)) 
   // d v_k
   //
   //
   // d a_i        n
   // ----- = -   sum    v_j* (Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
   // d v_i     j=1, j!=i
   //
   //         + 2*v_i*Bred[i,i]
   //         + 2*v_i*sum(b[s] for s=SShn[nonaux[i]])
   //
   //
   // [[[ i!=k ]]]
   //   d a_i
   // --------- = v_i*v_k *(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,k*sin(theta_i-theta_k))  
   // d theta_k
   //
   //  d a_i            n
   // --------- = -    sum    v_i*v_j *( Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
   // d theta_i     j=1, j!=i
   * 
   *    d a_i
   *  --------- = v_i^2, for s \in SShn[nonaux[i]] 
   *    d bs_s
   */
  bool PFReactiveBalanceKron::eval_Jac_eq(const OptVariables& x, bool new_x, 
					  const int& nxsparse, const int& nxdense,
					  const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
					  double** JacD)
  {
    //
    // sparse part
    //
    assert(nxsparse+nxdense == x.n());
    int row, *itnz;
 #ifdef DEBUG
    int nnz_loc=get_spJacob_eq_nnz();
    int row2=-1;
#endif
    if(iJacS && jJacS) {
      
      itnz = J_nz_idxs; row=this->index;
      for(int it=0; it<n; it++) {
	const int idxBusNonAux = bus_nonaux_idxs[it];
	const size_t sz = Gn_fs[idxBusNonAux].size();
	//q_g
	for(int ig=0; ig<sz; ++ig) {
	  iJacS[*itnz]=row; 
	  jJacS[*itnz++]=q_g->indexSparse+Gn_fs[idxBusNonAux][ig]; 
	}
	++row;
      }
#ifdef DEBUG
      assert(J_nz_idxs + nnz_loc == itnz);
      row2=row;
#endif
      
    }
    
    if(MJacS) {
      itnz = J_nz_idxs; 
      for(int it=0; it<n; it++) {
	//p_g 
	const size_t sz = Gn_fs[bus_nonaux_idxs[it]].size();
	for(int ig=0; ig<sz; ++ig) { 
	  MJacS[*itnz++] += 1.; 
	}
      }
    }

    //
    // dense part
    //
    if(JacD) {
      //values
      std::complex<double>** YredM = Ybus_red.local_data();
      double aux, aux_sin, aux_cos, aux_G_cos, aux_B_sin, aux_G_cos_B_sin, aux_G_sin, aux_B_cos;
      double vivj, Gcos_p_Bsin, Gsin__Bcos;

      assert(v_n->compute_indexDense() >=0);
      assert(theta_n->compute_indexDense() >= 0);
	
      assert(v_n->compute_indexDense()    +v_n->n     <= nxdense);
      assert(theta_n->compute_indexDense()+theta_n->n <= nxdense);
      assert(b_s->compute_indexDense()+b_s->n <= nxdense);
      
      assert(v_n->n == n);
      assert(theta_n->n == n);

      assert(SShn_fs.size() >= n);

      const int idx_col_of_v_n     = v_n->compute_indexDense();
      const int idx_col_of_theta_n = theta_n->compute_indexDense();
      const int idx_col_of_b_s     = b_s->compute_indexDense();

      assert(idx_col_of_v_n>=0 && idx_col_of_theta_n>=0 && idx_col_of_b_s>=0);

      assert(n==bus_nonaux_idxs.size());

      //in 'i2' we keep the  index of the constraint (row # in the Jacobian)
      for(int i=0, i2=this->index; i<n; ++i, ++i2) {
	
	//
	//partials w.r.t. to v_i and theta_i
	//

	//first term of da_i/v_i  -> we'll accumulate in here
	assert(JacD[i2][idx_col_of_v_n + i]==0. && "should not be written previously in this");
	JacD[i2][idx_col_of_v_n + i] = 2.0 * v_n->xref[i] * YredM[i][i].imag();
	//second term of da_i/v_i
	for(int issh : SShn_fs[bus_nonaux_idxs[i]]) {
	  assert(issh>=0 && issh<b_s->n);
	  JacD[i2][idx_col_of_v_n + i] += 2. * v_n->xref[i] * b_s->xref[issh];
	}
	
	//here we'll accumulate da_i/theta_i
	JacD[i2][idx_col_of_theta_n + i] = 0.;
	
	for(int j=0; j<i; j++) {

	  vivj = v_n->xref[j]*v_n->xref[i];
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  aux_G_cos = YredM[i][j].real()*aux_cos;
	  aux_B_cos = YredM[i][j].imag()*aux_cos;
	  aux_B_sin = YredM[i][j].imag()*aux_sin;
	  aux_G_sin = YredM[i][j].real()*aux_sin;
	  Gcos_p_Bsin = aux_G_cos + aux_B_sin;
	  Gsin__Bcos = aux_G_sin - aux_B_cos;

	  //for da_i/dv_k = - v_i * (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)) 
	  assert(JacD[i2][idx_col_of_v_n + j]==0. && "should not be written previously in this");
	  JacD[i2][idx_col_of_v_n + j] =  - v_n->xref[i]*Gsin__Bcos;

	  //for da_i/v_i: add - v_j* (Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i2][idx_col_of_v_n + i] -= v_n->xref[j]*Gsin__Bcos;
	  
	  //for da_i/dtheta_k = v_i*v_k *(Gred[i,k]*cos(theta_i-theta_k) - Bred*sin(theta_i-theta_k))  
	  assert(JacD[i2][idx_col_of_theta_n + j]==0. && "should not be written previously in this");
	  JacD[i2][idx_col_of_theta_n + j] = vivj*Gcos_p_Bsin;

	  //for da_i/theta_i: add
	  //      - v_i*v_j *( Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
	  JacD[i2][idx_col_of_theta_n + i] -= vivj*Gcos_p_Bsin;
	}
	
	for(int j=i+1; j<n; j++) {

	  vivj = v_n->xref[j]*v_n->xref[i];
	  aux = theta_n->xref[i]-theta_n->xref[j];
	  aux_sin = sin(aux);
	  aux_cos = cos(aux);

	  aux_G_cos = YredM[i][j].real()*aux_cos;
	  aux_B_cos = YredM[i][j].imag()*aux_cos;
	  aux_B_sin = YredM[i][j].imag()*aux_sin;
	  aux_G_sin = YredM[i][j].real()*aux_sin;
	  Gcos_p_Bsin = aux_G_cos + aux_B_sin;
	  Gsin__Bcos = aux_G_sin - aux_B_cos;

	  //for da_i/dv_k = - v_i * (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)) 
	  assert(JacD[i2][idx_col_of_v_n + j]==0. && "should not be written previously in this");
	  JacD[i2][idx_col_of_v_n + j] =  - v_n->xref[i]*Gsin__Bcos;

	  //for da_i/v_i: add - v_j* (Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i2][idx_col_of_v_n + i] -= v_n->xref[j]*Gsin__Bcos;
	  
	  //for da_i/dtheta_k = v_i*v_k *(Gred[i,k]*cos(theta_i-theta_k) - Bred*sin(theta_i-theta_k))  
	  assert(JacD[i2][idx_col_of_theta_n + j]==0. && "should not be written previously in this");
	  JacD[i2][idx_col_of_theta_n + j] = vivj*Gcos_p_Bsin;

	  //for da_i/theta_i: add
	  //      - v_i*v_j *( Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
	  JacD[i2][idx_col_of_theta_n + i] -= vivj*Gcos_p_Bsin;
	}

	//
	//w.r.t. b_s:  d a_i/bs_s = v_i^2 for s \in SShn[nonaux[i]]
	//
	const double vn_sq = v_n->xref[i] * v_n->xref[i];
	for(auto issh : SShn_fs[bus_nonaux_idxs[i]]) {
	  assert(issh>=0 && issh<b_s->n);
	  assert(JacD[i2][idx_col_of_b_s + issh]==0. && "should not be written previously in this");
	  JacD[i2][idx_col_of_b_s + issh] = vn_sq;
	}
      }
    }
    return true;
  }

  int PFReactiveBalanceKron::get_spJacob_eq_nnz(){ 
    int nnz = 0; 
    for(auto& i : bus_nonaux_idxs)
      nnz += Gn_fs[i].size();
    return nnz; 
  }
  
  bool PFReactiveBalanceKron::get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij)
  {
    if(n<=0) return true;
    
    int nnz = get_spJacob_eq_nnz();
    if(!J_nz_idxs) {
      J_nz_idxs = new int[nnz];
    }
#ifdef DEBUG
    int n_vij_in = vij.size();
#endif
    
    int row=this->index, *itnz=J_nz_idxs;
    for(int it=0; it<n; it++) {      
      //q_g
      for(auto g: Gn_fs[bus_nonaux_idxs[it]]) {
	vij.push_back(OptSparseEntry(row, q_g->indexSparse+g, itnz++));
      }
    
      //if(Gn_fs[bus_nonaux_idxs[it]].size()>0)
      ++row;
    }

#ifdef DEBUG
    assert(nnz+n_vij_in==vij.size());
#endif
    assert(J_nz_idxs+nnz == itnz);
    return true;
  }

 /**
   * sum(q_g[g] for g=Gn[nonaux[i]]) - N[:Qd][nonaux[i]] +
   * v_n[i]^2*sum(b_s[s] for s=SShn[nonaux[i]]) 
   *   - sum(v_n[i]*v_n[j]*(Gred[i,j]*sin(theta_n[i]-theta_n[j]) 
   *                        - Bred[i,j]*cos(theta_n[i]-theta_n[j])) for j=1:length(nonaux)))   :=   a(i) 
   * -----------------------------------------------------------------------------------------------------
   *
   * Sparse part easy: ones for each gen. at non-aux bus
   *
   * Dense part of the Jacobian w.r.t. v(=v_n) and theta(=theta_n)
   *
   // [[[ i!=k ]]]
   // d a_i
   // ----- =  - v_i * (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k)) 
   // d v_k
   // d a_i        n
   // ----- = -   sum    v_j* (Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
   // d v_i     j=1, j!=i
   //         + 2*v_i*Bred[i,i]
   //         + 2*v_i*sum(b[s] for s=SShn[nonaux[i]])
   //
   // [[[ i!=k ]]]
   //   d a_i
   // --------- = v_i*v_k *(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,k]*sin(theta_i-theta_k))  
   // d theta_k
   //  d a_i            n
   // --------- = -    sum    v_i*v_j *( Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j))
   // d theta_i     j=1, j!=i
   *    d a_i
   *  --------- = v_i^2, for s \in SShn[nonaux[i]] 
   *    d bs_s
   * ****************************************************
   * Dense part of Hessian of a_i w.r.t v_k,v_j  (j>=k)
   * k=1,...,i-1 all zeros except
   *  d2 a_i/dvk dvi = - (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k))
   * k=i
   *  d2 a_i/dvi dvi = 2*Bred[i,i] + 2*sum(b[s] for s=SShn[nonaux[i]])
   *  d2 a_i/dvi dvj = -(Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
   * k>i
   *  d2 a_i/dv_k dv_j = 0  (j>=k)
   * ***************************************************
   * ***************************************************
   * Dense part of the Hessian of a_i w.r.t. v_k, theta_j  (j =1,...,n)
   * k=1,...,i-1 all zeros except
   *  d2 a_i/dv_k dtheta_k =  v_i*(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,j]*sin(theta_i-theta_k))
   *  d2 a_i/dv_k dtheta_i = -v_i*(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,j]*sin(theta_i-theta_k))
   * k=i
   *  d2 a_i/dv_i dtheta_i = - sum (v_j*(Gred[i,j]*cos(theta_i-theta_j)+Bred[i,j]*sin(theta_i-theta_j))
   *                           j!=i
   *  d2 a_i/dv_i dtheta_j = + v_j * (Gred[i,j]*cos(theta_i-theta_j)+Bred[i,j]*sin(theta_i-theta_k))
   * k>i
   *  d2 a_i/dv_k dtheta_i = -v_i*(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,j]*sin(theta_i-theta_k))
   *  d2 a_i/dv_k dtheta_k =  v_i*(Gred[i,k]*cos(theta_i-theta_k) + Bred[i,j]*sin(theta_i-theta_k))
   *  d2 a_i/dv_k dtheta_j = 0 (j!=k j!=i)
   * ***************************************************
   * ***************************************************
   * Dense part of the Hessian of a_i w.r.t. v_k, bs_j  (j =1,...,n)
   * all zero except
   * d2 a_i / dv_i db_s = 2*v_i for s=SShn[nonaux[i]])
   * ***************************************************
   * ***************************************************
   * Dense part of the Hessian of a_i w.r.t. theta_k theta_j (j>=k)
   * k<i  zero for j!=i and j!=k
   *  d2 a_i/dtheta_k dtheta_k = vi*vk*( Gred[i,k]*sin(theta_i-theta_k)-Bred[i,k]*cos(theta_i-theta_k))
   *  d2 a_i/dtheta_k dtheta_i = vi*vk*(-Gred[i,k]*sin(theta_i-theta_k)+Bred[i,k]*cos(theta_i-theta_k))
   * k=i 
   *  d2 a_i/dtheta_i^2 = - sum vi*vj*(-Gred[i,j]*sin(theta_i-theta_j) + Bred[i,j]*cos(theta_i-theta_j))
   *                       j!=i
   *  d2 a_i/dtheta_i dtheta_j = - vi*vj*(Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
   * k>i  zero for j>0
   *  d2 a_i / dtheta_k dtheta_k = vi*vk*( Gred[i,k]*sin(theta_i-theta_k)-Bred[i,k]*cos(theta_i-theta_k))
   * ***************************************************
   * ***************************************************
   * Dense part of the Hessian of a_i w.r.t. theta_k bs_j (j=1,...,n)
   *  all zeros
   * ***************************************************
   * ***************************************************
   * Dense part of the Hessian of a_i w.r.t. bs_k bs_j (j>=k)
   * d2 a_i / dbs_k dbs_j = 0 
   */
  bool PFReactiveBalanceKron::eval_HessLagr(const OptVariables& x, bool new_x, 
					    const OptVariables& lambda_vars, bool new_lambda,
					    const int& nxsparse, const int& nxdense, 
					    const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
					    double** HDD,
					    const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //
    // sparse part is empty
    //
    
    //
    // dense part
    //
    // Note: Only upper triangle part of HDD is updated
    if(NULL!=iHSS && NULL!=jHSS) {
    }

    if(HDD) {
      const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
      assert(lambda != NULL);
      assert(lambda->n == n);
      assert(v_n->n == n);
      assert(theta_n->n == n);
      
      const int idx_col_of_v_n     = v_n->compute_indexDense();
      const int idx_col_of_theta_n = theta_n->compute_indexDense();
      const int idx_col_of_b_s     = b_s->compute_indexDense();
      
      assert(idx_col_of_v_n >= 0 && idx_col_of_v_n+v_n->n <= nxdense);
      assert(idx_col_of_theta_n >= 0 && idx_col_of_theta_n+theta_n->n <= nxdense);
      assert(idx_col_of_b_s >=0 && idx_col_of_b_s+b_s->n <= nxdense);

      std::complex<double>** YredM = Ybus_red.local_data();
      double theta_diff, aux_G_sin, aux_B_cos, Gcos, Bsin, res;
      
      assert(n==bus_nonaux_idxs.size());

      //loop over constraints/lambdas
      for(int i=0; i<n; i++) {
	
	const double& lambda_i = lambda->xref[i];

	//quick return for when the derivatives are checked
	if(lambda_i==0.) continue;
	
	int k,j;
	const int idx_col_of_v_n_elemi = idx_col_of_v_n+i;
	//********************************************************
	//Dense part of Hessian of a_i w.r.t v_k,v_j  (j>=k)
	//********************************************************

	//for k=1,...,i-1 Hessian entries all zeros except
	//d2 a_i/dvk dvi = - (Gred[i,k]*sin(theta_i-theta_k) - Bred[i,k]*cos(theta_i-theta_k))
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];

	  assert(idx_col_of_v_n+k <= idx_col_of_v_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_v_n_elemi] -= 
	    lambda_i * (YredM[i][k].real()*sin(theta_diff) - YredM[i][k].imag()*cos(theta_diff));
	}
	//---> k=i
	k=i;
	//d2 a_i/dvi dvi = 2*Bred[i,i] + 2*sum(b[s] for s=SShn[nonaux[i]])
	HDD[idx_col_of_v_n_elemi][idx_col_of_v_n_elemi] += 2.0*lambda_i*YredM[i][i].imag();
	for(int issh : SShn_fs[bus_nonaux_idxs[i]]) {
	  HDD[idx_col_of_v_n_elemi][idx_col_of_v_n_elemi] += 2.0*lambda_i*b_s->xref[issh];
	}
	//d2 a_i/dvi dvj = -(Gred[i,j]*sin(theta_i-theta_j) - Bred[i,j]*cos(theta_i-theta_j))
	for(j=k+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  
	  assert(idx_col_of_v_n_elemi <= idx_col_of_v_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_v_n+j] -=
	    lambda_i * (YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));
	}
	//---> k>i all zeros

	//********************************************************
	// Dense part of the Hessian of a_i w.r.t. v_k, theta_j
	//********************************************************
	
	// ----> k=1,...,i-1 all zeros except d2 a_i/dv_k dtheta_k  and  d2 a_i/dv_k dtheta_i
	// index of theta_k in dense part of the Hessian is idx_col_of_theta_n
	const int idx_col_of_theta_n_elemi = idx_col_of_theta_n+i;
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  Gcos = YredM[i][k].real()*cos(theta_diff);
	  Bsin = YredM[i][k].imag()*sin(theta_diff);
	  res = lambda_i * v_n->xref[i] * (Gcos + Bsin);
	  //d2 a_i/dv_k dtheta_k
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n+k);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n+k] += res;
	  
	  //d2 a_i/dv_k dtheta_i
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n_elemi] -= res;
	}

	// ---> k=i <---
	k=i;
	//                          n
	// d2 a_i/dv_i dtheta_i= - sum (v_j*(Gred[i,j]*cos(theta_i-theta_j)+Bred[i,j]*sin(theta_i-theta_j))
	//                       j=1, j!=i
	double sum=0.; 
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[j]*(YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff));
	}
	assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n_elemi);
	HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n_elemi] -= lambda_i*sum;

	// d2 a_i/dv_i dtheta_j = v_j * (Gred[i,j]*cos(theta_i-theta_j)+Bred[i,j]*sin(theta_i-theta_k))
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];

	  assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n+j] += lambda_i*v_n->xref[j]*
	    (YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff)); //!
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  
	  assert(idx_col_of_v_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_theta_n+j] += lambda_i*v_n->xref[j]*
	    (YredM[i][j].real()*cos(theta_diff) + YredM[i][j].imag()*sin(theta_diff)); //!
	}

	// ---> k=i+1, ..., n  <---
	for(k=i+1; k<n; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  res = lambda_i*v_n->xref[i]*(YredM[i][k].real()*cos(theta_diff) +
				       YredM[i][k].imag()*sin(theta_diff));
	  //d2 a_i/dv_k dtheta_k is nonzero
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n+k);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n+k] += res;
	    
	  //d2 a_i/dv_k dtheta_i
	  assert(idx_col_of_v_n+k <= idx_col_of_theta_n_elemi);
	  HDD[idx_col_of_v_n+k][idx_col_of_theta_n_elemi] -= res;
	}

	/* ***************************************************
	 * Dense part of the Hessian of a_i w.r.t. v_k, bs_j  (j =1,...,n)
	 * all zero except
	 * d2 a_i / dv_i db_s = 2*v_i for s=SShn[nonaux[i]])
	 * ***************************************************/
	for(int issh : SShn_fs[bus_nonaux_idxs[i]]) {
	  assert(issh>=0 && issh<b_s->n);
	  assert(idx_col_of_v_n_elemi < idx_col_of_b_s+issh);
	  HDD[idx_col_of_v_n_elemi][idx_col_of_b_s+issh] += lambda_i*2.0*v_n->xref[i];
	}

	//********************************************************
	// Dense part of the Hessian of a_i w.r.t theta_k, theta_j (j>=k)
	//********************************************************
	
	// ---> k<i <---
	// only d2 a_i / dtheta_k dtheta_k and d2 a_i / dtheta_k dtheta_i are nonzeros
	for(k=0; k<i; k++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  res = v_n->xref[i]*v_n->xref[k]*
	    (YredM[i][k].real()*sin(theta_diff) - YredM[i][k].imag()*cos(theta_diff));
	  const int idx_col_of_theta_n_elemk = idx_col_of_theta_n+k;
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemk] += lambda_i * res;
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemi] -= lambda_i * res;
	}
	// ---> k=i  <---
	k=i; 
	// d2 a_i / dtheta_i dtheta_i
	sum = 0.;
	for(j=0; j<i; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[i][j].real()*sin(theta_diff) + YredM[i][j].imag()*cos(theta_diff));
	}
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];
	  sum += v_n->xref[i]*v_n->xref[j]*
	    (-YredM[i][j].real()*sin(theta_diff) + YredM[i][j].imag()*cos(theta_diff));
	}
	HDD[idx_col_of_theta_n_elemi][idx_col_of_theta_n_elemi] -= lambda_i*sum;
	
	// d2 a_i / dtheta_i dtheta_j  (k=i) for j>i (j<i was updated by symmetry at ---> k<i <---)
	for(j=i+1; j<n; j++) {
	  theta_diff = theta_n->xref[i]-theta_n->xref[j];

	  assert(idx_col_of_theta_n_elemi <= idx_col_of_theta_n+j);
	  HDD[idx_col_of_theta_n_elemi][idx_col_of_theta_n+j] -=
	    lambda_i * v_n->xref[i] * v_n->xref[j] *
	    (YredM[i][j].real()*sin(theta_diff) - YredM[i][j].imag()*cos(theta_diff));
	}

	// ---> k>i : only d2 a_i / dtheta_k dtheta_k is nonzero
	for(k=i+1; k<n; k++) {
	  const int idx_col_of_theta_n_elemk = idx_col_of_theta_n+k;
	  theta_diff = theta_n->xref[i]-theta_n->xref[k];
	  
	  //assert(idx_col_of_theta_n_elemk <= idx_col_of_theta_n_elemk);
	  HDD[idx_col_of_theta_n_elemk][idx_col_of_theta_n_elemk] +=
	    lambda_i * v_n->xref[i] * v_n->xref[k] *
	    (YredM[i][k].real()*sin(theta_diff) - YredM[i][k].imag()*cos(theta_diff));
	}
      } // end of for over i=1,..,n
    }
    return true;
  }


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
  VoltageConsAuxBuses::VoltageConsAuxBuses(const std::string& id_in, int numcons,
					   OptVariablesBlock* v_n_in,
					   OptVariablesBlock* theta_n_in,
					   OptVariablesBlock* v_aux_n_in,
					   OptVariablesBlock* theta_aux_n_in,
					   const std::vector<int>& idxs_busviol_in_aux_buses_in,
					   const hiop::hiopMatrixComplexDense& vmap_in)
    : OptConstraintsBlockMDS(id_in, numcons),
      v_n_(v_n_in), theta_n_(theta_n_in),
      v_aux_n_(v_aux_n_in), theta_aux_n_(theta_aux_n_in),
      idxs_busviol_in_aux_buses_(idxs_busviol_in_aux_buses_in),
      vmap_(vmap_in) //v_n(v_n_),
  {
    for(int i=0; i<n; i++) lb[i]=0.;
    DCOPY(&n, lb, &ione, ub, &ione);

    assert(n == 2*idxs_busviol_in_aux_buses_.size());
    
    assert(v_n_->sparseBlock==false);
    assert(theta_n_->sparseBlock==false);

    assert(v_n_->indexSparse<=0);
    assert(theta_n_->indexSparse<=0);
  }
  
  VoltageConsAuxBuses::~VoltageConsAuxBuses()
  {
  }
  
  void VoltageConsAuxBuses::append_constraints(const std::vector<int>& idxs_busviol_in_aux_buses_new)
  {
    assert(v_n_->n == idxs_busviol_in_aux_buses_.size()+idxs_busviol_in_aux_buses_new.size());
    assert(v_n_->n == theta_n_->n);
    assert(vmap_.n() == v_n_->n);
    for(auto idx: idxs_busviol_in_aux_buses_new) {
      assert(idx >= 0);
      assert(idx < vmap_.m());
      idxs_busviol_in_aux_buses_.push_back(idx);
    }
    n = 2*idxs_busviol_in_aux_buses_.size();
    assert(false); //reallocate lb and ub
  }

  bool VoltageConsAuxBuses::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
  {
    std::complex<double>** vmap = vmap_.local_data();

    assert(vmap_.n() == v_n_->n);
    assert(vmap_.m() >= idxs_busviol_in_aux_buses_.size());
    assert(n == 2*idxs_busviol_in_aux_buses_.size());
    assert(theta_n_->n == v_n_->n);

    double* rhs = body + this->index;

    //loop over all buses with violations (for each there are two constraints)
    for(int b=0; b<idxs_busviol_in_aux_buses_.size(); ++b) {
      const int idx_in_aux_buses = idxs_busviol_in_aux_buses_[b];
      
      assert(idx_in_aux_buses>=0);
      assert(idx_in_aux_buses<vmap_.m());

      //
      //first equality for b
      //
      int idx_con = 2*b;
      rhs[idx_con] = - v_aux_n_->xref[b]*cos(theta_aux_n_->xref[b]);
      for(int i=0; i<v_n_->n; ++i) {
	rhs[idx_con] += vmap[idx_in_aux_buses][i].real() * v_n_->xref[i] * cos(theta_n_->xref[i]);
	rhs[idx_con] -= vmap[idx_in_aux_buses][i].imag() * v_n_->xref[i] * sin(theta_n_->xref[i]);
      }
      //
      //second equality for b
      //
      idx_con = idx_con + 1;
      rhs[idx_con] = - v_aux_n_->xref[b]*sin(theta_aux_n_->xref[b]);
      for(int i=0; i<v_n_->n; ++i) {
	rhs[idx_con] += vmap[idx_in_aux_buses][i].real() * v_n_->xref[i] * sin(theta_n_->xref[i]);
	rhs[idx_con] += vmap[idx_in_aux_buses][i].imag() * v_n_->xref[i] * cos(theta_n_->xref[i]);
      }
    }
    return true;
  }

  /*********************************************************************************************
   * for nix = idxs_busviol_in_aux_buses
   * a(2*nix)  = - vaux_n[nix]*cos(thetaaux_n[nix]) +
   *   sum((re_ynix[i]*v_n[i]*cos(theta_n[i]) - im_ynix[i]*v_n[i]*sin(theta_n[i])) for i=1:length(nonaux)))
   *
   * d a_{2*nix)
   * -----------  = 0, unless j=nix when is = - cos(thetaaux_n[nix])
   * d vaux_n[j]
   *
   * d a_{2*nix)
   * -----------  = 0, unless j=nix when is = vaux_n[nix]*sin(thetaaux_n[nix])
   * d thetaaux_n[j]
   * 
   * d a_{2*nix)
   * -----------  = re_ynix[j]*cos(theta_n[j]) - im_ynix[i]*sin(theta_n[j])
   * d v_n[j]
   *
   * d a_{2*nix)
   * -----------  = - re_ynix[j]*v_n[j]*sin(theta_n[j]) - im_ynix[j]*v_n[j]*cos(theta_n[j])
   * d theta_n[j]
   * .............................................................................
   *
   * a(2*nix+1)= - vaux_n[nix]*sin(thetaaux_n[nix]) +
   *   sum((re_ynix[i]*v_n[i]*sin(theta_n[i]) + im_ynix[i]*v_n[i]*cos(theta_n[i])) for i=1:length(nonaux)))
   *
   * d a_{2*nix+1}
   * -------------  = 0, unless j=nix when is = - sin(thetaaux_n[nix])
   * d vaux_n[j]
   *
   * d a_{2*nix+1}
   * -------------  = 0, unless j=nix when is = - vaux_n[nix]*cos(thetaaux_n[nix])
   * d thetaaux_n[j]
   * 
   * d a_{2*nix+1}
   * ------------  = re_ynix[j]*sin(theta_n[j]) + im_ynix[i]*cos(theta_n[j])
   * d v_n[j]
   *
   * d a_{2*nix+1}
   * ------------- = re_ynix[j]*v_n[j]*cos(theta_n[j]) - im_ynix[j]*v_n[j]*sin(theta_n[j])
   * d theta_n[j]
   *
   * where re_ynix=Re(vmap[nix,:]) and im_ynix=Im(vmap[nix,:])

   *********************************************************************************************/

  bool VoltageConsAuxBuses::eval_Jac_eq(const OptVariables& x, bool new_x, 
					const int& nxsparse, const int& nxdense,
					const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
					double** JacD)
  {
    std::complex<double>** vmap = vmap_.local_data();
    //
    // sparse part is empty
    //
    assert(0 == get_spJacob_eq_nnz());
    //printf("------- nnzJacS=%d\n", nnzJacS);
    //assert(nnzJacS==0);

    if(NULL==JacD) return true;
    //
    // dense spart
    //

    assert(2*idxs_busviol_in_aux_buses_.size() == n); 
    assert(idxs_busviol_in_aux_buses_.size() == theta_aux_n_->n);
    assert(idxs_busviol_in_aux_buses_.size() == v_aux_n_->n);

    const int idx_col_of_vaux     = v_aux_n_->compute_indexDense();
    const int idx_col_of_thetaaux = theta_aux_n_->compute_indexDense();
    const int idx_col_of_v        = v_n_->compute_indexDense();
    const int idx_col_of_theta    = theta_n_->compute_indexDense();

    for(int ii=0; ii<2*nxdense; ii++) JacD[this->index][ii] = 0.;
    
    for(int iauxb=0; iauxb<idxs_busviol_in_aux_buses_.size(); ++iauxb) {
      
      const int idx_in_aux_buses = idxs_busviol_in_aux_buses_[iauxb];
      assert(idx_in_aux_buses>=0); assert(idx_in_aux_buses<vmap_.m());

      assert(idx_col_of_vaux >=0 && idx_col_of_vaux + v_aux_n_->n <= nxdense);
      assert(idx_col_of_thetaaux >=0 && idx_col_of_thetaaux + theta_aux_n_->n <= nxdense);
      assert(idx_col_of_v >=0 && idx_col_of_v + v_n_->n <= nxdense);
      assert(idx_col_of_theta >=0 && idx_col_of_theta + theta_n_->n <= nxdense);

      //
      // first and second row combined to save on cos and sin evals 
      //

      const double cos_th_aux = cos(theta_aux_n_->xref[iauxb]);
      const double sin_th_aux = sin(theta_aux_n_->xref[iauxb]);
      
      int i = this->index+2*iauxb;

      //da_i / dvaux_n[nix] = - cos(thetaaux_n[nix]
      JacD[i][idx_col_of_vaux+iauxb] = - cos_th_aux;

      //da_i / dthetaaux_n[nix]
      JacD[i][idx_col_of_thetaaux+iauxb] = v_aux_n_->xref[iauxb] * sin_th_aux;

      //d_a{i+1} / dvaux_n and dthetaaux_n
      JacD[i+1][idx_col_of_vaux+iauxb] = - sin_th_aux;
      JacD[i+1][idx_col_of_thetaaux+iauxb] = - v_aux_n_->xref[iauxb] * cos_th_aux;

      //da_i     / dv_n[j] and theta_n[j] for j=1,...,length(nonaux)
      //da_{i+1} / dv_n[j] and theta_n[j] for j=1,...,length(nonaux)
      for(int j=0; j<v_n_->n; ++j) {
	const double cos_th = cos(theta_n_->xref[j]);
	const double sin_th = sin(theta_n_->xref[j]);
	JacD[i][idx_col_of_v + j] =
	  + vmap[idx_in_aux_buses][j].real() * cos_th
	  - vmap[idx_in_aux_buses][j].imag() * sin_th;

	JacD[i][idx_col_of_theta+j] =
	  v_n_->xref[j]*( - vmap[idx_in_aux_buses][j].real() * sin_th
			  - vmap[idx_in_aux_buses][j].imag() * cos_th);

	JacD[i+1][idx_col_of_v + j] =
	  + vmap[idx_in_aux_buses][j].real() * sin_th
	  + vmap[idx_in_aux_buses][j].imag() * cos_th;
	JacD[i+1][idx_col_of_theta+j] =
	  v_n_->xref[j]*( + vmap[idx_in_aux_buses][j].real() * cos_th
			  - vmap[idx_in_aux_buses][j].imag() * sin_th);

      }
      
    } // end of for over aux buses with violations
    return true;
  }

  int VoltageConsAuxBuses::get_spJacob_eq_nnz()
  {
    return 0;
  }
  bool VoltageConsAuxBuses::get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij)
  {
    return true;
  }

  /*********************************************************************************************
   * for nix = idxs_busviol_in_aux_buses
   * a(2*nix)  = - vaux_n[nix]*cos(thetaaux_n[nix]) +
   *   sum((re_ynix[i]*v_n[i]*cos(theta_n[i]) - im_ynix[i]*v_n[i]*sin(theta_n[i])) for i=1:length(nonaux)))
   *
   * d a_{2*nix)
   * -----------  = 0, unless j=nix when is = - cos(thetaaux_n[nix])
   * d vaux_n[j]
   *
   * d2 a_{2*nix}
   * -------------------------- = sin(thetaaux_n[nix])
   * dvaux[nix] dthetaaux[nix]
   * ==================================================================================================
   * d a_{2*nix)
   * -----------  = 0, unless j=nix when is = vaux_n[nix]*sin(thetaaux_n[nix])
   * d thetaaux_n[j]
   *
   * d2 a_{2*nix}
   * -------------------- = sin(thetaaux_n[nix])
   * dthaux[nix] dvaux[nix]
   *
   * d2 a_{2*nix}
   * ---------------------- = vaux_n[nix]*cos(thetaaux_n[nix])
   * dthaux[nix] dthaux[nix]
   * ==================================================================================================
   * d a_{2*nix)
   * -----------  = re_ynix[j]*cos(theta_n[j]) - im_ynix[i]*sin(theta_n[j])
   * d v_n[j]
   *
   * d2 a_{2*nix}           d2 a_{2*nix}           
   * ------------- = 0      ------------- = - re_ynix[j]*sin(theta_n[j]) - im_ynix[i]*cos(theta_n[j])
   * dv[j] dv[j]            dv[j] dth[j]
   * ==================================================================================================
   * d a_{2*nix)
   * -----------  = - re_ynix[j]*v_n[j]*sin(theta_n[j]) - im_ynix[j]*v_n[j]*cos(theta_n[j])
   * d theta_n[j]
   *
   * d2 a_{2*nix}
   * ------------- =  - re_ynix[j]*sin(theta_n[j]) - im_ynix[j]*cos(theta_n[j])
   * dth[j] dv[j]
   *
   * d2 a_{2*nix}
   * ------------- =  v_n[j] * [- re_ynix[j]*cos(theta_n[j]) + im_ynix[j]*sin(theta_n[j]) ]
   * dth[j] dth[j]
   *
   * ...................................................................................................
   * ...................................................................................................
   * ...................................................................................................
   *
   * a(2*nix+1)= - vaux_n[nix]*sin(thetaaux_n[nix]) +
   *   sum((re_ynix[i]*v_n[i]*sin(theta_n[i]) + im_ynix[i]*v_n[i]*cos(theta_n[i])) for i=1:length(nonaux)))
   *
   * d a_{2*nix+1}
   * -------------  = 0, unless j=nix when is = - sin(thetaaux_n[nix])
   * d vaux_n[j]
   *
   * d2 a_{2*nix+1}
   * --------------------- = - cos(thetaaux_n[nix])
   * d vaux[nix] dthaux[nix]
   * ==================================================================================================
   * d a_{2*nix+1}
   * -------------  = 0, unless j=nix when is = - vaux_n[nix]*cos(thetaaux_n[nix])
   * d thetaaux_n[j]
   * 
   * d2 a_{2*nix+1}
   * ------------------------ = - cos(thetaaux_n[nix])
   * d thaux[nix] dvaux[nix]
   * 
   * d2 a_{2*nix+1}
   * ------------------------ = + vaux_n[nix]*sin(thetaaux_n[nix])
   * d thaux[nix] dthaux[nix]
   * ==================================================================================================
   * d a_{2*nix+1}
   * ------------  = re_ynix[j]*sin(theta_n[j]) + im_ynix[i]*cos(theta_n[j])
   * d v_n[j]
   *
   * d2 a_{2*nix+1}
   * --------------  = re_ynix[j]*cos(theta_n[j]) - im_ynix[i]*sin(theta_n[j])
   * d v[j] dth[j]
   * ==================================================================================================
   * d a_{2*nix+1}
   * ------------- = re_ynix[j]*v_n[j]*cos(theta_n[j]) - im_ynix[j]*v_n[j]*sin(theta_n[j])
   * d theta_n[j]
   *
   * d2 a_{2*nix+1}
   * ---------------- = re_ynix[j]*cos(theta_n[j]) - im_ynix[j]*in(theta_n[j])
   * dth_n[j] dv_n[j]
   *
   * d2 a_{2*nix+1}
   * ---------------- = v_n[j] * [ - re_ynix[j]*sin(theta_n[j]) - im_ynix[j]*cos(theta_n[j]) ]
   * dth_n[j] dth_n[j]
   *
   * ==================================================================================================
   *****************************************************************************************************/


  bool VoltageConsAuxBuses::eval_HessLagr(const OptVariables& x, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nxsparse, const int& nxdense, 
			       const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
			       double** HDD,
			       const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    //
    //sparse part is empty -> do nothing
    //

    //
    //dense part
    //
    
    //quick return
    if(!HDD) return true;
    
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda != NULL);
    assert(lambda->n == n);
    assert(2*v_aux_n_->n == n);
    assert(2*theta_aux_n_->n == n);
    assert(lambda->n == 2*idxs_busviol_in_aux_buses_.size());
    
    const int idx_col_of_vaux     = v_aux_n_->compute_indexDense();
    const int idx_col_of_thetaaux = theta_aux_n_->compute_indexDense();
    const int idx_col_of_v        = v_n_->compute_indexDense();
    const int idx_col_of_theta    = theta_n_->compute_indexDense();

    std::complex<double>** vmap = vmap_.local_data();
    
    //iterate over constraints two by two
    for(size_t iauxb=0; iauxb<idxs_busviol_in_aux_buses_.size(); ++iauxb) { 

      const int idx_in_aux_buses = idxs_busviol_in_aux_buses_[iauxb];
      assert(idx_in_aux_buses>=0); assert(idx_in_aux_buses<vmap_.m());

      double lambda_i=0.;
      double lambda_ip1=0.;
      {
	const int i = 2*iauxb;
	lambda_i = lambda->xref[i];
	lambda_ip1 = lambda->xref[i+1];
      
	//quick return for when the derivatives are checked
	if(lambda_i==0. && lambda_ip1==0.) continue;

      }
      // d2 a_{i} / dvaux[iauxb] dthaux[iauxb]
      assert(idx_col_of_vaux+iauxb < nxdense);
      assert(idx_col_of_thetaaux+iauxb < nxdense);
      assert(idx_col_of_vaux+iauxb <= idx_col_of_thetaaux+iauxb);
      
      const double sin_th_aux = sin(theta_aux_n_->xref[iauxb]);
      const double cos_th_aux = cos(theta_aux_n_->xref[iauxb]);
      
      HDD[idx_col_of_vaux+iauxb][idx_col_of_thetaaux+iauxb] += lambda_i   * sin_th_aux;
      HDD[idx_col_of_vaux+iauxb][idx_col_of_thetaaux+iauxb] +=-lambda_ip1 * cos_th_aux;

      
      // d2 a_i / dthaux[nix] dthaux[nix]
      HDD[idx_col_of_thetaaux+iauxb][idx_col_of_thetaaux+iauxb] +=
	lambda_i * v_aux_n_->xref[iauxb] * cos_th_aux;
       HDD[idx_col_of_thetaaux+iauxb][idx_col_of_thetaaux+iauxb] +=
	lambda_ip1 * v_aux_n_->xref[iauxb] * sin_th_aux;
      

      // d2 a_i / dv[j]  dth[j]  for j=1,..., length(nonaux)
      // d2 a_i / dth[j] dth[j]  for j=1,..., length(nonaux)
      for(int j=0; j<v_n_->n; ++j) {
	assert(idx_col_of_v+j<idx_col_of_theta+j);
	assert(idx_col_of_v+j<nxdense);
	assert(idx_col_of_theta+j<nxdense);

	const int idx_vj = idx_col_of_v+j;
	const int idx_thj = idx_col_of_theta+j;
	
	const double sin_th = sin(theta_n_->xref[j]);
	const double cos_th = cos(theta_n_->xref[j]);

	//
	// d2 a_i / dv[j] dth[j]  and  d2 a_i+1 / dv[j] dth[j]
	HDD[idx_vj][idx_thj] +=
	  lambda_i * ( - vmap[idx_in_aux_buses][j].real() * sin_th
		       - vmap[idx_in_aux_buses][j].imag() * cos_th );

	HDD[idx_vj][idx_thj] +=
	  lambda_ip1 * ( + vmap[idx_in_aux_buses][j].real() * cos_th
			 - vmap[idx_in_aux_buses][j].imag() * sin_th );
	//
	// d2 a_i / dth[j] dth[j]  and  d2 a_i+1 / dth[j] dth[j]
	HDD[idx_thj][idx_thj] +=
	  lambda_i * v_n_->xref[j] * ( - vmap[idx_in_aux_buses][j].real() * cos_th
				       + vmap[idx_in_aux_buses][j].imag() * sin_th );

	HDD[idx_thj][idx_thj] +=
	  lambda_ip1 * v_n_->xref[j] * ( - vmap[idx_in_aux_buses][j].real() * sin_th
					 - vmap[idx_in_aux_buses][j].imag() * cos_th );
      }
    } // end outer loop 
    return true;
  }


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
   *
   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - (L[:RateBase][l]*vi + slack_li)^2 <=0
   *
   *
   *********************************************************************************************/
  
  LineThermalViolCons::LineThermalViolCons(const std::string& id_in,
					   int numcons,
					   const std::vector<int>& Lidx_overload_in,
					   const std::vector<int>& Lin_overload_in,
					   /*idxs of L_From and L_To in N_Bus (in L_Nidx[0] and L_Nidx[1])*/
					   const std::vector<std::vector<int> >& L_Nidx_in,
					   const std::vector<double>& L_Rate_in,
					   const std::vector<double>& L_G_in,
					   const std::vector<double>& L_B_in,
					   const std::vector<double>& L_Bch_in,
					   /*from N idxs to idxs in aux and nonaux optimiz vars*/
					   const std::vector<int>& map_idxbuses_idxsoptimiz_in, 
					   OptVariablesBlock* v_n_in,
					   OptVariablesBlock* theta_n_in,
					   OptVariablesBlock* v_aux_n_in,
					   OptVariablesBlock* theta_aux_n_in)
    : OptConstraintsBlockMDS(id_in, numcons),
      Lidx_overload_(Lidx_overload_in), Lin_overload_(Lin_overload_in),
      L_Nidx_(L_Nidx_in),
      map_idxbuses_idxsoptimiz_(map_idxbuses_idxsoptimiz_in),
      L_Rate_(L_Rate_in), L_G_(L_G_in), L_B_(L_B_in), L_Bch_(L_Bch_in),
      v_n_(v_n_in), theta_n_(theta_n_in),
      v_aux_n_(v_aux_n_in), theta_aux_n_(theta_aux_n_in),
      slacks_(NULL),
      J_nz_idxs_(NULL)
  {
    assert(false == v_aux_n_->sparseBlock);
    assert(false == theta_aux_n_->sparseBlock);

    assert(false == v_n_->sparseBlock);
    assert(false == theta_n_->sparseBlock);

    for(int i=0; i<n; i++) lb[i]=0.;
    DCOPY(&n, lb, &ione, ub, &ione);
  }
  LineThermalViolCons::~LineThermalViolCons()
  {
  }
  
  //add (append to existing block of) line thermal violations
  //indexes passed in 'vtheta_aux_idxs_new'
  void LineThermalViolCons::append_constraints(const std::vector<int>& Lidx_overload_pass_in,
					       const std::vector<int>& Lin_overload_pass_in)
  {
    assert(Lidx_overload_.size() == Lin_overload_.size());
    assert(Lidx_overload_pass_in.size() == Lin_overload_pass_in.size());

    const int num_new_cons = Lidx_overload_pass_in.size();
    if(num_new_cons==0) return;

    const int new_n = this->n+num_new_cons;
    
    double* lb_new = new double[new_n];
    DCOPY(&n, lb, &ione, lb_new, &ione);
    for(int i=n; i<new_n; ++i)
      lb_new[i] = 0.;
    delete[] lb;
    lb = lb_new;
    
    double* ub_new = new double[new_n];
    DCOPY(&n, ub, &ione, ub_new, &ione);
    for(int i=n; i<new_n; ++i)
      ub_new[i] = 0.;
    delete[] ub;
    ub = ub_new;

    for(auto& e : Lidx_overload_pass_in)
      Lidx_overload_.push_back(e);

    for(auto& e : Lin_overload_pass_in)
      Lin_overload_.push_back(e);

    n = new_n;
    assert(Lidx_overload_.size() == n);

    //add new variables in the slacks
    assert(slacks_);
    assert(owner_cons_);
    assert(owner_cons_->opt_problem());

    vector<double> s_lb(num_new_cons, 0.);
    owner_cons_->opt_problem()->primal_variables()->
      append_vars_to_varsblock(slacks_->id, num_new_cons, s_lb.data(), NULL, NULL);

  }

  // void LineThermalViolCons::get_v_theta_vals(const int& idx_bus, double& vval, double& thetaval)
  // {
  //   const int idx_aux_or_nonaux = map_idxbuses_idxsoptimiz_[idx_bus];
  //   //-1 are aux buses not included in optimization (no voltage and transmission violations at these)
  //   //thermal violation constraints should not appear at these "-1" buses
  //   assert(idx_aux_or_nonaux != -1);
  //   int idx_aux=-1, idx_nonaux=-1;
  //   if(idx_aux_or_nonaux<=-2) {
  //     idx_aux = -idx_aux_or_nonaux - 2;
  //     assert(idx_aux>=0);
  //     assert(idx_aux < v_aux_n_->n);
  //   } else {
  //     if(idx_aux_or_nonaux >= 0)
  // 	idx_nonaux = idx_aux_or_nonaux;
  //     else
  // 	assert(false);
  //   }
    
  //   if(idx_nonaux>=0) {
  //     assert(idx_aux == -1);
      
  //     vval = v_n_->xref[idx_nonaux];
  //     thetaval = theta_n_->xref[idx_nonaux];
  //   } else {
  //     assert(idx_aux>=0);
  //     assert(idx_aux < v_n_->n);
  //     assert(idx_nonaux == -1);
      
  //     vval = v_aux_n_->xref[idx_aux];
  //     thetaval = theta_aux_n_->xref[idx_aux];
  //   }
  // }

  void LineThermalViolCons::get_v_theta_vals_and_idxs(const int& idx_bus,
						      double& vval, double& thetaval,
						      int& v_idx_out, int& theta_idx_out)
  {
    v_idx_out = theta_idx_out = -1;
    
    const int idx_aux_or_nonaux = map_idxbuses_idxsoptimiz_[idx_bus];
    //-1 are aux buses not included in optimization (no voltage and transmission violations at these)
    //thermal violation constraints should not appear at these "-1" buses
    assert(idx_aux_or_nonaux != -1);
    int idx_aux=-1, idx_nonaux=-1;
    if(idx_aux_or_nonaux<=-2) {
      idx_aux = -idx_aux_or_nonaux - 2;
      assert(idx_aux>=0);
      assert(idx_aux < v_aux_n_->n);
    } else {
      if(idx_aux_or_nonaux >= 0)
	idx_nonaux = idx_aux_or_nonaux;
      else
	assert(false);
    }
    
    if(idx_nonaux>=0) {
      assert(idx_aux == -1);
      
      vval     = v_n_    ->xref[idx_nonaux];
      thetaval = theta_n_->xref[idx_nonaux];

      v_idx_out     = v_aux_n_    ->compute_indexDense();
      theta_idx_out = theta_aux_n_->compute_indexDense();

    } else {
      assert(idx_aux>=0);
      assert(idx_aux < v_n_->n);
      assert(idx_nonaux == -1);
      
      vval     = v_aux_n_->xref[idx_aux];
      thetaval = theta_aux_n_->xref[idx_aux];

      v_idx_out     = v_n_    ->compute_indexDense();
      theta_idx_out = theta_n_->compute_indexDense();
    }
  }

  
  /*****************************************************************************************************
   * for lix=1:length(Lidx_overload_pass)
   *   l = Lidx_overload_pass[lix]
   *   i = Lin_overload_pass[lix]
   *   vi = vall_n[L_Nidx[l,i]]
   *   thetai = thetaall_n[L_Nidx[l,i]]
   *   vj = vall_n[L_Nidx[l,3-i]]
   *   thetaj = thetaall_n[L_Nidx[l,3-i]]
   *   slack_li >=0 starts at max(0, abs(s_li[l,i]) - abs(v_n_all_complex[L_Nidx[l,i]])*L[:RateBase][l])
   *
   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - (L[:RateBase][l]*vi + slack_li)^2 <=0
   */
  bool LineThermalViolCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
  {
    double* rhs = body + this->index;
    
    assert(Lidx_overload_.size() == Lin_overload_.size());
    assert(Lidx_overload_.size() == this->n);
    assert(slacks_->n == this->n);

    double vi, vj, thetai, thetaj;
    
    for(int con=0; con<n; ++con) {
      const int l = Lidx_overload_[con];
      const int i = Lin_overload_[con];
      assert(i==0 || i==1);
      assert(L_Nidx_[i].size() > l);
      assert(L_Nidx_[0].size() == L_Nidx_[1].size());

      int idx_dummy1, idx_dummy2;
      
      //compute vi and theta i
      const int i_idx_bus = L_Nidx_[i][l];
      get_v_theta_vals_and_idxs(i_idx_bus, vi, thetai, idx_dummy1, idx_dummy2);
      
      //compute vj and thetaj
      const int j_idx_bus = L_Nidx_[2-i][i];
      get_v_theta_vals_and_idxs(j_idx_bus, vj, thetaj, idx_dummy1, idx_dummy2);

      //Ychi = im*L[:Bch][l]/2
      const auto Ychi = std::complex<double>(0., L_Bch_[l]/2);
      
      //Yij = L[:G][l] + im*L[:B][l]
      const auto Yij = std::complex<double>(L_G_[l], L_B_[l]);
      
      //Ychiyij = abs(Ychi+Yij)
      const double ychiyij =  std::abs(Ychi+Yij);
      //yij = abs(Yij)
      const double yij = std::abs(Yij);
      //phi = angle(Yij) - angle(Ychi+Yij)
      const double phi = std::arg(Yij)-std::arg(Ychi+Yij);

      rhs[con] = ychiyij*ychiyij*std::pow(vi,4) + yij*yij*vi*vi*vj*vj
	- 2*ychiyij*yij*pow(vi,3)*vj*cos(thetai-thetaj-phi)
	- std::pow(L_Rate_[l]*vi - slacks_->xref[con], 2);
      
    }
    return true;
  }

  /**********************************************************************************************
   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - (L[:RateBase][l]*vi + slack_li)^2  ==  c(vi,vj,thetai, thetaj, slack_li)
   *
   * dc/dvi =    4*ychiyij^2*vi^3 + 2*yij^2*vi*vj^2 
   *           - 6*ychiyij*yij*vi^2*vj*cos(thetai-thetaj-phi)
   *           -2*L[:RateBase][l]*(L[:RateBase][l]*vi + slack_li)
   *
   * dc/dvj  =     2*yij^2*vi^2*vj  
   *            - 2*ychiyij*yij*vi^3*cos(thetai-thetaj-phi) 
   *
   * dc/dthetai =   2*ychiyij*yij*vi^3*vj*sin(thetai-thetaj-phi)
   *
   * dc/dthetaj = - 2*ychiyij*yij*vi^3*vj*sin(thetai-thetaj-phi) 
   *
   * dc/dslack  = - 2*(L[:RateBase][l]*vi + slack_li)
   *
   */
  bool LineThermalViolCons::eval_Jac_eq(const OptVariables& x, bool new_x, 
					const int& nxsparse, const int& nxdense,
					const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
					double** JacD)
  {
    //
    // sparse part
    //
    assert(slacks_->indexSparse>=0);
    assert(nxsparse+nxdense == x.n());

    if(iJacS && jJacS) {
      int *itnz = J_nz_idxs_;
      for(int row=this->index; row<this->index+this->n; ++row) {
	iJacS[*itnz] = row;
	jJacS[*itnz++] = slacks_->indexSparse - this->index + row;
      }
    }

    if(MJacS) {
      int *itnz = J_nz_idxs_;
      double vi, thetai;
      for(int row=this->index; row<this->index+this->n; ++row) {

	const int con = row - this->index;
	const int l   = Lidx_overload_[con];
	const int i   = Lin_overload_[con];

	assert(i==0 || i==1);
	assert(L_Nidx_[i].size() > l);
	assert(L_Nidx_[0].size() == L_Nidx_[1].size());
	assert(con < slacks_->n);

	int idx_dummy1, idx_dummy2;
	
	const int i_idx_bus = L_Nidx_[i][l];
	get_v_theta_vals_and_idxs(i_idx_bus, vi, thetai, idx_dummy1, idx_dummy2);
	
	MJacS[*itnz++] -= 2*(L_Rate_[l] * vi + slacks_->xref[con]);
      }
    }

    //
    // dense part
    //
    if(NULL == JacD) return true;

    int idx_col_of_vi, idx_col_of_thetai;
    double vi, thetai;
    int idx_col_of_vj, idx_col_of_thetaj;
    double vj, thetaj;
    
    for(int con=0; con<this->n; con++) {
      const int row = con+this->index;

      const int l   = Lidx_overload_[con];
      const int i   = Lin_overload_[con];
      
      assert(i==0 || i==1);
      assert(L_Nidx_[i].size() > l);
      assert(L_Nidx_[0].size() == L_Nidx_[1].size());
      assert(con < slacks_->n);
      
      //Ychi = im*L[:Bch][l]/2
      const auto Ychi = std::complex<double>(0., L_Bch_[l]/2);
      
      //Yij = L[:G][l] + im*L[:B][l]
      const auto Yij = std::complex<double>(L_G_[l], L_B_[l]);
      
      //Ychiyij = abs(Ychi+Yij)
      const double ychiyij =  std::abs(Ychi+Yij);
      //yij = abs(Yij)
      const double yij = std::abs(Yij);
      //phi = angle(Yij) - angle(Ychi+Yij)
      const double phi = std::arg(Yij)-std::arg(Ychi+Yij);
      
      const int i_idx_bus = L_Nidx_[i][l];
      get_v_theta_vals_and_idxs(i_idx_bus, vi, thetai, idx_col_of_vi, idx_col_of_thetai);

      const int j_idx_bus = L_Nidx_[2-i][i];
      get_v_theta_vals_and_idxs(j_idx_bus, vj, thetaj, idx_col_of_vj, idx_col_of_thetaj);

      const double vi2 = vi*vi;
      const double vi3 = vi2*vi;
      const double ttp = thetai-thetaj-phi;
      const double costtp = cos(ttp);
      const double sinttp = sin(ttp);

      // dc/dvi
      assert(JacD[con][idx_col_of_vi]==0.);
      JacD[con][idx_col_of_vi] =
	+ 4 * ychiyij * ychiyij * vi3 + 2 * yij*yij *vi * vj*vj
	- 6 * ychiyij * yij * vi*vi * vj * costtp
        - 2 * L_Rate_[l] * (L_Rate_[l]*vi + slacks_->xref[con]);

      // dc/dvj
      assert(0. == JacD[con][idx_col_of_vj]);
      JacD[con][idx_col_of_vj] = + 2 * yij*yij * vi2 * vj - 2 * ychiyij * yij * vi3 * costtp;

      const double tmp = 2 * ychiyij * yij * vi3 * vj * sinttp;
      assert(0. == JacD[con][idx_col_of_thetai]);
      JacD[con][idx_col_of_thetai] = + tmp;

      assert(0. == JacD[con][idx_col_of_thetaj]);
      JacD[con][idx_col_of_thetaj] = - tmp;
    }
    
    
    return true;
  }

  bool LineThermalViolCons::eval_Jac_ineq(const OptVariables& x, bool new_x, 
					  const int& nxsparse, const int& nxdense,
					  const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
					  double** JacD)
  {
    return false;
  }
  
  int LineThermalViolCons::get_spJacob_eq_nnz()
  {
    return this->n;
  }
  int LineThermalViolCons::get_spJacob_ineq_nnz()
  {
    return 0;
  }
  bool LineThermalViolCons::get_spJacob_eq_ij(std::vector<OptSparseEntry>& vij)
  {
    if(n<=0) return true;
    
    int nnz = get_spJacob_eq_nnz();
    if(NULL == J_nz_idxs_) 
      J_nz_idxs_ = new int[nnz];
#ifdef DEBUG
    int n_vij_in = vij.size();
#endif
    
    int row=this->index, *itnz=J_nz_idxs_;
    for(int it=0; it<n; it++) {
      vij.push_back(OptSparseEntry(row++, slacks_->indexSparse+it, itnz++));
    }

#ifdef DEBUG
    assert(nnz+n_vij_in==vij.size());
#endif
    assert(J_nz_idxs_ + nnz == itnz);

    return true;
  }
  bool LineThermalViolCons::get_spJacob_ineq_ij(std::vector<OptSparseEntry>& vij)
  {
    assert(false);
    return true;
  }

  /**********************************************************************************************
   *   ychiyij^2*vi^4 + yij^2*vi^2*vj^2
   *    - 2*ychiyij*yij*vi^3*vj*cos(thetai-thetaj-phi) 
   *    - (L[:RateBase][l]*vi + slack_li)^2  ==  c(vi,vj,thetai, thetaj, slack_li)
   *
   *===============================================================================================
   * dc/dvi =    4*ychiyij^2*vi^3 + 2*yij^2*vi*vj^2 
   *           - 6*ychiyij*yij*vi^2*vj*cos(thetai-thetaj-phi)
   *           -2*L[:RateBase][l]*(L[:RateBase][l]*vi + slack_li)
   *
   * dc^2/dvi^2 = + 12*ychiyij^2*vi^2 + 2*yij^2*vj^2
   *              - 12*ychiyij*yij*vi*vj*cos(thetai-thetaj-phi)
   *              - 2*L[:RateBase][l]*L[:RateBase][l]
   *
   * dc^2/dvi dvj = + 4*yij^2*vi*vj
   *                - 6*ychiyij*yij*vi^2*cos(thetai-thetaj-phi)
   *
   * dc^2/dvi dthetai = + 6*ychiyij*yij*vi^2*vj*sin(thetai-thetaj-phi)
   *
   * dc^2/dvi dthetaj = - dc^2/dvi dthetai
   *
   * dc^2/dvi dslack = - 2*L[:RateBase][l]
   *===============================================================================================
   *
   * dc/dvj  =     2*yij^2*vi^2*vj  
   *            - 2*ychiyij*yij*vi^3*cos(thetai-thetaj-phi) 
   *
   * d2c/dvj dvi = 4*yij^2*vi*vj - 6*ychiyij*yij*vi^2*cos(thetai-thetaj-phi) 
   *
   * d2c/d2vj = 2*yij^2*vi^2
   *
   * d2c/dvj dthetai = 2*ychiyij*yij*vi^3*sin(thetai-thetaj-phi) 
   * d2c/dvj dthetaj = - dc/dvj dthetai
   *
   * d2c/dvj dslack = 0
   *================================================================================================
   * dc/dthetai =   2*ychiyij*yij*vi^3*vj*sin(thetai-thetaj-phi)
   *
   *================================================================================================
   * dc/dthetaj = - 2*ychiyij*yij*vi^3*vj*sin(thetai-thetaj-phi) 
   *
   *================================================================================================
   * dc/dslack  = - 2*(L[:RateBase][l]*vi + slack_li)
   *
   *================================================================================================
   */
  
  bool LineThermalViolCons::eval_HessLagr(const OptVariables& x, bool new_x, 
					  const OptVariables& lambda, bool new_lambda,
					  const int& nxsparse, const int& nxdense, 
					  const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
					  double** HDD,
					  const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {

    return true;
  }

  //nnz for sparse part (all zeros)
  int LineThermalViolCons::get_HessLagr_SSblock_nnz()
  {
    return 0;
  }
  
  bool LineThermalViolCons::get_HessLagr_SSblock_ij(std::vector<OptSparseEntry>& vij)
  {
    return true;
  }
  
  
} // end of namespace

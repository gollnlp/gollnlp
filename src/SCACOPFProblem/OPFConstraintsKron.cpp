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
  // --------- = -    sum    v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
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
    int row2=-1;
#endif
    if(iJacS && jJacS) {
      itnz = J_nz_idxs; row=0;
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
      row2=row;
#endif
      
    }
    if(MJacS) {
      itnz = J_nz_idxs; row=0;
      for(int it=0; it<n; it++) {
	//p_g 
	const size_t sz = Gn_fs[bus_nonaux_idxs[it]].size();
	for(int ig=0; ig<sz; ++ig) { 
	  MJacS[*itnz++] += 1; 
	}
	++row;
      }
    }
#ifdef DEBUG
    if(row2>=0) assert(row==row2);
#endif 
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
      
      for(int i=0; i<n; i++) {
	
	//
	//partials w.r.t. to v_i and theta_i
	//

	//first term of da_i/v_i  -> we'll accumulate in here
	JacD[i][idx_col_of_v_n + i] = - 2.0*v_n->xref[i]*YredM[i][i].real();
	//here we'll accumulate da_i/theta_i
	JacD[i][idx_col_of_theta_n + i] = 0.;
	
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
	  assert(JacD[i][idx_col_of_v_n + j]==0. && "should not written previously in this");
	  JacD[i][idx_col_of_v_n + j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/v_i: add  - v_j* (   Gred[i,j]*cos(theta_i-theta_j) + Bred[i,j]*sin(theta_i-theta_j)) 
	  JacD[i][idx_col_of_v_n + i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  
	  //for da_i/dtheta_k = - v_i*v_k *(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  assert(JacD[i][idx_col_of_theta_n + j]==0. && "should not written previously in this");
	  JacD[i][idx_col_of_theta_n + j] = - vivjGsin__Bcos;

	  //for da_i/theta_i: add
	  //        - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i][idx_col_of_theta_n + i] += vivjGsin__Bcos;
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
	  JacD[i][idx_col_of_v_n + i] -= v_n->xref[j]*aux_G_cos_B_sin;

	  //for da_i/dv_k =  - v_i * (Gred[i,k]*cos(theta_i-theta_k)+Bred[i,k]*sin(theta_i-theta_k))
	  assert(JacD[i][idx_col_of_v_n + j]==0. && "should not written previously in this");
	  JacD[i][idx_col_of_v_n + j] = -v_n->xref[i]*aux_G_cos_B_sin;

	  //for da_i/theta_i: add
	  //         - v_i*v_j *( - Gred[i,j]*sin(theta_i-theta_k) + Bred[i,j]*cos(theta_i-theta_j))
	  JacD[i][idx_col_of_theta_n + i] += vivjGsin__Bcos;

	  //for da_i/dtheta_k = - v_i*v_k*(Gred[i,k]*sin(theta_i-theta_k) - Bred*cos(theta_i-theta_k))
	  JacD[i][idx_col_of_theta_n + j] = - vivjGsin__Bcos;
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
    
    int row=0, *itnz=J_nz_idxs;
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
    if(NULL==MHSS) {
    } else {
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



} // end of namespace

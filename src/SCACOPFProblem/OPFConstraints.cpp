#include "OPFConstraints.hpp"

#include <string>
#include <cassert>
using namespace std;

namespace gollnlp {

//////////////////////////////////////////////////////////////////
// Power flows in rectangular form
// pq == A*vi^2 + B*vi*vj*cos(thetai - thetaj + Theta) + 
//       C*vi*vj*sin(thetai - thetaj + Theta)
//////////////////////////////////////////////////////////////////

PFConRectangular::PFConRectangular(const std::string& id_, int numcons,
		     OptVariablesBlock* pq_, 
		     OptVariablesBlock* v_n_, 
		     OptVariablesBlock* theta_n_,
		     const std::vector<int>& Nidx1, //T_Nidx or L_Nidx indexes
		     const std::vector<int>& Nidx2)
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

PFConRectangular::~PFConRectangular() 
{
  delete[] A; 
  delete[] B;
  delete[] C;
  delete[] T;
  delete[] E_Nidx1;
  delete[] E_Nidx2;
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}



bool PFConRectangular::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
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
#ifdef DEBUG
  //double r=DNRM2(&n, body+this->index, &ione);
  //printf("Evaluated constraint '%s' -> resid norm %g\n", id.c_str(), r);
#endif

  return true;
}

bool PFConRectangular::eval_Jac(const OptVariables& primal_vars, bool new_x, 
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

bool PFConRectangular::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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

bool PFConRectangular::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
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

bool PFConRectangular::get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
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
// pq := A*vi^2 + B*vi*vj*cos(thetai - thetaj + Theta) + 
//       C*vi*vj*sin(thetai - thetaj + Theta)
void PFConRectangular::compute_power(OptVariablesBlock* p_or_q)
{
  assert(p_or_q->n == n);
  double v1, v1v2, ththT, *it=p_or_q->x;
  for(int i=0; i<n; i++) {
    ththT = theta_n->x[E_Nidx1[i]] - theta_n->x[E_Nidx2[i]] + T[i];
    v1 = v_n->x[E_Nidx1[i]];
    v1v2 = v1 * v_n->x[E_Nidx2[i]];
    *it =  A[i]*v1*v1 + B[i]*v1v2*cos(ththT) + C[i]*v1v2*sin(ththT);
    assert(isfinite(*it)==true);
    it++;
  }
}


///////////////////////////////////////////////////////////////////////////////
// Active Balance constraints
//
// sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
// sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
// sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - 
// r*pslackp_n[n] + r*pslackm_n[n]    =   N[:Pd][n])
///////////////////////////////////////////////////////////////////////////////

PFActiveBalance::PFActiveBalance(const std::string& id_, int numcons,
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
				 const double& slacks_rescale/*=1.*/)
: OptConstraintsBlock(id_, numcons), 
  p_g(p_g_), v_n(v_n_), p_li1(p_li1_), p_li2(p_li2_), 
  p_ti1(p_ti1_), p_ti2(p_ti2_), pslack_n(NULL),
  N_Gsh(N_Gsh_), N_Pd(N_Pd_), Gn(Gn_),
  Lidxn1(Lidxn1_),  Lidxn2(Lidxn2_), Tidxn1(Tidxn1_),  Tidxn2(Tidxn2_)
{
  assert(N_Pd.size()==n);
  //rhs
  //!memcpy(lb, d.N_Pd.data(), n*sizeof(double));
  for(int i=0; i<n; i++) lb[i]=0.;
  DCOPY(&n, lb, &ione, ub, &ione);

  r = slacks_rescale>0 ? 1/slacks_rescale : 1.;

  J_nz_idxs = NULL;
  H_nz_idxs = NULL;
}
PFActiveBalance::~PFActiveBalance() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
};
bool PFActiveBalance::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
{
  assert(pslack_n); //assert(pslackm_n);
  double* body = body_ + this->index;
  double* slacks = const_cast<double*>(pslack_n->xref);

  r = 0-r;
  DAXPY(&n, &r, slacks,   &ione, body, &ione);
  r = 0-r;
  DAXPY(&n, &r, slacks+n, &ione, body, &ione);

  const double *NGsh=N_Gsh.data(), *NPd=N_Pd.data();
  for(int i=0; i<n; i++) {
    *body -= NPd[i];
    *body -= NGsh[i] * v_n->xref[i] * v_n->xref[i];
    body++;
  }
  body -= n;
  const int *Gnv; int nGn;
  for(int i=0; i<n; i++) {
    nGn = Gn[i].size(); 
    Gnv = Gn[i].data();
    for(int ig=0; ig<nGn; ig++) 
      *body += p_g->xref[Gnv[ig]];
    body++;
  }
  body -= n;
  {
    // - sum(p_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
    const int *Lidxn; int nLidx;
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn1[i].data(); nLidx = Lidxn1[i].size();
      for(int ilix=0; ilix<nLidx; ilix++)
	*body -= p_li1->xref[Lidxn[ilix]];
      body++;
    }
    body -= n;
    // - sum(p_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn2[i].data(); nLidx = Lidxn2[i].size();
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
      Tidxn=Tidxn1[i].data(); nTidx = Tidxn1[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= p_ti1->xref[Tidxn[itix]];
      body++;
    }
    body -= n;
    // - sum(p_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn2[i].data(); nTidx = Tidxn2[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= p_ti2->xref[Tidxn[itix]];
      body++;
    }
  }
#ifdef DEBUG
  //body -= n;
  //double r=DNRM2(&n, body, &ione);
  //printf("Evaluated constraint '%s' -> resid norm %g\n", id.c_str(), r);
#endif

  return true;
}
// sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
// sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
// sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - r*pslackp_n[n] + r*pslackm_n[n] = N[:Pd][n])
bool PFActiveBalance::eval_Jac(const OptVariables& primal_vars, bool new_x, 
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
      for(auto g: Gn[it]) { i[*itnz]=row; j[*itnz]=p_g->index+g; itnz++; }
      //p_li1 and p_l2
      for(auto l: Lidxn1[it]) { i[*itnz]=row; j[*itnz]=p_li1->index+l; itnz++; }	  
      for(auto l: Lidxn2[it]) { i[*itnz]=row; j[*itnz]=p_li2->index+l; itnz++; }
      //p_ti1 and p_ti2
      for(auto t: Tidxn1[it]) { i[*itnz]=row; j[*itnz]=p_ti1->index+t; itnz++; }	  
      for(auto t: Tidxn2[it]) { i[*itnz]=row; j[*itnz]=p_ti2->index+t; itnz++; }
 
      //slacks
      i[*itnz]=row; j[*itnz]=pslack_n->index+it;   itnz++;
      i[*itnz]=row; j[*itnz]=pslack_n->index+it+n; itnz++;
    }
#ifdef DEBUG
    assert(J_nz_idxs + nnz_loc == itnz);
#endif
  } else {
    const double* Gsh=N_Gsh.data(); int sz;
    for(int it=0; it<n; it++) {
      row = this->index+it;
      M[*itnz] -= 2*v_n->xref[it]*Gsh[it];  itnz++; //vn
	  
      sz = Gn[it].size();
      for(int ig=0; ig<sz; ig++) { M[*itnz] += 1; itnz++; } //p_g

      sz=Lidxn1[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_li_1
      sz=Lidxn2[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_l2_1

      sz=Tidxn1[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_ti_1
      sz=Tidxn2[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_t2_1

      //slacks
      M[*itnz] -= r; itnz++;
      M[*itnz] += r; itnz++;
    }
  }
  return true;
}

int PFActiveBalance::get_Jacob_nnz(){ 
  int nnz = 3*n; //slacks and v_n
  for(int i=0; i<n; i++) 
    nnz += Gn[i].size() + Lidxn1[i].size() + Lidxn2[i].size() + Tidxn1[i].size() + Tidxn2[i].size();
  return nnz; 
}

bool PFActiveBalance::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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
    for(auto g: Gn[it]) 
      vij.push_back(OptSparseEntry(row, p_g->index+g, itnz++));

    //p_li1
    for(auto l: Lidxn1[it])
      vij.push_back(OptSparseEntry(row, p_li1->index+l, itnz++));
    //p_li1
    for(auto l: Lidxn2[it])
      vij.push_back(OptSparseEntry(row, p_li2->index+l, itnz++));
			
    //p_ti1
    for(auto t: Tidxn1[it])
      vij.push_back(OptSparseEntry(row, p_ti1->index+t, itnz++));
    //p_ti1
    for(auto t: Tidxn2[it])
      vij.push_back(OptSparseEntry(row, p_ti2->index+t, itnz++));
			
    //slacks
    vij.push_back(OptSparseEntry(row, pslack_n->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, pslack_n->index+it+n, itnz++));
  }
  //printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
  assert(nnz+n_vij_in==vij.size());
#endif
  assert(J_nz_idxs+nnz == itnz);
  return true;
}

bool PFActiveBalance::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
				    const OptVariables& lambda_vars, bool new_lambda,
				    const int& nnz, int* ia, int* ja, double* M)
{
  int *itnz=H_nz_idxs, nend=v_n->index+n;
  if(NULL==M) {
    for(int it=v_n->index; it<nend; it++) {
      ia[*itnz] = ja[*itnz] = it; itnz++;
    }
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==n);
    const double* NGsh = N_Gsh.data();
    for(int it=0; it<n; it++) {
      M[*itnz++] -= 2*NGsh[it]*lambda->xref[it];
    }
  }
  return true;
}

bool PFActiveBalance::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)  
{
  if(n==0) return true;
      
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[get_HessLagr_nnz()];
  }

  int *itnz=H_nz_idxs, nend=v_n->index+n;
  for(int it=v_n->index; it<nend; it++) vij.push_back(OptSparseEntry(it,it, itnz++));

  return true;
}

//sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
// sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
// sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) -   N[:Pd][n]) ==
// + r*pslackp_n[n] - r*pslackm_n[n]    
//
//
void PFActiveBalance::compute_slacks(OptVariablesBlock* slacksv) const
{
  assert(slacksv->n == 2*n);
  double* body = slacksv->x; //use first n as buffer

  const double *NGsh=N_Gsh.data(), *NPd = N_Pd.data();
  for(int i=0; i<n; i++) {
    *body = - NPd[i] - NGsh[i] * v_n->x[i] * v_n->x[i];
    body++;
  }
  body -= n;
  const int *Gnv; int nGn;
  for(int i=0; i<n; i++) {
    nGn = Gn[i].size(); Gnv = Gn[i].data();
    for(int ig=0; ig<nGn; ig++) {
      *body += p_g->x[Gnv[ig]];
      assert(isfinite(*body)==true);
    }
    body++;
  }
  body -= n;
  {
    // - sum(p_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
    const int *Lidxn; int nLidx;
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn1[i].data(); nLidx = Lidxn1[i].size();
      for(int ilix=0; ilix<nLidx; ilix++) {
	*body -= p_li1->x[Lidxn[ilix]];
	assert(isfinite(*body)==true);
      }
      body++;
    }
    body -= n;
    // - sum(p_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn2[i].data(); nLidx = Lidxn2[i].size();
      for(int ilix=0; ilix<nLidx; ilix++) {
	*body -= p_li2->x[Lidxn[ilix]];
	assert(isfinite(*body)==true);
      }
      body++;
    }
    body -= n;
  }
  {
    const int *Tidxn; int nTidx;
    // - sum(p_ti1[Tidxn1[n][tix]] for tix=1:length(Tidxn1[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn1[i].data(); nTidx = Tidxn1[i].size();
      for(int itix=0; itix<nTidx; itix++) {
	*body -= p_ti1->x[Tidxn[itix]];
	assert(isfinite(*body)==true);
      }
      body++;
    }
    body -= n;
    // - sum(p_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn2[i].data(); nTidx=Tidxn2[i].size();
      for(int itix=0; itix<nTidx; itix++) {
	*body -= p_ti2->x[Tidxn[itix]];
	assert(isfinite(*body)==true);
      }
      body++;
    }
  }
  double* pslackp_n = slacksv->x, *pslackm_n = slacksv->x+n;
  for(int i=0; i<n; i++) {
    //pslackm_n[n] = max(0.0, -pslack)
    //pslackp_n[n] = max(0.0, pslack)

    assert(std::isnan(pslackp_n[i])==false);
    assert(isfinite(pslackp_n[i])==true);
    assert(std::isinf(pslackp_n[i])==false);

    if(pslackp_n[i]<0) {
      pslackm_n[i] = -pslackp_n[i]/r;
      pslackp_n[i] = 0.;
    } else {
      pslackp_n[i] = pslackp_n[i]/r;
      pslackm_n[i] = 0.;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////
// Reactive Balance constraints
//
//sum(q_g[g] for g=Gn[n]) - 
// (-N[:Bsh][n] - sum(b_s[s] for s=SShn[n]))*v_n[n]^2 -
// sum(q_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
// sum(q_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - 
// r*qslackp_n[n] + r*qslackm_n[n])  ==  N[:Qd][n]
////////////////////////////////////////////////////////////////////////////////////////////
PFReactiveBalance::
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
		  const double& slacks_rescale)
: OptConstraintsBlock(id_, numcons), 
  q_g(q_g_), v_n(v_n_), 
  q_li1(q_li1_), q_li2(q_li2_), 
  q_ti1(q_ti1_), q_ti2(q_ti2_), 
  b_s(b_s_), qslack_n(NULL),
  N_Bsh(N_Bsh_), N_Qd(N_Qd_), Gn(Gn_), SShn(SShn_),
  Lidxn1(Lidxn1_),  Lidxn2(Lidxn2_), Tidxn1(Tidxn1_),  Tidxn2(Tidxn2_)
{
  assert(N_Qd.size()==n);
  assert(slacks_rescale>0);
  r = slacks_rescale>0 ? 1/slacks_rescale: 1.;

  //rhs
  //!memcpy(lb, d.N_Qd.data(), n*sizeof(double));
  for(int i=0; i<n; i++) lb[i]=0.;
  DCOPY(&n, lb, &ione, ub, &ione);
  J_nz_idxs = NULL;
  H_nz_idxs = NULL;
}
PFReactiveBalance::~PFReactiveBalance() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

bool PFReactiveBalance::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
{
  assert(qslack_n); 
  double* body = body_ + this->index;
  double* slacks = const_cast<double*>(qslack_n->xref);
  r = 0-r;
  DAXPY(&n, &r, slacks,   &ione, body, &ione);
  r = 0-r;
  DAXPY(&n, &r, slacks+n, &ione, body, &ione);

  const int *Gnv; int nGn; const double *NQd=N_Qd.data();
  for(int i=0; i<n; i++) {
    *body -= NQd[i];
    nGn = Gn[i].size(); Gnv = Gn[i].data();
    for(int ig=0; ig<nGn; ig++) 
      *body += q_g->xref[Gnv[ig]];
    body++;
  }
  body -= n;
  {
    //(N[:Bsh][n]  + sum(b_s[s] for s=SShn[n]))*v_n[n]^2 
    const double *NBsh=N_Bsh.data(); const int *SShnv; int sz; double aux;
    for(int i=0; i<n; i++) {
      aux = NBsh[i];
      SShnv = SShn[i].data(); sz = SShn[i].size();
      for(int is=0; is<sz; is++) aux += b_s->xref[SShnv[is]];
      *body++ += aux*v_n->xref[i]*v_n->xref[i];
    }
  }
  body -= n;
  {
    // - sum(q_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
    const int *Lidxn; int nLidx;
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn1[i].data(); nLidx = Lidxn1[i].size();
      for(int ilix=0; ilix<nLidx; ilix++)
	*body -= q_li1->xref[Lidxn[ilix]];
      body++;
    }
    body -= n;
    // - sum(q_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn2[i].data(); nLidx = Lidxn2[i].size();
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
      Tidxn=Tidxn1[i].data(); nTidx = Tidxn1[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= q_ti1->xref[Tidxn[itix]];
      body++;
    }
    body -= n;
    // - sum(q_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn2[i].data(); nTidx = Tidxn2[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= q_ti2->xref[Tidxn[itix]];
      body++;
    }
  }

#ifdef DEBUG
  body -= n;
  //for(int i=0; i<n; i++) printf("%12.5e %12.5e| ", pslack_n->xref[i], pslack_n->xref[i+n]);
  //printf("\n");

  //for(int i=0; i<n; i++) body[i]=0.;

  //double resid[n];
  //DCOPY(&n, body, &ione, resid, &ione);
  //DAXPY(&n, &dminusone, lb, &ione, resid, &ione);
  double r=DNRM2(&n, body, &ione);
  //printf("Evaluated constraint '%s' -> resid norm %g\n", id.c_str(), r);
#endif

  return true;
} 

void PFReactiveBalance::compute_slacks(OptVariablesBlock* qslacks_n)
{
  double* body = qslacks_n->x;

  const int *Gnv; int nGn;
  for(int i=0; i<n; i++) {
    *body=0;
    nGn = Gn[i].size(); Gnv = Gn[i].data();
    for(int ig=0; ig<nGn; ig++) 
      *body += q_g->x[Gnv[ig]];
    body++;
  }
  body -= n;
  {
    //(N[:Bsh][n]  + sum(b_s[s] for s=SShn[n]))*v_n[n]^2 
    const double *NBsh=N_Bsh.data(); const int *SShnv; int sz; double aux;
    for(int i=0; i<n; i++) {
      aux = NBsh[i];
      SShnv = SShn[i].data(); sz = SShn[i].size();
      for(int is=0; is<sz; is++) aux += b_s->x[SShnv[is]];
      *body++ += aux*v_n->x[i]*v_n->x[i] - N_Qd[i];
    }
  }
  body -= n;
  {
    // - sum(q_li1[Lidxn1[n][lix]] for lix=1:length(Lidxn1[n])) 
    const int *Lidxn; int nLidx;
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn1[i].data(); nLidx = Lidxn1[i].size();
      for(int ilix=0; ilix<nLidx; ilix++)
	*body -= q_li1->x[Lidxn[ilix]];
      body++;
    }
    body -= n;
    // - sum(q_li2[Lidxn1[n][lix]] for lix=1:length(Lidxn2[n])) 
    for(int i=0; i<n; i++) {
      Lidxn=Lidxn2[i].data(); nLidx = Lidxn2[i].size();
      for(int ilix=0; ilix<nLidx; ilix++)
	*body -= q_li2->x[Lidxn[ilix]];
      body++;
    }
    body -= n;
  }
  {
    const int *Tidxn; int nTidx;
    // - sum(q_ti1[Tidxn1[n][tix]] for tix=1:length(Tidxn1[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn1[i].data(); nTidx = Tidxn1[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= q_ti1->x[Tidxn[itix]];
      body++;
    }
    body -= n;
    // - sum(q_ti2[Tidxn2[n][tix]] for tix=1:length(Tidxn2[n]))
    for(int i=0; i<n; i++) {
      Tidxn=Tidxn2[i].data(); nTidx = Tidxn2[i].size();
      for(int itix=0; itix<nTidx; itix++)
	*body -= q_ti2->x[Tidxn[itix]];
      body++;
    }
  }

  double *qslackp_n=qslacks_n->x, *qslackm_n=qslacks_n->x+n;
  for(int i=0; i<n; i++) {
    if(qslackp_n[i]<0) {
      qslackm_n[i] = - qslackp_n[i]/r;
      qslackp_n[i] = 0.;
    } else {
      qslackp_n[i] = qslackp_n[i]/r;
      qslackm_n[i] = 0.;
    }
  }
}

//sum(q_g[g] for g=Gn[n]) - 
// (-N[:Bsh][n] - sum(b_s[s] for s=SShn[n]))*v_n[n]^2 -
// sum(q_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
// sum(q_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - qslackp_n[n] + qslackm_n[n]) == N[:Qd][n]
bool PFReactiveBalance::eval_Jac(const OptVariables& primal_vars, bool new_x, 
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
      for(auto g: Gn[it]) { i[*itnz]=row; j[*itnz]=q_g->index+g; itnz++; }
      //q_li1 and p_l2
      for(auto l: Lidxn1[it]) { i[*itnz]=row; j[*itnz]=q_li1->index+l; itnz++; }	  
      for(auto l: Lidxn2[it]) { i[*itnz]=row; j[*itnz]=q_li2->index+l; itnz++; }
      //q_ti1 and q_ti2
      for(auto t: Tidxn1[it]) { i[*itnz]=row; j[*itnz]=q_ti1->index+t; itnz++; }	  
      for(auto t: Tidxn2[it]) { i[*itnz]=row; j[*itnz]=q_ti2->index+t; itnz++; }
	  
      //b_s
      for(auto ssh: SShn[it])  { i[*itnz]=row; j[*itnz]=b_s->index+ssh; itnz++; }

      //slacks
      i[*itnz]=row; j[*itnz]=qslack_n->index+it;   itnz++;
      i[*itnz]=row; j[*itnz]=qslack_n->index+it+n; itnz++;
    }
#ifdef DEBUG
    assert(J_nz_idxs + nnz_loc == itnz);
#endif
  } else {
    const double* Bsh=N_Bsh.data(); int sz, szsshn; const int *sshn; double aux;
    for(int it=0; it<n; it++) {
      row = this->index+it;

      aux=Bsh[it];
      szsshn = SShn[it].size(); sshn = SShn[it].data();
      for(int s=0; s<szsshn; s++) aux += b_s->xref[sshn[s]];
      M[*itnz] += 2*v_n->xref[it]*aux;  itnz++; //vn
	  
      sz = Gn[it].size();
      for(int ig=0; ig<sz; ig++) { M[*itnz] += 1; itnz++; } //q_g

      sz=Lidxn1[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //q_li_1
      sz=Lidxn2[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_l2_1

      sz=Tidxn1[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //q_ti_1
      sz=Tidxn2[it].size();
      for(int i=0; i<sz; i++) { M[*itnz] -= 1; itnz++;} //p_t2_1

      aux = v_n->xref[it]*v_n->xref[it];
      for(int s=0; s<szsshn; s++) {  M[*itnz] += aux; itnz++;} //b_s

      //slacks
      M[*itnz] -= r; itnz++;
      M[*itnz] += r; itnz++;
    }
  }
  return true;
}
int PFReactiveBalance::get_Jacob_nnz(){ 
  int nnz = 2*n; //slacks
  for(int i=0; i<n; i++) 
    nnz += Gn[i].size() + (1+SShn[i].size()) + Lidxn1[i].size() + Lidxn2[i].size() + Tidxn1[i].size() + Tidxn2[i].size();
  return nnz; 
}

bool PFReactiveBalance::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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
    for(auto g: Gn[it]) 
      vij.push_back(OptSparseEntry(row, q_g->index+g, itnz++));

    //q_li1
    for(auto l: Lidxn1[it])
      vij.push_back(OptSparseEntry(row, q_li1->index+l, itnz++));
    //q_li1
    for(auto l: Lidxn2[it])
      vij.push_back(OptSparseEntry(row, q_li2->index+l, itnz++));
			
    //q_ti1
    for(auto t: Tidxn1[it])
      vij.push_back(OptSparseEntry(row, q_ti1->index+t, itnz++));
    //q_ti1
    for(auto t: Tidxn2[it])
      vij.push_back(OptSparseEntry(row, q_ti2->index+t, itnz++));

    //b_s
    for(auto is: SShn[it])
      vij.push_back(OptSparseEntry(row, b_s->index+is, itnz++));
		
    //slacks
    vij.push_back(OptSparseEntry(row, qslack_n->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, qslack_n->index+it+n, itnz++));
  }
  //printf("nnz=%d vijsize=%d\n", nnz, vij.size());
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
bool PFReactiveBalance::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
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
      for(auto is: SShn[it]) {
	i = uppertr_swap(row,j=b_s->index+is, aux); 
	ia[*itnz] = i; ja[*itnz] = j; itnz++; // w.r.t. v_n,b_s[s])
      }
    }
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==n);
    
    const double* NBsh = N_Bsh.data(); const int* ssh; int sz; double aux;
    for(int it=0; it<n; it++) {
      aux = NBsh[it];

      sz=SShn[it].size(); ssh = SShn[it].data();
      for(int is=0; is<sz; is++) {
	aux += b_s->xref[ssh[is]];
      }
      M[*itnz] += 2*aux*lambda->xref[it]; itnz++; //w.r.t. (v_n, v_n)

      for(int is=0; is<sz; is++) {
	M[*itnz] += 2*v_n->xref[it]*lambda->xref[it]; itnz++; //w.r.t. (v_n, v_n)
      }
    }
  }
#ifdef DEBUG
  assert(H_nz_idxs+nnz_loc==itnz);
#endif
  return true;
}

int PFReactiveBalance::get_HessLagr_nnz() 
{ 
  int nnz=n; //v_n
  for(int i=0; i<n; i++) nnz += SShn[i].size();
  return nnz; 
}

bool PFReactiveBalance::get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
{
  if(n==0) return true;
      
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[get_HessLagr_nnz()];
  }

  int *itnz=H_nz_idxs, i, j, row, aux;
  for(int it=0; it<n; it++) {
    row = v_n->index+it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t. v_n,v_n
	
    for(auto is: SShn[it]) {
      i = uppertr_swap(row,j=b_s->index+is, aux); 
      vij.push_back(OptSparseEntry(i, j, itnz++)); // w.r.t. v_n,b_s[s])
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////
// PFLineLimits - Line thermal limits
//////////////////////////////////////////////////////////////////
PFLineLimits::PFLineLimits(const std::string& id_, int numcons,
			   OptVariablesBlock* p_li_, 
			   OptVariablesBlock* q_li_,
			   OptVariablesBlock* v_n_,
			   const std::vector<int>& L_Nidx_,
			   const std::vector<double>& L_Rate_,
			   const double& slacks_rescale)
  : OptConstraintsBlock(id_, numcons), 
    p_li(p_li_), q_li(q_li_), v_n(v_n_), Nidx(L_Nidx_), L_Rate(L_Rate_), sslack_li(NULL)
{
  assert(L_Rate.size()==n);
  assert(slacks_rescale>0);
  r = slacks_rescale ? 1/slacks_rescale : 1.;
  //rhs
  for(int i=0; i<n; i++)
    lb[i]=-1e+20;
  for(int i=0; i<n; i++)
    ub[i]=0.;
  J_nz_idxs = NULL;
  H_nz_idxs = NULL;

  //!
  //for(int i=0; i<L_Rate.size(); i++)
  //  L_Rate[i] *= L_Rate_reduction_factor;
}
PFLineLimits::~PFLineLimits() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

// p_li[l,i]^2 + q_li[l,i]^2 - (L[RateSymb][l]*v_n[L_Nidx[l,i]] + r*sslack_li[l,i])^2) <=0
bool PFLineLimits::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
{
  assert(sslack_li); 
  assert(n == L_Rate.size());
  assert(n == Nidx.size());
  double* body = body_ + this->index;
  double* slacks = const_cast<double*>(sslack_li->xref);

  for(int i=0; i<n; i++)
    *body++ += p_li->xref[i]*p_li->xref[i];
  body -= n;
  for(int i=0; i<n; i++)
    *body++ += q_li->xref[i]*q_li->xref[i];
  body -= n;

  const double* Rate = L_Rate.data();
  const int* L_Nidx = Nidx.data();
  double aux;
  for(int i=0; i<n; i++) {
    aux = Rate[i]*v_n->xref[L_Nidx[i]] + r*sslack_li->xref[i];
    *body++ -= aux*aux;
  }
  return true;
}
 
void PFLineLimits::compute_slacks(OptVariablesBlock* sslacks)
{
  assert(n == sslacks->n);
  assert(n == L_Rate.size());
  assert(n == Nidx.size());

  double* body = sslacks->x;
  for(int i=0; i<n; i++)
    *body++ = p_li->x[i]*p_li->x[i];
  body -= n;
  for(int i=0; i<n; i++)
    *body++ += q_li->x[i]*q_li->x[i];
  body -= n;

  const double* Rate = L_Rate.data();
  const int* L_Nidx = Nidx.data();
  double aux;
  for(int i=0; i<n; i++) {
    aux = body[i]>0. ? sqrt(body[i]) : 0.;
    aux -= Rate[i]*v_n->x[L_Nidx[i]];
    body[i] = aux>0. ? aux/r : 0.;
  }
}


bool PFLineLimits::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			    const int& nnz, int* i, int* j, double* M)
{
#ifdef DEBUG
  int nnz_loc=get_Jacob_nnz();
#endif
  int row, *itnz=J_nz_idxs; const int* L_Nidx = Nidx.data();
  if(NULL==M) {
    for(int it=0; it<n; it++) {
      row = this->index+it;
      //v_n, p_li, q_li, and slack_li in this order
      i[*itnz]=row; j[*itnz]=v_n->index+L_Nidx[it]; itnz++;
      i[*itnz]=row; j[*itnz]=p_li->index+it; itnz++;
      i[*itnz]=row; j[*itnz]=q_li->index+it; itnz++;
      i[*itnz]=row; j[*itnz]=sslack_li->index+it; itnz++;
    }
#ifdef DEBUG
    assert(J_nz_idxs + nnz_loc == itnz);
#endif
  } else {
    const double* Rate = L_Rate.data(); double cc, R;
    for(int it=0; it<n; it++) {
      R=Rate[it];
      cc = R * v_n->xref[L_Nidx[it]] + r*sslack_li->xref[it];
      cc *= 2;

      M[*itnz] -= R*cc;              itnz++; //vn
      M[*itnz] += 2*p_li->xref[it];  itnz++; //p_li
      M[*itnz] += 2*q_li->xref[it];  itnz++; //q_li
      M[*itnz] -= r*cc;              itnz++; //sslack_li
    }
  }
  return true;
}
// p_li[l,i]^2 + q_li[l,i]^2 - (L[Rate][l]*v_n[L_Nidx[l,i]] + r*sslack_li[l,i])^2) <=0
// Jacobian : let  c = L[Rate][l]*v_n[L_Nidx[l,i]] + r*sslack_li[l,i]
// w.r.t to:   v_n        p_li    q_li   sslack_li
//          -2*Rate*c   2*p_li   2*q_li    -2*c*r
int PFLineLimits::get_Jacob_nnz(){ 
  return 4*n; 
}
bool PFLineLimits::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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
    //v_n, p_li, q_li, and slack_li in this order
    vij.push_back(OptSparseEntry(row, v_n->index+Nidx[it], itnz++));
    vij.push_back(OptSparseEntry(row, p_li->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, q_li->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, sslack_li->index+it, itnz++));
  }
  //printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
  assert(nnz+n_vij_in==vij.size());
#endif
  assert(J_nz_idxs+nnz == itnz);
  return true;
}


bool PFLineLimits::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
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
      row = v_n->index+Nidx[it];
      ia[*itnz] = ja[*itnz] = row; itnz++; // w.r.t. (v_n,v_n)

      i = uppertr_swap(row,j=sslack_li->index+it, aux);
      ia[*itnz] = i; ja[*itnz] = j; itnz++; // w.r.t. (v_n,sslack_n)

      ia[*itnz] = ja[*itnz] = p_li->index + it; itnz++; // w.r.t. (p_li,p_li)
      ia[*itnz] = ja[*itnz] = q_li->index + it; itnz++; // w.r.t. (q_li,q_li)
      ia[*itnz] = ja[*itnz] = sslack_li->index + it; itnz++; // w.r.t. (sslack_li,sslack_li)
    }
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==n);
    const double* Rate = L_Rate.data(); double aux, R, lam;
    for(int it=0; it<n; it++) {
      lam=lambda->xref[it]; lam *= 2;
      R=Rate[it]; aux = R*lam*r; //aux = 2*R*lam*r
      M[*itnz++] -= aux*R; //w.r.t. (v_n, v_n)
      M[*itnz++] -= aux;   //w.r.t. (v_n, sslack_li)
      M[*itnz++] += lam;   //w.r.t. (p_li...
      M[*itnz++] += lam;   //w.r.t. (q_li...
      M[*itnz++] -= lam*r; //w.r.t. (sslack_li...
    }
  }
#ifdef DEBUG
  assert(H_nz_idxs+nnz_loc==itnz);
#endif
  return true;
}

// p_li[l,i]^2 + q_li[l,i]^2 - (L[Rate][l]*v_n[L_Nidx[l,i]] + r*sslack_li[l,i])^2) <=0
// Jacobian : let  c = L[Rate][l]*v_n[L_Nidx[l,i]] + r*sslack_li[l,i]
// w.r.t to:   v_n        p_li    q_li   sslack_li
//          -2*Rate*c   2*p_li   2*q_li    -2*r*c
// Hessian
// -2*Rate^2    (v_n,v_n)
// -2*Rate*r    (v_n,sslack_li)
//  2           (p_li,p_li)
//  2           (q_li,q_li)
// -2*r*r       (sslack_li, sslack_li)
int PFLineLimits::get_HessLagr_nnz() 
{ 
  return 5*n;
}
bool PFLineLimits::get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
{
  if(n==0) return true;
  assert(sslack_li->n == n);
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[get_HessLagr_nnz()];
  }
  
  int *itnz=H_nz_idxs, i, j, row, aux;
  for(int it=0; it<n; it++) {
    row = v_n->index+Nidx[it];
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t. v_n,v_n
    i = uppertr_swap(row,j=sslack_li->index+it, aux);
    vij.push_back(OptSparseEntry(i, j, itnz++)); // w.r.t. v_n,sslack_li

    row = p_li->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t.  (p_li,p_li)

    row = q_li->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t. (q_li,q_li)

    row = sslack_li->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t.  (sslack_li, sslack_li)
  }
  return true;
}
  
OptVariablesBlock* PFLineLimits::create_varsblock() 
{ 
  assert(sslack_li==NULL);
  sslack_li = new OptVariablesBlock(n, string("sslack_li_")+id, 0, 1e+20);
  return sslack_li; 
}
OptObjectiveTerm* PFLineLimits::create_objterm() 
{ 
  return NULL;
  //return new DummySingleVarQuadrObjTerm("pen_sslack_li"+id, sslack_li); 
}


//////////////////////////////////////////////////////////////////
// PFTransfLimits - Transformer thermal limits
//////////////////////////////////////////////////////////////////
PFTransfLimits::PFTransfLimits(const std::string& id_, int numcons,
			       OptVariablesBlock* p_ti_, 
			       OptVariablesBlock* q_ti_,
			       const std::vector<double>& T_Rate_,
			       const double& slacks_rescale)
  : OptConstraintsBlock(id_, numcons), 
    p_ti(p_ti_), q_ti(q_ti_), T_Rate(T_Rate_)
{
  assert(T_Rate.size()==n);
  r = slacks_rescale>0 ? 1/slacks_rescale : 1.;
  //rhs
  for(int i=0; i<n; i++)
    lb[i]=-1e+20;
  for(int i=0; i<n; i++)
    ub[i]=0.;

  sslack_ti = new OptVariablesBlock(n, string("sslack_ti_")+id, 0, 1e+20);

  J_nz_idxs = NULL;
  H_nz_idxs = NULL;
}
PFTransfLimits::~PFTransfLimits() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

// p_ti[t,i]^2 + q_ti[t,i]^2 - (T_Rate][t] + r*sslack_ti[t,i])^2) <=0
bool PFTransfLimits::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
{
  assert(sslack_ti); 
  assert(n == T_Rate.size());
  double* body = body_ + this->index;

  for(int i=0; i<n; i++)
    *body++ += p_ti->xref[i]*p_ti->xref[i];
  body -= n;
  for(int i=0; i<n; i++)
    *body++ += q_ti->xref[i]*q_ti->xref[i];
  body -= n;

  const double* Rate = T_Rate.data();
  double aux;
  for(int i=0; i<n; i++) {
    aux = Rate[i] + r*sslack_ti->xref[i];
    *body++ -= aux*aux;
  }
  return true;
}
// p_ti[t,i]^2 + q_ti[t,i]^2 - (T_Rate][t] + r*sslack_ti[t,i])^2) <=0
void PFTransfLimits::compute_slacks(OptVariablesBlock* sslackti)
{
  assert(sslack_ti->n == n); 
  assert(n == T_Rate.size());
  double* body  = sslackti->x;

  for(int i=0; i<n; i++)
    *body++ = p_ti->x[i]*p_ti->x[i];
  body -= n;

  for(int i=0; i<n; i++) {
    *body += q_ti->x[i]*q_ti->x[i];
    *body = sqrt(*body);
    body++;
  }
  body -= n;

  const double* Rate = T_Rate.data();
  for(int i=0; i<n; i++) {
    *body++ -= Rate[i];
  }
  body -= n;

  for(int i=0; i<n; i++) {
    body[i] = body[i]>0 ? body[i]/r : 0;
  }
}

  
bool PFTransfLimits::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			      const int& nnz, int* i, int* j, double* M)
{
#ifdef DEBUG
  int nnz_loc=get_Jacob_nnz();
#endif
  int row, *itnz=J_nz_idxs; 
  if(NULL==M) {
    for(int it=0; it<n; it++) {
      row = this->index+it;
      //p_ti, q_ti, and slack_li in this order
      i[*itnz]=row; j[*itnz]=p_ti->index+it; itnz++;
      i[*itnz]=row; j[*itnz]=q_ti->index+it; itnz++;
      i[*itnz]=row; j[*itnz]=sslack_ti->index+it; itnz++;
    }
#ifdef DEBUG
    assert(J_nz_idxs + nnz_loc == itnz);
#endif
  } else {
    const double* Rate = T_Rate.data(); double c;
    for(int it=0; it<n; it++) {
      c = Rate[it] + r*sslack_ti->xref[it];

      M[*itnz] += 2*p_ti->xref[it];  itnz++; //p_ti
      M[*itnz] += 2*q_ti->xref[it];  itnz++; //q_ti
      M[*itnz] -= 2*c*r;             itnz++; //sslack_ti
    }
  }
  return true;
}

// p_ti[t,i]^2 + q_ti[t,i]^2 - (Rate[t] + r*sslack_ti[t,i])^2) <=0
// Jacobian : let  c = Rate[t] + r*sslack_ti[t,i]
// w.r.t to:   p_ti    q_ti   sslack_ti
//            2*p_ti   2*q_ti    -2*c*r
int PFTransfLimits::get_Jacob_nnz(){ 
  return 3*n; 
}
bool PFTransfLimits::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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
    //v_n, p_ti, q_ti, and slack_li in this order
    vij.push_back(OptSparseEntry(row, p_ti->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, q_ti->index+it, itnz++));
    vij.push_back(OptSparseEntry(row, sslack_ti->index+it, itnz++));
  }
  //printf("nnz=%d vijsize=%d\n", nnz, vij.size());
#ifdef DEBUG
  assert(nnz+n_vij_in==vij.size());
#endif
  assert(J_nz_idxs+nnz == itnz);
  return true;
}


bool PFTransfLimits::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
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
      ia[*itnz] = ja[*itnz] = p_ti->index + it; itnz++; // w.r.t. (p_ti,p_ti)
      ia[*itnz] = ja[*itnz] = q_ti->index + it; itnz++; // w.r.t. (q_ti,q_ti)
      ia[*itnz] = ja[*itnz] = sslack_ti->index + it; itnz++; // w.r.t. (sslack_ti,sslack_ti)
    }
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL);
    assert(lambda->n==n);
    double aux, lam, rsq=r*r;
    for(int it=0; it<n; it++) {
      lam=lambda->xref[it]; lam = 2*lam;
      M[*itnz++] += lam;     //w.r.t. (p_ti...
      M[*itnz++] += lam;     //w.r.t. (q_ti...
      M[*itnz++] -= lam*rsq; //w.r.t. (sslack_ti...
    }
  }
#ifdef DEBUG
  assert(H_nz_idxs+nnz_loc==itnz);
#endif
  return true;
}

// p_ti[t,i]^2 + q_ti[t,i]^2 - (T[Rate][t] + r*sslack_ti[t,i])^2) <=0
// Jacobian : let  c = Rate[t] + r*sslack_ti[t,i]
// w.r.t to:    p_ti    q_ti   sslack_ti
//             2*p_ti   2*q_ti    -2*c*r
// Hessian
//  2           (p_ti,p_ti)
//  2           (q_ti,q_ti)
// -2*r*r       (sslack_ti, sslack_ti)
int PFTransfLimits::get_HessLagr_nnz() 
{ 
  return 3*n;
}
bool PFTransfLimits::get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
{
  if(n==0) return true;
      
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[get_HessLagr_nnz()];
  }
  
  int *itnz=H_nz_idxs, i, j, row, aux;
  for(int it=0; it<n; it++) {

    row = p_ti->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t.  (p_ti,p_ti)

    row = q_ti->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t. (q_ti,q_ti)

    row = sslack_ti->index + it;
    vij.push_back(OptSparseEntry(row, row, itnz++)); // w.r.t.  (sslack_ti, sslack_ti)
  }
  return true;
}
  
OptVariablesBlock* PFTransfLimits::create_varsblock() 
{ 
  return sslack_ti; 
}
OptObjectiveTerm* PFTransfLimits::create_objterm() 
{ 
  return NULL;//new DummySingleVarQuadrObjTerm("pen_sslack_ti"+id, sslack_ti); 
}
 

////////////////////////////////////////////////////////////////////////////
// PFProdCostAffineCons - constraints needed by the piecewise linear 
// production cost function  
// min sum_g( sum_h CostCi[g][h]^T t[g][h])
// constraints (handled outside) are
//   t>=0, sum_h t[g][h]=1, for g=1,2,...
//   p_g[g] - sum_h CostPi[g][h]*t[g][h] =0 , for all for g=1,2,...
///////////////////////////////////////////////////////////////////////////
PFProdCostAffineCons::
PFProdCostAffineCons(const std::string& id_, int numcons,
		     OptVariablesBlock* p_g_, 
		     const std::vector<int>& G_idx_,
		     const std::vector<std::vector<double> >& G_CostCi_,
		     const std::vector<std::vector<double> >& G_CostPi_)
  : OptConstraintsBlock(id_,2*p_g_->n), p_g(p_g_), G_CostPi(G_CostPi_), J_nz_idxs(NULL)
{
  assert(p_g->n == G_idx_.size());

  //rhs of this block
  //lb = new double[n];
  for(int i=0; i<n; ) {
    lb[i++] = 1.; lb[i++] = 0.;
  }

  //ub = new double[n];
  DCOPY(&n, lb, &ione, ub, &ione);

  int sz_t_h = 0;
  //we create here the extra variables and the objective term
  for(auto idx: G_idx_) 
    sz_t_h += G_CostPi[idx].size();
  t_h = new OptVariablesBlock(sz_t_h, "t_h_0", 0., 1e+20);

  obj_term = new PFProdCostPcLinObjTerm(id+"_cons", t_h, G_idx_, G_CostCi_);
}
PFProdCostAffineCons::~PFProdCostAffineCons()
{
  delete[] J_nz_idxs;
  //do not delete t_h, obj_term; OptProblem frees them by convention
}

bool PFProdCostAffineCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  assert(p_g->n*2 == n);
  const double* Pi; int sz, i; 
  double* itbody=body+this->index; const double *it_t_h=t_h->xref;
  for(int it=0; it<p_g->n; it++) {
    sz = G_CostPi[obj_term->Gidx[it]].size(); 
    Pi = G_CostPi[obj_term->Gidx[it]].data();
    for(i=0; i<sz; i++) {
      *itbody     += *it_t_h;
      *(itbody+1) -= Pi[i]* (*it_t_h);

      it_t_h++;
    }
    itbody++;

    *itbody += p_g->xref[it];
    itbody++;
  }
  assert(body+this->index+n == itbody);
  assert(t_h->xref+t_h->n == it_t_h);
#ifdef DEBUG
  //for(int i=0; i<n/2; i++) 
  //  printf("residual %12.5e %12.5e \n", body[2*i]-1, body[2*i+1]);
  //printf("\n");
  //assert(false);
#endif
  

  return true;
}
// given p_g[g] computes t[g][h]
void PFProdCostAffineCons::compute_t_h(OptVariablesBlock* th)
{
  int sz, i; double *it_th=th->x;
  for(int it=0; it<p_g->n; it++) {
    sz = G_CostPi[obj_term->Gidx[it]].size(); 
    //!copy
    vector<double> Pi = G_CostPi[obj_term->Gidx[it]];

    assert(sz>=2 && "need to have at least two cost points");
    sort(Pi.begin(), Pi.end());
    assert(Pi == G_CostPi[obj_term->Gidx[it]] && "we expect sorted generation cost points for any given generator");

    int idx_min = 0; //distance(begin(Pi), min_element(begin(Pi), end(Pi)));
    int idx_max = sz-1; //distance(begin(Pi), max_element(begin(Pi), end(Pi)));
    if(p_g->x[it]<Pi[idx_min]) {
      printf("Warning: p_g[%d] is too much outside the min of generation cost points. will adjust it.\n", it);
      p_g->x[it] = Pi[idx_min];
    } 
    if(p_g->x[it]>Pi[idx_max]) {
      printf("Warning: p_g[%d] is too much outside the max of generation cost points. will adjust it.\n", it);
      p_g->x[it] = Pi[idx_max];
    }
    for(i=0; i<sz; i++, it_th++) {
      if(Pi[i]>p_g->x[it]) break;
      *it_th = 0.;
    }
    if(i==sz) {
      assert(p_g->x[it] == Pi[sz-1]);
      i = i-1;
      it_th--;
    }
    assert(i>=1);
    assert(i<sz);
    assert(Pi[i]   >= p_g->x[it]);
    assert(Pi[i-1] <= p_g->x[it]);
    assert(Pi[i]-Pi[i-1] >=1e-10);
    *(it_th-1) = 1 - (p_g->x[it] - Pi[i-1]) / (Pi[i]-Pi[i-1]);
    *it_th     = 1 - *(it_th-1);

    i++; it_th++;
    for(; i<sz; i++, it_th++) *it_th = 0.;
  }
  assert(th->x+th->n == it_th);
}

bool PFProdCostAffineCons::eval_Jac(const OptVariables& primal_vars, bool new_x, 
				    const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz, sz; int t_h_idx=t_h->index;
  if(NULL==M) {
    for(int it=0; it<p_g->n; it++) {
      row = this->index+2*it;
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      //for sum_h t[g][h]=1
      //w.r.t. p_g, t[g][h]
      sz = G_CostPi[obj_term->Gidx[it]].size(); 
      for(int i=0; i<sz; i++) {
	ia[idxnz]=row; ja[idxnz]=t_h_idx+i; idxnz++;
      }
      
      //for p_g[g] - sum_h CostPi[g][h]*t[g][h] =0
      //w.r.t. t[g][h]
      row++;
      ia[idxnz]=row; ja[idxnz]=p_g->index+it; idxnz++; //p_g
      for(int i=0; i<sz; i++) {
	ia[idxnz]=row; ja[idxnz]=t_h_idx+i; idxnz++;
      }
      t_h_idx += sz;
    }
    assert(row+1 == this->index+this->n);
    assert(t_h_idx == t_h->index+t_h->n);
  } else {
    const double* Pi;
    for(int it=0; it<p_g->n; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      //for sum_h t[g][h]=1
      //w.r.t. p_g, t[g][h]
      sz = G_CostPi[obj_term->Gidx[it]].size(); 
      for(int i=0; i<sz; i++) {
	M[idxnz++] += 1.;
      }
      
      Pi = G_CostPi[obj_term->Gidx[it]].data();
      //for p_g[g] - sum_h CostPi[g][h]*t[g][h] =0
      //w.r.t. t[g][h]
      row++;
      M[idxnz++] += 1.; //p_g
      for(int i=0; i<sz; i++) {
	M[idxnz++] -= Pi[i];
      }
    }
  }
  return true;
}
int PFProdCostAffineCons::get_Jacob_nnz()
{
  int nnz = p_g->n; //pg from second constraint
  for(int it=0; it<p_g->n; it++) nnz+= 2 * G_CostPi[obj_term->Gidx[it]].size(); 
  return nnz;
}
bool PFProdCostAffineCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
  int vij_sz_in = vij.size();
#endif

  if(!J_nz_idxs) 
    J_nz_idxs = new int[p_g->n];

  int row=this->index; int sz; int t_h_idx=t_h->index;
  for(int it=0; it<p_g->n; it++) {
    row = this->index+2*it;

    //for sum_h t[g][h]=1
    //w.r.t. p_g, t[g][h]
    vij.push_back(OptSparseEntry(row, p_g->index+it, J_nz_idxs+it));

    sz = G_CostPi[obj_term->Gidx[it]].size(); 

    for(int i=0; i<sz; i++)
      vij.push_back(OptSparseEntry(row, t_h_idx+i, NULL));

    //for p_g[g] - sum_h CostPi[g][h]*t[g][h] =0
    //w.r.t. t[g][h]
    row++;
    for(int i=0; i<sz; i++)
      vij.push_back(OptSparseEntry(row, t_h_idx+i, NULL));

    t_h_idx += sz;
  }
#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(t_h_idx == t_h->index+t_h->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}


//////////////////////////////////////////////////////////////////////////////
// Slack penalty constraints block
// min sum_i( sum_h P[i][h] sigma_h[i][h])
// constraints are
//   0<= sigma[i][h] <= S_h, 
//   slacks[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
// 
// Upon rescaling
// min sum_i( sum_h P[i][h]*f sigma_h[i][h])
//  s.t. 0<= sigma[i][h] <= S_h/f
//       slacks[i] - sum_h sigma[i][h] =0
// where f = 1/slacks_rescale, usually 1/256 or 1/512
//////////////////////////////////////////////////////////////////////////////
PFPenaltyAffineCons::
PFPenaltyAffineCons(const std::string& id_, int numcons,
		    OptVariablesBlock* slack_, 
		    const std::vector<double>& pen_coeff,
		    const std::vector<double>& pen_segm,
		    const double& obj_weight, // DELTA or (1-DELTA)/NumK
		    const double& slacks_rescale)
  : OptConstraintsBlock(id_,numcons), slack(slack_), J_nz_idxs(NULL)
{
  assert(pen_segm.size()==3);
  assert(pen_coeff.size()==3);

  assert(slacks_rescale>1e-6);
  double rescale = slacks_rescale>0 ? slacks_rescale : 1.; 

  P1=pen_coeff[0]/rescale; P2=pen_coeff[1]/rescale; P3=pen_coeff[2]/rescale;
  S1=pen_segm[0]*rescale;  S2=pen_segm[1]*rescale;  S3=pen_segm[2]*rescale;

  //rhs of this block
  //lb = new double[n];
  for(int i=0; i<n; i++) lb[i] = 0.; 

  //ub = new double[n];
  DCOPY(&n, lb, &ione, ub, &ione);

  //for(int i=0; i<n; ) {ub[i++] = S1; ub[i++] = S2; ub[i++] = S3;} 
  sigma = new OptVariablesBlock(3*n, id+"_sigma", 0., S3);
  for(int i=0; i<sigma->n; ) { sigma->ub[i++]=S1; sigma->ub[i++]=S2; i++;}

  obj_term = new PFPenaltyPcLinObjTerm(id+"_obj", sigma, pen_coeff, obj_weight, slacks_rescale);
}
PFPenaltyAffineCons::~PFPenaltyAffineCons()
{
  delete[] J_nz_idxs;
  //do not delete t_h, obj_term; OptProblem frees them by convention
}
  
bool PFPenaltyAffineCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  double* itbody=body+this->index; 
  for(int it=0; it<sigma->n; it+=3) {
    *itbody++ -= sigma->xref[it]+sigma->xref[it+1]+sigma->xref[it+2];
  }
  assert(body+this->index+n == itbody);

  DAXPY(&n, &done, const_cast<double*>(slack->xref), &ione, body+this->index, &ione);
  return true;
}

void PFPenaltyAffineCons::compute_sigma(OptVariablesBlock *sigmav)
{
  assert(sigmav==sigma);
  double slack_val;
  for(int it=0; it<sigma->n; it+=3) {
    assert(3*(it/3) == it);
    slack_val=slack->x[it/3];
    assert(slack_val>=0);

    sigma->x[it] = min(slack_val, S1);
    slack_val -= sigma->x[it];

    sigma->x[it+1] = min(slack_val, S2);
    slack_val -= sigma->x[it+1];

    sigma->x[it+2] = min(slack_val, S3);
  }
}

bool PFPenaltyAffineCons::eval_Jac(const OptVariables& primal_vars, bool new_x, 
				    const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz;
  if(NULL==M) {
    assert(slack->n == n);
    for(int it=0; it<n; it++) {
      row = this->index+it;
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      ia[idxnz]=row; ja[idxnz]=slack->index+it;     idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it;   idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it+1; idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it+2; idxnz++; 
    }
    assert(row+1 == this->index+this->n);
  } else {

    for(int it=0; it<n; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      M[idxnz++] += 1.; //slack
      M[idxnz++] -= 1.; //sigma1
      M[idxnz++] -= 1.; //sigma2
      M[idxnz++] -= 1.; //sigma3
    }
  }
  return true;
}
int PFPenaltyAffineCons::get_Jacob_nnz()
{
  return 4*n;
}

//slacks[i] - sum_h sigma[i][h] =0, i=1,2, size(slacks)
bool PFPenaltyAffineCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
  int vij_sz_in = vij.size();
#endif

  if(!J_nz_idxs) 
    J_nz_idxs = new int[n];

  int row=0; 
  for(int it=0; it<n; it++) {
    row = this->index+it;

    vij.push_back(OptSparseEntry(row, slack->index+it, J_nz_idxs+it));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it, NULL));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it+1, NULL));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it+2, NULL));
  }
#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// (double) Slacks penalty constraints block
// min sum_i( sum_h P[i][h] sigma_h[i][h])
// constraints (handled outside) are
//   0<= sigma[i][h] <= Pseg_h, 
//   slacksp[i] + slacksm[i] - sum_h sigma[i][h] =0, i=1,2, size(slacksp)
//
// the two slacks are kept vectorized: slack_ = [slacksp; slackm]
//////////////////////////////////////////////////////////////////////////////
bool PFPenaltyAffineConsTwoSlacks::
eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  double* itbody=body+this->index; 
  for(int it=0; it<sigma->n; it+=3) {
    *itbody++ -= sigma->xref[it]+sigma->xref[it+1]+sigma->xref[it+2];
  }
  assert(body+this->index+n == itbody);

  DAXPY(&n, &done, const_cast<double*>(slack->xref),   &ione, body+this->index, &ione);
  DAXPY(&n, &done, const_cast<double*>(slack->xref)+n, &ione, body+this->index, &ione);
  return true;
}

void PFPenaltyAffineConsTwoSlacks::compute_sigma(OptVariablesBlock *sigmav)
{
  assert(sigmav==sigma);
  double slack_val; int i;
  for(int it=0; it<sigma->n; it+=3) {
    i = it/3;
    assert(3*i == it);
    slack_val = slack->x[i] + slack->x[i+n];
    assert(slack_val>=0);

    sigma->x[it] = min(slack_val, S1);
    slack_val -= sigma->x[it];

    sigma->x[it+1] = min(slack_val, S2);
    slack_val -= sigma->x[it+1];

    sigma->x[it+2] = min(slack_val, S3);
  }
}

bool PFPenaltyAffineConsTwoSlacks::eval_Jac(const OptVariables& primal_vars, bool new_x, 
					    const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz;
  if(NULL==M) {
    assert(slack->n == 2*n);
    for(int it=0; it<n; it++) {
      row = this->index+it;
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      ia[idxnz]=row; ja[idxnz]=slack->index+it;     idxnz++; 
      ia[idxnz]=row; ja[idxnz]=slack->index+it+n;   idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it;   idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it+1; idxnz++; 
      ia[idxnz]=row; ja[idxnz]=sigma->index+3*it+2; idxnz++; 
    }
    assert(row+1 == this->index+this->n);
  } else {

    for(int it=0; it<n; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      M[idxnz++] += 1.; //slackp
      M[idxnz++] += 1.; //slackm
      M[idxnz++] -= 1.; //sigma1
      M[idxnz++] -= 1.; //sigma2
      M[idxnz++] -= 1.; //sigma3
    }
  }
  return true;
}
int PFPenaltyAffineConsTwoSlacks::get_Jacob_nnz()
{
  return 5*n;
}

//slackp[i]+slackm[i] - sum_h sigma[i][h] =0, i=1,2, size(slacksp)
bool PFPenaltyAffineConsTwoSlacks::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
  int vij_sz_in = vij.size();
#endif

  if(!J_nz_idxs) 
    J_nz_idxs = new int[n];

  int row=0; 
  for(int it=0; it<n; it++) {
    row = this->index+it;

    vij.push_back(OptSparseEntry(row, slack->index+it, J_nz_idxs+it));
    vij.push_back(OptSparseEntry(row, slack->index+n+it, NULL));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it, NULL));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it+1, NULL));
    vij.push_back(OptSparseEntry(row, sigma->index+3*it+2, NULL));
  }
#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// AGC reserve constraints 
// i. loss reserve
//   a. max loss:  sum(Pub[i]-pg[i]: i in AGC) - f*max_loss + s >=0
//   b. Kgen loss: sum(Pub[i]-pg[i]: i in AGC) - f*pg[Kgen] + s >=0
//
// ii. gain reserve
//   a. max gain:  sum(pg[i]-Plb[i]: i in AGC) - f*max_gain + s >=0 
//   b. Kgen gain: sum(pg[i]-Plb[i]: i in AGC) + f*pg[Kgen] + s >=0 
//
// 'f' is the percentage of the loss/gain that should be covered by the AGC gens
// usually 1. or closely to the left of 1. (0.95 or 0.99)

AGCReservesCons::AGCReservesCons(const std::string& id_, OptVariablesBlock* p_g_)
  : OptConstraintsBlock(id_,0), p_g(p_g_), J_nz_idxs(NULL)
{
  lb=lb_.data(); ub=ub_.data();
  slacks = NULL;
  obj_penalty = NULL;
#ifdef DEBUG
  isAssembled = false;
#endif
}

void AGCReservesCons::add_max_loss_reserve(const std::vector<int>& idxs_agc, 
					   const double& max_loss, const double& f,
					   const std::vector<double>& Pub)
{
  idxs_.push_back(idxs_agc);
  coeff_.push_back(vector<double>());
  vector<double>& coeff = coeff_.back();
  
  ub_.push_back(1e+20);
  lb_.push_back(f*max_loss); 
  double& dlb=lb_.back();
  
  assert(lb_.size() == idxs_.size());
  assert(lb_.size() == coeff_.size());
  
  for(int idx: idxs_agc) {
    coeff.push_back(-1.);
    dlb -= Pub[idx];
  }
  ub = ub_.data();
  lb = lb_.data();
  n = ub_.size();
}
void AGCReservesCons::add_Kgen_loss_reserve(const std::vector<int>& idxs_agc, 
					    const int& idx_Kgen, const double& f,
					    const std::vector<double>& Pub)
{
  this->add_max_loss_reserve(idxs_agc, 0., 1., Pub);
  idxs_.back().push_back(idx_Kgen);
  coeff_.back().push_back(-f);
}

// ii. gain reserve
//   a. max gain:  sum(pg[i]-Plb[i]: i in AGC) - f*max_gain + s >=0 
//   b. Kgen gain: sum(pg[i]-Plb[i]: i in AGC) + f*pg[Kgen] + s >=0 
void AGCReservesCons::add_max_gain_reserve(const std::vector<int>& idxs_agc, 
					   const double& max_gain, const double& f,
					   const std::vector<double>& Plb)
{
  idxs_.push_back(idxs_agc);
  coeff_.push_back(std::vector<double>());
  vector<double>& coeff = coeff_.back();
  
  ub_.push_back(1e+20);
  lb_.push_back(f*max_gain); 
  double& dlb=lb_.back();
  
  assert(lb_.size() == idxs_.size());
  assert(lb_.size() == coeff_.size());
  
  for(int idx: idxs_agc) {
    coeff.push_back(1.);
    dlb += Plb[idx];
  }
  ub = ub_.data();
  lb = lb_.data();
  n = ub_.size();
}

void AGCReservesCons::add_Kgen_gain_reserve(const std::vector<int>& idxs_agc, 
					    const int& idx_Kgen, const double& f,
					    const std::vector<double>& Plb)
{
  this->add_max_gain_reserve(idxs_agc, 0., 1., Plb);
  idxs_.back().push_back(idx_Kgen);
  coeff_.back().push_back(f);
}

void AGCReservesCons::finalize_setup()
{
  n = lb_.size();
  assert(n==ub_.size());
  lb = new double[n];
  ub = new double[n];
  memcpy(lb, lb_.data(), n*sizeof(double));
  memcpy(ub, ub_.data(), n*sizeof(double));

  //printvec(lb_, "lower bound");
  //printvec(ub_, "upper bound");

  //printvecvec(idxs_, "idxs");
  //printvecvec(coeff_, "coeff");


  hardclear(lb_); hardclear(ub_);

  assert(NULL!=slacks);
  assert(NULL!=obj_penalty);

#ifdef DEBUG
  isAssembled=true;
#endif
}

void AGCReservesCons::add_penalty_objterm(const std::vector<double>& P_pen,
					  const std::vector<double>& P_qua,
					  const double& obj_weight,
					  const double& slacks_scale)
{
  assert(slacks==NULL);
  slacks = new OptVariablesBlock(n, string("sslack_")+id, 0., 1e+20);

  slacks->set_start_to(0.0001);
  slacks->providesStartingPoint=true;

  obj_penalty = new PFPenaltyQuadrApproxObjTerm("quadr_pen_" + slacks->id, 
						slacks, 
						P_pen, P_qua, obj_weight, 
						slacks_scale);

}

bool AGCReservesCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body_)
{
  assert(isAssembled);
  assert(slacks);
  assert(n==coeff_.size());
  assert(n==idxs_.size());
  double* body = body_ + this->index;
  double* s = const_cast<double*>(slacks->xref);

  for(int i=0; i<n; i++) {
    for(int it=0; it<idxs_[i].size(); it++) {
      assert(coeff_[i].size()>=1);
      assert(coeff_[i].size() == idxs_[i].size());
      body[i] += p_g->xref[idxs_[i][it]]*coeff_[i][it];
    }
    body[i] += s[i];
  }

#ifdef DEBUG

#endif

  return true;
}

int AGCReservesCons::get_Jacob_nnz()
{
  int nnz=0;
  for(int i=0; i<n; i++) nnz += idxs_[i].size();
  //nz for slacks
  nnz+= n; 
  return nnz;
}

bool AGCReservesCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
  //if(n<=0) return true;
  assert(isAssembled);
  int nnz = get_Jacob_nnz();

  if(!J_nz_idxs) 
    J_nz_idxs = new int[nnz];

  int row, *itnz=J_nz_idxs;
  for(int it=0; it<n; it++) {
    row=this->index+it;
    for(int idx_gen: idxs_[it])
      vij.push_back(OptSparseEntry(row, p_g->index+idx_gen, itnz++));
    
    vij.push_back(OptSparseEntry(row, slacks->index+it, itnz++));
    
  }
  assert(J_nz_idxs + nnz == itnz);
  return true;
}

bool AGCReservesCons::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			       const int& nnz, int* i, int* j, double* M)
{
  assert(isAssembled);
  int row, *itnz=J_nz_idxs;
  if(NULL==M) {
    for(int it=0; it<n; it++) {
      row = this->index+it;
      for(int idx_gen: idxs_[it]) {
	i[*itnz]=row; j[*itnz]=p_g->index+idx_gen; itnz++;
      }
      i[*itnz]=row; j[*itnz]=slacks->index+it; itnz++;

    }
    assert(J_nz_idxs + get_Jacob_nnz() == itnz);
  } else {
    //values

    for(int it=0; it<n; it++) {
      for(double coeff : coeff_[it]) {
	M[*itnz] += coeff; itnz++; //w.r.t. agc p_g
      }
      M[*itnz] += 1.; itnz++; //w.r.t. slacks
    }
    assert(J_nz_idxs + get_Jacob_nnz() == itnz);
  }

  return true;
}

AGCReservesCons::~AGCReservesCons() 
{
  delete[] J_nz_idxs;
}


}//end namespace 

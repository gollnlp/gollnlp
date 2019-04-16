#include "OPFConstraints.hpp"

#include <string>

using namespace std;

namespace gollnlp {

//
// Thermal line limits
//
PFLineLimits::PFLineLimits(const std::string& id_, int numcons,
			   OptVariablesBlock* p_li_, 
			   OptVariablesBlock* q_li_,
			   OptVariablesBlock* v_n_,
			   const std::vector<int>& L_Nidx_,
			   const std::vector<double>& L_Rate_,
			   const SCACOPFData& d_)
  : OptConstraintsBlock(id_, numcons), 
    p_li(p_li_), q_li(q_li_), v_n(v_n_), Nidx(L_Nidx_), L_Rate(L_Rate_), d(d_), sslack_li(NULL)
{
  assert(d.L_Line.size()==n);
  //rhs
  for(int i=0; i<n; i++)
    lb[i]=-1e+20;
  for(int i=0; i<n; i++)
    ub[i]=0;
  J_nz_idxs = NULL;
  H_nz_idxs = NULL;
}
PFLineLimits::~PFLineLimits() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

// p_li[l,i]^2 + q_li[l,i]^2 - (L[RateSymb][l]*v_n[L_Nidx[l,i]] + sslack_li[l,i])^2) <=0
bool PFLineLimits::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  assert(sslack_li); 
  assert(n == L_Rate.size());
  assert(n == Nidx.size());
  body += this->index;
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
    aux = Rate[i]*v_n->xref[L_Nidx[i]] + sslack_li->xref[i];
    *body++ -= aux*aux;
  }
  return true;
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
      cc = R * v_n->xref[L_Nidx[it]] + sslack_li->xref[it];
      cc *= 2;

      M[*itnz] -= R*cc;              itnz++; //vn
      M[*itnz] += 2*p_li->xref[it];  itnz++; //p_li
      M[*itnz] += 2*q_li->xref[it];  itnz++; //q_li
      M[*itnz] -= cc;                itnz++; //sslack_li
    }
  }
  return true;
}
// p_li[l,i]^2 + q_li[l,i]^2 - (L[Rate][l]*v_n[L_Nidx[l,i]] + sslack_li[l,i])^2) <=0
// Jacobian : let  c = L[Rate][l]*v_n[L_Nidx[l,i]] + sslack_li[l,i]
// w.r.t to:   v_n        p_li    q_li   sslack_li
//          -2*Rate*c   2*p_li   2*q_li    -2*c
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
  printf("nnz=%d vijsize=%d\n", nnz, vij.size());
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
      lam=lambda->xref[it]; lam *= 2*lam;
      R=Rate[it]; aux = R*lam; 
      M[*itnz++] -= aux*R; //w.r.t. (v_n, v_n)
      M[*itnz++] -= aux;   //w.r.t. (v_n, sslack_li)
      M[*itnz++] += lam;   //w.r.t. (p_li...
      M[*itnz++] += lam;   //w.r.t. (q_li...
      M[*itnz++] -= lam;   //w.r.t. (sslack_li...
    }
  }
#ifdef DEBUG
  assert(H_nz_idxs+nnz_loc==itnz);
#endif
  return true;
}

// p_li[l,i]^2 + q_li[l,i]^2 - (L[Rate][l]*v_n[L_Nidx[l,i]] + sslack_li[l,i])^2) <=0
// Jacobian : let  c = L[Rate][l]*v_n[L_Nidx[l,i]] + sslack_li[l,i]
// w.r.t to:   v_n        p_li    q_li   sslack_li
//          -2*Rate*c   2*p_li   2*q_li    -2*c
// Hessian
// -2*Rate^2    (v_n,v_n)
// -2*Rate      (v_n,sslack_li)
//  2           (p_li,p_li)
//  2           (q_li,q_li)
// -2           (sslack_li, sslack_li)
int PFLineLimits::get_HessLagr_nnz() 
{ 
  return 5*n;
}
bool PFLineLimits::get_HessLagr_ij(std::vector<OptSparseEntry>& vij) 
{
  if(n==0) return true;
      
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
  return new DummySingleVarQuadrObjTerm("pen_sslack_li"+id, sslack_li); 
}
  
}//end namespace 

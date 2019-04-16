#include "OPFConstraints.hpp"

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
    lb[i]=0.;
  for(int i=0; i<n; i++)
    ub[i]=+1e+20;
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
      row = this->index+it;
      
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
  
}//end namespace 

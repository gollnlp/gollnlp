#include "OPFConstraints.hpp"

#include <string>

using namespace std;

namespace gollnlp {

//////////////////////////////////////////////////////////////////
// PFLineLimits - Line thermal limits
//////////////////////////////////////////////////////////////////
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


//////////////////////////////////////////////////////////////////
// PFTransfLimits - Transformer thermal limits
//////////////////////////////////////////////////////////////////
PFTransfLimits::PFTransfLimits(const std::string& id_, int numcons,
			   OptVariablesBlock* p_ti_, 
			   OptVariablesBlock* q_ti_,
			   const std::vector<int>& T_Nidx_,
			   const std::vector<double>& T_Rate_,
			   const SCACOPFData& d_)
  : OptConstraintsBlock(id_, numcons), 
    p_ti(p_ti_), q_ti(q_ti_), Nidx(T_Nidx_), T_Rate(T_Rate_), d(d_), sslack_ti(NULL)
{
  assert(d.T_Transformer.size()==n);
  //rhs
  for(int i=0; i<n; i++)
    lb[i]=-1e+20;
  for(int i=0; i<n; i++)
    ub[i]=0;
  J_nz_idxs = NULL;
  H_nz_idxs = NULL;
}
PFTransfLimits::~PFTransfLimits() 
{
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

// p_ti[t,i]^2 + q_ti[t,i]^2 - (T_Rate][t] + sslack_ti[t,i])^2) <=0
bool PFTransfLimits::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  assert(sslack_ti); 
  assert(n == T_Rate.size());
  assert(n == Nidx.size());
  body += this->index;
  double* slacks = const_cast<double*>(sslack_ti->xref);

  for(int i=0; i<n; i++)
    *body++ += p_ti->xref[i]*p_ti->xref[i];
  body -= n;
  for(int i=0; i<n; i++)
    *body++ += q_ti->xref[i]*q_ti->xref[i];
  body -= n;

  const double* Rate = T_Rate.data();
  const int* L_Nidx = Nidx.data();
  double aux;
  for(int i=0; i<n; i++) {
    aux = Rate[i] + sslack_ti->xref[i];
    *body++ -= aux*aux;
  }
  return true;
}
  
bool PFTransfLimits::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			      const int& nnz, int* i, int* j, double* M)
{
#ifdef DEBUG
  int nnz_loc=get_Jacob_nnz();
#endif
  int row, *itnz=J_nz_idxs; const int* L_Nidx = Nidx.data();
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
      c = Rate[it] + sslack_ti->xref[it];

      M[*itnz] += 2*p_ti->xref[it];  itnz++; //p_ti
      M[*itnz] += 2*q_ti->xref[it];  itnz++; //q_ti
      M[*itnz] -= 2*c;               itnz++; //sslack_ti
    }
  }
  return true;
}
// p_ti[t,i]^2 + q_ti[t,i]^2 - (Rate[t] + sslack_ti[t,i])^2) <=0
// Jacobian : let  c = Rate[l] + sslack_ti[l,i]
// w.r.t to:   p_ti    q_ti   sslack_ti
//            2*p_ti   2*q_ti    -2*c
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
  printf("nnz=%d vijsize=%d\n", nnz, vij.size());
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
    double aux, R, lam;
    for(int it=0; it<n; it++) {
      lam=lambda->xref[it]; lam *= 2*lam;
      M[*itnz++] += lam;   //w.r.t. (p_ti...
      M[*itnz++] += lam;   //w.r.t. (q_ti...
      M[*itnz++] -= lam;   //w.r.t. (sslack_ti...
    }
  }
#ifdef DEBUG
  assert(H_nz_idxs+nnz_loc==itnz);
#endif
  return true;
}

// p_ti[l,i]^2 + q_ti[l,i]^2 - (L[Rate][l]*v_n[L_Nidx[l,i]] + sslack_ti[l,i])^2) <=0
// Jacobian : let  c = Rate[l] + sslack_ti[l,i]
// w.r.t to:    p_ti    q_ti   sslack_ti
//             2*p_ti   2*q_ti    -2*c
// Hessian
//  2           (p_ti,p_ti)
//  2           (q_ti,q_ti)
// -2           (sslack_ti, sslack_ti)
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
  assert(sslack_ti==NULL);
  sslack_ti = new OptVariablesBlock(n, string("sslack_ti_")+id, 0, 1e+20);
  return sslack_ti; 
}
OptObjectiveTerm* PFTransfLimits::create_objterm() 
{ 
  return new DummySingleVarQuadrObjTerm("pen_sslack_ti"+id, sslack_ti); 
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
		     const SCACOPFData& d_)
  : OptConstraintsBlock(id_,2*p_g_->n), d(d_), p_g(p_g_), J_nz_idxs(NULL)
{
  assert(p_g->n == G_idx_.size());

  //rhs of this block
  lb = new double[n];
  for(int i=0; i<n; ) {
    lb[i++] = 1.; lb[i++] = 0.;
  }

  ub = new double[n];
  DCOPY(&n, lb, &ione, ub, &ione);

  int sz_t_h = 0;
  //we create here the extra variables and the objective term
  for(auto idx: G_idx_) 
    sz_t_h += d.G_CostPi[idx].size();
  t_h = new OptVariablesBlock(sz_t_h, "t_h", 0., 1e+20);
  obj_term = new PFProdCostPcLinObjTerm(id+"_cons", t_h, G_idx_, d);
}
PFProdCostAffineCons::~PFProdCostAffineCons()
{
  delete[] J_nz_idxs;
  //do not delete t_h, obj_term; OptProblem frees them by convention
}
  
bool PFProdCostAffineCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  const double* Pi; int sz; double* itbody=body+this->index; const double *it_t_h=t_h->xref;
  for(int it=0; it<p_g->n; it++) {
    sz = d.G_CostPi[obj_term->Gidx[it]].size(); 
    Pi = d.G_CostPi[obj_term->Gidx[it]].data();
    for(int i=0; i<sz; i++) {
      *itbody += *it_t_h;
      *(itbody+1) -= Pi[i]* (*it_t_h);
      it_t_h++;
    }
    itbody++;
    *itbody++ += p_g->xref[it];
  }
  assert(body+this->index+n == itbody);
  assert(t_h->xref+t_h->n == it_t_h);
  return true;
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
      sz = d.G_CostPi[obj_term->Gidx[it]].size(); 
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
      sz = d.G_CostPi[obj_term->Gidx[it]].size(); 
      for(int i=0; i<sz; i++) {
	M[idxnz++] += 1.;
      }
      
      Pi = d.G_CostPi[obj_term->Gidx[it]].data();
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
  for(int it=0; it<p_g->n; it++) nnz+= 2 * d.G_CostPi[obj_term->Gidx[it]].size(); 
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

    sz = d.G_CostPi[obj_term->Gidx[it]].size(); 

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

}//end namespace 

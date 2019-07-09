#include "CouplingConstraints.hpp"

#include "goUtils.hpp"

#include <string>
#include <cassert>
using namespace std;

namespace gollnlp {
/////////////////////////////////////////////////////////////////////////////////
// non-anticipativity-like constraints 
/////////////////////////////////////////////////////////////////////////////////
NonAnticipCons::NonAnticipCons(const std::string& id_, int numcons,
			       OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, 
			       const std::vector<int>& idx0_, const std::vector<int>& idxK_)
  : OptConstraintsBlock(id_,numcons), pg0(pg0_), pgK(pgK_), J_nz_idxs(NULL)
{
  assert(numcons==idx0_.size());
  assert(numcons==idxK_.size());
  assert(pg0_->n >= numcons);
  assert(pgK_->n >= numcons);
  idx0 = new int[idx0_.size()];
  memcpy(idx0, idx0_.data(), numcons*sizeof(int));
  idxK = new int[idxK_.size()];
  memcpy(idxK, idxK_.data(), numcons*sizeof(int));

  //rhs of this block
  lb = new double[n];
  for(int i=0; i<n; i++) lb[i] = 0.; 
  
  ub = new double[n];
  DCOPY(&n, lb, &ione, ub, &ione);
}

NonAnticipCons::~NonAnticipCons()
{
  delete [] idx0;
  delete [] idxK;
  delete [] J_nz_idxs;
}

bool NonAnticipCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  double* g = body+this->index;
  for(int i=0; i<n; i++)
    g[i] += pg0->xref[idx0[i]] - pgK->xref[idxK[i]];
  return true;
}


bool NonAnticipCons::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			      const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz;
  if(NULL==M) {
    for(int it=0; it<n; it++) {
      row = this->index+it;
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      ia[idxnz]=row; ja[idxnz]=pg0->index+idx0[it]; idxnz++; 
      ia[idxnz]=row; ja[idxnz]=pgK->index+idxK[it]; idxnz++; 
    }
    assert(row+1 == this->index+this->n);
  } else {

    for(int it=0; it<n; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      M[idxnz++] += 1.; //pg0
      assert(idxnz<nnz && idxnz>=0);
      M[idxnz]   -= 1.; //pgK
    }
  }
  return true;
}
int NonAnticipCons::NonAnticipCons::get_Jacob_nnz()
{
  return 2*n;
}
bool NonAnticipCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
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

    vij.push_back(OptSparseEntry(row, pg0->index+idx0[it], J_nz_idxs+it));
    vij.push_back(OptSparseEntry(row, pgK->index+idxK[it], NULL));
  }
#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}

/////////////////////////////////////////////////////////////////////////////////
// simplified AGC constraints 
// p0 + alpha*deltak - pk = 0 
/////////////////////////////////////////////////////////////////////////////////

AGCSimpleCons::AGCSimpleCons(const std::string& id_, int numcons,
			     OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, 
			     OptVariablesBlock* deltaK_,
			     const std::vector<int>& idx0_, const std::vector<int>& idxK_,
			     const std::vector<double>& G_alpha_)
  : OptConstraintsBlock(id_, numcons), pg0(pg0_), pgK(pgK_), deltaK(deltaK_),
    J_nz_idxs(NULL)
{
  assert(idx0_.size()==idxK_.size());
  assert(idx0_.size()==n);
  assert(n<=pg0->n); assert(n<=pgK->n);
  assert(1==deltaK->n);
  assert(G_alpha_.size()==pg0_->n);

  G_alpha = G_alpha_.data();

  int dim = n;
  idx0 = new int[dim];
  memcpy(idx0, idx0_.data(), dim*sizeof(int));
  idxK = new int[dim];
  memcpy(idxK, idxK_.data(), dim*sizeof(int));

  ub = new double[n];
  for(int i=0; i<n; i++) ub[i] = 0.;  
  DCOPY(&n, ub, &ione, lb, &ione);
}

AGCSimpleCons::~AGCSimpleCons()
{
  delete [] idx0;
  delete [] idxK;
  delete [] J_nz_idxs;
}

bool AGCSimpleCons::
eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  double* g = body+this->index;
  // p0 + alpha*deltak - pk  = 0
  int dim = n;
  for(int it=0; it<dim; it++) {
    g[it] += pg0->xref[idx0[it]] + deltaK->xref[0]*G_alpha[idx0[it]] - pgK->xref[idxK[it]];
  }
  return true;
}

bool AGCSimpleCons::
eval_Jac(const OptVariables& primal_vars, bool new_x, 
	 const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz, dim = n;
  if(NULL==M) {
    for(int it=0; it<dim; it++) {
      row = this->index+it;
      idxnz = J_nz_idxs[it];   

      assert(idxnz+3<nnz && idxnz>=0);

      ia[idxnz]=row; ja[idxnz]=pg0->index+idx0[it];   idxnz++; // w.r.t. po
      ia[idxnz]=row; ja[idxnz]=pgK->index+idxK[it];   idxnz++; // w.r.t. pk
      assert(idxnz<nnz);
      ia[idxnz]=row; ja[idxnz]=deltaK->index;        idxnz++; // w.r.t. delta
    }
    assert(row+1 == this->index+this->n);
  } else {
    for(int it=0; it<dim; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz+3<nnz && idxnz>=0);
      
      M[idxnz++] += 1.; // w.r.t. p0
      M[idxnz++] -= 1.; // w.r.t. pk
      M[idxnz++] += G_alpha[idx0[it]]; // w.r.t. delta
    }
  }
  return true;
}

int AGCSimpleCons::get_Jacob_nnz()
{
  return 3*n; 
}

bool AGCSimpleCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
  int vij_sz_in = vij.size();
#endif
  
  if(!J_nz_idxs) 
    J_nz_idxs = new int[n];

  //p0 + alpha*deltak - pk = 0
  int row=0; 
  for(int it=0; it<n; it++) {
    row = this->index+it;
    vij.push_back(OptSparseEntry(row, pg0->index+idx0[it], J_nz_idxs+it)); //p0
    vij.push_back(OptSparseEntry(row, pgK->index+idxK[it], NULL));         //pk
    vij.push_back(OptSparseEntry(row, deltaK->index, NULL)); //delta
  }

#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}



/////////////////////////////////////////////////////////////////////////////////
// AGC smoothing using complementarity function
// 
// p + alpha*deltak - pk - gb * rhop + gb * rhom = 0
// rhop, rhom >=0
// -r <= (pk-Pub)/gb * rhop <= r (or +inf)
// -r (or -inf) <= (pk-Plb)/gb * rhom <= r
//
// when r=0, the last two constraints are enforced as equalities ==0
// scaling parameter gb = generation band = Pub-Plb
//
// also, a penalty objective term can be added
// min M * [ rhop*(Pub-pk)/gb + rhom*(pk-Plb)/gb ];
/////////////////////////////////////////////////////////////////////////////////
AGCComplementarityCons::
AGCComplementarityCons(const std::string& id_, int numcons,
		       OptVariablesBlock* pg0_,
		       OptVariablesBlock* pgK_, OptVariablesBlock* deltaK_,
		       const std::vector<int>& idx0_, const std::vector<int>& idxK_,
		       const std::vector<double>& Plb_, const std::vector<double>& Pub_, 
		       const std::vector<double>& G_alpha_,
		       const double& r_,
		       bool add_penalty_obj/*=false*/, const double& bigM/*=0*/,
		       bool fix_p_g0_/*=false*/)
  : OptConstraintsBlock(id_, numcons), p0(pg0_), pk(pgK_), deltak(deltaK_), r(r_),
  J_nz_idxs(NULL), H_nz_idxs(NULL), fixed_p_g0(fix_p_g0_)
{
  assert(idx0_.size()==idxK_.size());
  assert(idx0_.size()==n/3);
  assert(n/3<=p0->n); assert(n/3<=pk->n);
  assert(1==deltak->n);
  assert(Plb_.size()==n/3);
  assert(Pub_.size()==n/3);
  assert(3*(n/3)==n);
  assert(G_alpha_.size()==pg0_->n);

  G_alpha = G_alpha_.data();

  int dim=n/3;
  idx0 = new int[dim];
  memcpy(idx0, idx0_.data(), dim*sizeof(int));
  idxk = new int[dim];
  memcpy(idxk, idxK_.data(), dim*sizeof(int));

  Plb = new double[dim];
  memcpy(Plb, Plb_.data(), dim*sizeof(double));
  Pub = new double[dim];
  memcpy(Pub, Pub_.data(), dim*sizeof(double));
  gb = new double[dim]; //gb = Pub-Plb
  DCOPY(&dim, Pub, &ione, gb, &ione);
  DAXPY(&dim, &dminusone, Plb, &ione, gb, &ione);

  //rhs of this constraints block
  assert(r>=0);
  ub = new double[n];
  for(int i=0; i<n/3; i++) ub[i] = 0.;
  for(int i=n/3; i<n; i++) ub[i] = r;
  
  //for(int i=n/3; i<n; ) { ub[i++] = 1e+20; ub[i++] = r;}
  //assert(r>0); 
  //for(int i=n/3; i<n; ) { ub[i++] = -r; ub[i++] = r;}
  
  lb = new double[n];
  if(r==0)
    DCOPY(&n, ub, &ione, lb, &ione);
  else {
    for(int i=0; i<n/3; i++) lb[i] = 0.;
    for(int i=n/3; i<n; i++) lb[i] = -r;

    //for(int i=n/3; i<n; ) { lb[i++] = -r; lb[i++] = -1e+20; }
    //for(int i=n/3; i<n; ) { lb[i++] = -r; lb[i++] = r; }
  }

  rhop = new OptVariablesBlock(n/3, string("rhop_")+id, 0, 1e+20);
  rhom = new OptVariablesBlock(n/3, string("rhom_")+id, 0, 1e+20);

  assert(bigM>=0);
  if(add_penalty_obj) {
    //add obj term
    assert(false && "not yet implemented");
  } else {
    assert(bigM==0);
  } 
}
AGCComplementarityCons::~AGCComplementarityCons()
{
  delete[] idx0; 
  delete[] idxk;
  delete[] Plb;
  delete[] Pub;
  delete[] gb;
  delete[] H_nz_idxs;
  delete[] J_nz_idxs;
} 

bool AGCComplementarityCons::
eval_body (const OptVariables& vars_primal, bool new_x, double* body)
{
  double* g = body+this->index;
  // p0 + alpha*deltak - pk - gb * rhop + gb * rhom = 0
  int dim = n/3;
  for(int it=0; it<dim; it++) {
    g[it] += p0->xref[idx0[it]] + deltak->xref[0]*G_alpha[idx0[it]] - pk->xref[idxk[it]];
  }
  for(int it=0; it<dim; it++) {
    g[it] += (rhom->xref[it] - rhop->xref[it])*gb[it];
  }
  int it=0;
  //body of the next 2n/3 constraints
  // -r <= (pk-Pub)/gb * rhop <= r  and  -r <= (pk-Plb)/gb * rhom <= r
  for(int conidx=dim; conidx<n; ) {
    assert(it<dim);
    g[conidx] += (pk->xref[idxk[it]]-Pub[it])/gb[it] * rhop->xref[it];
    conidx++;
    g[conidx] += (pk->xref[idxk[it]]-Plb[it])/gb[it] * rhom->xref[it];
    conidx++; it++;
  }

  return true;
}
void AGCComplementarityCons::compute_rhos(OptVariablesBlock* rp, OptVariablesBlock* rm)
{
  //compute from   p0 + alpha*deltak - pk - gb * rhop + gb * rhom = 0
  //use rhom->x as buffer
  double* g=rhom->x;
  int dim = n/3;
  assert(dim==rhom->n);assert(dim==rhop->n);
  for(int it=0; it<dim; it++) {
    assert(gb[it]>1e-8);
    //if(gb[it]<=1e-8) {
    //  printf(" it=%d  gb[it]=%15.8e\n", it, gb[it]);
    //}
    g[it] = ( p0->x[idx0[it]] + deltak->x[0]*G_alpha[idx0[it]] - pk->x[idxk[it]] ) / gb[it];
  }
  for(int it=0; it<dim; it++) {
    //carefull: g is rhom->x
    if(g[it] > 0) {
      rhop->x[it] = g[it]; rhom->x[it] = 0.;
    } else {
      rhop->x[it] = 0.; rhom->x[it] = - g[it]; // same as  rhom->x[it] = -rhom->x[it]
    }
  }
}


bool AGCComplementarityCons::
eval_Jac(const OptVariables& primal_vars, bool new_x, 
	 const int& nnz, int* ia, int* ja, double* M)
{
  int row=0, idxnz, dim=n/3;
  if(NULL==M) {
    assert(rhom->n == n/3);
    assert(rhop->n == n/3);
    for(int it=0; it<dim; it++) {
      row = this->index+it;
      idxnz = J_nz_idxs[it];   
   
      if(!fixed_p_g0) {
	assert(idxnz+4<nnz && idxnz>=0);
	ia[idxnz]=row; ja[idxnz]=p0->index+idx0[it];   idxnz++; // w.r.t. p0
      } else {
	assert(idxnz+3<nnz && idxnz>=0);
      }
      ia[idxnz]=row; ja[idxnz]=pk->index+idxk[it];   idxnz++; // w.r.t. pk
      ia[idxnz]=row; ja[idxnz]=deltak->index;        idxnz++; // w.r.t. delta
      ia[idxnz]=row; ja[idxnz]=rhop->index+it;       idxnz++; // w.r.t. rhop
      assert(idxnz<nnz);
      ia[idxnz]=row; ja[idxnz]=rhom->index+it;       idxnz++; // w.r.t. rhom
    }
    int idx;
    for(int it=0; it<dim; it++) {
      idx = dim+2*it;
      row = this->index + idx;
      idxnz = J_nz_idxs[idx];
      assert(idxnz<nnz && idxnz>=0);
      assert(idxnz+2<nnz);
      ia[idxnz]=row; ja[idxnz]=pk->index+idxk[it];   idxnz++; // w.r.t. pk
      ia[idxnz]=row; ja[idxnz]=rhop->index+it;       idxnz++; // w.r.t. rhop
      row++; idx++;
      
      idxnz = J_nz_idxs[idx]; assert(idxnz<nnz && idxnz>=0);  assert(idxnz+1<nnz); 
      ia[idxnz]=row; ja[idxnz]=pk->index+idxk[it];   idxnz++; // w.r.t. pk
      ia[idxnz]=row; ja[idxnz]=rhom->index+it;       idxnz++; // w.r.t. rhom
      row++;
    }
    assert(row == this->index+this->n);
  } else {
    if(!fixed_p_g0) {
      for(int it=0; it<dim; it++) {
	idxnz = J_nz_idxs[it];
	assert(idxnz<nnz && idxnz>=0);
	
	M[idxnz++] += 1.; // w.r.t. p0
	M[idxnz++] -= 1.; // w.r.t. pk
	M[idxnz++] += G_alpha[idx0[it]]; // w.r.t. delta
	M[idxnz++] -= gb[it]; // w.r.t. rhop
	M[idxnz++] += gb[it]; // w.r.t. rhom
      }
    } else { //fixed_p_g0 == true
      for(int it=0; it<dim; it++) {
	idxnz = J_nz_idxs[it];
	assert(idxnz<nnz && idxnz>=0);
	
	//M[idxnz++] += 1.; // w.r.t. p0
	M[idxnz++] -= 1.; // w.r.t. pk
	M[idxnz++] += G_alpha[idx0[it]]; // w.r.t. delta
	M[idxnz++] -= gb[it]; // w.r.t. rhop
	M[idxnz++] += gb[it]; // w.r.t. rhom
      }
    }
    int idx;
    for(int it=0; it<dim; it++) {
      idx = dim+2*it;
      idxnz = J_nz_idxs[idx];
      assert(idxnz+2<nnz);
      M[idxnz++] += rhop->xref[it]/gb[it];               // w.r.t. pk
      M[idxnz++] += (pk->xref[idxk[it]]-Pub[it])/gb[it]; // w.r.t. rhop
      idx++;
      
      idxnz = J_nz_idxs[idx]; assert(idxnz<nnz && idxnz>=0);  assert(idxnz+1<nnz); 
      M[idxnz++] += rhom->xref[it]/gb[it];               // w.r.t. pk
      M[idxnz++] += (pk->xref[idxk[it]]-Plb[it])/gb[it]; // w.r.t. rhom
    }
  }
  return true;
}

int AGCComplementarityCons::get_Jacob_nnz()
{
  if(!fixed_p_g0) {
    return 3*n; //  = n/3 * (5+2+2);
  } else {
    return (n/3)*8;
  }
}

bool AGCComplementarityCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
  int vij_sz_in = vij.size();
#endif
  
  if(!J_nz_idxs) 
    J_nz_idxs = new int[n];

  //p0 + alpha*deltak - pk - gb * rhop + gb * rhom = 0
  int row=0; 
  for(int it=0; it<n/3; it++) {
    row = this->index+it;
    if(!fixed_p_g0) {
      vij.push_back(OptSparseEntry(row, p0->index+idx0[it], J_nz_idxs+it)); //p0
      vij.push_back(OptSparseEntry(row, pk->index+idxk[it], NULL));         //pk
    } else {
      vij.push_back(OptSparseEntry(row, pk->index+idxk[it], J_nz_idxs+it)); //pk
    }
    vij.push_back(OptSparseEntry(row, deltak->index, NULL)); //delta
    vij.push_back(OptSparseEntry(row, rhop->index+it, NULL)); //rhop
    vij.push_back(OptSparseEntry(row, rhom->index+it, NULL)); //rhom
  }
  int idx=n/3;
  // -r <= (pk-Pub)/gb * rhop <= r and  -r <= (pk-Plb)/gb * rhom <= r
  for(int it=0; it<n/3; it++) {
    idx = n/3+2*it;
    row = this->index+idx;
    vij.push_back(OptSparseEntry(row, pk->index+idxk[it], J_nz_idxs+idx)); //pk
    vij.push_back(OptSparseEntry(row, rhop->index+it, NULL)); //rhop
    row++; idx++;
    vij.push_back(OptSparseEntry(row, pk->index+idxk[it], J_nz_idxs+idx)); //pk
    vij.push_back(OptSparseEntry(row, rhom->index+it, NULL)); //rhom
  }
  assert(idx==n-1);

#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}
// Jacobian is linear for the first n/3
// for the next 2n/3 has 2 elems per row
//   rhop/gb   (pk-Pub)/gb
//   rhom/gb   (pg-Plb)/gb
//
// Hessian has one elem per row (in the upper triangle)
//  1/gb w.r.t. (pk,rhop)
//  1/gb w.r.t. (pk,rhom)
int AGCComplementarityCons::get_HessLagr_nnz()
{
  return 2*(n/3);
}

bool AGCComplementarityCons::
eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
	      const OptVariables& lambda_vars, bool new_lambda,
	      const int& nnz, int* ia, int* ja, double* M)
{
  int *itnz=H_nz_idxs, dim=n/3;
  if(NULL==M) {
    int i,j, aux;
    for(int it=0; it<dim; it++) {
      i = pk->index+idxk[it]; j = rhop->index+it;
      i = uppertr_swap(i,j,aux);
      ia[*itnz]=i; ja[*itnz]=j; itnz++; //w.r.t. (pk,rhop)
      
      i = pk->index+idxk[it]; j = rhom->index+it;
      i = uppertr_swap(i,j,aux); 
      ia[*itnz]=i; ja[*itnz]=j; itnz++; //w.r.t. (pk,rhom)
    }
    assert(H_nz_idxs + 2*n/3 == itnz);
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL); assert(lambda->n==n);
    int it=dim;
    for(int i=0; i<dim; i++) {
      M[*itnz] += lambda->xref[it]/gb[i]; //w.r.t. (pk,rhop)
      itnz++; it++;
      M[*itnz] += lambda->xref[it]/gb[i]; //w.r.t. (pk,rhom)
      itnz++; it++;
    }
    assert(H_nz_idxs + 2*n/3 == itnz);
    assert(it == n);
  }

  return true;
}
bool AGCComplementarityCons::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)
{
  if(n==0) return true;
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[get_HessLagr_nnz()];
  }
  int i,j, dim=n/3, aux;
  for(int it=0; it<n/3; it++) {
    i = pk->index+idxk[it];
    j = rhop->index+it;
    i = uppertr_swap(i,j,aux);
    vij.push_back(OptSparseEntry(i,j,H_nz_idxs+2*it)); //w.r.t. (pk,rhop)

    i = pk->index+idxk[it];
    j = rhom->index+it;
    i = uppertr_swap(i,j,aux);
    vij.push_back(OptSparseEntry(i,j,H_nz_idxs+2*it+1)); //w.r.t. (pk,rhom)
  }
  return true;
}

OptVariablesBlock* AGCComplementarityCons::create_varsblock()
{
  return NULL; //the two slacks blocks are returned in 'create_multiple_varsblocks'
}

vector<OptVariablesBlock*> AGCComplementarityCons:: create_multiple_varsblocks()
{
  return std::vector<OptVariablesBlock*>({rhop,rhom});
}

OptObjectiveTerm* AGCComplementarityCons::create_objterm()
{
  return NULL;
}

///////////////////////////////////////////////////////////////////////////////////
// PVPQ smoothing using complementarity function
// 
// v[n] - vk[n] - nup[n]+num[n] = 0, for all n=idxs_bus
// nup, num >=0
// -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
// -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
//
// when r=0, the last two constraints are enforced as equalities == 0
// scaling parameter gb[n] = generation band at the bus = Qub[n]-Qlb[n]
//
// also, a big-M penalty objective term can be added to penalize 
// -(qk-Qub)/gb * nup and  (qk-Qlb)/gb * num
///////////////////////////////////////////////////////////////////////////////////

PVPQComplementarityCons::
PVPQComplementarityCons(const std::string& id_, int numcons,
			OptVariablesBlock* v0_, OptVariablesBlock* vK_, 
			OptVariablesBlock* qK_,
			const std::vector<int>& idxs_bus_,
			const std::vector<vector<int> >& idxs_gen_, 
			const std::vector<double>& Qlb_, const std::vector<double>& Qub_, 
			const double& r_,
			bool add_penalty_obj, const double& bigM)
  : OptConstraintsBlock(id_, numcons), v0(v0_), vk(vK_), qk(qK_), idxs_gen(idxs_gen_),
    r(r_), J_nz_idxs(NULL), H_nz_idxs(NULL)
{
  assert(v0->n == vk->n);
  assert(n/3 == idxs_gen.size());
  assert(n/3 == idxs_bus_.size());
  assert(Qlb_.size() == Qub_.size());
  assert(n/3 == Qlb_.size());
  assert(3*(n/3) == n);
  
  int nbus = n/3;
  idxs_bus = new int[nbus];
  memcpy(idxs_bus, idxs_bus_.data(), nbus*sizeof(int));

  Qlb = new double[nbus];
  memcpy(Qlb, Qlb_.data(), nbus*sizeof(double));
  Qub = new double[nbus];
  memcpy(Qub, Qub_.data(), nbus*sizeof(double));
  gb = new double[nbus]; //gb = Pub-Plb
  DCOPY(&nbus, Qub, &ione, gb, &ione);
  DAXPY(&nbus, &dminusone, Qlb, &ione, gb, &ione);

  //rhs of this constraints block
  assert(r>=0);
  ub = new double[n];
  for(int i=0; i<n/3; i++) ub[i] = 0.;
  for(int i=n/3; i<n; i++) ub[i] = r; 
  
  lb = new double[n];
  if(r==0)
    DCOPY(&n, ub, &ione, lb, &ione);
  else {
    DCOPY(&nbus, ub, &ione, lb, &ione);//for(int i=0; i<n/3; i++) lb[i] = 0.;
    for(int i=n/3; i<n; i++) lb[i] = -r; 
  }

  nup = new OptVariablesBlock(n/3, string("nup_")+id, 0, 1e+20);
  num = new OptVariablesBlock(n/3, string("num_")+id, 0, 1e+20);

  assert(bigM>=0);
  if(add_penalty_obj) {
    //add obj term
    assert(false && "not yet implemented");
  } else {
    assert(bigM==0.);
  } 
}
PVPQComplementarityCons::~PVPQComplementarityCons()
{
  delete[] idxs_bus;
  delete[] gb;
  delete[] Qlb;
  delete[] Qub;
  delete[] J_nz_idxs;
  delete[] H_nz_idxs;
}

void PVPQComplementarityCons::compute_nus(OptVariablesBlock* np, OptVariablesBlock* nm)
{
  assert(v0->n == vk->n);
  //compute from  v[n] - vk[n] - nup[n]+num[n] = 0
  //use rhom->x as buffer
  double* g=num->x;
  int dim = n/3;
  assert(dim==num->n);assert(dim==nup->n);
  for(int it=0; it<dim; it++) {
    g[it] = v0->x[idxs_bus[it]] - vk->x[idxs_bus[it]];
  }
  for(int it=0; it<dim; it++) {
    //carefull: g is num->x
    if(g[it] > 0) {
      nup->x[it] = g[it]; num->x[it] = 0.;
    } else {
      nup->x[it] = 0.; num->x[it] = - g[it]; // same as  rhom->x[it] = -rhom->x[it]
    }
  }
}

bool PVPQComplementarityCons::eval_body(const OptVariables& vars_primal, bool new_x, double* body)
{
  int nbus = n/3, it;
  double* g = body+this->index;

  assert(v0->n == vk->n);
  assert(vk->n >= nbus);

  // v[n] - vk[n] - nup[n]+num[n] = 0, for all n=idxs_bus
  for(it=0; it<nbus; it++) {
    assert(idxs_bus[it] < v0->n);
    assert(idxs_bus[it] >= 0);
    g[it] += v0->xref[idxs_bus[it]];
  }
  for(it=0; it<nbus; it++) {
    g[it] -= vk->xref[idxs_bus[it]];
  }
  DAXPY(&nbus, &dminusone, const_cast<double*>(nup->xref), &ione, g, &ione);
  DAXPY(&nbus, &done,      const_cast<double*>(num->xref), &ione, g, &ione);

  const int* idxs_g; int gi, ngen; double aux;
  //body of the next 2n/3 constraints
  // -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
  // -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
  it=0;
  for(int conidx=nbus; conidx<n; ) {
    idxs_g = idxs_gen[it].data(); ngen=idxs_gen[it].size();
    assert(ngen >= 1);
    aux=qk->xref[idxs_g[0]];
    for(gi=1; gi<ngen; gi++) 
      aux += qk->xref[idxs_g[gi]];

    assert(gb[it]>1e-8);    

    g[conidx] += (aux-Qub[it]) / gb[it] * nup->xref[it];
    conidx++;
    g[conidx] += (aux-Qlb[it]) / gb[it] * num->xref[it];
    conidx++; it++;
  }
  return true;
}

bool PVPQComplementarityCons::eval_Jac(const OptVariables& primal_vars, bool new_x, 
			  const int& nnz, int* ia, int* ja, double* M)
{
#ifdef DEBUG
  int loc_nnz = get_Jacob_nnz();
#endif
  int row=0, dim=n/3, idx_v, ngen, gi; const int* idxs_g;
  const int *pidxnz=J_nz_idxs;
  if(NULL==M) {
    assert(num->n == n/3);
    assert(nup->n == n/3);
    for(int it=0; it<dim; it++) {
      row = this->index+it;

      assert(pidxnz[0]<nnz && pidxnz[1]<nnz && pidxnz[2]<nnz && pidxnz[3]<nnz);
      assert(pidxnz[0]>=0   && pidxnz[1]>=0 && pidxnz[2]>=0  && pidxnz[3]>=0);
      // v[idxs_bus[n]] - vk[idxs_bus[n]] - nup[n]+num[n] = 0
      idx_v = idxs_bus[it];
      ia[*pidxnz]=row; ja[*pidxnz]=v0->index+idx_v;      pidxnz++; // w.r.t. v0
      ia[*pidxnz]=row; ja[*pidxnz]=vk->index+idx_v;      pidxnz++; // w.r.t. vk
      ia[*pidxnz]=row; ja[*pidxnz]=nup->index+it;        pidxnz++; // w.r.t. nup
      ia[*pidxnz]=row; ja[*pidxnz]=num->index+it;        pidxnz++; // w.r.t. num
    }
    assert(pidxnz == J_nz_idxs+4*(n/3));

    int idx;
    for(int it=0; it<dim; it++) {
      idx = dim+2*it;
      row = this->index + idx;

      ngen = idxs_gen[it].size(); 
      idxs_g = idxs_gen[it].data();
      assert(ngen>=1);

      assert(pidxnz[0]<nnz && pidxnz[1]<nnz);
      assert(pidxnz[0]>=0  && pidxnz[1]>=0);

      // -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
      ia[*pidxnz]=row; ja[*pidxnz]=nup->index+it;           pidxnz++; // w.r.t. nup
      ia[*pidxnz]=row; ja[*pidxnz]=qk->index+idxs_g[0];     pidxnz++; // w.r.t. qk's
      for(gi=1; gi<ngen; gi++) {
	assert(*pidxnz>=0 && *pidxnz<nnz);
	ia[*pidxnz]=row; ja[*pidxnz]=qk->index+idxs_g[gi];  pidxnz++; // w.r.t. qk's
      }
      row++; idx++;

      // -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
      assert(pidxnz[0]<nnz && pidxnz[1]<nnz);
      assert(pidxnz[0]>=0  && pidxnz[1]>=0);

      ia[*pidxnz]=row; ja[*pidxnz]=num->index+it;            pidxnz++; // w.r.t. num
      ia[*pidxnz]=row; ja[*pidxnz]=qk->index+idxs_g[0];      pidxnz++; // w.r.t. qk's
      for(gi=1; gi<ngen; gi++) {
	assert(*pidxnz>=0 && *pidxnz<nnz);
	ia[*pidxnz]=row; ja[*pidxnz]=qk->index+idxs_g[gi];   pidxnz++; // w.r.t. qk's
      }
      row++;
    }
#ifdef DEBUG
    assert(row == this->index+this->n);
    assert(pidxnz == J_nz_idxs+loc_nnz);
#endif
  } else {

    for(int it=0; it<dim; it++) {
#ifdef DEBUG
      assert(pidxnz[0]<nnz && pidxnz[1]<nnz && pidxnz[2]<nnz && pidxnz[3]<nnz);
      assert(pidxnz[0]>=0   && pidxnz[1]>=0 && pidxnz[2]>=0  && pidxnz[3]>=0);
#endif
      M[*pidxnz++] += 1.; // w.r.t. v0
      M[*pidxnz++] -= 1.; // w.r.t. vk
      M[*pidxnz++] -= 1.; // w.r.t. nup
      M[*pidxnz++] += 1.; // w.r.t. num
    }
    assert(pidxnz == J_nz_idxs+4*(n/3));
    int idx; double qsum, aux1;
    for(int it=0; it<dim; it++) {
      idx = dim+2*it;

      ngen = idxs_gen[it].size(); 
      idxs_g = idxs_gen[it].data();
      assert(ngen>=1);
      
      
      qsum = qk->xref[idxs_g[0]];
      for(gi=1; gi<ngen; gi++) 
	qsum += qk->xref[idxs_g[gi]];

      // -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
      aux1 = nup->xref[it]/gb[it];
      M[*pidxnz++] += (qsum-Qub[it])/gb[it]; // w.r.t. nup
      for(gi=0; gi<ngen; gi++)
	M[*pidxnz++] += aux1;                // w.r.t. qk's
      idx++;
      
      // -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
      aux1 = num->xref[it]/gb[it];
      M[*pidxnz++] += (qsum-Qlb[it])/gb[it]; // w.r.t. num
      for(gi=0; gi<ngen; gi++)
	M[*pidxnz++] += aux1;                // w.r.t. qk's
    }
#ifdef DEBUG
    assert(pidxnz == J_nz_idxs+loc_nnz);
#endif
  }

  return true;
}

int PVPQComplementarityCons::get_Jacob_nnz()
{
  //for the first n/3 constraints we have 4*n/3 nonzeros
  int nnz = 4*(n/3);
  for(int ni=0; ni<n/3; ni++)
    nnz += 2*(1+idxs_gen[ni].size()); //1 for nup/num[ni] and idxs_gen[ni].size() for sum(qk[g])
  return nnz;
}

bool PVPQComplementarityCons::get_Jacob_ij(std::vector<OptSparseEntry>& vij)
{
  int loc_nnz = get_Jacob_nnz();
#ifdef DEBUG
  int vij_sz_in = vij.size();
#endif
  
  if(!J_nz_idxs) 
    J_nz_idxs = new int[loc_nnz];
  int *itnz=J_nz_idxs;

  // v[idxs_bus[n]] - vk[idxs_bus[n]] - nup[n]+num[n] = 0
  int row=0; 
  for(int it=0; it<n/3; it++) {
    row = this->index+it;
    vij.push_back(OptSparseEntry(row, v0->index+idxs_bus[it], itnz++)); //v0
    vij.push_back(OptSparseEntry(row, vk->index+idxs_bus[it], itnz++)); //vk
    vij.push_back(OptSparseEntry(row, nup->index+it, itnz++));          //nup
    vij.push_back(OptSparseEntry(row, num->index+it, itnz++));          //num
  }
  assert(J_nz_idxs+4*(n/3)==itnz);
  int idx=n/3; ;
  for(int it=0; it<n/3; it++) {
    idx = n/3+2*it;
    row = this->index+idx;
    // -r <= ( sum(qk[g])-Qub[n] ) / gb[n] * nup[n] <= r
    vij.push_back(OptSparseEntry(row, nup->index+it, itnz++)); //nup
    for(auto gi: idxs_gen[it]) {
      assert(gi >= 0);
      assert(gi < qk->n);
      vij.push_back(OptSparseEntry(row, qk->index+gi, itnz++)); //qk
    }
    row++; idx++;

    // -r <= ( sum(qk[g])-Qlb[n] ) / gb[n] * num[n] <= r
    vij.push_back(OptSparseEntry(row, num->index+it, itnz++));  //num
    for(auto gi: idxs_gen[it]) {
      vij.push_back(OptSparseEntry(row, qk->index+gi, itnz++)); //qk
    }
  }
  assert(idx==n-1);
  assert(J_nz_idxs+loc_nnz==itnz);
#ifdef DEBUG
  assert(row+1 == this->index+this->n);
  assert(vij.size() == loc_nnz+vij_sz_in);
#endif
  return true;
}

bool PVPQComplementarityCons::eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
			       const OptVariables& lambda_vars, bool new_lambda,
			       const int& nnz, int* ia, int* ja, double* M)
{
#ifdef DEBUG
  int nnz_loc = get_HessLagr_nnz();
#endif
  int *itnz=H_nz_idxs, dim=n/3;
  if(NULL==M) {
    int i,j, aux;
    for(int it=0; it<dim; it++) {
      aux = nup->index+it;
      for(auto gi : idxs_gen[it]) {
	i = qk->index+gi; 
	j = aux;
	i = uppertr_swap(i,j,aux);
	ia[*itnz]=i; ja[*itnz]=j; itnz++; //w.r.t. (qk's,nup)
      }
      aux = num->index+it;
      for(auto gi : idxs_gen[it]) {
	i = qk->index+gi; 
	j = aux;
	i = uppertr_swap(i,j,aux); 
	ia[*itnz]=i; ja[*itnz]=j; itnz++; //w.r.t. (qk's,num)
      }
    }
  } else {
    const OptVariablesBlock* lambda = lambda_vars.get_block(std::string("duals_") + this->id);
    assert(lambda!=NULL); assert(lambda->n==n);
    const int* idxs_g; int ngen;
    int it=dim;
    for(int i=0; i<dim; i++) {
      idxs_g = idxs_gen[i].data(); ngen = idxs_gen[i].size();
      assert(ngen>=1);

      // 1/gb -> w.r.t. (q1,nup), (q2,nup), ..., (qs,nup)
      M[*itnz] += lambda->xref[it]/gb[i]; itnz++;   //w.r.t. (qk'q,nup)
      for(int gi=1; gi<ngen; gi++) {
	M[*itnz] += lambda->xref[it]/gb[i]; itnz++; //w.r.t. (qk'q,nup)
      }
      it++;

      // /gb -> w.r.t. (q1,num), (q2,num), ..., (qs,num)
      M[*itnz] += lambda->xref[it]/gb[i]; itnz++;   //w.r.t. (pk's,num)
      for(int gi=1; gi<ngen; gi++) {
	M[*itnz] += lambda->xref[it]/gb[i]; itnz++; //w.r.t. (pk's,num)
      }
      it++;
    }
    assert(it == n);
  }
#ifdef DEBUG
  assert(H_nz_idxs + nnz_loc == itnz);
#endif
  return true;
}

// Hessian is zero for the first n/3
// for each bilinear constraint of the following 2*n/3, there are s=idx_gen[n].size() 
// non-zeros, all off-diagonal
//                q1      q2        qs       num                   
// Jacobian1:  nup/gb  nup/gb ... nup/gb  (sum(q)-lb)/gb
// Jacobian2:  num/gb  num/gb ... num/gb  (sum(q)-lb)/gb

// Hessian: 
//    1/gb -> w.r.t. (q1,nup), (q2,nup), ..., (qs,nup)
//    1/gb -> w.r.t. (q1,num), (q2,num), ..., (qs,num)
int PVPQComplementarityCons::get_HessLagr_nnz()
{
  int nnz=0;
  for(int ni=0; ni<n/3; ni++)
    nnz += 2*idxs_gen[ni].size(); 
  return nnz;
}

bool PVPQComplementarityCons::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)
{
  if(n==0) return true;
  int nnz=get_HessLagr_nnz();
  if(NULL==H_nz_idxs) {
    H_nz_idxs = new int[nnz];
  }
  int i,j, dim=n/3, aux, itnz=0;
  for(int it=0; it<n/3; it++) {
    for(auto gi : idxs_gen[it]) {
      j = nup->index+it;
      i = qk->index+gi;
      i = uppertr_swap(i,j,aux);
      vij.push_back(OptSparseEntry(i,j,H_nz_idxs+itnz)); //w.r.t. (qk's,nup)
      itnz++;
    }
    for(auto gi : idxs_gen[it]) {
      j = num->index+it;
      i = qk->index+gi;
      i = uppertr_swap(i,j,aux);
      vij.push_back(OptSparseEntry(i,j,H_nz_idxs+itnz)); //w.r.t. (qk's,nup)
      itnz++;
    }
  }
  assert(itnz==nnz);
  return true;
}
    
OptVariablesBlock* PVPQComplementarityCons::create_varsblock()
{
  return NULL; //the two slacks blocks are returned in 'create_multiple_varsblocks'
}

std::vector<OptVariablesBlock*> PVPQComplementarityCons::create_multiple_varsblocks()
{
  return std::vector<OptVariablesBlock*>({nup,num});
}

OptObjectiveTerm* PVPQComplementarityCons::create_objterm()
{
  return NULL;
}

}//end namespace 

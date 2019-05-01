#include "CouplingConstraints.hpp"


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
// AGC smoothing using complementarity function
// 
// p + alpha*deltak - pk - gb * rhop + gb * rhom = 0
// rhop, rhom >=0
// -r <= (pk-Pub)/gb * rhop <= r
// -r <= (pk-Plb)/gb * rhom <= r
//
// when r=0, the last two constraints are enforced as equalities ==0
// scaling parameter gb = generation band = Pub-Plb
//
// also, a penalty objective term can be added
// min M * [ rhop*(Pub-pk)/gb + rhom*(pk-Plb)/gb ];
/////////////////////////////////////////////////////////////////////////////////
AGCComplementarityCons::
AGCComplementarityCons(const std::string& id_, int numcons,
		       OptVariablesBlock* pg0_, OptVariablesBlock* pgK_, OptVariablesBlock* deltaK_,
		       const std::vector<int>& idx0_, const std::vector<int>& idxK_,
		       const std::vector<double>& Plb_, const std::vector<double>& Pub_, 
		       const std::vector<double>& G_alpha_,
		       const double& r_,
		       bool add_penalty_obj/*=false*/, const double& bigM/*=0*/)
  : OptConstraintsBlock(id_, numcons), p0(pg0_), pk(pgK_), deltak(deltaK_), r(r_),
    J_nz_idxs(NULL), H_nz_idxs(NULL)
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

  idx0 = new int[n];
  memcpy(idx0, idx0_.data(), n*sizeof(int));
  idxk = new int[n];
  memcpy(idxk, idxK_.data(), n*sizeof(int));

  Plb = new double[n];
  memcpy(Plb, Plb_.data(), n*sizeof(double));
  Pub = new double[n];
  memcpy(Pub, Pub_.data(), n*sizeof(double));
  gb = new double[n]; //gb = Pub-Plb
  DCOPY(&n, Pub, &ione, gb, &ione);
  DAXPY(&n, &dminusone, Plb, &ione, gb, &ione);

  //rhs of this constraints block
  assert(r>=0);
  ub = new double[n];
  for(int i=0; i<n/3; i++) ub[i] = 0.;
  for(int i=n/3; i<n; i++) ub[i] = r; 
  
  lb = new double[n];
  if(r==0)
    DCOPY(&n, ub, &ione, lb, &ione);
  else {
    for(int i=0; i<n/3; i++) lb[i] = 0.;
    for(int i=n/3; i<n; i++) lb[i] = -r; 
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

bool AGCComplementarityCons::eval_body (const OptVariables& vars_primal, bool new_x, double* body)
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
      assert(idxnz<nnz && idxnz>=0);
      
      ia[idxnz]=row; ja[idxnz]=p0->index+idx0[it];   idxnz++; // w.r.t. p0
      ia[idxnz]=row; ja[idxnz]=pk->index+idxk[it];   idxnz++; // w.r.t. pk
      ia[idxnz]=row; ja[idxnz]=deltak->index;        idxnz++; // w.r.t. delta
      ia[idxnz]=row; ja[idxnz]=rhop->index+it;       idxnz++; // w.r.t. rhop
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
#ifdef DEBUG
      row++;
#endif
    }
    assert(row == this->index+this->n);
  } else {

    for(int it=0; it<dim; it++) {
      idxnz = J_nz_idxs[it];
      assert(idxnz<nnz && idxnz>=0);
      
      M[idxnz++] += 1.; // w.r.t. p0
      M[idxnz++] -= 1.; // w.r.t. pk
      M[idxnz++] += G_alpha[idx0[it]]; // w.r.t. delta
      M[idxnz++] -= gb[it]; // w.r.t. rhop
      M[idxnz++] += gb[it]; // w.r.t. rhom
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
  return 3*n; //  = n/3 * (5+2+2);
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
    vij.push_back(OptSparseEntry(row, p0->index+idx0[it], J_nz_idxs+it)); //p0
    vij.push_back(OptSparseEntry(row, pk->index+idxk[it], NULL)); //pk
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
    for(int it=0; it<n/3; it++) {
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
    for(int i=0; i<n/3; i++) {
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

}//end namespace 

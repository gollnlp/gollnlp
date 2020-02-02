#ifndef MATRIX_SSTRIP_COMPLEX
#define MATRIX_SSTRIP_COMPLEX

#include <vector>
#include <complex>

#include <numeric>
#include <algorithm>

//container for sparse matrices in triplet format; implements minimal functionality for matrix ops
template <class Tidx, class Tval>
class MatrixSparseTripletStorage
{
public:
  MatrixSparseTripletStorage()
    : nrows_(0), ncols_(0), nnz_(0), irow_(NULL), jcol_(NULL), values_(NULL)
  {
    
  }
  MatrixSparseTripletStorage(Tidx num_rows, Tidx num_cols, Tidx num_nz)
    : nrows_(num_rows), ncols_(num_cols), nnz_(num_nz)
  {
    irow_ = new Tidx[nnz_];
    jcol_ = new Tidx[nnz_];
    values_ = new Tval[nnz_];
  }

  virtual ~MatrixSparseTripletStorage()
  {
    if(values_) delete[] values_;
    if(jcol_) delete[] jcol_;
    if(irow_) delete[] irow_;
  }

  //sorts the (i,j) in increasing order of 'i' and for equal 'i's in increasing order of 'j'
  //Complexity: n*log(n)
  //
  // Warning: irow_, jcol_, and values_ pointers will changes inside this method. Corresponding
  // accessor methods i(), j(), M() should be called again to get the correct pointers
  void sort_indexes() {
    std::vector<Tidx> vIdx(nnz);
    std::iota(vIdx.begin(), vIdx.end(), 0);
    sort(vIdx.begin(), vIdx.end(), 
	 [&](const int& i1, const int& i2) { 
	   if(irow_[i1]<irow_[i2]) return true;
	   if(irow_[i1]>irow_[i2]) return false;
	   return jcol_[i1]<jcol_[i2];
	 });

    //permute irow, jcol, and M using additional storage

    //irow and jcol can use the same  buffer 
    {
      Tidx* buffer = new Tidx[nnz];
      for(int itnz=0; itnz<nnz; itnz++)
	buffer[itnz] = irow_[vIdx[itnz]];
      
      //avoid copy back
      Tidx* buffer2 = irow_;
      irow_ = buffer;
      buffer = buffer2; 
      
      for(int itnz=0; itnz<nnz; itnz++)
	buffer[itnz] = jcol_[vIdx[itnz]];
      
      delete[] jcol_;
      jcol_ = buffer;
    }

    //M
    {
      Tval* buffer = new Tval[nnz];
      
      for(int itnz=0; itnz<nnz; itnz++)
	buffer[itnz] = values_[vIdx[itnz]];

      delete[] values_;
      values_ = buffer;
    }
  }

  //add elements with identical (i,j) and update nnz_, irow_, jcol_, and values_ array accordingly
  // Precondition: (irow_,jcol_) are assumed to be sorted (see sort_indexes())
  void sum_up_duplicates()
  {
    if(nnz_<=0) return;
    Tidx itleft=0, itright=1;
    Tidx currI=irow_[0], currJ=jcol_[0];
    Tval val1 = values_[0];
    
    while(itright<nnz) {
      values_[itleft] = val1;
      
      while(itright<nnz && irow_[itright]==currI && jcol_[itright]==currJ) {
	values_[itleft] += values_[itright];
	itright++;
      }
      irow_[itleft] = currI;
      jcol_[itleft] = currJ;

      if(itright<nnz) {
	currI = irow_[itright];
	currJ = jcol_[itright];
	val1 = values_[itright];
	itright++;
      }
      itleft++;
      assert(itleft<=nnz);
      assert(itleft<itright);
    }
    
    nnz_ = itleft;
  }
  
  Tidx n() const { return nrows_; }
  Tidx m() const { return ncols_; }
  Tidx nnz() const { return nnz_; }
  Tidx* i() const { return irow_; }
  Tidx* j() const { return jcol_; }
  Tval* M() const { return values_; }
protected:
  Tidx nrows_, ncols_, nnz_;
  
  Tidx *irow_, *jcol_;
  Tval* values_; 
};

#endif

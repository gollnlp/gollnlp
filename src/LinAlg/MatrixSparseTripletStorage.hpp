#ifndef MATRIX_SSTRIP_COMPLEX
#define MATRIX_SSTRIP_COMPLEX

#include <vector>
#include <complex>

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
  void sort_indexes() {
    std::vector<Tidx> vIdx(nnz);
    std::iota(vIdx.begin(), vIdx.end(), 0);
    sort(vIdx.begin(), vIdx.end(), 
	 [&](const int& i1, const int& i2) { 
	   if(irow_[i1]<irow_[i2]) return true;
	   if(irow_[i1]>irow_[i2]) return false;
	   return jcol_[i1]<jcol_[i2];
	 });

    //for(int itnz=0; itnz
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

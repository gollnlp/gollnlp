#ifndef MATRIX_SSTRIP_COMPLEX
#define MATRIX_SSTRIP_COMPLEX

#include <complex>

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

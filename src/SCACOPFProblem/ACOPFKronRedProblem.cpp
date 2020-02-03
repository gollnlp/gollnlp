#include "ACOPFKronRedProblem.hpp"

#include "MatrixSparseTripletStorage.hpp"

namespace gollnlp {

  ACOPFKronRedProblem::~ACOPFKronRedProblem() {}
  
  /* initialization method: performs Kron reduction and builds the OptProblem */
  bool ACOPFKronRedProblem::assemble()
  {
    
    YBus_ = construct_YBus_matrix();

    

    return true;
  }
  
  void ACOPFKronRedProblem::add_variables()
  {
  }
    
  void ACOPFKronRedProblem::add_cons_pf()
  {
  }
    
  void ACOPFKronRedProblem::add_obj_prod_cost()
  {
  }

  MatrixSpTComplex* ACOPFKronRedProblem::construct_YBus_matrix()
  {
    //
    // details at 
    //  https://gitlab.pnnl.gov/exasgd/frameworks/hiop-framework/blob/master/modules/DenseACOPF.jl
    //

    const int& N = data_sc_.N_Bus.size();

    int nnz=N;
    // go over (L_Nidx1, L_Nidx2) and increase nnz when idx1>idx2
    assert(data_sc_.L_Nidx[0].size() == data_sc_.L_Nidx[1].size());
    for(int it=0; it<data_sc_.L_Nidx[0].size(); it++) {
      if(data_sc_.L_Nidx[0][it]>data_sc_.L_Nidx[1][it]) 
	nnz++;
    }
    //same for transformers
    assert(data_sc_.T_Nidx[0].size() == data_sc_.T_Nidx[1].size());
    for(int it=0; it<data_sc_.T_Nidx[0].size(); it++) {
      if(data_sc_.T_Nidx[0][it]>data_sc_.T_Nidx[1][it]) 
	nnz++;
    }


    //alocate Matrix
    auto Ybus = new MatrixSparseTripletStorage<int, std::complex<double> >(N, N, nnz);
    int *Ii=Ybus->i_row(), *Ji=Ybus->j_col(); dcomplex *M=Ybus->M();

    for(int busidx=0; busidx<N; busidx++) {
      Ii[busidx] = Ji[busidx] = busidx;

      // shunt contribution to Ybus
      M[busidx] = dcomplex(data_sc_.N_Gsh[busidx], data_sc_.N_Bsh[busidx]);
    }
    int nnz_count = N;
    
    //
    // go over (L_Nidx1, L_Nidx2) and populate the matrix
    //
    for(int l=0; l<data_sc_.L_Nidx[0].size(); l++) {
      const int& Nidxfrom=data_sc_.L_Nidx[0][l], Nidxto=data_sc_.L_Nidx[1][l];

      dcomplex ye(data_sc_.L_G[l], data_sc_.L_B[l]);
      {
	//yCHe = L[:Bch][l]*im;
	dcomplex res(0.0, data_sc_.L_Bch[l]/2); //this is yCHe/2
	res += ye; 

	//Ybus(Nidxfrom,Nidxfrom) = ye + yCHe/2
	M[Nidxfrom] += res;
	//Ybus(Nidxto,Nidxto) = ye + yCHe/2
	M[Nidxto] += res;

      }
      //M[i,j] -= ye
      if(Nidxfrom>Nidxto) {
	Ii[nnz_count] = Nidxfrom;
	Ji[nnz_count] = Nidxto;
	M [nnz_count] = -ye;
	nnz_count++;
      }
    }
    //
    //same as above but for (T_Nidx1, T_Nidx2)
    //
    for(int t=0; t<data_sc_.T_Nidx[0].size(); t++) {
      const int& Nidxfrom=data_sc_.T_Nidx[0][t], Nidxto=data_sc_.L_Nidx[1][t];
      dcomplex yf(data_sc_.T_G[t], data_sc_.T_B[t]);
      dcomplex yMf(data_sc_.T_Gm[t], data_sc_.T_Bm[t]);
      const double& tauf = data_sc_.T_Tau[t];
      
      M[Nidxfrom] += yf/(tauf*tauf) + yMf;
      M[Nidxto]   += yf;

      if(Nidxfrom>Nidxto) {
	Ii[nnz_count] = Nidxfrom;
	Ji[nnz_count] = Nidxto;
	M[nnz_count] = -yf/tauf;
	nnz_count++;
      }
    }

    assert(nnz_count==nnz);
    return Ybus;
  }
} //end namespace
    



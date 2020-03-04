#include "ACOPFKronRedProblem.hpp"

#include "goUtils.hpp"

using namespace hiop;
using namespace std;

namespace gollnlp {

  ACOPFKronRedProblem::~ACOPFKronRedProblem() {}
  
  /* initialization method: performs Kron reduction and builds the OptProblem */
  bool ACOPFKronRedProblem::assemble()
  {   
    hiopMatrixComplexSparseTriplet* YBus_ = construct_YBus_matrix();
    //YBus_->print();


    std::vector<int> idxs_nonaux, idxs_aux;
    construct_buses_idxs(idxs_nonaux, idxs_aux);


    hiopMatrixComplexDense Ybus_red(idxs_nonaux.size(),idxs_nonaux.size());

    hiopKronReduction reduction;
    if(!reduction.go(idxs_nonaux, idxs_aux, *YBus_, Ybus_red)) {
      return false;
    }
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

  void ACOPFKronRedProblem::construct_buses_idxs(std::vector<int>& idxs_nonaux, std::vector<int>& idxs_aux)
  {
    const double SMALL=1e-8;

    idxs_nonaux.clear(); idxs_aux.clear();

    for(int n=0; n<data_sc_.N_Pd.size(); n++) {
      if(data_sc_.Gn[n].size()>0 || data_sc_.SShn[n].size()>0 || 
	 magnitude(data_sc_.N_Pd[n], data_sc_.N_Qd[n])>SMALL) {

	idxs_nonaux.push_back(n);
      } else {
	idxs_aux.push_back(n);
      }
    }

    assert(data_sc_.Gn.size() == idxs_nonaux.size()+idxs_aux.size());
  }

  hiopMatrixComplexSparseTriplet* ACOPFKronRedProblem::construct_YBus_matrix()
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
      if(data_sc_.L_Nidx[0][it]!=data_sc_.L_Nidx[1][it]) 
	nnz++;
    }
    //same for transformers
    assert(data_sc_.T_Nidx[0].size() == data_sc_.T_Nidx[1].size());
    for(int it=0; it<data_sc_.T_Nidx[0].size(); it++) {
      if(data_sc_.T_Nidx[0][it]!=data_sc_.T_Nidx[1][it]) 
	nnz++;
    }

    //alocate Matrix
    auto Ybus = new hiopMatrixComplexSparseTriplet(N, N, nnz);
    int *Ii=Ybus->storage()->i_row(), *Ji=Ybus->storage()->j_col(); 
    complex<double> *M=Ybus->storage()->M();

    for(int busidx=0; busidx<N; busidx++) {
      Ii[busidx] = Ji[busidx] = busidx;

      // shunt contribution to Ybus
      M[busidx] = complex<double>(data_sc_.N_Gsh[busidx], data_sc_.N_Bsh[busidx]);
    }

    int nnz_count = N;
    //
    // go over (L_Nidx1, L_Nidx2) and populate the matrix
    //
    for(int l=0; l<data_sc_.L_Nidx[0].size(); l++) {
      const int& Nidxfrom=data_sc_.L_Nidx[0][l], Nidxto=data_sc_.L_Nidx[1][l];

      complex<double> ye(data_sc_.L_G[l], data_sc_.L_B[l]);
      {
	//yCHe = L[:Bch][l]*im;
	complex<double> res(0.0, data_sc_.L_Bch[l]/2); //this is yCHe/2
	res += ye; 

	//Ybus(Nidxfrom,Nidxfrom) = ye + yCHe/2
	M[Nidxfrom] += res;
	//Ybus(Nidxto,Nidxto) = ye + yCHe/2
	M[Nidxto] += res;

      }
      //M[i,j] -= ye
      assert(Nidxfrom!=Nidxto);
      //if(Nidxfrom>Nidxto) 
      {
	Ji[nnz_count] = std::max(Nidxfrom, Nidxto);
	Ii[nnz_count] = std::min(Nidxfrom, Nidxto);
	M [nnz_count] = -ye;
	nnz_count++;
      }
    }
    //
    //same as above but for (T_Nidx1, T_Nidx2)
    //
    for(int t=0; t<data_sc_.T_Nidx[0].size(); t++) {
      const int& Nidxfrom=data_sc_.T_Nidx[0][t], Nidxto=data_sc_.T_Nidx[1][t];
      complex<double> yf(data_sc_.T_G[t], data_sc_.T_B[t]);
      complex<double> yMf(data_sc_.T_Gm[t], data_sc_.T_Bm[t]);
      const double& tauf = data_sc_.T_Tau[t];
      
      M[Nidxfrom] += yf/(tauf*tauf) + yMf;
      M[Nidxto]   += yf;

      assert(Nidxfrom!=Nidxto);
      //if(Nidxfrom>Nidxto) 
      {
	Ji[nnz_count] = std::max(Nidxfrom, Nidxto);
	Ii[nnz_count] = std::min(Nidxfrom, Nidxto);
	M[nnz_count] = -yf/tauf;
	nnz_count++;
      }
    }
    assert(nnz_count==nnz);
    Ybus->storage()->sort_indexes();
    Ybus->storage()->sum_up_duplicates();

    return Ybus;
  }
} //end namespace
    



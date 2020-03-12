#include "ACOPFKronRedProblem.hpp"

#include "OPFObjectiveTerms.hpp"
#include "OPFConstraints.hpp"
#include "OPFConstraintsKron.hpp"

#include "SCACOPFUtils.hpp"
#include "goUtils.hpp"

using namespace hiop;
using namespace std;

namespace gollnlp {

  ACOPFKronRedProblem::~ACOPFKronRedProblem() 
  {
    delete Ybus_red;
  }
  
  /* initialization method: performs Kron reduction and builds the OptProblem */
  bool ACOPFKronRedProblem::assemble()
  {   
    hiopMatrixComplexSparseTriplet* YBus_ = construct_YBus_matrix();
    //YBus_->print();
    construct_buses_idxs(idxs_buses_nonaux, idxs_buses_aux);

    if(Ybus_red) delete Ybus_red;

    Ybus_red = new hiopMatrixComplexDense(idxs_buses_nonaux.size(),idxs_buses_aux.size());

    hiopKronReduction reduction;
    if(!reduction.go(idxs_buses_nonaux, idxs_buses_aux, *YBus_, *Ybus_red)) {
      return false;
    }

    add_variables(data_sc);
    add_cons_pf(data_sc);

    return true;
  }
  
  void ACOPFKronRedProblem::add_variables(SCACOPFData& d, bool SysCond_BaseCase/* = true*/)
  {
    { //voltages
      vector<double>& vlb = SysCond_BaseCase==true ?  data_sc.N_Vlb : data_sc.N_EVlb;
      vector<double>& vub = SysCond_BaseCase==true ?  data_sc.N_Vub : data_sc.N_EVub;
      
      vector<double> vlb_na, vub_na, v0_na;
      selectfrom(vlb, idxs_buses_nonaux, vlb_na);
      selectfrom(vub, idxs_buses_nonaux, vub_na);
      selectfrom(data_sc.N_v0, idxs_buses_nonaux, v0_na);
      
      auto v_n = new OptVariablesBlock(idxs_buses_nonaux.size(), var_name("v_n",d), vlb_na.data(), vub_na.data());
      append_variables(v_n);
      v_n->set_start_to(v0_na.data());
    }

    { //theta
      auto theta_n = new OptVariablesBlock(idxs_buses_nonaux.size(), var_name("v_n",d));
      append_variables(theta_n);
      
      vector<double> theta0_n;
      selectfrom(data_sc.N_theta0, idxs_buses_nonaux, theta0_n);
      theta_n->set_start_to(theta0_n.data());

      int RefBusIdx = data_sc.bus_with_largest_gen(), RefBusIdx_nonaux;
      auto it = std::find(idxs_buses_nonaux.begin(),  idxs_buses_nonaux.end(), RefBusIdx);
      if(it==idxs_buses_nonaux.end()) {
	assert(false && "check this");
	RefBusIdx_nonaux=0;
      } else {
	RefBusIdx_nonaux = std::distance(idxs_buses_nonaux.begin(), it);
      }

      const double& theta0_ref = data_sc.N_theta0[RefBusIdx];
      if(theta0_ref!=0.) {
	//check indexing again
	assert(theta_n->x[RefBusIdx_nonaux] == theta0_ref);

	for(int b=0; b<theta_n->n; b++) {
	  theta_n->x[b] -= theta0_ref;
	  assert( theta_n->x[b] >= theta_n->lb[b]);
	  assert( theta_n->x[b] <= theta_n->ub[b]);
	}
      }
      theta_n->lb[RefBusIdx_nonaux] = theta_n->ub[RefBusIdx_nonaux] = 0.;
      assert(theta_n->x[RefBusIdx_nonaux]==0.);
    }

    { //b_s
      vector<double> lb, ub, x0;
      selectfrom(data_sc.SSh_Blb, idxs_buses_nonaux, lb);
      selectfrom(data_sc.SSh_Bub, idxs_buses_nonaux, ub);

      auto b_s = new OptVariablesBlock(idxs_buses_nonaux.size(), var_name("b_s",d), lb.data(), ub.data());

      selectfrom(data_sc.SSh_B0, idxs_buses_nonaux, x0);
      b_s->set_start_to(x0.data());

      append_variables(b_s);
    }

    { // generation p_g and q_g
      auto p_g = new OptVariablesBlock(d.G_Generator.size(), var_name("p_g",d), 
				       d.G_Plb.data(), d.G_Pub.data());
      append_variables(p_g); 
      p_g->set_start_to(d.G_p0.data());


      auto q_g = new OptVariablesBlock(d.G_Generator.size(), var_name("q_g",d), 
				       d.G_Qlb.data(), d.G_Qub.data());
      q_g->set_start_to(d.G_q0.data());
      append_variables(q_g); 
    }
  }
    
  void ACOPFKronRedProblem::add_cons_pf(SCACOPFData& d)
  {
    auto p_g = vars_block(var_name("p_g",d)), 
      v_n = vars_block(var_name("v_n",d)), 
      theta_n = vars_block(var_name("theta_n",d));
    {
      //active power balance
      auto pf_p_bal = new PFActiveBalanceKron(con_name("p_balance_kron",d), 
					      idxs_buses_nonaux.size(),
					      p_g, v_n, theta_n,
					      idxs_buses_nonaux,
					      d.Gn, *Ybus_red, data_sc.N_Pd);
      append_constraints(pf_p_bal);
    }
  }
    
  void ACOPFKronRedProblem::add_obj_prod_cost(SCACOPFData& d)
  {
    vector<int> gens(d.G_Generator.size()); iota(gens.begin(), gens.end(), 0);
    auto p_g = vars_block(var_name("p_g", d));
    PFProdCostAffineCons* prod_cost_cons = 
      new PFProdCostAffineCons(con_name("prodcost_cons",d), 2*gens.size(), 
			       p_g, gens, d.G_CostCi, d.G_CostPi);
    append_constraints(prod_cost_cons);
    
    OptVariablesBlock* t_h = prod_cost_cons->get_t_h();
    prod_cost_cons->compute_t_h(t_h); t_h->providesStartingPoint = true;
  }

  void ACOPFKronRedProblem::construct_buses_idxs(std::vector<int>& idxs_nonaux, std::vector<int>& idxs_aux)
  {
    const double SMALL=1e-8;

    idxs_nonaux.clear(); idxs_aux.clear();

    for(int n=0; n<data_sc.N_Pd.size(); n++) {
      if(data_sc.Gn[n].size()>0 || data_sc.SShn[n].size()>0 || 
	 magnitude(data_sc.N_Pd[n], data_sc.N_Qd[n])>SMALL) {

	idxs_nonaux.push_back(n);
      } else {
	idxs_aux.push_back(n);
      }
    }

    assert(data_sc.Gn.size() == idxs_nonaux.size()+idxs_aux.size());
  }

  hiopMatrixComplexSparseTriplet* ACOPFKronRedProblem::construct_YBus_matrix()
  {
    //
    // details at 
    //  https://gitlab.pnnl.gov/exasgd/frameworks/hiop-framework/blob/master/modules/DenseACOPF.jl
    //

    const int& N = data_sc.N_Bus.size();

    int nnz=N;
    // go over (L_Nidx1, L_Nidx2) and increase nnz when idx1>idx2
    assert(data_sc.L_Nidx[0].size() == data_sc.L_Nidx[1].size());
    for(int it=0; it<data_sc.L_Nidx[0].size(); it++) {
      if(data_sc.L_Nidx[0][it]!=data_sc.L_Nidx[1][it]) 
	nnz++;
    }
    //same for transformers
    assert(data_sc.T_Nidx[0].size() == data_sc.T_Nidx[1].size());
    for(int it=0; it<data_sc.T_Nidx[0].size(); it++) {
      if(data_sc.T_Nidx[0][it]!=data_sc.T_Nidx[1][it]) 
	nnz++;
    }

    //alocate Matrix
    auto Ybus = new hiopMatrixComplexSparseTriplet(N, N, nnz);
    int *Ii=Ybus->storage()->i_row(), *Ji=Ybus->storage()->j_col(); 
    complex<double> *M=Ybus->storage()->M();

    for(int busidx=0; busidx<N; busidx++) {
      Ii[busidx] = Ji[busidx] = busidx;

      // shunt contribution to Ybus
      M[busidx] = complex<double>(data_sc.N_Gsh[busidx], data_sc.N_Bsh[busidx]);
    }

    int nnz_count = N;
    //
    // go over (L_Nidx1, L_Nidx2) and populate the matrix
    //
    for(int l=0; l<data_sc.L_Nidx[0].size(); l++) {
      const int& Nidxfrom=data_sc.L_Nidx[0][l], Nidxto=data_sc.L_Nidx[1][l];

      complex<double> ye(data_sc.L_G[l], data_sc.L_B[l]);
      {
	//yCHe = L[:Bch][l]*im;
	complex<double> res(0.0, data_sc.L_Bch[l]/2); //this is yCHe/2
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
    for(int t=0; t<data_sc.T_Nidx[0].size(); t++) {
      const int& Nidxfrom=data_sc.T_Nidx[0][t], Nidxto=data_sc.T_Nidx[1][t];
      complex<double> yf(data_sc.T_G[t], data_sc.T_B[t]);
      complex<double> yMf(data_sc.T_Gm[t], data_sc.T_Bm[t]);
      const double& tauf = data_sc.T_Tau[t];
      
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
    


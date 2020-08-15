#include "ACOPFKronRedProblem.hpp"

#include "OPFObjectiveTerms.hpp"
#include "OPFConstraints.hpp"
#include "OPFConstraintsKron.hpp"

#include "SCACOPFUtils.hpp"
#include "goUtils.hpp"

using namespace hiop;
using namespace std;

namespace gollnlp {

  ACOPFKronRedProblem::ACOPFKronRedProblem(SCACOPFData& d_in) 
    : data_sc(d_in), Ybus_red_(NULL)
  {
  }
  
  ACOPFKronRedProblem::~ACOPFKronRedProblem() 
  {
    delete Ybus_red_;
  }
  
  /* initialization method: performs Kron reduction and builds the OptProblem */
  bool ACOPFKronRedProblem::assemble()
  {   
    hiopMatrixComplexSparseTriplet* YBus_ = construct_YBus_matrix();
    //YBus_->print();
    construct_buses_idxs(idxs_buses_nonaux, idxs_buses_aux);

    printf("Total %lld buses: %lu nonaux    %lu aux\n",
	   YBus_->m(),
	   idxs_buses_nonaux.size(),
	   idxs_buses_aux.size());
    
    if(Ybus_red_) delete Ybus_red_;

    Ybus_red_ = new hiopMatrixComplexDense(idxs_buses_nonaux.size(),idxs_buses_nonaux.size());

    if(!reduction_.go(idxs_buses_nonaux, idxs_buses_aux, *YBus_, *Ybus_red_)) {
      return false;
    }

    
#ifdef DEBUG
    if(Ybus_red_->assertSymmetry(1e-12))
      printf("!!!!! Ybus matrix is symmetric\n");
    else
      printf("!!!!! Ybus matrix is NOT symmetric\n");
#endif
    add_variables(data_sc);
    add_cons_pf(data_sc);

    //objective
    add_obj_prod_cost(data_sc);
    
    print_summary();
    
    return true;
  }
  
  bool ACOPFKronRedProblem::optimize(const std::string& nlpsolver)
  {
    //set_solver_option("tol", 1e-2);
    //set_solver_option("mu_target", 1e-4);
    bool bret = OptProblemMDS::optimize(nlpsolver);

    int n_iter = 1;
    do {
      if(!bret) return false;

      if(n_iter>=100) {
	printf("error: Max number (%d) of loopings has been reached\n", n_iter);
	return false;
      }
      
      vector<complex<double> > v_n_all_complex;
      compute_voltages_nonaux(vars_block(var_name("v_n", data_sc)), 
			      vars_block(var_name("theta_n", data_sc)),
			      v_n_all_complex);

      //!
      //printvec(v_n_all_complex, "v_n_all_complex");
      
      //s_li, s_ti = power_flows(v_n_all_complex, L, L_Nidx, T, T_Nidx)
      vector<vector<complex<double> > > s_li, s_ti;
      compute_power_flows(v_n_all_complex, s_li, s_ti);

      //!
      //printvecvec(s_li, "s_li");
      //printvecvec(s_ti, "s_ti");
      
      vector<int> Nidx_voutofobounds_pass;
      find_voltage_viol_busidxs(v_n_all_complex, Nidx_voutofobounds_pass);
      if(Nidx_voutofobounds_pass.size()>0) 
	printvec(Nidx_voutofobounds_pass, "Bus indexes with voltage out of bounds: ");
      
      std::vector<int> Lidx_overload_pass, Lin_overload_pass, Tidx_overload_pass, Tin_overload_pass;
      find_power_viol_LTidxs(v_n_all_complex, s_li, s_ti,
			     Lidx_overload_pass, Lin_overload_pass,
			     Tidx_overload_pass, Tin_overload_pass);
      
      // verify that all included constraints are respected
      auto Lfails = Lidx_overload_pass;
      for(int i=0; i<Lfails.size(); i++) Lfails[i] = 10*Lfails[i]+Lin_overload_pass[i];
      auto Lfails_prev_pass = Lidx_overload_;
      for(int i=0; Lfails_prev_pass.size(); i++)
	Lfails_prev_pass[i] = 10*Lfails_prev_pass[i]+Lin_overload_[i];
      auto LfailIdxs = gollnlp::indexin(Lfails, Lfails_prev_pass);
      LfailIdxs = gollnlp::findall(LfailIdxs, [](int val) {return val!=-1;});
      
      auto Tfails = Tidx_overload_pass;
      for(int i=0; i<Tfails.size(); i++) Tfails[i] = 10*Tfails[i]+Tin_overload_pass[i];
      auto Tfails_prev_pass = Tidx_overload_;
      for(int i=0; i<Tfails_prev_pass.size(); i++)
	Tfails_prev_pass[i] = 10*Tfails_prev_pass[i]+Tin_overload_[i];
      auto TfailIdxs = gollnlp::indexin(Tfails, Tfails_prev_pass);
      TfailIdxs = gollnlp::findall(TfailIdxs, [](int val) {return val!=-1;});
      
      //warnings 
      if(LfailIdxs.size()) {
	printvec(LfailIdxs, "warning / error: unexpected lines overloaded - they should not be");
      }
      if(TfailIdxs.size()) {
	printvec(TfailIdxs, "warning / error: unexpected transformers overloaded - they should not be");
      }
      
      //remove failed transmission elements that were previously added to avoid adding
      //redundant constraints for them later on
      for(auto idx : LfailIdxs) {
	erase_idx_from(Lidx_overload_pass, idx);
	erase_idx_from(Lin_overload_pass, idx);
      }
      for(auto idx : TfailIdxs) {
	erase_idx_from(Tidx_overload_pass, idx);
	erase_idx_from(Tin_overload_pass, idx);
      }

      //
      // exit loop if no new lines/transformers appear
      //
      if(Nidx_voutofobounds_pass.size() + Lidx_overload_pass.size() + Tidx_overload_pass.size()==0) {
	cout << "No new violations detected. Exiting loop. Total lazy constraints added: "
	     << (2*(Nidx_voutofobounds_pass.size() + Lidx_overload_pass.size() + Tidx_overload_pass.size()))
	     << endl;
	break;
      }

      //create/get auxiliary voltages and theta variables
      OptVariablesBlock* v_aux_n = vars_block(var_name("v_aux_n",data_sc));
      if(NULL == v_aux_n) {
	assert(1==n_iter);
	v_aux_n = new OptVariablesBlock(0, var_name("v_aux_n", data_sc));
	v_aux_n->sparseBlock = false;
	append_varsblock(v_aux_n);
      }

      OptVariablesBlock* theta_aux_n = vars_block(var_name("theta_aux_n",data_sc));
      if(NULL == theta_aux_n) {
	assert(n_iter==1);
	theta_aux_n = new OptVariablesBlock(0, var_name("theta_aux_n", data_sc));
	theta_aux_n->sparseBlock = false;
	append_varsblock(theta_aux_n);
      }
      
      // reqbuses will hold (aux)bus idxs that have violations and for which (aux) v_n and theta_n
      // will be later added

      //add necessary auxiliary bus voltages (if any)
      auto reqbuses = Nidx_voutofobounds_pass;
      
      //buses part of lines/transformers that are violated
      for(auto idx : Lidx_overload_pass) {
	reqbuses.push_back(data_sc.L_Nidx[Lin_overload_pass[idx]][Lidx_overload_pass[idx]]);
      }
      for(auto idx : Tidx_overload_pass) {
	reqbuses.push_back(data_sc.T_Nidx[Tin_overload_pass[idx]][Tidx_overload_pass[idx]]);
      }

      // sort and remove duplicates
      remove_duplicates(reqbuses);
      //printvec(reqbuses, "reqbuses1: ");
      
      //indexes in idxs_buses_aux of the (aux) buses for which v_n and theta_n will be added
      auto reqauxidxs = indexin(set_diff(reqbuses, idxs_buses_nonaux), idxs_buses_aux);

      //printvec(reqbuses, "reqbuses2: ");
      //reqauxidxs = { 0 };//reqauxidxs[1] };

      //add constraints for voltage violations at auxiliary buses with voltage violations
      {
	vector<double> v_lb, v_ub, v_start;
	vector<double> theta_start;

	//first build information for buses; skip the ones already added (previously with
	//voltage violations)
	for(auto nix : reqauxidxs) {
	  assert(-1 != nix);
			    
	  const int bus_idx = idxs_buses_aux[nix];

	  assert(map_idxbuses_idxsoptimiz_.size() == data_sc.N_Bus.size());
	  assert(map_idxbuses_idxsoptimiz_.size() == v_n_all_complex.size());
	  assert(bus_idx < map_idxbuses_idxsoptimiz_.size());

	  //should not be optimized over previously
	  //((if it was, the voltage violation constraints are already part of the model))
	  if(-1 == map_idxbuses_idxsoptimiz_[bus_idx]) {
	    assert(v_lb.size() == v_ub.size());
	    assert(v_ub.size() == v_start.size());
	    assert(v_start.size() == theta_start.size());
	    
	    //we will add v_aux_n and theta_aux_n for this bus, therefore in map_idxbuses_idxsoptimiz_
	    //we will keep the index in v_aux_n and theta_aux_n
	    assert(v_aux_n->n == theta_aux_n->n);
	    map_idxbuses_idxsoptimiz_[bus_idx] = -(v_aux_n->n + v_lb.size())-2;

	    v_lb.push_back(data_sc.N_Vlb[bus_idx]);
	    v_ub.push_back(data_sc.N_Vub[bus_idx]);
	    assert(bus_idx<v_n_all_complex.size());

	    assert(data_sc.N_v0.size() == v_n_all_complex.size());
	    v_start.push_back(std::abs(v_n_all_complex[bus_idx]));
	    
	    theta_start.push_back(std::arg(v_n_all_complex[bus_idx]));

	    //printf("add bus %d  low/upp %g,%g start=%g   theta start=%g\n",
	    //	   bus_idx, v_lb.back(), v_ub.back(), v_start.back(), theta_start.back());

	    //const hiop::hiopMatrixComplexDense& vmap = reduction_.map_nonaux_to_aux();
	    //auto vmapd = vmap.local_data();

	    // printf("re :");
	    // for(int i=0; i<vmap.n(); ++i) {
	    //   printf("%g ", vmapd[nix][i].real());
	    // }
	    // printf("im :");
	    // for(int i=0; i<vmap.n(); ++i) {
	    //   printf("%g ", vmapd[nix][i].imag());
	    // }
	    // printf("\n");

	  }
	} // end of for loop over reqauxidxs
	
	//
	//append these buses to v_aux_n and theta_aux_n
	//
	//printvec(v_lb, "v_lb");
	//printvec(v_ub, "v_ub");
	//printvec(v_start, "v_start");
	this->primal_variables()->
	  append_vars_to_varsblock(v_aux_n->id, v_lb.size(),
				   v_lb.data(), v_ub.data(),
				   v_start.data());
	this->primal_variables()->
	  append_vars_to_varsblock(theta_aux_n->id, theta_start.size(),
				   NULL, NULL,
				   theta_start.data());
	this->primal_problem_changed();

	//
	// cold starting 
	//
	if(false) {
	  vars_block(var_name("p_g",data_sc))->set_start_to(data_sc.G_p0.data());
	  vars_block(var_name("q_g",data_sc))->set_start_to(data_sc.G_q0.data());
	  
	  vector<double> v0_na;
	  selectfrom(data_sc.N_v0, idxs_buses_nonaux, v0_na);
	  vars_block(var_name("v_n",data_sc))->set_start_to(v0_na.data());

	  //
	  //for theta is a bit more complicated
	  //
	  auto theta_n = vars_block(var_name("theta_n",data_sc));
	  vector<double> theta0_n;
	  selectfrom(data_sc.N_theta0, idxs_buses_nonaux, theta0_n);
	  theta_n->set_start_to(theta0_n.data());

	  int RefBusIdx = data_sc.bus_with_largest_gen(), RefBusIdx_nonaux;
	  auto it = std::find(idxs_buses_nonaux.begin(), idxs_buses_nonaux.end(), RefBusIdx);
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
	      assert(theta_n->x[b] >= theta_n->lb[b]);
	      assert(theta_n->x[b] <= theta_n->ub[b]);
	    }
	  }
	  theta_n->lb[RefBusIdx_nonaux] = 0.;
	  theta_n->ub[RefBusIdx_nonaux] = 0.;
	  assert(theta_n->x[RefBusIdx_nonaux]==0.);
	  

	  vars_block(var_name("b_s",data_sc))->set_start_to(data_sc.SSh_B0.data());
	} // end of cold starting
	
	//
	//append the constraints for voltage violations
	//
	assert(reqauxidxs.size() == v_lb.size());
	
	const hiop::hiopMatrixComplexDense& vmap = reduction_.map_nonaux_to_aux();
	
	auto cons_volt_viol = this->constraints_block(con_name("voltage_viol_aux", data_sc));
	if(cons_volt_viol) {
	  VoltageConsAuxBuses* cons = dynamic_cast<VoltageConsAuxBuses*>(cons_volt_viol);
	  assert(NULL!=cons);

	  if(cons) {
	    cons->append_constraints(reqauxidxs);
	  }
	} else {
	  VoltageConsAuxBuses* cons_block =
	    new VoltageConsAuxBuses(con_name("voltage_viol_aux", data_sc),
				    2*reqauxidxs.size(),
				    vars_block(var_name("v_n", data_sc)), 
				    vars_block(var_name("theta_n", data_sc)),
				    v_aux_n, theta_aux_n,
				    reqauxidxs,
				    vmap);

	  append_constraints(cons_block);
	}	
      } // end of voltage violations block

      
      //
      // append constraints for line thermal violations
      //
      assert(Lidx_overload_pass.size() == Lin_overload_pass.size());
      if(Lidx_overload_pass.size()>0) {

	auto cons_block = this->constraints_block(con_name("line_thermal_viol", data_sc));
	if(cons_block) {
	  LineThermalViolCons* cons_viol = dynamic_cast<LineThermalViolCons*>(cons_block);
	  if(cons_viol) {
	    cons_viol->append_constraints(Lidx_overload_pass, Lin_overload_pass);
	  } else assert(false);
	} else {
	  //idxs_buses_nonaux
	  //idxs_buses_aux
	  //map_idxbuses_idxsoptimiz_
	  LineThermalViolCons* cons_block =
	    new LineThermalViolCons(con_name("line_thermal_viol", data_sc),				   
				    Lidx_overload_pass.size(),
	  			    Lidx_overload_pass,
	  			    Lin_overload_pass,
	  			    data_sc.L_Nidx,
				    data_sc.L_RateBase,
				    data_sc.L_G,
				    data_sc.L_B,
				    data_sc.L_Bch,
				    map_idxbuses_idxsoptimiz_,
				    vars_block(var_name("v_n", data_sc)), 
				    vars_block(var_name("theta_n", data_sc)),
				    v_aux_n,
				    theta_aux_n);
	  append_constraints(cons_block);
	}
      } // end of cons block for line thermal violations
      
      this->dual_problem_changed();


      //
      // resolve
      //
 
      print_summary();

      //derivative_test first-order
      //derivative_test only-second-order
      //set_solver_option("derivative_test", "first-order");
      //set_solver_option("derivative_test", "only-second-order");
      //OptProblem::pass2 = true;

      //set_solver_option("start_with_resto", "yes");
      //set_solver_option("max_hessian_perturbation", 1e+30);
      //set_solver_option("mu_init", 1e+1);
      
      //bret = OptProblemMDS::reoptimize(primalRestart);
      bret = OptProblemMDS::optimize("hiop");

      
      n_iter++;
    } while(true);
    return bret; 
  }

  bool ACOPFKronRedProblem::reoptimize(RestartType t/*=primalRestart*/)
  {
    bool bret = OptProblemMDS::reoptimize(t);

    return bret;
  }
  
  
  void ACOPFKronRedProblem::add_variables(SCACOPFData& d, bool SysCond_BaseCase/* = true*/)
  {
    
    /******************************************************************************
     * WARNING: "sparse" variables need to be added first, before "dense" variables
     ******************************************************************************
     */
    { // generation p_g and q_g
      auto p_g = new OptVariablesBlock(d.G_Generator.size(), var_name("p_g",d), 
				       d.G_Plb.data(), d.G_Pub.data());
      append_varsblock(p_g); 
      p_g->set_start_to(d.G_p0.data());


      auto q_g = new OptVariablesBlock(d.G_Generator.size(), var_name("q_g",d), 
				       d.G_Qlb.data(), d.G_Qub.data());
      q_g->set_start_to(d.G_q0.data());
      append_varsblock(q_g); 
    }

    /******************************************************************************
     * WARNING: voltage variables need to be added first, before theta variables
     * otherwise the Hessian evaluation of PFActiveBalanceKron will not work
     ******************************************************************************
     */
    { //voltages
      vector<double>& vlb = SysCond_BaseCase==true ? data_sc.N_Vlb : data_sc.N_EVlb;
      vector<double>& vub = SysCond_BaseCase==true ? data_sc.N_Vub : data_sc.N_EVub;
      
      vector<double> vlb_na, vub_na, v0_na;
      selectfrom(vlb, idxs_buses_nonaux, vlb_na);
      selectfrom(vub, idxs_buses_nonaux, vub_na);
      selectfrom(data_sc.N_v0, idxs_buses_nonaux, v0_na);
      
      auto v_n = new OptVariablesBlock(idxs_buses_nonaux.size(),
				       var_name("v_n",d),
				       vlb_na.data(),
				       vub_na.data());
      v_n->sparseBlock = false;
      append_varsblock(v_n);
      v_n->set_start_to(v0_na.data());

      assert(map_idxbuses_idxsoptimiz_.size() == 0);
      map_idxbuses_idxsoptimiz_ = vector<int>(data_sc.N_Bus.size(), -1);
      for(int i=0; i<idxs_buses_nonaux.size(); i++) {
	assert(0<=idxs_buses_nonaux[i] && idxs_buses_nonaux[i]<=data_sc.N_Bus.size());
	map_idxbuses_idxsoptimiz_[idxs_buses_nonaux[i]] = i;
      }
    }

    { //theta
      auto theta_n = new OptVariablesBlock(idxs_buses_nonaux.size(), var_name("theta_n",d));
      theta_n->sparseBlock = false;
      append_varsblock(theta_n);
      
      vector<double> theta0_n;
      selectfrom(data_sc.N_theta0, idxs_buses_nonaux, theta0_n);
      theta_n->set_start_to(theta0_n.data());

      int RefBusIdx = data_sc.bus_with_largest_gen(), RefBusIdx_nonaux;
      auto it = std::find(idxs_buses_nonaux.begin(), idxs_buses_nonaux.end(), RefBusIdx);
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
      theta_n->lb[RefBusIdx_nonaux] = 0.;
      theta_n->ub[RefBusIdx_nonaux] = 0.;
      assert(theta_n->x[RefBusIdx_nonaux]==0.);
    }

    { //b_s

      // vector<int> count_SSh_at_nonaux_buses(data_sc.SSh_SShunt.size(), 0);
      // for(int bus_nonaux : idxs_buses_nonaux) {
      //	for(int idx_ssh : data_sc.SShn[bus_nonaux])
      //	  count_SSh_at_nonaux_buses[idx_ssh]++;
      // }
      // printvec(count_SSh_at_nonaux_buses);
      
      auto b_s = new OptVariablesBlock(data_sc.SSh_Blb.size(),
				       var_name("b_s",d),
				       data_sc.SSh_Blb.data(),
				       data_sc.SSh_Bub.data());
      b_s->sparseBlock = false;
      b_s->set_start_to(data_sc.SSh_B0.data());

      append_varsblock(b_s);
    }
  }

  void ACOPFKronRedProblem::add_cons_pf(SCACOPFData& d)
  {
    auto p_g = vars_block(var_name("p_g",d)), 
      v_n = vars_block(var_name("v_n",d)), 
      theta_n = vars_block(var_name("theta_n",d));

    if(true) {
      //active power balance
      auto pf_p_bal = new PFActiveBalanceKron(con_name("p_balance_kron",d), 
					      idxs_buses_nonaux.size(),
					      p_g, v_n, theta_n,
					      idxs_buses_nonaux,
					      d.Gn, *Ybus_red_, data_sc.N_Pd);
      append_constraints(pf_p_bal);
    }
    
    auto b_s = vars_block(var_name("b_s", d));
    auto q_g = vars_block(var_name("q_g",d));
    if(true) {
      auto pf_q_bal = new PFReactiveBalanceKron(con_name("q_balance_kron", d), 
						idxs_buses_nonaux.size(),
						q_g, v_n, theta_n, b_s,
						idxs_buses_nonaux,
						d.Gn, d.SShn,
						*Ybus_red_, data_sc.N_Qd);
      append_constraints(pf_q_bal);
    }
  }
    
  void ACOPFKronRedProblem::add_obj_prod_cost(SCACOPFData& d)
  {
    vector<int> gens(d.G_Generator.size()); iota(gens.begin(), gens.end(), 0);
    auto p_g = vars_block(var_name("p_g", d));
    //PFProdCostAffineCons* prod_cost_cons = 
    // new PFProdCostAffineCons(con_name("prodcost_cons",d), 2*gens.size(), 
    //			       p_g, gens, d.G_CostCi, d.G_CostPi);
    //append_constraints(prod_cost_cons);
    //
    //OptVariablesBlock* t_h = prod_cost_cons->get_t_h();
    //prod_cost_cons->compute_t_h(t_h); t_h->providesStartingPoint = true;

    append_objterm(new PFProdCostApproxAffineObjTerm("prod_cost",
						     p_g,
						     gens,
						     d.G_CostCi,
						     d.G_CostPi));
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
    // go over (L_Nidx1, L_Nidx2) and count nnz 
    assert(data_sc.L_Nidx[0].size() == data_sc.L_Nidx[1].size());
    for(int it=0; it<data_sc.L_Nidx[0].size(); it++) {
      if(data_sc.L_Nidx[0][it]!=data_sc.L_Nidx[1][it]) 
	nnz+=2;
    }
    //same for transformers
    assert(data_sc.T_Nidx[0].size() == data_sc.T_Nidx[1].size());
    for(int it=0; it<data_sc.T_Nidx[0].size(); it++) {
      if(data_sc.T_Nidx[0][it]!=data_sc.T_Nidx[1][it]) 
	nnz+=2;
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
      //M[i,j] -= ye  and M[j,i] -= ye
      assert(Nidxfrom!=Nidxto);
      //if(Nidxfrom>Nidxto) 
      {
	//Ji[nnz_count] = std::max(Nidxfrom, Nidxto);
	//Ii[nnz_count] = std::min(Nidxfrom, Nidxto);
	//M [nnz_count] = -ye;

	Ji[nnz_count] = Nidxfrom;
	Ii[nnz_count] = Nidxto;
	M [nnz_count] = -ye;
	nnz_count++;

	Ji[nnz_count] = Nidxto;
	Ii[nnz_count] = Nidxfrom;
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
	//Ji[nnz_count] = std::max(Nidxfrom, Nidxto);
	//Ii[nnz_count] = std::min(Nidxfrom, Nidxto);
	//M[nnz_count] = -yf/tauf;

	Ji[nnz_count] = Nidxfrom;
	Ii[nnz_count] = Nidxto;
	M[nnz_count] = -yf/tauf;
	nnz_count++;

      	Ji[nnz_count] = Nidxto;
	Ii[nnz_count] = Nidxfrom;
	M[nnz_count] = -yf/tauf;
	nnz_count++;
      }
    }
    assert(nnz_count==nnz);
    Ybus->storage()->sort_indexes();
    Ybus->storage()->sum_up_duplicates();

    return Ybus;
  }

  void ACOPFKronRedProblem::compute_voltages_nonaux(const OptVariablesBlock* v_nonaux,
						    const OptVariablesBlock* theta_nonaux,
						    std::vector<std::complex<double> >& v_complex_all)
  {
    v_complex_all.resize(idxs_buses_nonaux.size() + idxs_buses_aux.size());

    //vector<double> a(v_nonaux->x, v_nonaux->x+v_nonaux->n);
    //printvec(a, "v_nonaux->x");


    std::vector<std::complex<double> > v_complex_nonaux(v_nonaux->n);

    assert(v_nonaux->n == idxs_buses_nonaux.size());
    assert(theta_nonaux->n == v_nonaux->n);
    
    for(int i=0; i<v_nonaux->n; i++) {
      v_complex_nonaux[i] = v_nonaux->x[i] * complex<double>(cos(theta_nonaux->x[i]),
							     sin(theta_nonaux->x[i]));
    }
    
    // v_complex_aux = -(Ybb\Yba)*v_complex_nonaux
    std::vector<complex<double> > v_complex_aux(idxs_buses_aux.size());
    reduction_.apply_nonaux_to_aux(v_complex_nonaux, v_complex_aux);
    //for(auto& it : v_complex_aux) it = -it;

    assert(v_complex_aux.size() == idxs_buses_aux.size());
 
    for(int i=0; i<idxs_buses_nonaux.size(); i++) {
      assert(idxs_buses_nonaux[i] < v_complex_all.size());
      v_complex_all[idxs_buses_nonaux[i]] = v_complex_nonaux[i];
    }
	
    for(int i=0; i<idxs_buses_aux.size(); i++) {
      assert(idxs_buses_aux[i] < v_complex_all.size());
      v_complex_all[idxs_buses_aux[i]] = v_complex_aux[i];
    }
  }

  void ACOPFKronRedProblem::compute_power_flows(const std::vector<std::complex<double> >& v_complex_all,
						std::vector<std::vector<std::complex<double> > >& pli,
						std::vector<std::vector<std::complex<double> > >& pti)
  {
    pli.clear();
    pli.push_back(vector<complex<double> >(data_sc.L_Line.size()));
    pli.push_back(vector<complex<double> >(data_sc.L_Line.size()));
    for(int l=0; l<data_sc.L_Line.size(); ++l) {
      // ye = L[:G][l] + L[:B][l]*im;
      complex<double> ye(data_sc.L_G[l], data_sc.L_B[l]); 
      // yCHe = L[:Bch][l]*im;
      complex<double> yCHe(0., data_sc.L_Bch[l]);
      for(int i=0; i<2; i++) {
	// v_i = v_n[L_Nidx[l,i]]
	auto v_i = v_complex_all[data_sc.L_Nidx[i][l]];
	
	// v_j = v_n[L_Nidx[l,3-i]]
	auto v_j = v_complex_all[data_sc.L_Nidx[1-i][l]];
				 
	// s_li[l, i] = v_i*conj(yCHe/2*v_i) + v_i*conj(ye*(v_j - v_i))
	pli[i][l] = v_i * std::conj(yCHe/2.0*v_i) + v_i * std::conj(ye*(v_j-v_i));
      }
    }

    pti.clear();
    pti.push_back(vector<complex<double> >(data_sc.T_Transformer.size()));
    pti.push_back(vector<complex<double> >(data_sc.T_Transformer.size()));

    for(int t=0; t<data_sc.T_Transformer.size(); ++t) {
      //yf = T[:G][t] + T[:B][t]*im;
      complex<double> yf(data_sc.T_G[t], data_sc.T_B[t]);
      //yMf = T[:Gm][t] + T[:Bm][t]*im;
      complex<double> yMf(data_sc.T_Gm[t], data_sc.T_Bm[t]);
      //tauf = T[:Tau][t];
      double tauf = data_sc.T_Tau[t];
      //v_1 = v_n[T_Nidx[t,1]]
      auto v_1 = v_complex_all[data_sc.T_Nidx[0][t]];
      //v_2 = v_n[T_Nidx[t,2]]
      auto v_2 = v_complex_all[data_sc.T_Nidx[1][t]];
	    
      //s_ti[t, 1] = v_1*conj(yMf*v_1) + v_1/tauf*conj(yf*(v_2 - v_1/tauf))
      pti[0][t] = v_1 * std::conj(yMf*v_1) + v_1/tauf*std::conj(yf*(v_2-v_1/tauf));
      //s_ti[t, 2] = v_2*conj(yf*(v_1/tauf - v_2))
      pti[1][t] = v_2 * std::conj(yf*(v_1/tauf-v_2));
    }
  }

#define EPSILON 1e-6
  void ACOPFKronRedProblem::find_voltage_viol_busidxs(const std::vector<std::complex<double> >& v_complex_all,
						      std::vector<int>& Nidx_voltoutofbnds)
  {
    Nidx_voltoutofbnds.clear();
    for(auto n : idxs_buses_aux) {
      const double v_abs = std::abs(v_complex_all[n]);
      if(data_sc.N_Vlb[n] > v_abs + EPSILON || v_abs > data_sc.N_Vub[n] + EPSILON) {
	//printf("!!!!! viol bus %d -> [%20.14f, %20.14f] val %20.14f\n",
	//       n, data_sc.N_Vlb[n], data_sc.N_Vub[n], v_abs);
	Nidx_voltoutofbnds.push_back(n);	
      }
    }
  }

  /** Finds indexes in lines/transformers and in to/from arrays corresponding to lines/transformers
   * that are overloaded
   */
  void ACOPFKronRedProblem::find_power_viol_LTidxs(const std::vector<std::complex<double> >& v_complex_all,
						   const std::vector<std::vector<std::complex<double> > >& pli,
						   const std::vector<std::vector<std::complex<double> > >& pti,
						   std::vector<int>& Lidx_overload,
						   std::vector<int>& Lin_overload,
						   std::vector<int>& Tidx_overload,
						   std::vector<int>& Tin_overload)
  {
    Lidx_overload.clear();
    Lin_overload.clear();

    for(int l=0; l<data_sc.L_Line.size(); l++) {
      for(int i=0; i<2; i++) {
	const double viol = std::abs(pli[i][l]) -
	  data_sc.L_RateBase[l] * std::abs(v_complex_all[data_sc.L_Nidx[i][l]]);
	if(viol>EPSILON) {
	  Lidx_overload.push_back(l);
	  Lin_overload.push_back(i);
	}
      }
    }

    Tidx_overload.clear();
    Tin_overload.clear();

    for(int t=0; t<data_sc.T_Transformer.size(); t++) {
      for(int i=0; i<2; i++) {
	const double viol = std::abs(pti[i][t]) - data_sc.T_RateEmer[t];
	if(viol>EPSILON) {
	  Tidx_overload.push_back(t);
	  Tin_overload.push_back(i);
	}
      }
    }
  }
  
} //end namespace
    



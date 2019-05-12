#include "SCRecourseProblem.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"

using namespace std;

namespace gollnlp {
  SCRecourseObjTerm::SCRecourseObjTerm(SCACOPFData& d_in,
				       OptVariablesBlock* pg0, OptVariablesBlock* vn0,
				       const std::vector<int>& K_Cont_) 
    : OptObjectiveTerm("recourse_term"), data_sc(d_in), 
      p_g0(pg0), v_n0(vn0), f(0.), grad_p_g0(NULL), grad_v_n0(NULL), H_nz_idxs(NULL)
  {
    auto K_Cont = K_Cont_;
    if(0==K_Cont.size())
      K_Cont = data_sc.K_Contingency;

    for(auto K : K_Cont)
      recou_probs.push_back(new SCRecourseProblem(data_sc, K));


    for(auto prob : recou_probs) {
      //prob->useQPen=true;
      prob->default_assembly(p_g0, v_n0);
    }

    for(auto prob : recou_probs) {
      prob->use_nlp_solver("ipopt");
      prob->set_solver_option("mu_init", 1e-2);
      prob->set_solver_option("mu_target", 1e-9);
      
      prob->set_solver_option("linear_solver", "ma57"); //master_prob.set_solver_option("mu_init", 1.);
      prob->set_solver_option("print_frequency_iter", 20);
      prob->set_solver_option("print_level", 2);
      prob->set_solver_option("tol", 1e-10);
      //prob->set_solver_option("fixed_variable_treatment", "relax_bounds");
      prob->set_solver_option("fixed_variable_treatment", "make_parameter");
    }
    stop_evals = false;
  }
double lala=0.;
  SCRecourseObjTerm::~SCRecourseObjTerm()
  {
    delete [] grad_p_g0;
    delete [] grad_v_n0;
    for(auto p : recou_probs) 
      delete p;
  }
  bool SCRecourseObjTerm::eval_f_grad()
  {
    if(grad_p_g0==NULL) grad_p_g0 = new double[p_g0->n];
    //if(grad_v_n0==NULL) grad_v_n0 = new double[v_n0->n];

    if(!stop_evals) {
      f =0.;
      for(int i=0; i<p_g0->n; i++) grad_p_g0[i]=0.;
      //for(int i=0; i<v_n0->n; i++) grad_v_n0[i]=0.;
      
      //p_g0->print();
      for(auto prob : recou_probs) {
	if(!prob->eval_recourse(p_g0, v_n0, f, grad_p_g0, grad_v_n0))
	  return false;
      }

      for(int i=0; i<p_g0->n; i++) f += 0.5*lala*p_g0->xref[i]*p_g0->xref[i];
      //if(f<1) stop_evals=true;

    }

    return true;
  }

  bool SCRecourseObjTerm::
  eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
  {
    if(new_x) {
      if(!eval_f_grad()) return false;
    }
    obj_val += f;
    return true;
  }
  bool SCRecourseObjTerm::
  eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
  {
    if(new_x) {
      if(!eval_f_grad()) return false;
    }
    double a=1.;
    
    DAXPY(&(p_g0->n), &a, grad_p_g0, &ione, grad+p_g0->index, &ione);

    a=lala;
    DAXPY(&(p_g0->n), &a, const_cast<double*>(p_g0->xref),
	  &ione, grad+p_g0->index, &ione);
    //DAXPY(&(v_n0->n), &a, grad_v_n0, &ione, grad+v_n0->index, &ione);
    return true;
  }
  
  bool SCRecourseObjTerm::
  eval_HessLagr(const OptVariables& vars_primal, bool new_x, 
		const double& obj_factor,
		const int& nnz, int* i, int* j, double* M)
  {
    if(NULL==M) {
      int idx;
      for(int it=0; it<p_g0->n; it++) {
	idx = H_nz_idxs[it]; 
	assert(idx>=0);
	//if(idx<0) return false;
	i[idx] = j[idx] = p_g0->index+it;
      }
    } else {
      double aux = lala*obj_factor;
      for(int it=0; it<p_g0->n; it++) {
	assert(H_nz_idxs[it]>=0);
	assert(H_nz_idxs[it]<nnz);
	M[H_nz_idxs[it]] += aux;
      }
    }
    return true;
  }
  int SCRecourseObjTerm::get_HessLagr_nnz() {
    return p_g0->n;
  }
  bool SCRecourseObjTerm::get_HessLagr_ij(std::vector<OptSparseEntry>& vij)
  {
    if(NULL==H_nz_idxs)
      H_nz_idxs = new int[p_g0->n];
    
    int i;
    for(int it=0; it<p_g0->n; it++) {
      i = p_g0->index+it;
      vij.push_back(OptSparseEntry(i,i,H_nz_idxs+it));
    }
		    
    return true;
  }


  //////////////////////////////////////////////////////////////////////////////////////////
  // SCRecourseProblem
  //////////////////////////////////////////////////////////////////////////////////////////
  SCRecourseProblem::SCRecourseProblem(SCACOPFData& d_in, int K_idx_) 
    : SCACOPFProblem(d_in), K_idx(K_idx_), restart(false)
  {
    int numK = 1; //!

    assert(0==data_K.size());
    //data_sc = d_in (member of the parent)
    data_K.push_back(new SCACOPFData(data_sc)); 
    data_K[0]->rebuild_for_conting(K_idx,numK);

    relax_factor_nonanticip_fixing = 5e-5;
  }

  SCRecourseProblem::~SCRecourseProblem()
  {
  }
  bool SCRecourseProblem::eval_recourse(OptVariablesBlock* pg0, OptVariablesBlock* vn0,
					double& f, double* grad_pg0, double *grad_vn0)
  {

    goTimer tmrec; tmrec.start();
    update_cons_nonanticip_using(pg0);
    //update_cons_AGC_using(pg0);
    //update_cons_PVPQ_using(vn0);

    if(!restart) {
      if(!optimize("ipopt"))
	return false;
      restart = true;
    } else {


      //set_solver_option("bound_push", 1e-16);
      //set_solver_option("slack_bound_push", 1e-16);
      if(!reoptimize(OptProblem::primalDualRestart))
	return false;
    }

    // objective value
    f += this->obj_value;
    //update the grad based on the multipliers
    add_grad_pg0_nonanticip_part_to(grad_pg0);
    //add_grad_pg0_AGC_part_to(grad_pg0);
    //add_grad_vn0_to(grad_vn0);

    tmrec.stop();
    //printf("SCRecourseProblem K_id %d: eval_recourse took %g sec\n", K_idx, tmrec.getElapsedTime());
    printf("SCRecourseProblem K_id %d: recourse obj_value %g\n\n", K_idx, this->obj_value);

    if(false) {
      int dim = pg0->n;
      printf("p_g0 in\n");
      for(int i=0; i<pg0->n; i++)
	printf("%12.5e ", pg0->xref[i]);
      printf("\n**************************************************\n");
      printf("grad_p_g0 out\n");
      for(int i=0; i<pg0->n; i++)
	printf("%12.5e ", grad_pg0[i]);
      printf("\n\n");
    }
    return true;
  }
  bool SCRecourseProblem::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    printf("SCRecourseProblem K_id %d: assembly IDOut=%d outidx=%d Type=%s\n",
	   K_idx, data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str());

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];

    useQPen = true;

    add_variables(dK);
    add_cons_lines_pf(dK);
    add_cons_transformers_pf(dK);
    add_cons_active_powbal(dK);
    add_cons_reactive_powbal(dK);
    bool SysCond_BaseCase = false;
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    //
    // setup for indexes used in non-anticip and AGC coupling 
    //
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    vector<int> Gk;
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);
    assert(pg0->n == Gk.size() || pg0->n == 1+Gk.size());

    pg0_nonpartic_idxs=Gk;
    pg0_partic_idxs={};

    // indexes in data_K (for the recourse's contingency)
    auto ids_no_AGC = selectfrom(data_sc.G_Generator, pg0_nonpartic_idxs);
    pgK_nonpartic_idxs = indexin(dK.G_Generator, ids_no_AGC);
    pgK_nonpartic_idxs = findall(pgK_nonpartic_idxs, [](int val) {return val!=-1;});

    auto ids_AGC = selectfrom(data_sc.G_Generator, pg0_partic_idxs);
    pgK_partic_idxs = indexin(dK.G_Generator, ids_AGC);
    pgK_partic_idxs = findall(pgK_partic_idxs, [](int val) {return val!=-1;});
#ifdef DEBUG
    assert(pg0_nonpartic_idxs.size() == pgK_nonpartic_idxs.size());
    for(int i0=0, iK=0; i0<pg0_nonpartic_idxs.size(); i0++, iK++) {
      //all dB.G_Generator should be in data_sc.G_Generator
      assert(pgK_nonpartic_idxs[iK]>=0); 
      //all ids should match in order
      assert(dK.G_Generator[pgK_nonpartic_idxs[iK]] ==
	     data_sc.G_Generator[pg0_nonpartic_idxs[i0]]);
    }
    assert(pg0_partic_idxs.size() == pgK_partic_idxs.size());
    for(int i=0; i<pg0_partic_idxs.size(); i++) {
      assert(pgK_partic_idxs[i]>=0); 
      //all ids should match in order
      assert(dK.G_Generator[pgK_partic_idxs[i]] ==
	     data_sc.G_Generator[pg0_partic_idxs[i]]);
    }
      
#endif
    add_cons_nonanticip_using(pg0);
    //add_cons_AGC_using(pg0);
    //PVPQSmoothing = AGCSmoothing = 1e-2;
    //coupling AGC and PVPQ; also creates delta_k
    //add_cons_coupling(dK);
    return true;
  }

  void SCRecourseProblem::bodyof_cons_nonanticip_using(OptVariablesBlock* pg0)
  {
    SCACOPFData& dK = *data_K[0]; assert(dK.id-1 == K_idx);
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] SCRecourseProblem K_id %d: p_g var not found in contingency  "
	     "recourse problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    
    int sz = pgK_nonpartic_idxs.size();
    assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    int idxK; double pg0_val, lb, ub; double& f = relax_factor_nonanticip_fixing;
  
    for(int i=0; i<sz; i++) {
      assert(pg0_idxs[i]<pg0->n && pg0_idxs[i]>=0);
      assert(pgK_idxs[i]<pgK->n && pgK_idxs[i]>=0);
      idxK = pgK_idxs[i];
      pgK->lb[idxK] = pgK->ub[idxK] = pg0->xref[pg0_idxs[i]];
    }
    /* 
    // code that set p_gK=p_g0 and relax the bounds a bit; if the bounds are not
    // relaxed, Ipopt seems to return large bound multiplies (for both bounds!?!)
    for(int i=0; i<sz; i++) {
      //pgK->lb[idxK] = pgK->ub[idxK] = pg0->xref[pg0_idxs[i]];
      //pgK->lb[idxK] -= 1e-6;
      assert(pg0_idxs[i]<pg0->n && pg0_idxs[i]>=0);
      assert(pgK_idxs[i]<pgK->n && pgK_idxs[i]>=0);

      idxK = pgK_idxs[i];
      pg0_val = pg0->xref[pg0_idxs[i]];
      lb = pg0->lb[pg0_idxs[i]];
      ub = pg0->ub[pg0_idxs[i]];
      aux = ub-lb;
      assert(aux>1e-6);
      if(aux<1e-2) aux = 1e-2;
      aux = aux*f;
      if(fabs(pg0_val-lb)<1e-8) {
	pgK->lb[idxK] = pg0_val; 
	pgK->ub[idxK] = pg0_val+aux;
	continue;
      }
      if(fabs(ub-pg0_val)<1e-8) {
	pgK->ub[idxK] = pg0_val;
	pgK->lb[idxK] = pg0_val-aux;
	continue;
      }
      aux = aux/2;
      pgK->lb[idxK] = pg0_val-aux;
      pgK->ub[idxK] = pg0_val+aux;
    }
    */
  }

  void SCRecourseProblem::add_grad_pg0_nonanticip_part_to(double* grad_pg0)
  {
    SCACOPFData& dK = *data_K[0];
    assert(pgK_nonpartic_idxs.size() == pg0_nonpartic_idxs.size());
    OptVariablesBlock* pgK = variable("p_g", dK);
    
    // Active Power Balance constraints
    //
    // sum(p_g[g] for g=Gn[n])  - N[:Gsh][n]*v_n[n]^2 -
    // sum(p_li[Lidxn[n][lix],Lin[n][lix]] for lix=1:length(Lidxn[n])) -
    // sum(p_ti[Tidxn[n][tix],Tin[n][tix]] for tix=1:length(Tidxn[n])) - 
    // r*pslackp_n[n] + r*pslackm_n[n]    =   N[:Pd][n])
    // 
    // since p_gK[non-particip] were fixed, the gradient of the objective 
    // with respect to these p_gKs is given by multipliers of the active
    // power balance constraints
    //
    // iterate over pgK_nonpartic_idxs, find the bus idx, and set the gradient
    // entry to the multiplier of the balance constraints at the found bus idx.

    const OptVariablesBlock* duals = // of p_balance constraints in recourse
      vars_duals_cons->get_block(string("duals_") + con_name("p_balance", dK));
    assert(duals->n == data_sc.N_Bus.size());
    int sz = pgK_nonpartic_idxs.size(); assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    int bus_idx;
    for(int i=0; i<sz; i++) {
      bus_idx = dK.G_Nidx[pgK_idxs[i]];
      assert(bus_idx >=0 && bus_idx < data_sc.N_Bus.size());
      assert(bus_idx < duals->n);
      grad_pg0[pg0_idxs[i]] += duals->x[bus_idx];
    }

    //printf("SCRecourseProblem K_id %d: evaluated grad_pg0_nonanticip\n", K_idx);

    /*
    // code that gets the gradient of the recourse based on duals bounds;
    // not very reliable for some reason: multipliers/gradient seems to 
    // change drastically for different values of 'relax_factor_nonanticip_fixing'
    const OptVariablesBlock* duals_pgK_bounds = 
      vars_duals_bounds->get_block(string("duals_bnd_") + pgK->id);
    assert(duals_pgK_bounds);
    assert(duals_pgK_bounds->n == pgK->n);
    //assert(false);
    int sz = pgK_nonpartic_idxs.size(); assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    for(int i=0; i<sz; i++) {
      grad_pg0[pg0_idxs[i]] += duals_pgK_bounds->x[pgK_idxs[i]];
    }
    */

  }

  void SCRecourseProblem::add_cons_AGC_using(OptVariablesBlock* pg0)
  {
    SCACOPFData& dK = *data_K[0];
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] SCRecourseProblem K_id %d: p_g var not found in contingency  "
	     "recourse problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dK));
    append_variables(deltaK);
    deltaK->set_start_to(0.);
    
    AGCSmoothing = 1e-3;
    auto cons = new AGCComplementarityCons(con_name("AGC", dK), 3*pgK_partic_idxs.size(),
					   pg0, pgK, deltaK, 
					   pg0_partic_idxs, pgK_partic_idxs, 
					   selectfrom(data_sc.G_Plb, pgK_partic_idxs), 
					   selectfrom(data_sc.G_Pub, pgK_partic_idxs),
					   data_sc.G_alpha, AGCSmoothing,
					   false, 0., //no internal penalty
					   true); //fixed p_g0 
    append_constraints(cons);

    //starting point for rhop and rhom that were added by AGCComplementarityCons
    auto rhop=cons->get_rhop(), rhom=cons->get_rhom();
    cons->compute_rhos(rhop, rhom);
    rhop->providesStartingPoint=true; rhom->providesStartingPoint=true;

    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhom->id, rhom, 1.));
    append_objterm(new LinearPenaltyObjTerm(string("bigMpen_")+rhop->id, rhop, 1.));
    
    //for(int i=0; i<rhop->n; i++) 
    //  printf("%g %g   %g\n", rhop->x[i], rhom->x[i], cons->gb[i]);
    //printf("\n");
    printf("SCRecourseProblem K_id %d: AGC %lu gens participating (out of %d)\n", 
	   K_idx, pgK_partic_idxs.size(), pgK->n);
  }
  void SCRecourseProblem::update_cons_AGC_using(OptVariablesBlock* pg0)
  {
    //pg0 pointer that AGCComplementarityCons should not change
#ifdef DEBUG
    SCACOPFData& dK = *data_K[0];
    auto cons_AGC = dynamic_cast<AGCComplementarityCons*>(constraint("AGC", dK));
    assert(cons_AGC);
    if(pg0 != cons_AGC->get_p_g0()) {
      assert(false);
    }
#endif
  }

  void SCRecourseProblem::add_grad_pg0_AGC_part_to(double* grad_pg0)
  {
    SCACOPFData& dK = *data_K[0];
    auto duals = vars_duals_cons->get_block(string("duals_") + con_name("AGC", dK));
    assert(duals);
    assert(duals->n == 3*pgK_partic_idxs.size());
    // only need the multipliers for p0 + alpha*deltak - pk - gb * rhop + gb * rhom = 0
    // these are the first duals->n/3 equations
    int sz = pgK_partic_idxs.size(); assert(sz == pg0_partic_idxs.size());
    int *pgK_idxs = pgK_partic_idxs.data(), *pg0_idxs = pg0_partic_idxs.data();
    for(int i=0; i<sz; i++) {
      grad_pg0[pg0_idxs[i]] += duals->x[i];
    }
  }

  void SCRecourseProblem::add_cons_PVPQ_using(OptVariablesBlock* vn0)
  {
  }
  void SCRecourseProblem::update_cons_PVPQ_using(OptVariablesBlock* vn0)
  {
    
  }
  void SCRecourseProblem::add_grad_vn0_to(double* grad)
  {
    
  }
				       
} //end namespace

#include "SCRecourseProblem.hpp"

#include "goUtils.hpp"
#include "goTimer.hpp"

using namespace std;

namespace gollnlp {
  SCRecourseObjTerm::SCRecourseObjTerm(SCACOPFData& d_in,
				       OptVariablesBlock* pg0, OptVariablesBlock* vn0,
				       const std::vector<int>& K_Cont_) 
    : OptObjectiveTerm("recourse_term"), data_sc(d_in), 
      p_g0(pg0), v_n0(vn0), f(0.), grad_p_g0(NULL), grad_v_n0(NULL)
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
      prob->set_solver_option("linear_solver", "ma57"); //master_prob.set_solver_option("mu_init", 1.);
      prob->set_solver_option("print_frequency_iter", 10);
      prob->set_solver_option("print_level", 2);
      prob->set_solver_option("fixed_variable_treatment", "relax_bounds");
    }
  }

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
    if(grad_v_n0==NULL) grad_v_n0 = new double[v_n0->n];
    f =0.;
    for(int i=0; i<p_g0->n; i++) grad_p_g0[i]=0.;
    for(int i=0; i<v_n0->n; i++) grad_v_n0[i]=0.;

    //p_g0->print();
    for(auto prob : recou_probs) {
      if(!prob->eval_recourse(p_g0, v_n0, f, grad_p_g0, grad_v_n0))
	return false;
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
    DAXPY(&(p_g0->n), &done, grad_p_g0, &ione, grad+p_g0->index, &ione);
    DAXPY(&(v_n0->n), &done, grad_v_n0, &ione, grad+v_n0->index, &ione);
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

    relax_factor_nonanticip_fixing = 1e-4;
  }

  SCRecourseProblem::~SCRecourseProblem()
  {
  }
  bool SCRecourseProblem::eval_recourse(OptVariablesBlock* pg0, OptVariablesBlock* vn0,
					double& f, double* grad_pg0, double *grad_vn0)
  {

    goTimer tmrec; tmrec.start();
    update_cons_nonanticip_using(pg0);
    update_cons_AGC_using(pg0);
    update_cons_PVPQ_using(vn0);

    if(!restart) {
      if(!optimize("ipopt"))
	return false;
      restart = true;
    } else {

      set_solver_option("mu_init", 1e-8);
      set_solver_option("bound_push", 1e-16);
      set_solver_option("slack_bound_push", 1e-16);
      if(!reoptimize(OptProblem::primalDualRestart))
	return false;
    }

    // objective value
    f += this->obj_value;
    //update the grad based on the multipliers
    add_grad_pg0_nonanticip_part_to(grad_pg0);
    add_grad_pg0_AGC_part_to(grad_pg0);
    add_grad_vn0_to(grad_vn0);

    tmrec.stop();
    printf("SCRecourseProblem K_id %d: eval_recourse took %g sec\n", K_idx, tmrec.getElapsedTime());
    printf("SCRecourseProblem K_id %d: recourse obj_value %g\n", K_idx, this->obj_value);

    if(true) {
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

    // coupling 
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    vector<int> Gk;
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);
    assert(pg0->n == Gk.size() || pg0->n == 1+Gk.size());

    // indexes in data_K (for the recourse's contingency)
    auto ids_no_AGC = selectfrom(data_sc.G_Generator, pg0_nonpartic_idxs);
    pgK_nonpartic_idxs = indexin(dK.G_Generator, ids_no_AGC);
    pgK_nonpartic_idxs = findall(pgK_nonpartic_idxs, [](int val) {return val!=-1;});

#ifdef DEBUG
    assert(pg0_nonpartic_idxs.size() == pgK_nonpartic_idxs.size());
    for(int i0=0, iK=0; i0<pg0_nonpartic_idxs.size(); i0++, iK++) {
      //all dB.G_Generator should be in data_sc.G_Generator
      assert(pgK_nonpartic_idxs[iK]>=0); 
      //all ids should match in order
      assert(dK.G_Generator[pgK_nonpartic_idxs[iK]] ==
	     data_sc.G_Generator[pg0_nonpartic_idxs[i0]]);
    }
#endif
    add_cons_nonanticip_using(pg0);

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
    int idxK; double pg0_val, lb, ub, aux; double& f = relax_factor_nonanticip_fixing;
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
    
  }

  void SCRecourseProblem::add_grad_pg0_nonanticip_part_to(double* grad_pg0)
  {
    SCACOPFData& dK = *data_K[0];
    assert(pgK_nonpartic_idxs.size() == pg0_nonpartic_idxs.size());
    OptVariablesBlock* pgK = variable("p_g", dK);
    const OptVariablesBlock* duals_pgK_bounds = vars_duals_bounds->get_block(string("duals_bnd_") + pgK->id);
    assert(duals_pgK_bounds);
    assert(duals_pgK_bounds->n == pgK->n);
    //assert(false);
    int sz = pgK_nonpartic_idxs.size(); assert(sz == pg0_nonpartic_idxs.size());
    int *pgK_idxs = pgK_nonpartic_idxs.data(), *pg0_idxs = pg0_nonpartic_idxs.data();
    for(int i=0; i<sz; i++) {
      grad_pg0[pg0_idxs[i]] += duals_pgK_bounds->x[pgK_idxs[i]];
    }
    //printf("SCRecourseProblem K_id %d: evaluated grad_pg0_nonanticip\n", K_idx);
  }

  void SCRecourseProblem::add_cons_AGC_using(OptVariablesBlock* pg0)
  {
    
  }
  void SCRecourseProblem::update_cons_AGC_using(OptVariablesBlock* pg0)
  {
    
  }

  void SCRecourseProblem::add_grad_pg0_AGC_part_to(double* grad)
  {
    
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

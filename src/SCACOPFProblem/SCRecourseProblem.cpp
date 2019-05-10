#include "SCRecourseProblem.hpp"

#include "goUtils.hpp"

using namespace std;

namespace gollnlp {
  SCRecourseObjTerm::SCRecourseObjTerm(SCACOPFData& d_in,
				       OptVariablesBlock* pg0, OptVariablesBlock* vn0,
				       const std::vector<int>& K_Cont_) 
    : OptObjectiveTerm("recourse_term"), data_sc(d_in), p_g0(pg0), v_n0(vn0)
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
      
      prob->optimize("ipopt");
      printf("SOLVED conting problem %d\n\n", prob->data_K[0]->K_Contingency[0]);
    }
  }

  SCRecourseObjTerm::~SCRecourseObjTerm()
  {
    for(auto p : recou_probs) 
      delete p;
  }

  bool SCRecourseObjTerm::
  eval_f(const OptVariables& vars_primal, bool new_x, double& obj_val)
  {
    return true;
  }
  bool SCRecourseObjTerm::
  eval_grad(const OptVariables& vars_primal, bool new_x, double* grad)
  {
    return true;
  }


  //////////////////////////////////////////////////////////////////////////////////////////
  // SCRecourseProblem
  //////////////////////////////////////////////////////////////////////////////////////////
  SCRecourseProblem::SCRecourseProblem(SCACOPFData& d_in, int K_idx_) 
    : SCACOPFProblem(d_in), K_idx(K_idx_)
  {
    int numK = 1; //!

    assert(0==data_K.size());
    //data_sc = d_in (member of the parent)
    data_K.push_back(new SCACOPFData(data_sc)); 
    data_K[0]->rebuild_for_conting(K_idx,numK);
  }

  SCRecourseProblem::~SCRecourseProblem()
  {
  }
  bool SCRecourseProblem::eval_recourse(OptVariablesBlock* pg0, OptVariablesBlock* vn0,
					double& f, double* grad)
  {
    return true;
  }
  bool SCRecourseProblem::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {

    printf("SCRecourseProblem: assemblying for contingency K=%d IDOut=%d "
	   "outidx=%d Type=%s\n",
	   K_idx, data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str());

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];
    
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
    enforce_nonanticip_coupling(pg0);

    //PVPQSmoothing = AGCSmoothing = 1e-2;
    //coupling AGC and PVPQ; also creates delta_k
    //add_cons_coupling(dK);
    return true;
  }

  void SCRecourseProblem::enforce_nonanticip_coupling(OptVariablesBlock* pg0)
  {
    SCACOPFData& dK = *data_K[0];
    OptVariablesBlock* pgK = variable("p_g", dK);
    if(NULL==pgK) {
      printf("[warning] Contingency %d: p_g var not found in contingency recourse"
	     " problem; will not enforce non-ACG coupling constraints.\n", dK.id);
      return;
    }
    assert(pgK_nonpartic_idxs.size() == pg0_nonpartic_idxs.size());
    for(int i0=0, iK=0; i0<pg0_nonpartic_idxs.size(); i0++, iK++) {
      pgK->lb[iK] = pgK->ub[iK] = pg0->x[i0];
      assert(i0<pg0->n);
      assert(iK<pgK->n);
    }
    printf("Recourse K_id %d: AGC: %lu gens NOT participating: fixed all of "
	   "them.\n", K_idx, pg0_nonpartic_idxs.size());
  }

  void SCRecourseProblem::add_cons_AGC(OptVariablesBlock* pg0)
  {
    
  }
				       
} //end namespace

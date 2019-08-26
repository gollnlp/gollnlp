#include "ContingencyProblemWithFixing.hpp"

#include "CouplingConstraints.hpp"
#include "OPFObjectiveTerms.hpp"
#include "OptObjTerms.hpp"
#include "OPFConstraints.hpp"

#include <string>

#include "goUtils.hpp"
#include "goTimer.hpp"
#include "unistd.h"
using namespace std;

#define BE_VERBOSE

namespace gollnlp {
  
  ContingencyProblemWithFixing::~ContingencyProblemWithFixing()
  {
  }

  bool ContingencyProblemWithFixing::do_fixing_for_AGC(const double& smoothing, bool fixVoltage, 
						       OptVariablesBlock* pgk, OptVariablesBlock* delta)
  {
    SCACOPFData& d = *data_K[0];
    assert(delta); assert(delta->n==1);
    const double deltak = delta->x[0];

    vector<int> Gk_simplified;

    double rtol  = sqrt(smoothing);
    double rtolp = sqrt(smoothing)/10.;

    for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
      const int iK=pgK_nonpartic_idxs[i];
      const int i0=pg0_nonpartic_idxs[i];

      double gen_band = d.G_Pub[iK] - d.G_Plb[iK];
      assert(d.G_Pub[iK] == data_sc.G_Pub[i0]); assert(d.G_Plb[iK] == data_sc.G_Plb[i0]);
      assert(gen_band>=1e-6);

      double dist_lower = (pgk->x[iK]-d.G_Plb[iK])/gen_band;
      double dist_upper = (d.G_Pub[iK]-pgk->x[iK])/gen_band;
	
      assert(dist_lower<0); assert(dist_upper<0);

      double distp = (pgk->x[iK] - p_g0->x[i0] - d.G_alpha[i0]*deltak)/gen_band;

      if(dist_lower > rtol && dist_upper > rtol) {
	//p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
	//
	//!@constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
	printf("  inside, enf. eq\n");
      } else if(dist_lower <= rtol) {
	if(distp > rtolp) { // && deltaIsFeas) {
	  //strict complementarity and delta is feasible -> fixing is clear
	  pgk->ub[iK] = d.G_Plb[iK];
	  printf("  fixed p_gk at lower\n");
	} else {
	  //degenerate complementarity or delta is not feasible , enforce equality

	  //p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
	  
          //! @constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
	  Gk_simplified.push_back(i0);
	  printf("  p_gk at lower but enforcing eq.; degen compl\n");
	}  
      } else {
	//#dist_upper <= rtol
	if(distp < -rtolp) { // && deltaIsFeas
	  //strict complementarity and delta is feasible, fixing is clear
	  pgk->lb[iK] = d.G_Pub[iK];
	  printf("  fixed p_gk at upper\n");
	} else {
	  //degenerate complementarity or delta is infeasible, enforce equality
	  //p_gk[g] = @variable(m, lower_bound=G[:Plb][g], upper_bound=G[:Pub][g], start=papprox_gk[g])
	  //!@constraint(m, p_gk[g]   == p_g[g] + G[:alpha][g]*delta_k)
	  Gk_simplified.push_back(i0);
	  printf("  p_gk at upper but enforcing eq.; degen compl\n");
	}
      }
    }
  

    if(Gk_simplified.size() > 0) {
      //!add_cons_AGC_simplified(d, Gk_simplified);
    }

    return true;
  }


  bool ContingencyProblemWithFixing::do_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
							OptVariablesBlock* vnk, OptVariablesBlock* qgk)
  {
    SCACOPFData& d = *data_K[0];

    //(aggregated) non-fixed q_g generator ids at each node/bus
    // with PVPQ generators that have at least one non-fixed q_g 
    vector<vector<int> > idxs_gen_agg;
    //bus indexes that have at least one non-fixed q_g
    vector<int> idxs_bus_pvpq;
    //aggregated lb and ub on reactive power at each PVPQ bus
    vector<double> Qlb, Qub;
    int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;
    
    get_idxs_PVPQ(d, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
		  nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);
    assert(idxs_gen_agg.size() == idxs_bus_pvpq.size());
    assert(vnk->n == v_n0->n);

    for(int itpvpq=0; itpvpq<idxs_bus_pvpq.size(); itpvpq++) {
      const int busidx = idxs_bus_pvpq[itpvpq];
      double vdev = (vnk->x[busidx]-v_n0->x[busidx]) / std::max(1., fabs(v_n0->x[busidx]));
      double Qlbn=0., Qubn=0., qapprox_nk=0.;
      for(int gidx : idxs_gen_agg[itpvpq]) {
	Qlbn += d.G_Qlb[gidx];
	Qubn += d.G_Qub[gidx];
	qapprox_nk += qgk->x[gidx];
      }

      double gen_band = Qubn - Qlbn; 
      double dist_lower = (qapprox_nk - Qlbn)/gen_band; 
      double dist_upper = (Qubn - qapprox_nk)/gen_band; 
      

      if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
	printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
	       busidx, gen_band, dist_lower,  dist_upper,
	       Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);

      assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
      assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

      const double rtol = 1e-2, rtolv=1e-3;
      if(dist_lower > rtol && dist_upper > rtol) {
	//inside -> fix v_nk
	if(fabs(vdev)>=sqrt(smoothing)) {
	  printf("[warning] sum(q_gk) in is inside the bounds (dist_lower,dist_upper)=(%g,%g), "
		 "but volt dev is large %g. %d gens at the busidx %d.", 
		 dist_lower, dist_upper, vdev, idxs_gen_agg[itpvpq].size(), busidx);
	}
	//printf("  fixing v_nk\n");
	vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
        //for g=Gnk
	//    q_gk[g] = @variable(m, lower_bound=G[:Qlb][g], upper_bound=G[:Qub][g], start=qapprox_gk[g])
	//end

      } else if(dist_lower <= rtol) {
	if(vdev >= rtolv) {
	  //strict complementarity -> fix q_gk     to Qlb               
	  
          //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

	  vnk->lb[busidx] = v_n0->x[busidx]; 
	  for(int g : idxs_gen_agg[itpvpq])
	    qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
	}  else {
	  //degenerate complementarity 
	  
	  if(fixVoltage) {
	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will fix voltage\n", busidx); 
	    vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
	  } else {
	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d; will put q_g close to lower\n", busidx); 
	    vnk->lb[busidx] = v_n0->x[busidx]; 
	    for(int g : idxs_gen_agg[itpvpq])
	      qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
	  }
	}
      } else { // if(dist_upper <= rtol)
	assert(dist_upper <= rtol);
	if(vdev <= - rtolv) {
	  //strict complementarity -> fix q_gk to Qub 
	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
	  vnk->ub[busidx] = v_n0->x[busidx]; 
	    for(int g : idxs_gen_agg[itpvpq])
	      qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
	} else {
	  //degenerate complementarity 
	  
	  if(fixVoltage) 
	    {
	      printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will fix voltage\n", busidx); 
	      vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
	    } else {
	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d; will put q_g to the upper\n", busidx); 
	    vnk->ub[busidx] = v_n0->x[busidx]; 
	    for(int g : idxs_gen_agg[itpvpq])
		qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
	  }
	}
      }
      
    }

    return true;
  }

  bool ContingencyProblemWithFixing::attempt_fixing_for_PVPQ(const double& smoothing, bool fixVoltage,
							     OptVariablesBlock* vnk, OptVariablesBlock* qgk)
  {
    SCACOPFData& d = *data_K[0];
    bool needs_another_fixing=false;

    //(aggregated) non-fixed q_g generator ids at each node/bus
    // with PVPQ generators that have at least one non-fixed q_g 
    vector<vector<int> > idxs_gen_agg;
    //bus indexes that have at least one non-fixed q_g
    vector<int> idxs_bus_pvpq;
    //aggregated lb and ub on reactive power at each PVPQ bus
    vector<double> Qlb, Qub;
    int nPVPQGens=0,  num_qgens_fixed=0, num_N_PVPQ=0, num_buses_all_qgen_fixed=0;
    
    get_idxs_PVPQ(d, Gk, idxs_gen_agg, idxs_bus_pvpq, Qlb, Qub, 
		  nPVPQGens, num_qgens_fixed, num_N_PVPQ, num_buses_all_qgen_fixed);
    assert(idxs_gen_agg.size() == idxs_bus_pvpq.size());
    assert(vnk->n == v_n0->n);

    for(int itpvpq=0; itpvpq<idxs_bus_pvpq.size(); itpvpq++) {
      const int busidx = idxs_bus_pvpq[itpvpq];
      double vdev = (vnk->x[busidx]-v_n0->x[busidx]) / std::max(1., fabs(v_n0->x[busidx]));
      double Qlbn=0., Qubn=0., qapprox_nk=0.;
      for(int gidx : idxs_gen_agg[itpvpq]) {
	Qlbn += d.G_Qlb[gidx];
	Qubn += d.G_Qub[gidx];
	qapprox_nk += qgk->x[gidx];
      }

      double gen_band = Qubn - Qlbn; 
      double dist_lower = (qapprox_nk - Qlbn)/gen_band; 
      double dist_upper = (Qubn - qapprox_nk)/gen_band; 
      

      if(dist_lower<=0 || dist_upper<=0 || gen_band<1e-6)
	printf("busidx=%d %g %g %g qlb[%g %g] qub[%g %g]\n", 
	       busidx, gen_band, dist_lower,  dist_upper,
	       Qlbn, Qlb[itpvpq], Qubn, Qub[itpvpq]);

      assert(dist_lower>=0); assert(dist_upper>=0); assert(gen_band>=0);
      assert(fabs(Qlbn-Qlb[itpvpq])<1e-10);  assert(fabs(Qubn-Qub[itpvpq])<1e-10);

      const double rtol = 1e-2, rtolv=1e-3;
      if(dist_lower > rtol && dist_upper > rtol) {
	//inside -> fix v_nk
	if(fabs(vdev)>=sqrt(smoothing)) {
	  printf("[warning] sum(q_gk) in is inside the bounds (dist_lower,dist_upper)=(%g,%g), "
		 "but volt dev is large %g. %d gens at the busidx %d.", 
		 dist_lower, dist_upper, vdev, idxs_gen_agg[itpvpq].size(), busidx);
	}
	//printf("  fixing v_nk\n");
	//!vnk->lb[busidx] = vnk->ub[busidx] = v_n0->x[busidx];
        //for g=Gnk
	//    q_gk[g] = @variable(m, lower_bound=G[:Qlb][g], upper_bound=G[:Qub][g], start=qapprox_gk[g])
	//end

      } else if(dist_lower <= rtol) {
	if(vdev >= rtolv) {
	  //strict complementarity -> fix q_gk     to Qlb               
	  
          //printf("  fixing q_gk to Qlb;  lower bound for v_nk updated\n");

	  //!vnk->lb[busidx] = v_n0->x[busidx]; 
	  //!for(int g : idxs_gen_agg[itpvpq])
	  //!  qgk->lb[g] = qgk->ub[g] = d.G_Qlb[g];
	}  else {
	  //degenerate complementarity 
	  needs_another_fixing=true;
	  if(fixVoltage) {
	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d;\n", busidx); 
	    //!!vnk->lb[busidx] = v_n0->x[busidx];
	    //!1for(int g : idxs_gen_agg[itpvpq])
	    //!!  qgk->ub[g] = qgk->x[g];
	  } else {
	    printf("  degenerate complementarity (q_g close to lower) at busidx=%d\n", busidx); 
	    //!!vnk->lb[busidx] = v_n0->x[busidx]; 
	    //!!for(int g : idxs_gen_agg[itpvpq])
	    //!!  qgk->ub[g] = qgk->x[g];
	  }
	}
      } else { // if(dist_upper <= rtol)
	assert(dist_upper <= rtol);
	if(vdev <= - rtolv) {
	  //strict complementarity -> fix q_gk to Qub 
	  //printf("  fixing q_gk to Qub;  upper bound for v_nk updated\n");
	  
	  //!vnk->ub[busidx] = v_n0->x[busidx]; 
	  //!for(int g : idxs_gen_agg[itpvpq])
	  //!  qgk->lb[g] = qgk->ub[g] = d.G_Qub[g];
	} else {
	  //degenerate complementarity 
	  needs_another_fixing=true;
	  if(fixVoltage) {
	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d\n", busidx); 
	    //!!vnk->ub[busidx] = v_n0->x[busidx];
	    //!!for(int g : idxs_gen_agg[itpvpq])
	//!!	qgk->lb[g] = qgk->x[g];
	  } else {
	    printf("  degenerate complementarity (q_g close to upper) at busidx=%d\n", busidx); 
	    //!!vnk->ub[busidx] = v_n0->x[busidx]; 
	    //!!for(int g : idxs_gen_agg[itpvpq])
	//!!	qgk->lb[g] = qgk->x[g];
	  }
	}
      }
      
    }

    return needs_another_fixing;
  }

  bool ContingencyProblemWithFixing::optimize(OptVariablesBlock* pg0, OptVariablesBlock* vn0, double& f)
  {
    goTimer tmrec; tmrec.start();

    assert(p_g0 == pg0); assert(v_n0 == vn0);
    p_g0 = pg0; v_n0=vn0;

    update_cons_nonanticip_using(pg0);
    //update_cons_AGC_using(pg0);

    f = -1e+20;
    vector<double> smoothing_continu = {1e-4};
    int smoothing_status = 0;
    vector<int> hist_iter, hist_obj;

    for(auto smoothing : smoothing_continu) {
      smoothing_status = 0;
      update_AGC_smoothing_param(smoothing);
      update_PVPQ_smoothing_param(smoothing);
      
      if(smoothing == smoothing_continu[0]) {
	this->set_solver_option("mu_init", 1e-5);
	//if(!OptProblem::optimize("ipopt")) {
	  if(!OptProblem::reoptimize(OptProblem::primalDualRestart)) {
	  smoothing_status=-1;
	}
      } else {
	this->set_solver_option("mu_init", 1e-4);
	if(!OptProblem::reoptimize(OptProblem::primalDualRestart)) {
	  smoothing_status=-1;
	}
      }
      hist_iter.push_back(number_of_iterations());

      // objective value
      hist_obj.push_back(this->obj_value);
      f = this->obj_value;
#ifdef BE_VERBOSE
      if(obj_value>100)
	print_objterms_evals();
#endif
    }

    bool fixVoltage=true;

    if(false && attempt_fixing_for_PVPQ(smoothing_continu.back(), fixVoltage, 
			       variable("v_n", *data_K[0]), variable("q_g", *data_K[0]))) {
      
      printf("--------------------------------\n");
      update_AGC_smoothing_param(1e-6);
      //primal_problem_changed();
      OptProblem::reoptimize(OptProblem::primalDualRestart);
      hist_obj.push_back(this->obj_value);
      hist_iter.push_back(number_of_iterations());

      //print_PVPQ_info(*data_K[0], vn0);
      do_fixing_for_PVPQ(smoothing_continu.back(), fixVoltage, 
			 variable("v_n", *data_K[0]), variable("q_g", *data_K[0]));
    } 



    if(false) {
    do_fixing_for_PVPQ(smoothing_continu.back(), fixVoltage, 
		       variable("v_n", *data_K[0]), variable("q_g", *data_K[0]));

    update_AGC_smoothing_param(1e-9);
    //primal_problem_changed();
    OptProblem::reoptimize(OptProblem::primalDualRestart);
    hist_obj.push_back(this->obj_value);
    hist_iter.push_back(number_of_iterations());

    }
#ifdef BE_VERBOSE
      if(obj_value>100)
	print_objterms_evals();
#endif

    //print_PVPQ_info(*data_K[0], vn0);
      print_p_g_with_coupling_info(*data_K[0], pg0);
    tmrec.stop();
#ifdef BE_VERBOSE
    string sit = "["; for(auto iter:  hist_iter) sit += to_string(iter)+'/'; sit[sit.size()-1] = ']';
    string sobj="["; for(auto obj: hist_obj) sobj += to_string(obj)+'/'; sobj[sobj.size()-1]=']';
    printf("ContProbWithFixing K_idx=%d: optimize took %g sec - iters %s objs %s on rank=%d\n", 
	   K_idx, tmrec.getElapsedTime(), sit.c_str(), sobj.c_str(), my_rank);
    fflush(stdout);
#endif

    //print_summary();

    return true;

  }


  bool ContingencyProblemWithFixing::default_assembly(OptVariablesBlock* pg0, OptVariablesBlock* vn0) 
  {
    printf("ContProbWithFixing K_idx=%d: IDOut=%d outidx=%d Type=%s\n",
	   K_idx, data_sc.K_IDout[K_idx], data_sc.K_outidx[K_idx],
	   data_sc.cont_type_string(K_idx).c_str());
    fflush(stdout);

    p_g0=pg0; v_n0=vn0;

    assert(data_K.size()==1);
    SCACOPFData& dK = *data_K[0];
    useQPen = true;

    ////////////////////////////////////////////////////////////
    // setup for indexes used in non-anticip and AGC coupling 
    ////////////////////////////////////////////////////////////
    //indexes in data_sc.G_Generator; exclude 'outidx' if K_idx is a generator contingency
    data_sc.get_AGC_participation(K_idx, Gk, pg0_partic_idxs, pg0_nonpartic_idxs);
    assert(pg0->n == Gk.size() || pg0->n == 1+Gk.size());

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


    add_variables(dK,false);

    if(!warm_start_variable_from_basecase(*vars_primal)) {
      assert(false);
      return false;
    }

    add_cons_lines_pf(dK);
    add_cons_transformers_pf(dK);
    add_cons_active_powbal(dK);
    add_cons_reactive_powbal(dK);
    bool SysCond_BaseCase = false;
    add_cons_thermal_li_lims(dK,SysCond_BaseCase);
    add_cons_thermal_ti_lims(dK,SysCond_BaseCase);

    add_cons_nonanticip_using(pg0);
    //add_cons_AGC_using(pg0);
    add_cons_AGC_simplified(dK, pg0_partic_idxs, pgK_partic_idxs, pg0);

    add_const_nonanticip_v_n_using(vn0, Gk);
    //add_cons_PVPQ_using(vn0, Gk);

    //depending on reg_vn, reg_thetan, reg_bs, reg_pg, and reg_qg
    add_regularizations();


    if(NULL==vars_duals_bounds_L || NULL==vars_duals_bounds_U || NULL==vars_duals_cons) {
      //force allocation of duals
      dual_problem_changed();
    }
    if(!warm_start_variable_from_basecase(*vars_duals_bounds_L)) return false;
    if(!warm_start_variable_from_basecase(*vars_duals_bounds_U)) return false;
    if(!warm_start_variable_from_basecase(*vars_duals_cons)) return false;
    return true;
  }

  bool ContingencyProblemWithFixing::warm_start_variable_from_basecase(OptVariables& v)
  {
    SCACOPFData& dK = *data_K[0];
    for(auto& b : v.vblocks) {
      
      size_t pos = b->id.find_last_of("_");
      if(pos == string::npos) { 
	assert(false);
	b->providesStartingPoint = false; 
	continue; 
      }
      const string b0_name = b->id.substr(0, pos+1) + "0";
      auto b0p = dict_basecase_vars.find(b0_name);
      if(b0p == dict_basecase_vars.end()) {
	assert(b->id.find("delta") != string::npos || 
	       b->id.find("AGC") != string::npos); //!remove agc later
	continue;
      }

      auto b0 = b0p->second; assert(b0);
      if(b0->n == b->n) {
	b->set_start_to(*b0);
      } else {
	assert(b0->n - 1 == b->n);
	if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
	  assert(b->id.find("_g_") != string::npos);
	  for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	    assert(pgK_nonpartic_idxs[i] < b->n);
	    assert(pg0_nonpartic_idxs[i] < b0->n);
	    b->x[pgK_nonpartic_idxs[i]] = b0->x[pg0_nonpartic_idxs[i]];
	  }

	  for(int i=0; i<pg0_partic_idxs.size(); i++) {
	    assert(pgK_partic_idxs[i] < b->n);
	    assert(pg0_partic_idxs[i] < b0->n);
	    b->x[pgK_partic_idxs[i]] = b0->x[pg0_partic_idxs[i]];
	  }

	} else if(dK.K_ConType[0] == SCACOPFData::kLine) {
	  assert(b->id.find("_li") != string::npos);
	  int i=0, i0=0;
	  for(; i0<b0->n; i0++) {
	    if(i0 != dK.K_outidx[0]) {
	      b->x[i] = b0->x[i0];
	      i++;
	    }
	  }
	  assert(i0 == b0->n);
	  assert(i  == b->n);
	  
	} else if(dK.K_ConType[0] == SCACOPFData::kTransformer) {
	  assert(b->id.find("_ti") != string::npos || b->id.find("_trans_") != string::npos);
	  int i=0, i0=0;
	  for(; i0<b->n; i0++) {
	    if(i0 != dK.K_outidx[0]) {
	      b->x[i] = b0->x[i0];
	      i0++;
	    }
	  }
	  assert(i0 == b0->n);
	  assert(i  == b->n);
	}
      }
    }
    return true;
  }
  bool ContingencyProblemWithFixing::set_warm_start_from_basecase()
  {
    assert(false && "do not use");
    SCACOPFData& dK = *data_K[0];
if(false) {
    

    if(!warm_start_variable_from_basecase(*vars_primal)) return false;

    if(NULL==vars_duals_bounds_L || NULL==vars_duals_bounds_U || NULL==vars_duals_cons) {
      //force allocation of duals
      dual_problem_changed();
    }
    if(!warm_start_variable_from_basecase(*vars_duals_bounds_L)) return false;
    if(!warm_start_variable_from_basecase(*vars_duals_bounds_U)) return false;
    if(!warm_start_variable_from_basecase(*vars_duals_cons)) return false;
    return true;

} else {
    variable("v_n", dK)->set_start_to(*v_n0);
    variable("theta_n", dK)->set_start_to(*theta_n0);
    variable("b_s", dK)->set_start_to(*b_s0);

    if(dK.K_ConType[0] == SCACOPFData::kGenerator) {
      auto p_gK = variable("p_g", dK);
      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	p_gK->x[pgK_nonpartic_idxs[i]] = p_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	p_gK->x[pgK_partic_idxs[i]] = p_g0->x[pg0_partic_idxs[i]];
      }
      p_gK->providesStartingPoint = true;
      
      auto q_gK = variable("q_g", dK);
      for(int i=0; i<pg0_nonpartic_idxs.size(); i++) {
	q_gK->x[pgK_nonpartic_idxs[i]] = q_g0->x[pg0_nonpartic_idxs[i]];
      }
      for(int i=0; i<pg0_partic_idxs.size(); i++) {
	q_gK->x[pgK_partic_idxs[i]] = q_g0->x[pg0_partic_idxs[i]];
      }
      q_gK->providesStartingPoint = true;
      
    } else {
#ifdef DEBUG
      assert(variable("p_g", dK)->n == p_g0->n);
      assert(variable("q_g", dK)->n == q_g0->n);
#endif
      variable("p_g", dK)->set_start_to(*p_g0);
      variable("q_g", dK)->set_start_to(*q_g0);
    }
}

  }

  bool ContingencyProblemWithFixing::
  add_cons_AGC_simplified(SCACOPFData& dB, 
			  const std::vector<int>& idxs_pg0_AGC_particip, 
			  const std::vector<int>& idxs_pgK_AGC_particip,
			  OptVariablesBlock* pg0)
  {
    assert(idxs_pg0_AGC_particip.size()==idxs_pgK_AGC_particip.size());
    assert(variable("p_g", dB));

    if(idxs_pg0_AGC_particip.size()==0) {
      printf("[warning] ContingencyProblemWithFixing: add_cons_AGC_simplified: NO gens participating !?! in contingency %d\n", dB.id);
      return true;
    }

    OptVariablesBlock* deltaK = new OptVariablesBlock(1, var_name("delta", dB));
    append_variables(deltaK);
    deltaK->set_start_to(0.);

    auto cons = new AGCSimpleCons_pg0Fixed(con_name("AGC_simple_fixedpg0", dB), idxs_pg0_AGC_particip.size(), 
					   pg0, variable("p_g", dB), deltaK, 
					   idxs_pg0_AGC_particip, idxs_pgK_AGC_particip, 
					   data_sc.G_alpha);

    append_constraints(cons);

    return true;
  }
} //end of namespace

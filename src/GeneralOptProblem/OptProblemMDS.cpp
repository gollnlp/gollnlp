#include "OptProblemMDS.hpp"

#include "HiopSolver.hpp"

namespace gollnlp {


  bool OptProblemMDS::eval_Jaccons_eq(const double* x, bool new_x, 
				      const int& nxsparse, const int& nxdense,
				      const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
				      double** JacD)
  {
    if(new_x) {
      new_x_fgradf=true;
      vars_primal->attach_to(x);
    }
    // if(MJacS==NULL) {
    //   for(auto& con_gen: cons->vblocks) {
	
    // 	OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
    // 	assert(NULL!=con);
    // 	if(!con) continue;
	
    // 	if(!con->eval_Jac_eq(*vars_primal, new_x,
    // 			     nxsparse, nxdense,
    // 			     nnzJacS, iJacS, jJacS, MJacS,
    // 			     JacD)) {
    // 	  assert(false && "eval_Jaccons should be called after get_nnzJaccons");
    // 	}
    //   }
    //   return true;
    // }
    
    //M!=NULL > just fill in the values
    if(MJacS)
      for(int i=0; i<nnzJacS; ++i) MJacS[i]=0.;
    if(JacD)
      for(int i=0; i<nxdense*cons->m(); ++i) JacD[0][i]=0.;
  
    for(auto& con_gen: cons->vblocks) {
      
      OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
      assert(NULL!=con);
      if(!con) continue;
      
      if(!con->eval_Jac_eq(*vars_primal, new_x,
			   nxsparse, nxdense,
			   nnzJacS, iJacS, jJacS, MJacS,
			   JacD)) {
	return false;
      }
      //print_spmat_triplet(nnzJacS, cons->m(), nxsparse, iJacS, jJacS, MJacS, "Jac_eq");

    }
    return true;
  }

  bool OptProblemMDS::eval_HessLagr(const double* x, bool new_x, 
				    const double& obj_factor, 
				    const double* lambda, bool new_lambda, 
				    const int& nxsparse, const int& nxdense, 
				    const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
				    double** HDD,
				    const int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    if(new_x) {
      new_x_fgradf=true; 
      vars_primal->attach_to(x);
    }
    if(new_lambda) vars_duals_cons->attach_to(lambda);
      
    if(NULL==MHSS) {
      //
      // Objective terms
      //
      for(auto& ot_gen: obj->vterms) {

	OptObjectiveTermMDS* ot = dynamic_cast<OptObjectiveTermMDS*>(ot_gen);
	if(NULL==ot) {

	  if(!ot_gen->eval_HessLagr(*vars_primal, new_x, obj_factor,
				    nnzHSS, iHSS, jHSS, MHSS)) {
	    assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
	  }
	} else {
	  if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, 
				nxsparse, nxdense,
				nnzHSS, iHSS, jHSS, MHSS,
				HDD,
				nnzHSD, iHSD, jHSD, MHSD)) {
	    assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
	  }
	}
      }
      //
      // constraints
      //
      for(auto& con_gen: cons->vblocks) {
	OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	if(NULL==con) {
	  assert(false && "check this");
	  continue;
	}
	if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, 
			       nxsparse, nxdense,
			       nnzHSS, iHSS, jHSS, MHSS,
			       HDD,
			       nnzHSD, iHSD, jHSD, MHSD)) {
	  assert(false && "eval_HessLagr should be called after get_nnzHessLagr");
	}
      }
    } else {
      // case of M!=NULL > just fill in the values
      for(int it=0; it<nnzHSS; ++it) MHSS[it]=0.;
      for(int it=0; it<nxdense*nxdense; ++it) HDD[0][it] = 0.;
      assert(nnzHSD==0);
	
      for(auto& ot_gen: obj->vterms) {
	  
	OptObjectiveTermMDS* ot = dynamic_cast<OptObjectiveTermMDS*>(ot_gen);
	if(NULL==ot) {
	  //this is a general 'OptObjectiveTerm' which is sparse and contributes only to the
	  //sparse Hessian
	  if(!ot_gen->eval_HessLagr(*vars_primal, new_x, obj_factor, nnzHSS, iHSS, jHSS, MHSS)) {
	    return false;
	  }
	} else {
	  
	  if(!ot->eval_HessLagr(*vars_primal, new_x, obj_factor, 
				nxsparse, nxdense,
				nnzHSS, iHSS, jHSS, MHSS,
				HDD,
				nnzHSD, iHSD, jHSD, MHSD))
	    return false;
	}
      }
	
      for(auto& con_gen: cons->vblocks) {
	OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	if(NULL==con) {
	  assert(false && "check this");
	  continue;
	}

	if(!con->eval_HessLagr(*vars_primal, new_x, *vars_duals_cons, new_lambda, 
			       nxsparse, nxdense,
			       nnzHSS, iHSS, jHSS, MHSS,
			       HDD,
			       nnzHSD, iHSD, jHSD, MHSD))
	  return false;
      }
    }
    return true;
  } // end of eval_HessLagr

  int OptProblemMDS::compute_num_variables_sparse() const
  {
    int nsparse=0;
    for(auto& var_block: vars_primal->vblocks)
      if(var_block->sparseBlock)
	nsparse += var_block->n;
    return nsparse;
  }
  int OptProblemMDS::compute_num_variables_dense() const
  {
    int ndense=0;
    for(auto& var_block: vars_primal->vblocks)
      if(false==var_block->sparseBlock)
	ndense += var_block->n;
    return ndense;
  }
  bool OptProblemMDS::compute_num_variables_dense_sparse(int& ndense, int& nsparse) const
  {
    ndense = nsparse = 0;
    for(auto& var_block: vars_primal->vblocks)
      if(var_block->sparseBlock)
	nsparse += var_block->n;
      else
	ndense += var_block->n;
    return true;
  }

  int OptProblemMDS::compute_nnzJac_eq()
  {
    if(nnz_Jac_eq<0) {
      //goTimer tm; tm.start();
      
      for(auto& con_gen: cons->vblocks) {
	
	OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
	if(NULL==con) {
	  assert(false && "check this: incorrect/unsupported type for constraints");
	  continue;
	}
	con->get_spJacob_eq_ij(ij_Jac_eq);
      }
      nnz_Jac_eq = uniquely_indexise_spTripletIdxs(ij_Jac_eq);
      
      //tm.stop();
      //printf("Jacobian structure took %g sec\n", tm.getElapsedTime());
    }
    return nnz_Jac_eq;
  }

  int OptProblemMDS::compute_nnzHessLagr_SSblock()
  {
    if(nnz_HessLagr_SSblock>=0)
      return nnz_HessLagr_SSblock;

    for(auto& ot_gen: obj->vterms) {
      OptObjectiveTermMDS* ot = dynamic_cast<OptObjectiveTermMDS*>(ot_gen);
      if(NULL==ot) {
	//treat general objective terms (which are sparse) as sparse blocks in the MDS
	ot_gen->get_HessLagr_ij(ij_HessLagr_SSblock);
      } else {
	ot->get_HessLagr_SSblock_ij(ij_HessLagr_SSblock);
      }
      
#ifdef DEBUG
      if(false==check_is_upper(ij_HessLagr_SSblock)) {
	printf("[Warning] Objective term %s returned nonzero elements in the lower triangular "
	       "part of the Hessian.", ot->id.c_str());
	//assert(false);
      }
#endif
    }
    
    for(auto& con_gen: cons->vblocks) {
      OptConstraintsBlockMDS* con = dynamic_cast<OptConstraintsBlockMDS*>(con_gen);
      if(NULL==con) {
	assert(false && "check this");
	continue;
      }
      con->get_HessLagr_SSblock_ij(ij_HessLagr_SSblock);
#ifdef DEBUG
      if(false==check_is_upper(ij_HessLagr_SSblock)) {
	printf("[Warning] Constraint term %s returned nonzero elements in the lower triangular "
	       "part of the Hessian.", con->id.c_str());
	//assert(false);
      }
#endif      
    }
    nnz_HessLagr_SSblock = uniquely_indexise_spTripletIdxs(ij_HessLagr_SSblock);
    return nnz_HessLagr_SSblock;
  }
  
  void OptProblemMDS::use_nlp_solver(const std::string& name)
  {
    if(NULL == nlp_solver) {
      if(gollnlp::tolower(name) == "ipopt") {
	printf("Ipopt called for a MDS formulation: this is not recommended since the "
	       "IpoptSolver for MDS is for testing purposes only; use HiopSolver instead\n");

	nlp_solver = new IpoptSolver_HiopMDS(this);
	nlp_solver->initialize();
      } else {
	assert(gollnlp::tolower(name) == "hiop");
	nlp_solver = new HiopSolverMDS(this);
	nlp_solver->initialize();
      }
    }
  }
  
} //end namespace

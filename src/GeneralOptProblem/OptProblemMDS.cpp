#include "OptProblemMDS.hpp"

#include "HiopSolver.hpp"

namespace gollnlp {

  void OptProblemMDS::use_nlp_solver(const std::string& name)
  {
    if(NULL == nlp_solver) {
      if(gollnlp::tolower(name) == "ipopt") {
	assert(false && "no Ipopt solver class for OptProblemMDS is available");
	//nlp_solver = new IpoptSolver(this);
	//nlp_solver->initialize();
      } else {
	assert(gollnlp::tolower(name) == "hiop");
	nlp_solver = new HiopSolverMDS(this);
	nlp_solver->initialize();
      }
    }
  }
  
} //end namespace

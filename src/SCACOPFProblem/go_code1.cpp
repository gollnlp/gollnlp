#include "go_code1.hpp"

#include "ACOPFProblem.hpp"
#include "goTimer.hpp"

using namespace std;
using namespace gollnlp;

int myexe1_function(const std::string& InFile1, const std::string& InFile2,
		    const std::string& InFile3, const std::string& InFile4,
		    double TimeLimitInSeconds, 
		    int ScoringMethod, 
		    const std::string& NetworkModel)
{
  goTimer ttot; ttot.start();
  
  SCACOPFData d;
  d.readinstance(InFile1, InFile2, InFile3, InFile4);

  ACOPFProblem ac_prob(d);
  ac_prob.default_assembly();

  ttot.stop();
  printf("MyExe1 took %g sec.\n", ttot.getElapsedTime());
  return 0;
}

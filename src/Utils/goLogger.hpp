#ifndef GOLLNLP_LOGGER
#define GOLLNLP_LOGGER

#include <cstdio>
#include <cstdarg>

#include "mpi.h"

namespace gollnlp
{
class goOptions;

/* Verbosity 0 to 9 */
enum goOutVerbosity {
  hovError=-1,
  hovVerySilent=0,
  hovWarning=1,
  hovNoOutput=2,
  hovSummary=3, //summary of the problem and each iteration
  hovScalars=4, //additional, usually scalars, such as norm of resids, nlp and log bar errors, etc
  hovFcnEval=5, //the above plus info about the number of function, gradient and Hessians
  hovLinesearch=6, //linesearch info
  hovLinAlgScalars=7, //print out various scalars: e.g., linear systems residuals
  hovLinesearchVerb=8, //linesearch with more output
  hovLinAlgScalarsVerb=9, //additional scalars, e.g., BFGS updating info
  hovIteration=10, //print out iteration
  hovMatrices=11,
  hovMaxVerbose=12
};

class goLogger
{
public:
  goLogger(FILE* f, int masterrank=0) 
    : _f(f), _master_rank(masterrank), _my_rank(-1) {};
  virtual ~goLogger() {};
  /* outputs a vector. loggerid indicates which logger should be used, by default stdout*/
  void write(const char* msg, const goOptions& options,     goOutVerbosity v, int loggerid=0);
  void write(const char* msg, goOutVerbosity v, int loggerid=0);

  //only for loggerid=0 for now
  void printf(goOutVerbosity v, const char* format, ...); 

  /* This static method is to be used before NLP created its internal instance of goLogger. To be
   * used for displaying errors (on stderr) that occur during initialization of the NLP class 
   */
  static void printf_error(goOutVerbosity v, const char* format, ...); 

protected:
  FILE* _f;
  char _buff[1024];
private:
  inline int get_my_rank() {

    if(_my_rank<0) {
      int ierr = MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank);
      if(ierr!=MPI_SUCCESS) {
	_my_rank = -1;
	return 0;
      } else {
	return _my_rank;
      }
    }
    return _my_rank;
  }
  int _master_rank;
  int _my_rank;
};
}
#endif

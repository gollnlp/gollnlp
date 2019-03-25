#ifndef GOLLNLP_TIMER
#define GOLLNLP_TIMER

#ifdef GOLLNLP_USE_MPI
#include "mpi.h"
#else
#include <sys/time.h>
#endif

#include <cassert>

//to do: sys time: getrusage(RUSAGE_SELF,&usage);

namespace gollnlp
{

class goTimer
{
public:
  goTimer() : tmElapsed(0.0), tmStart(0.0) {};

  //returns the elapsed time (accumulated between start/stop) in seconds
  inline double getElapsedTime() const { return tmElapsed; }

  inline void start() 
  {
#ifdef GOLLNLP_USE_MPI 
    tmStart = MPI_Wtime();
#else
    gettimeofday(&tv, NULL);
    tmStart = ( static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec)/1000000.0 );
#endif
  }

  inline void stop()
  {
#ifdef GOLLNLP_USE_MPI
    tmElapsed += ( MPI_Wtime()-tmStart );
#else
    gettimeofday(&tv, NULL);
    tmElapsed += ( static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec)/1000000.0  - tmStart );
#endif
  }

  inline void reset() {
    tmElapsed=0.0; tmStart=0.0;
  }

  inline goTimer& operator=(const double& zero) {
    assert(0==zero);
    this->reset(); 
    return *this;
  }
private:
  double tmElapsed; //in seconds
  double tmStart;

#ifdef GOLLNLP_USE_MPI
#else
  struct timeval tv;
#endif
};
}
#endif

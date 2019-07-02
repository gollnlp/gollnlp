#include "goLogger.hpp"

#include "goOptions.hpp"
#include <cstring>

#define GOLLNLP_USE_MPI 1

namespace gollnlp
{

void goLogger::write(const char* msg, goOutVerbosity v, int loggerid/*=0*/) 
{ 
#ifdef GOLLNLP_USE_MPI
  if(_master_rank != get_my_rank()) return;
#endif
  goOutVerbosity _verb = hovSummary; // = (goOutVerbosity) _nlp->options->GetInteger("verbosity_level");
  if(v>_verb) return;
  fprintf(_f, "%s\n", msg); 
}


void goLogger::write(const char* msg, const goOptions& options,     goOutVerbosity v, int loggerid/*=0*/)
{
#ifdef GOLLNLP_USE_MPI
  if(_master_rank != get_my_rank()) return;//if(_master_rank != _nlp->get_rank()) return;
#endif
  goOutVerbosity _verb = hovSummary; //(goOutVerbosity) _nlp->options->GetInteger("verbosity_level");
  if(v>_verb) return;
  options.print(_f, msg);
}



  //only for loggerid=0 for now
void goLogger::printf(goOutVerbosity v, const char* format, ...)
{
#ifdef GOLLNLP_USE_MPI
  if(_master_rank != get_my_rank()) return;//if(_master_rank != _nlp->get_rank()) return;
#endif
  goOutVerbosity _verb = hovSummary; // = (goOutVerbosity) _nlp->options->GetInteger("verbosity_level");
  if(v>_verb) return;

  char label[16];label[0]='\0';
  if(v==hovError) strcpy(label, "[Error] ");
  else if(v==hovWarning) strcpy(label, "[Warning] ");
  fprintf(_f, "%s", label);

  va_list args;
  va_start (args, format);
  vsprintf (_buff,format, args);
  fprintf(_f,"%s",_buff);
  va_end (args);

};

void goLogger::printf_error(goOutVerbosity v, const char* format, ...)
{
  char buff[1024];
  va_list args;
  va_start (args, format);
  vsprintf (buff,format, args);
  fprintf(stderr,"%s",buff);
  va_end (args);
};

};

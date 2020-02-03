#include "goSignalHandling.hpp"

#include <string.h>
#include <string>
#include <assert.h>
#include <unistd.h>

#ifdef GOLLNLP_FAULT_HANDLING
//extern "C" void gollnlp_fault_handler(int nsignum, siginfo_t* si,void* vcontext)

#define MSG_MAX_SZ 128 
static char msg_fault[MSG_MAX_SZ];
static int sz_msg_fault = 0;

// handler function for selected faults
// !!! use only "safe" functions in this function
extern "C" void gollnlp_fault_handler(int nsignum)
{
  size_t dummy = write(2, msg_fault, sz_msg_fault);
  sleep(30000);
  sleep(30000);
}

void set_fault_signal_message(const char * msg) 
{
  strncpy(msg_fault, msg, MSG_MAX_SZ-2);
  msg_fault[MSG_MAX_SZ-2] = '\n';
  msg_fault[MSG_MAX_SZ-1] = '\0';
  sz_msg_fault = strlen(msg_fault);
}

void enable_fault_signal_handling(void (*handler)(int))
{
  //doc on sigaction 
  //https://www.ibm.com/support/knowledgecenter/en/SSLTBW_2.1.0/com.ibm.zos.v2r1.bpxbd00/rtsigac.htm
  struct sigaction new_action, old_action;

  int signal;
  std::string msg;
  
  new_action.sa_handler = handler;
  sigemptyset (&new_action.sa_mask);
  new_action.sa_flags = 0;

  signal = SIGABRT; //Abnormal termination (sent by abort())
  sigaction (signal, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
    sigaction (signal, &new_action, NULL);
  else 
    printf("[warning] SIGABRT is set to ignore -- handler was not set for it\n");

  signal = SIGFPE; //Arithmetic exceptions that are not masked, for example, overflow, division by zero, and incorrect operation.
  sigaction (signal, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
    sigaction (signal, &new_action, NULL);
  else 
    printf("[warning] SIGPFE is set to ignore -- handler was not set for it\n");

  
  signal = SIGBUS; //Bus error (available only when running on MVS™ 5.2 or higher).
  sigaction (signal, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
    sigaction (signal, &new_action, NULL);
  else 
    printf("[warning] SIGBUS is set to ignore -- handler was not set for it\n");

  signal = SIGSEGV; //Incorrect access to memory
  sigaction (signal, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
    sigaction (signal, &new_action, NULL);
  else 
    printf("[warning] SIGSEGV is set to ignore -- handler was not set for it\n");

  //do nothing about this since we want the job to be terminated
  signal = SIGTERM; //Termination request sent to the program

  signal = SIGFPE; //Arithmetic exceptions that are not masked, for example, overflow, division by zero, and incorrect operation.
  sigaction (signal, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN)
    sigaction (signal, &new_action, NULL);
  else 
    printf("[warning] SIGPFE is set to ignore -- handler was not set for it\n");


  //different handler for virtual timers
  //signal = SIGVTALRM;
}

//
// timer - alarm signals
//
// handler defined in "locally" in the cpp file(s)
// set_timer_message also defined locally

//docs on setjmp/longjmp
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/longjmp.html#
// https://stackoverflow.com/questions/38842951/why-is-sigalrm-not-working-second-time/38843103#38843103
// https://stackoverflow.com/questions/20647808/c-unix-siglongjmp-and-sigsetjmp

// void enable_timer_handling(void (*handler)(int))
// {
//   struct sigaction new_action, old_action;
//   int signal = SIGALRM;
//   sigemptyset (&new_action.sa_mask);
//   new_action.sa_handler = handler;
//   new_action.sa_flags = 0;

//   sigaction(signal, NULL, &old_action);
//   if(old_action.sa_handler != SIG_IGN)
//     sigaction(signal, &new_action, NULL);
//   else
//     printf("[warning] SIGALRM is set to ignore -- alarm handler was not set for it\n");
// }

#endif

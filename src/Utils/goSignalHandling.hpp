#ifndef GO_SIGNALS_HANDLING
#define GO_SIGNALS_HANDLING

#ifdef GOLLNLP_FAULT_HANDLING
#include <signal.h>
#include <setjmp.h>

void set_fault_signal_message(const char * msg);
void enable_fault_signal_handling(void (*handler)(int));
extern "C" void gollnlp_fault_handler(int nsignum);


void set_timer_message(const char* msg);
void enable_timer_handling(int seconds, void (*handler)(int));
extern "C" void gollnlp_timer_handler(int nsignum);
extern jmp_buf jmpbuf_K_solve;

#endif


#endif

#ifndef SCACOPF_IO
#define SCACOPF_IO

#include "SCACOPFProblem.hpp"

#include "SCACOPFData.hpp"

#include <cstring>

namespace gollnlp {
  
  class SCACOPFIO 
  {
  public:
    static 
    void write_append_solution_block(const double* v_n, const double* theta_n, const double* b_s,
				     const double* p_g, const double* q_g,
				     SCACOPFData& data,
				     const std::string& filename="solution2.txt",
				     const std::string& fileopenflags="a+");
    static 
    void write_append_solution_block(OptVariablesBlock* v_n, OptVariablesBlock* theta_n, 
				     OptVariablesBlock* b_s,
				     OptVariablesBlock* p_g, OptVariablesBlock* q_g,
				     SCACOPFData& data,
				     const std::string& filename="solution2.txt",
				     const std::string& fileopenflags="a+")
    {
      write_append_solution_block(v_n->x, theta_n->x, b_s->x, p_g->x, q_g->x,
				  data, filename, fileopenflags);
    }

    static
    void read_solution1(OptVariablesBlock** v_n, OptVariablesBlock** theta_n, OptVariablesBlock** b_s,
			OptVariablesBlock** p_g, OptVariablesBlock** q_g,
			SCACOPFData& data,
			const std::string& filename="solution1.txt");

  private:
    static 
    bool read_solution1(std::vector<int>& I_n,  std::vector<double>& v_n, 
			std::vector<double>& theta_n, std::vector<double>& b_n,
			std::vector<int>& I_g, std::vector<std::string>& ID_g,
			std::vector<double>& p_g, std::vector<double>& q_g,
			const std::string& filename="solution1.txt");

    static std::vector<int> SSh_Nidx;
    static std::vector<int> gmap;
    static std::vector<double> bcsn;

  }; // end of SCACOPFIO

} //end namespace

#endif

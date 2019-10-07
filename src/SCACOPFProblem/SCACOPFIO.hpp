#ifndef SCACOPF_IO
#define SCACOPF_IO

#include "SCACOPFProblem.hpp"
#include "SCACOPFData.hpp"

#include <unordered_map>

#include <cstring>

namespace gollnlp {
  
  class SCACOPFIO 
  {
  public:

    static
    void write_solution1(const double* v_n, const double* theta_n, const double* b_s,
			 const double* p_g, const double* q_g,
			 SCACOPFData& data,
			 const std::string& filename="solution1.txt")
    {
      write_append_solution_block(v_n, theta_n, b_s, p_g, q_g, data, filename, "w");
    }

    static
    void write_solution2_block(int Kidx,
			       const double* v_n, const double* theta_n, const double* b_s,
			       const double* p_g, const double* q_g, const double& delta,
			       SCACOPFData& data,
			       const std::string& filename="solution2.txt", 
			       bool open_file=true, bool close_file=true)
    {
      std::string fileopenflags = sol2_write_1st_call==true ? "w" : "a+";
      sol2_write_1st_call=false;

      FILE* file;
      if(open_file) {
	assert(sol2_file==NULL);
	file=fopen(filename.c_str(), fileopenflags.c_str());
      }
      else {
	assert(sol2_file!=NULL);
	file=sol2_file;
      }
      
      if(NULL==file) {
	printf("[warning] could not open [%s] file for writing (flags '%s')\n", 
	       filename.c_str(), fileopenflags.c_str());
	return;
      }
      fprintf(file, "--contingency\nlabel\n");
      fprintf(file, "\'%s\'\n", data.K_Label[Kidx].c_str());

      write_append_solution_block(v_n, theta_n, b_s, p_g, q_g, data, filename, fileopenflags, file);

      fprintf(file, "--delta section\ndelta(MW)\n%g\n", data.MVAbase*delta);
      if(close_file) {
	fclose(file);
	sol2_file = NULL;
      } else {
	sol2_file = file;
      }
      printf("sol2--\n");
    }

    static
    void write_variable_block(OptVariablesBlock* var, SCACOPFData& data, FILE* file);
    

    //will read from "solution_b_pd.txt"
    static
    void read_variables_blocks(SCACOPFData& data, 
			       std::unordered_map<std::string, OptVariablesBlock*>& map_basecase_vars);
    // static 
    // void write_append_solution_block(OptVariablesBlock* v_n, OptVariablesBlock* theta_n, 
    // 				     OptVariablesBlock* b_s,
    // 				     OptVariablesBlock* p_g, OptVariablesBlock* q_g,
    // 				     SCACOPFData& data,
    // 				     const std::string& filename="solution2.txt",
    // 				     const std::string& fileopenflags="a+")
    // {
    //   write_append_solution_block(v_n->x, theta_n->x, b_s->x, p_g->x, q_g->x,
    // 				  data, filename, fileopenflags);
    // }

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
    static 
    void write_append_solution_block(const double* v_n, const double* theta_n, const double* b_s,
				     const double* p_g, const double* q_g,
				     SCACOPFData& data,
				     const std::string& filename="solution2.txt",
				     const std::string& fileopenflags="a+", FILE* f=NULL);
    static std::vector<int> SSh_Nidx;
    static std::vector<int> gmap;
    static std::vector<double> bcsn;
    static bool sol2_write_1st_call;
    static FILE* sol2_file;
  }; // end of SCACOPFIO

} //end namespace

#endif

#ifndef SCACOPF_UTILS_GOLLNLP
#define SCACOPF_UTILS_GOLLNLP

namespace gollnlp
{
  //variables and constraints accessers
  inline std::string var_name(const std::string& prefix, int Kid) { 
    return prefix+"_"+std::to_string(Kid); 
  }
  inline std::string var_name(const std::string& prefix, const SCACOPFData& d) { 
    return var_name(prefix, d.id); 
  }
  inline std::string con_name(const std::string& prefix, int Kid) { 
    return prefix+"_"+std::to_string(Kid); 
  }
  inline std::string con_name(const std::string& prefix, const SCACOPFData& d) { 
    return con_name(prefix, d.id);
  }
  inline std::string objterm_name(const std::string& prefix, const SCACOPFData& d) { 
    return prefix+"_"+std::to_string(d.id); 
  }
}

#endif

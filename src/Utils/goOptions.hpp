#ifndef GOLLNLP_OPTIONS
#define GOLLNLP_OPTIONS

#include "goLogger.hpp"

#include <string>
#include <vector>
#include <map>

namespace gollnlp
{
class goLogger;
 
class goOptions
{
public:
  goOptions(const char* szOptionsFilename=NULL);
  virtual ~goOptions();

  //Seters for options values that should be self explanatory with the exception of the last parameter.
  //
  //Passing 'setFromFile' with non-default, 'true' value is for expert use-only. It indicates that the option 
  //value comes from the options file (go.options) and will overwrite any options set at runtime by the 
  //user's code. However, passing 'setFromFile' with 'true' during runtime is perfectly fine and will 
  //conviniently "overwrite the overwriting" file options  

  virtual bool SetNumericValue (const char* name, const double& value, const bool& setFromFile=false);
  virtual bool SetIntegerValue(const char* name, const int& value, const bool& setFromFile=false);
  virtual bool SetStringValue (const char* name,  const char* value, const bool& setFromFile=false);
  
  virtual double      GetNumeric(const char* name) const;
  virtual int         GetInteger(const char* name) const;
  virtual std::string GetString (const char* name) const;

  void SetLog(goLogger* log_) { log=log_; ensureConsistence(); }
  virtual void print(FILE* file, const char* msg=NULL) const;
protected:
  /* internal use only */

  void registerNumOption(const std::string& name, double defaultValue, double rangeLo, double rangeUp, const char* description);
  void registerIntOption(const std::string& name, int    defaultValue, int    rangeLo, int    rangeUp, const char* description);
  //void registerBooOption(const std::string& name, bool defaultValue);
  void registerStrOption(const std::string& name, const std::string& defaultValue, const std::vector<std::string>& range, const char* description);
  void registerOptions();

  //sets the (name, value) pair accordingly to the type registered in mOptions, or prints an warning message and leaves
  //the 'name' option to the default value. This method is for internal use.
  bool setNameValuePair(const std::string& name, const std::string& value);

  void loadFromFile(const char* szFilename);

  void ensureConsistence();

  void log_printf(goOutVerbosity v, const char* format, ...);

  struct _O { // option entry
    _O(const char* description) : descr(description), specifiedInFile(false) {};
    virtual ~_O() {};
    std::string descr;
    bool specifiedInFile;
    virtual void print(FILE* f) const =0;
  };
  struct _OInt : public _O { 
    _OInt(int    v, int    low,    int upp, const char* description) : _O(description), val(v), lb(low), ub(upp) {}; 
    int    val, lb, ub; 
    void print(FILE* f) const;
  };
  struct _ONum : public _O { 
    _ONum(double v, double low, double upp, const char* description) : _O(description), val(v), lb(low), ub(upp) {}; 
    double val, lb, ub; 
    void print(FILE* f) const;
  };

  struct _OStr : public _O { 
    _OStr(std::string v, const std::vector<std::string>& range_, const char* description) 
      : _O(description), val(v), range(range_) {};
    std::string val;
    std::vector<std::string> range;
    void print(FILE* f) const;
  };

  std::map<std::string, _O*> mOptions;

  goLogger* log;
};


} // ~namespace
#endif 

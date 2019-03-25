#include "goOptions.hpp"

#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <cstring>

namespace gollnlp
{

using namespace std;
const char* szDefaultFilename = "gollnlp.options";

goOptions::goOptions(const char* szOptionsFilename/*=NULL*/)
  : log(NULL)
{
  registerOptions();
  loadFromFile(szOptionsFilename==NULL?szDefaultFilename:szOptionsFilename);
  //ensureConsistence();
}

goOptions::~goOptions()
{
  map<std::string, _O*>::iterator it = mOptions.begin();
  for(;it!=mOptions.end(); it++) delete it->second;
}

double goOptions::GetNumeric(const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _ONum* option = dynamic_cast<_ONum*>(it->second);
  assert(option!=NULL);
  return option->val;
}

int goOptions::GetInteger(const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _OInt* option = dynamic_cast<_OInt*>(it->second);
  assert(option!=NULL);
  return option->val;
}

string goOptions::GetString (const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _OStr* option = dynamic_cast<_OStr*>(it->second);
  assert(option!=NULL);
  return option->val;
}

void goOptions::registerOptions()
{
  registerNumOption("mu0", 1., 1e-6, 1000., "Initial log-barrier parameter mu (default 1.)");
  registerNumOption("tolerance", 1e-8, 1e-14, 1e-1, "Absolute error tolerance for the NLP (default 1e-8)");
  registerNumOption("rel_tolerance", 0., 0., 0.1, "Error tolerance for the NLP relative to errors at the initial point. A null value disables this option (default 0.)");


  {
    vector<string> range(2); range[0]="lsq"; range[1]="linear";
    registerStrOption("dualsUpdateType", "lsq", range, "Type of update of the multipliers of the eq. cons. (default lsq)"); //
  }
  
  registerIntOption("max_iter", 3000, 1, 1e6, "Max number of iterations (default 3000)");
  
  registerNumOption("acceptable_tolerance", 1e-6, 1e-14, 1e-1, "Go will terminate if the NLP residuals are below for 'acceptable_iterations' many consecutive iterations (default 1e-6)");   
  registerIntOption("acceptable_iterations", 10, 1, 1e6, "Number of iterations of acceptable tolerance after which Go terminates (default 10)");
  
  
  registerIntOption("verbosity_level", 3, 0, 12, "Verbosity level: 0 no output (only errors), 1=0+warnings, 2=1 (reserved), 3=2+optimization output, 4=3+scalars; larger values explained in goLogger.hpp"); 

  
  registerNumOption("fixed_var_tolerance", 1e-15, 1e-30, 0.01, "A variable is considered fixed if |upp_bnd-low_bnd| < fixed_var_tolerance * max(abs(upp_bnd),1) (default 1e-15)");
  
  registerNumOption("fixed_var_perturb", 1e-8, 1e-14, 0.1, "Perturbation of the lower and upper bounds for fixed variables relative to its magnitude: lower/upper_bound -=/+= max(abs(upper_bound),1)*fixed_var_perturb (default 1e-8)");
}

void goOptions::registerNumOption(const std::string& name, double defaultValue, double low, double upp, const char* description)
{
  mOptions[name]=new _ONum(defaultValue, low, upp, description);
}

void goOptions::registerStrOption(const std::string& name, const std::string& defaultValue, const std::vector<std::string>& range, const char* description)
{
  mOptions[name]=new _OStr(defaultValue, range, description);
}

void goOptions::registerIntOption(const std::string& name, int    defaultValue, int low, int upp, const char* description)
{
  mOptions[name]=new _OInt(defaultValue, low, upp, description);
}

void goOptions::ensureConsistence()
{
  //check that the values of different options are consistent 
  //do not check is the values of a particular option is valid; this is done in the Set methods

  double eps_tol_accep = GetNumeric("acceptable_tolerance");
  double eps_tol  =      GetNumeric("tolerance");     
  if(eps_tol_accep < eps_tol) {
    log_printf(hovWarning, "There is no reason to set 'acceptable_tolerance' tighter than 'tolerance'. Will set the two to 'tolerance'.\n");
    SetNumericValue("acceptable_tolerance", eps_tol);
  }
}

static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), 
	  std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

void goOptions::loadFromFile(const char* filename)
{
  if(NULL==filename) { 
    log_printf(hovError, "Option file name not valid"); 
    return;
  }

  ifstream input( filename );

  if(input.fail()) 
    if(strcmp(szDefaultFilename, filename)) {
      log_printf(hovError, "Failed to read option file '%s'. Go will use default options.\n", filename);
      return;
    }

  string line; string name, value;
  for( std::string line; getline( input, line ); ) {

    line = ltrim(line);

    if(line.size()==0) continue;
    if(line[0]=='#') continue;

    istringstream iss(line);
    if(!(iss >> name >> value)) {
      log_printf(hovWarning, "Go could not parse and ignored line '%s' from the option file\n", line.c_str());
      continue;
    }
    
    //find the _O object in mOptions corresponding to 'optname' and set his value to 'optval'
    _ONum* on; _OInt* oi; _OStr* os;

    map<string, _O*>::iterator it = mOptions.find(name);
    if(it!=mOptions.end()) {
      _O* option = it->second;
      on = dynamic_cast<_ONum*>(option);
      if(on!=NULL) {
	stringstream ss(value); double val;
	if(ss>>val) { SetNumericValue(name.c_str(), val, true); }
	else 
	  log_printf(hovWarning, 
		      "Go could not parse value '%s' as double for option '%s' specified in the option file and will use default value '%g'\n", 
		      value.c_str(), name.c_str(), on->val);
      } else {
	os = dynamic_cast<_OStr*>(option);
	if(os!=NULL) {
	  SetStringValue(name.c_str(), value.c_str(), true);
	} else {
	  oi = dynamic_cast<_OInt*>(option);
	  if(oi!=NULL) {
	    stringstream ss(value); int val;
	    if(ss>>val) { SetIntegerValue(name.c_str(), val, true); }
	    else {
	      log_printf(hovWarning, 
			  "Go could not parse value '%s' as int for option '%s' specified in the option file and will use default value '%d'\n",
			  value.c_str(), name.c_str(), oi->val);
	    }
	  } else {
	    // not one of the expected types? Can't happen
	    assert(false);
	  }
	}
      }

    } else { // else from it!=mOptions.end()
      // option not recognized/found/registered
      log_printf(hovWarning, 
		  "Go does not understand option '%s' specified in the option file and will ignore its value '%s'.\n",
		  name.c_str(), value.c_str());
    }
  } //end of the for over the lines
}

bool goOptions::SetNumericValue (const char* name, const double& value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _ONum* option = dynamic_cast<_ONum*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning, 
		"Go does not know option '%s' as 'numeric'. Maybe it is an 'integer' or 'string' value? The option will be ignored.\n",
		name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning, 
		     "Go will ignore value '%g' set for option '%s' since this option is already specified in an option file.\n", value, name);
	  return true;
	}
      } 

      if(setFromFile)
	option->specifiedInFile=true;

      if(value<option->lb || value>option->ub) {
	log_printf(hovWarning, 
		    "Go: option '%s' must be in [%g,%g]. Default value %g will be used.\n",
		    name, option->lb, option->ub, option->val);
      } else option->val = value;
    }
  } else {
    log_printf(hovWarning, 
		"Go does not understand option '%s' and will ignore its value '%g'.\n",
		name, value);
  }
  return true;
}

bool goOptions::SetIntegerValue(const char* name, const int& value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OInt* option = dynamic_cast<_OInt*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning, 
		  "Go does not know option '%s' as 'integer'. Maybe it is an 'numeric' or a 'string' option? The option will be ignored.\n",
		  name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning, 
		     "Go will ignore value '%d' set for option '%s' since this option is already specified in an option file.\n", value, name);
	  return true;
	}
      } 

      if(setFromFile)
	option->specifiedInFile=true;


      if(value<option->lb || value>option->ub) {
	log_printf(hovWarning, 
		    "Go: option '%s' must be in [%d, %d]. Default value %d will be used.\n",
		    name, option->lb, option->ub, option->val);
      } else option->val = value;
    }
  } else {
    log_printf(hovWarning, 
		"Go does not understand option '%s' and will ignore its value '%d'.\n",
		name, value);
  }
  return true;
}

bool goOptions::SetStringValue (const char* name,  const char* value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OStr* option = dynamic_cast<_OStr*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning, 
		  "Go does not know option '%s' as 'string'. Maybe it is an 'integer' or a 'string' option? The option will be ignored.\n",
		  name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning, 
		     "Go will ignore value '%s' set for option '%s' since this option is already specified in an option file.\n", value, name);
	  return true;
	}
      } 

      if(setFromFile)
	option->specifiedInFile=true;

      string strValue(value);
      transform(strValue.begin(), strValue.end(), strValue.begin(), ::tolower);
      //see if it is in the range (of supported values)
      bool inrange=false;
      for(int it=0; it<option->range.size() && !inrange; it++) inrange = (option->range[it]==strValue);

      if(!inrange) {
	stringstream ssRange; ssRange << " ";
	for(int it=0; it<option->range.size(); it++) ssRange << option->range[it] << " ";

	log_printf(hovWarning, 
		    "Go: value '%s' for option '%s' must be one of [%s]. Default value '%s' will be used.\n",
		    value, name, ssRange.str().c_str(), option->val.c_str());
      }
      else option->val = value;
    }
  } else {
    log_printf(hovWarning, 
		"Go does not understand option '%s' and will ignore its value '%s'.\n",
		name, value);
  }
  return true;
}

void goOptions::log_printf(goOutVerbosity v, const char* format, ...)
{
  char buff[1024];
  va_list args;
  va_start (args, format);
  vsprintf (buff,format, args);
  if(log)
    log->printf(v,buff);
  else
    goLogger::printf_error(v,buff);
  //fprintf(stderr,buff);
  va_end (args);
}

void goOptions::print(FILE* file, const char* msg) const
{
  if(NULL==msg) fprintf(file, "#\n# Go options\n#\n");
  else          fprintf(file, "%s ", msg);
 
  map<string,_O*>::const_iterator it = mOptions.begin();
  for(; it!=mOptions.end(); it++) {
    fprintf(file, "%s ", it->first.c_str());
    it->second->print(file);
    fprintf(file, "\n");
  }
  fprintf(file, "# end of Go options\n\n");
}

void goOptions::_ONum::print(FILE* f) const
{
  fprintf(f, "%.3e \t# (numeric)  %g to %g   [%s]", val, lb, ub, descr.c_str());
}
void goOptions::_OInt::print(FILE* f) const
{
  fprintf(f, "%d \t# (integer)  %d to %d   [%s]", val, lb, ub, descr.c_str());
}

void goOptions::_OStr::print(FILE* f) const
{
  stringstream ssRange; ssRange << " ";
  for(int i=0; i<range.size(); i++) ssRange << range[i] << " ";
  fprintf(f, "%s \t# (string) one of [%s]   [%s]", val.c_str(), ssRange.str().c_str(), descr.c_str());
}



} //~end namespace

#include "goSCACOPFData.hpp"

#include "goLogger.hpp"
#include <cstdlib>
#include <cassert>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
using namespace std;

namespace gollnlp {

//temporary log object
goLogger log(stdout);

goSCACOPFData::goSCACOPFData()
{
  
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

static vector<string> split (const string &s, char delim) {
  vector<string> result;
  stringstream ss(s);
  string item;
  
  while (getline (ss, item, delim)) result.push_back (item);
  
  return result;
}

static inline bool isEndOrStartOfSection(const string& l)
{
  if(l.size()==0) return false;
  if(l[0] != '0' && l[0] != ' ') return false;
  if(l.size() == 1 && l[1] == '0') return true;
  if(l.size() >= 2 && l[0] == '0' && l[1] == ' ') return true;
  if(l.size() >= 3 && l[0] == ' ' && l[1] == '0' && l[0] == ' ') return true;
  return false;
}

bool goSCACOPFData::
readinstance(const std::string& raw, const std::string& rop, const std::string& inl, const std::string& con)
{
  double MVAbase;
  VVStr buses, loads,  fixedbusshunts, generators, ntbranches, tbranches, switchedshunts;
  if(!readRAW(raw, MVAbase, buses, loads,  fixedbusshunts, generators, ntbranches, tbranches, switchedshunts)) return false;

  VVStr generatordsp, activedsptables;
  VInt costcurves_ltbl; VStr costcurves_label; VVDou costcurves_xi; VVDou costcurves_yicostcurves;
  if(!readROP(rop, generatordsp, activedsptables, 
	      costcurves_ltbl, costcurves_label, costcurves_xi, costcurves_yicostcurves))
    return false;

  
  return true;
}

bool goSCACOPFData::
readRAW(const std::string& raw, double& MVAbase,
	VVStr& buses,  VVStr& loads, VVStr& fixedbusshunts,
	VVStr& generators, VVStr& ntbranches, VVStr& tbranches,
	VVStr& switchedshunts)
{
  ifstream rawfile(raw.c_str());
  if(!rawfile.is_open()) {
    log.printf(hovError, "failed to load raw file %s\n", raw.c_str());
    return false;
  }
  bool ret; string line; 
  ret = getline(rawfile, line); assert(ret);
  MVAbase = strtod(split(line, ',')[1].c_str(), NULL);

  //skip the next two lines
  ret = getline(rawfile, line); assert(ret);
  ret = getline(rawfile, line); assert(ret);

  size_t pos; string delimiter=","; int i;

  //
  //bus data
  //
  //buses is a vector of 13 column vectors of string
  for(int i=0; i<=12; i++) buses.push_back(vector<string>());
  
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<=12; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	buses[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	buses[i].push_back(line);
      }
    }
    assert(i==13);
  }
#ifdef DEBUG
  int n=buses[0].size();
  for(i=1; i<=12; i++) {
    assert(buses[i].size()==n);
  }
#endif
  log.printf(hovSummary, "loaded data for %d buses\n", buses[0].size());

  //
  // load data
  //
  for(int i=0; i<14; i++) loads.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<14; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	loads[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	loads[i].push_back(line);
      }
    }
    trim(loads[2].back()); //j. loads[:ID] = strip.(string.(loads[:ID]))
    assert(i==14);
  }
#ifdef DEBUG
  int nloads=loads[0].size();
  for(i=1; i<14; i++) {
    assert(loads[i].size()==nloads);
  }
#endif
  log.printf(hovSummary, "loaded data for %d loads\n", loads[0].size());

  //
  //fixed bus shunt data
  //
  for(int i=0; i<5; i++) fixedbusshunts.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<5; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	fixedbusshunts[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	fixedbusshunts[i].push_back(line);
      }
    }
    //j. fixedbusshunts[:ID] = strip.(string.(fixedbusshunts[:ID]))
    trim(fixedbusshunts[1].back());
    assert(i==5);
  }
#ifdef DEBUG
  int nfbsh=fixedbusshunts[0].size();
  for(i=1; i<5; i++) {
    assert(fixedbusshunts[i].size()==nfbsh);
  }
#endif
  log.printf(hovSummary, "loaded data for %d fixed bus shunts\n", fixedbusshunts[0].size());

  //
  // generator data
  //
  for(int i=0; i<28; i++) generators.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<28; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	generators[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	generators[i].push_back(line);
      }
    }
    //j. generators[:ID] = strip.(string.(generators[:ID]))
    trim(generators[1].back());
    assert(i==28);
  }
#ifdef DEBUG
  n=generators[0].size();
  for(i=1; i<28; i++) {
    assert(generators[i].size()==n);
  }
#endif
  log.printf(hovSummary, "loaded data for %d generators\n", generators[0].size());
  //
  //non-transformer branch data
  //
  for(i=0; i<24; i++) ntbranches.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<24; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	ntbranches[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	ntbranches[i].push_back(line);
      }
    }
    //j. ntbranches[:CKT] = strip.(string.(ntbranches[:CKT]))
    trim(ntbranches[2].back());
    assert(i==24);
  }
#ifdef DEBUG
  n=ntbranches[0].size();
  for(i=1; i<24; i++) {
    assert(ntbranches[i].size()==n);
  }
#endif
  log.printf(hovSummary, "loaded data for %d non-transformer branches\n", ntbranches[0].size());
  

  //
  // transformer data
  //
  for(i=0; i<43; i++) tbranches.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;

    for(i=0; i<21; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	tbranches[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	tbranches[i].push_back(line);
      }
    }
    //line 2
    ret = getline(rawfile, line); assert(ret);
    assert(!isEndOrStartOfSection(line));
    for(i=21; i<24; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	tbranches[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	tbranches[i].push_back(line);
      }
    }
    //line 3
    ret = getline(rawfile, line); assert(ret);
    assert(!isEndOrStartOfSection(line));
    for(i=24; i<41; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	tbranches[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	tbranches[i].push_back(line);
      }
    }
    //line 4
    ret = getline(rawfile, line); assert(ret);
    assert(!isEndOrStartOfSection(line));
    for(i=41; i<43; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	tbranches[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	tbranches[i].push_back(line);
      }
    }

    //j. transformers[:CKT] .= strip.(transformers[:CKT])
    trim(tbranches[3].back());

  } //end while -> end of transformer data tbranches
#ifdef DEBUG
  n=tbranches[0].size();
  for(i=1; i<43; i++) {
    assert(tbranches[i].size()==n);
  }
#endif
  log.printf(hovSummary, "loaded data for %d transformer branches\n", tbranches[0].size());

  int section=8;
  while(section<18) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) section++;
  }
#ifdef DEBUG
  std::transform(line.begin(), line.end(), line.begin(), ::toupper);
  assert(string::npos != line.find("BEGIN SWITCHED SHUNT"));
#endif

  //
  // switched shunt data 26
  //
  for(i=0; i<26; i++) switchedshunts.push_back(vector<string>());
  while(true) {
    ret = getline(rawfile, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;
    
    for(i=0; i<26; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	switchedshunts[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	switchedshunts[i].push_back(line);
      }
    }
  }
#ifdef DEBUG
  n=switchedshunts[0].size();
  for(i=1; i<26; i++) {
    assert(switchedshunts[i].size()==n);
  }
#endif
  log.printf(hovSummary, "loaded data for %d switched shunts\n", switchedshunts[0].size());
  return true;
}

bool goSCACOPFData::
readROP(const std::string& rop, VVStr& generatordsp, VVStr& activedsptables, 
	VInt& costcurves_ltbl, VStr& costcurves_label, VVDou& costcurves_xi, VVDou& costcurves_yi)
{
  ifstream file(rop.c_str());
  if(!file.is_open()) {
    log.printf(hovError, "failed to load rop file %s\n", rop.c_str());
    return false;
  }
  int i,n; string delimiter=","; size_t pos; 
  for(i=0; i<4; i++) generatordsp.push_back(vector<string>());
  for(i=0; i<7; i++) activedsptables.push_back(vector<string>());

  bool ret; string line; 
  bool isGenDispSec=false, isCostCurvesSec=false, isPowerDispSec=false;
  while(getline(file, line)) {
    if(isEndOrStartOfSection(line)) {
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      if(line.find("generator dispatch") != string::npos) isGenDispSec=true;
      else if(line.find("active power dispatch") != string::npos) isPowerDispSec=true;
      else if(line.find("piece-wise linear cost")!= string::npos) isCostCurvesSec=true;
    }
    
    if(isGenDispSec) {
      while(true) {
	ret = getline(file, line); assert(ret);
	if(isEndOrStartOfSection(line)) {
	  isGenDispSec=false;
	  break;
	}
	for(i=0; i<4; i++) {
	  if( (pos = line.find(delimiter)) != string::npos ) {
	    generatordsp[i].push_back(line.substr(0,pos));
	    line.erase(0, pos+delimiter.length());
	  } else {
	    generatordsp[i].push_back(line);
	  }
	}
      }
#ifdef DEBUG
      n=generatordsp[0].size();
      for(i=1; i<4; i++) {
	assert(generatordsp[i].size()==n);
      }
#endif
      log.printf(hovSummary, "loaded dispatch data for %d generators\n", generatordsp[0].size());
    }//end if(isGenDispSec)
    
  }

  return true;
}
}//end namespace

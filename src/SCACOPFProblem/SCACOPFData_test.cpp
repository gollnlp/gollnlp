#include "SCACOPFData.hpp"
#include "goUtils.hpp"
#include "goTimer.hpp"
#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
using namespace gollnlp;
using namespace std;

void usage(const string& smain);
vector<vector<string> > loadinstances(const string& file);
string computechecks(const SCACOPFData& data);
vector<string> getchecks();

int main(int argc, char* argv[])
{
  if(argc>=3) { usage(argv[0]); return -1; }
  string sinstfile = "test_instanceslist.txt";
  if(argc==2) sinstfile = argv[1];

  goTimer t; t.start();

  vector<vector<string> > vInst = loadinstances(sinstfile);

  int ncheck=0; vector<string> vChecks=getchecks();
  for(auto& inst: vInst) {
    
    cout<< "\nreading " << inst[0] << endl;

    SCACOPFData data;
    data.readinstance(inst[1], inst[2], inst[3], inst[4]);
    data.buildindexsets();

    //check with the saved result
    string sCheck = computechecks(data);
    if(sCheck != vChecks[ncheck]) {
      cout << "computed  [" << sCheck << "]\n"
	   << "should be [" << vChecks[ncheck] << "]\n";
      return -1;
    }
    ncheck++;
  }
  
  t.stop();
  cout << "Total time elapsed: " << t. getElapsedTime() << endl;
  return 0;
}

void usage(const string& smain) {
  
  cout << "Usage: \n  smain [file]" << endl;
  cout << "'file' contains the instances and their stats against which "<< smain << " will auto-check."<< endl;
  cout << "'file' is optional; if not provided 'test_instanceslist.txt' in the current director will be loaded" << endl;
}

vector<vector<string> > loadinstances(const string& file)
{
  vector<vector<string> > vInst;

  ifstream f(file);
  if(!f.is_open()) {
    cerr << "failed to load raw file " << file << "\n";
    return vInst;
  }
  string line; 
  bool ret = getline(f, line); 
  while(ret) {
    trim(line);
    //root
    vInst.push_back(vector<string>());
    vInst.back().push_back(line);
    //raw - relative to root
    ret = getline(f, line); assert(ret); trim(line);
    vInst.back().push_back(vInst.back()[0]+"/"+line);
    //rop - relative to root
    ret = getline(f, line); assert(ret); trim(line);
    vInst.back().push_back(vInst.back()[0]+"/"+line);
    //inl - relative to root
    ret = getline(f, line); assert(ret); trim(line);
    vInst.back().push_back(vInst.back()[0]+"/"+line);
    //con- relative to root
    ret = getline(f, line); assert(ret); trim(line);
    vInst.back().push_back(vInst.back()[0]+"/"+line);
    //results line
    ret = getline(f, line); assert(ret); trim(line);
    vInst.back().push_back(line);

    ret = getline(f, line);
  }
}

string computechecks(const SCACOPFData& d)
{
  double chk=0; stringstream ret; ret.precision(5);
  for(auto& v: d.L_Nidx[0]) chk += (1+v)/16.0;
  for(auto& v: d.L_Nidx[1]) chk += (1+v)/16.0;
  ret << "L_Nidx=" << scientific << chk;

  chk=0.;
  for(auto& v: d.T_Nidx[0]) chk += (1+v)/16.0;
  for(auto& v: d.T_Nidx[1]) chk += (1+v)/16.0;
  ret << " T_Nidx=" << scientific << chk;

  chk=0.;
  for(auto& v: d.SSh_Nidx) chk += (1+v)/16.0;
  ret << " SSh_Nidx=" << scientific << chk;

  chk=0.;
  for(auto& v: d.G_Nidx) chk += (1+v)/16.0;
  ret << " G_Nidx=" << scientific << chk;

  chk=0.;
  for(auto& u: d.Lidxn) for(auto& v: u) chk += (1+v)/16.0;
  ret << " Lidxn=" << scientific << chk;
  chk=0.;
  for(auto& u: d.Lin) for(auto& v: u) chk += (1+v)/16.0;
  ret << " Lin=" << scientific << chk;

  chk=0.;
  for(auto& u: d.Tidxn) for(auto& v: u) chk += (1+v)/16.0;
  ret << " Tidxn=" << scientific << chk;
  chk=0.;
  for(auto& u: d.Tin) for(auto& v: u) chk += (1+v)/16.0;
  ret << " Tin=" << scientific << chk;

  chk=0.;
  for(auto& u: d.SShn) for(auto& v: u) chk += (1+v)/16.0;
  ret << " SShn=" << scientific << chk;

  chk=0.;
  for(auto& u: d.Gn) for(auto& v: u) chk += (1+v)/16.0;
  ret << " Gn=" << scientific << chk;

  return ret.str();
}

vector<string> getchecks()
{
  vector<string> ret;
  ret.push_back("L_Nidx=1.40543e+04 T_Nidx=4.55162e+03 SSh_Nidx=2.21750e+02 G_Nidx=8.96750e+02 Lidxn=1.33691e+04 Lin=8.66250e+01 Tidxn=1.08075e+03 Tin=2.45625e+01 SShn=4.12500e+00 Gn=8.28750e+01");
  ret.push_back("L_Nidx=4.35477e+06 T_Nidx=1.88365e+06 SSh_Nidx=1.05403e+05 G_Nidx=5.04319e+03 Lidxn=4.95897e+06 Lin=1.67006e+03 Tidxn=4.84590e+05 Tin=5.22000e+02 SShn=6.37038e+03 Gn=1.01709e+04");
  ret.push_back("L_Nidx=4.35477e+06 T_Nidx=1.88365e+06 SSh_Nidx=1.05403e+05 G_Nidx=5.03806e+03 Lidxn=4.95897e+06 Lin=1.67006e+03 Tidxn=4.84590e+05 Tin=5.22000e+02 SShn=6.37038e+03 Gn=1.01353e+04");
  ret.push_back("L_Nidx=4.35477e+06 T_Nidx=1.88365e+06 SSh_Nidx=1.05403e+05 G_Nidx=5.04319e+03 Lidxn=4.95897e+06 Lin=1.67006e+03 Tidxn=4.84590e+05 Tin=5.22000e+02 SShn=6.37038e+03 Gn=1.01709e+04");
  ret.push_back("L_Nidx=3.95574e+04 T_Nidx=6.16244e+03 SSh_Nidx=1.19762e+03 G_Nidx=2.04469e+03 Lidxn=3.70081e+04 Lin=1.44188e+02 Tidxn=1.28700e+03 Tin=2.68125e+01 SShn=7.96875e+01 Gn=2.12688e+02");
  ret.push_back("L_Nidx=3.95574e+04 T_Nidx=6.16244e+03 SSh_Nidx=1.19762e+03 G_Nidx=2.39525e+03 Lidxn=3.70081e+04 Lin=1.44188e+02 Tidxn=1.28700e+03 Tin=2.68125e+01 SShn=7.96875e+01 Gn=2.73188e+02");
  ret.push_back("L_Nidx=3.01436e+05 T_Nidx=1.13911e+05 SSh_Nidx=8.01825e+03 G_Nidx=2.80622e+04 Lidxn=3.36545e+05 Lin=4.35000e+02 Tidxn=4.48910e+04 Tin=1.58812e+02 SShn=7.07812e+02 Gn=5.84550e+03");
  ret.push_back("L_Nidx=3.03972e+05 T_Nidx=1.40239e+05 SSh_Nidx=1.49339e+04 G_Nidx=1.78208e+04 Lidxn=2.90656e+05 Lin=4.04250e+02 Tidxn=4.59566e+04 Tin=1.60688e+02 SShn=1.26881e+03 Gn=1.96094e+03");
  ret.push_back("L_Nidx=3.01075e+05 T_Nidx=1.13911e+05 SSh_Nidx=8.01825e+03 G_Nidx=2.80622e+04 Lidxn=3.35965e+05 Lin=4.34625e+02 Tidxn=4.48910e+04 Tin=1.58812e+02 SShn=7.07812e+02 Gn=5.84550e+03");
  return ret;
}

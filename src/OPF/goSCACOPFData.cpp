#include "goSCACOPFData.hpp"

#include "goLogger.hpp"
#include <cstdlib>
#include <cassert>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <numeric>
using namespace std;

#include <cmath>
#include "blasdefs.hpp"

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

// //trim with return / copy
// inline std::string trim(const std::string &s)
// {
//   auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
//   return std::string(wsfront,
// 		     std::find_if_not(s.rbegin(),
// 				      std::string::const_reverse_iterator(wsfront),
// 				      [](int c){return std::isspace(c);}).base());
// }

static vector<string> split(const string &s, char delim) {
  vector<string> result;
  stringstream ss(s);
  string item;
  
  while(getline(ss, item, delim)) result.push_back (item);
  
  return result;
}

static vector<string> split_skipempty(const string &s, char delim) {
  vector<string> result;
  stringstream ss(s);
  string item;
  
  while(getline(ss, item, delim)) {
    if(!item.empty())
      result.push_back(item);
  }
  return result;
}

static inline bool isEndOrStartOfSection(const string& l)
{
  if(l.size()==0) return false;
  if(l[0] != '0' && l[0] != ' ') return false;
  if(l.size() == 1 && l[0] == '0') return true;
  if(l.size() == 2 && l[0] == '0' && l[1] =='\r') return true;
  if(l.size() >= 2 && l[0] == '0' && l[1] == ' ') return true;
  if(l.size() >= 3 && l[0] == ' ' && l[1] == '0' && l[0] == ' ') return true;
  return false;
}

static inline bool mygetline(ifstream& file, string& line)
{
  if(!getline(file,line)) return false;
  if(line.size()==0) return true;
  string::iterator last = line.end()-1;
  if(*last=='\r') line.erase(last);
}

template<class T> inline void hardclear(vector<T>& in) { vector<T>().swap(in); };

template<class T> inline void printvec(const vector<T>& v, const string& msg="") 
{ 
  cout.precision(6); 
  cout << msg << " size:" << v.size() << endl;
  cout << scientific;
  typename vector<T>::const_iterator it=v.begin();
  for(;it!=v.end(); ++it) cout << (*it) << " ";
  cout << endl;
}

  template<class T> inline void printvecvec(const vector<vector<T> >& v, const string& msg="") 
{ 
  cout.precision(6); 
  cout << msg << " size:" << v.size() << endl;
  cout << scientific;
  for(auto& l: v) {
    for(auto& c: l) cout << c << " ";
    cout << endl;
  }
}


// for entries of 'v' that are not present in 'in', the indexes will be set to -1
template<class T> inline vector<int> indexin(vector<T>& v, vector<T>& in)
{
  vector<int> vIdx(v.size());
  iota(vIdx.begin(), vIdx.end(), 0);
  //sort permutation for v
  sort(vIdx.begin(), vIdx.end(), [&](const int& a, const int& b) { return (v[a] < v[b]); } );

  vector<int> inIdx(in.size());
  iota(inIdx.begin(), inIdx.end(), 0);
  //sort permutation for in
  sort(inIdx.begin(), inIdx.end(), [&](const int& a, const int& b) { return (in[a] < in[b]); } );

  size_t szv=v.size(), szin=in.size();
  vector<int> idxs(szv, -1);
  
  for(int iv=0, iin=0; iv<szv && iin<szin;) {
    //cout << iv << "|" << iin << "  " << v[vIdx[iv]] <<"|" << in[inIdx[iin]] << endl;
    if(v[vIdx[iv]]==in[inIdx[iin]]) {
	idxs[vIdx[iv]]=inIdx[iin];
	iin++; iv++;
      } else v[vIdx[iv]]>in[inIdx[iin]] ? iin++: iv++;
  }

  // T *vv = v.data(), *vin = in.data();
  // for(int iv=0, iin=0, *div=vIdx.data(), *diin=inIdx.data(), *didxs=idxs.data(); iv<szv && iin<szin;) {
  //   cout << "iv=" << iv << "  iin=" << iin << " | " << div[iv] << " " << diin[iin] << endl;
  //   if(vv[div[iv]]==vin[diin[iin]]) {
  //     didxs[div[iv]]=diin[iin];
  //     iin++; iv++;
  //   } else vv[div[iv]]>vin[diin[iin]]? iin++: iv++;
  // }
  return idxs;
}

// returns the indexes 'i' in 'v', for which 'v[i]' satisfies (unary) predicate
template<class T>
vector<int> findall(const vector<T>& v, std::function<bool(const int&)> pred)
{
  vector<int> ret; int count=0;
  for(auto& it : v) {
    if(pred(it)) ret.push_back(count);
    count++;
  }
  return ret;
}

template<class T> vector<T> select(vector<T>& v, const vector<int>& idx)
{
  vector<T> ret;
  for(auto& keep: idx) ret.push_back(v[keep]);
  return ret;
}

enum Bheader{BI=0,BNAME,BBASKV,BIDE,BAREA,BZONE,BOWNER,BVM,BVA,BNVHI,BNVLO,BEVHI,BEVLO};
enum Lheader{LI=0,LID,LSTATUS,LAREA,LZONE,LPL,LQL,LIP,LIQ,LYP,LYQ,LOWNER,LSCALE,LINTRPT};
enum FSheader{FSI=0,FSID,FSSTATUS,FSGL,FSBL}; //fixedbusshunts
enum NTBheader{NTBI=0,NTBJ,NTBCKT,NTBR,NTBX,NTBB,NTBRATEA,NTBRATEB,NTBRATEC,
	       NTBGI,NTBBI,NTBGJ,NTBBJ,NTBST,NTBMET,NTBLEN,
	       NTBO1,NTBF1,NTBO2,NTBF2,NTBO3,NTBF3,NTBO4,NTBF4}; //non-transformer branches
enum TBheader{TBI=0,TBJ,TBK,TBCKT,TBCW,TBCZ,TBCM,TBMAG1,TBMAG2,
	      TBNMETR,TBNAME,TBSTAT,TBO1,TBF1,TBO2,TBF2,TBO3,TBF3,TBO4,TBF4,
	      TBVECGRP,TBR12,TBX12, TBSBASE12,TBWINDV1,TBNOMV1,TBANG1,
	      TBRATA1,TBRATB1,TBRATC1,TBCOD1,TBCONT1,TBRMA1,
	      TBRMI1,TBVMA1,TBVMI1,TBNTP1,TBTAB1,TBCR1,TBCX1,
	      TBCNXA1,TBWINDV2,TBNOMV2}; // transformer branches
enum SShheader{SSI,SSMODSW,SSADJM,SSSTAT,SSVSWHI,SSVSWLO,SSSWREM,SSRMPCT,
	       SSRMIDNT,SSBINIT,SSN1,SSB1,SSN2,SSB2,SSN3,SSB3,SSN4,SSB4,
	       SSN5,SSB5,SSN6,SSB6,SSN7,SSB7,SSN8,SSB8};
enum Gheader{GI=0,GID,GPG,GQG,GQT,GQB,GVS,GIREG,GMBASE,GZR,GZX,GRT,GXT,
	     GGTAP,GSTAT,GRMPCT, GPT,GPB,GO1,GF1,GO2,GF2,GO3,GF3,GO4,
	     GF4,GWMOD,GWPF}; //generators
enum GDSPheader{GDBUS=0,GDGENID,GDDISP,GDDSPTBL}; //generator dispatch tables

enum GADSPheader{GADSPTBL=0,GADSPPMAX,GADSPPMIN,GADSPFUELCOST,
		 GADSPCTYP,GADSPSTATUS,GADSPCTBL}; //generator active dispatch 
enum GORheader{GORI=0,GORID,GORH,GORPMAX,GORPMIN,GORR,GORD}; //governor response
  //enum Contheader{COLABEL=0, COCTYPE, COCON}; //contingencies

bool goSCACOPFData::
readinstance(const std::string& raw, const std::string& rop, const std::string& inl, const std::string& con)
{
  double MVAbase;
  VVStr buses, loads,  fixedbusshunts, generators, ntbranches, tbranches, switchedshunts;
  if(!readRAW(raw, MVAbase, buses, loads, fixedbusshunts, generators, ntbranches, tbranches, switchedshunts)) return false;

  //here we (should) deallocate columns that are not currently used
  //for example
  vector<int> cols = {TBK,TBCW,TBCZ,TBCM, TBNMETR,TBNAME,TBO1,TBF1,TBO2,TBF2,TBO3,TBF3,TBO4,TBF4,
		      TBVECGRP,TBSBASE12,TBNOMV1,
		      TBRATB1,TBCOD1,TBCONT1,TBRMA1,
		      TBRMI1,TBVMA1,TBVMI1,TBNTP1,TBTAB1,TBCR1,TBCX1,
		      TBCNXA1,TBNOMV2};
  for(auto c: cols) hardclear(tbranches[c]);

  cols = {NTBRATEB,NTBGI,NTBBI,NTBGJ,NTBBJ,NTBMET,NTBLEN,
	       NTBO1,NTBF1,NTBO2,NTBF2,NTBO3,NTBF3,NTBO4,NTBF4};
  for(auto c: cols) hardclear(ntbranches[c]);
  //end of deallocation

  int n, one=1; double scale=M_PI/180.;

  convert(buses[BI],    N_Bus);    hardclear(buses[BI]);
  convert(buses[BAREA], N_Area);   hardclear(buses[BAREA]);
  convert(buses[BNVLO], N_Vlb);    hardclear(buses[BNVLO]);
  convert(buses[BNVHI], N_Vub);    hardclear(buses[BNVHI]);
  convert(buses[BEVLO], N_EVlb);   hardclear(buses[BEVLO]); 
  convert(buses[BEVHI], N_EVub);   hardclear(buses[BEVHI]);
  convert(buses[BVM],   N_v0);     hardclear(buses[BVM]);
  convert(buses[BVA],   N_theta0); hardclear(buses[BVA]);
  n=N_theta0.size(); DSCAL(&n, &scale, N_theta0.data(), &one);

  //initialize rest of N_ members
  N_Pd = vector<double>(n, 0.);
  N_Qd = vector<double>(n, 0.);
  N_Gsh= vector<double>(n, 0.);
  N_Bsh= vector<double>(n, 0.);

  //loads 
  vector<int> loads_I, loads_status; 
  vector<double> loads_PL, loads_QL;
  convert(loads[LI], loads_I);           hardclear(loads[LI]);
  convert(loads[LSTATUS], loads_status); hardclear(loads[LSTATUS]);
  convert(loads[LPL], loads_PL);         hardclear(loads[LPL]);
  convert(loads[LQL], loads_QL);         hardclear(loads[LQL]);
  {
    vector<int> BusLoad = indexin(loads_I, N_Bus);
    for(int l=0; l<loads_I.size(); l++) {
      assert(BusLoad[l]>=0);
      if(loads_status[l]==1) {
	N_Pd[BusLoad[l]] += loads_PL[l]/MVAbase;
	N_Qd[BusLoad[l]] += loads_QL[l]/MVAbase;
      }
    }
  }
  //fixedbusshunts
  vector<int> fixedbusshunts_I, fixedbusshunts_GL, fixedbusshunts_BL, fixedbusshunts_status;
  convert(fixedbusshunts[FSI], fixedbusshunts_I);   hardclear(fixedbusshunts[FSI]);
  convert(fixedbusshunts[FSGL], fixedbusshunts_GL); hardclear(fixedbusshunts[FSGL]);
  convert(fixedbusshunts[FSBL], fixedbusshunts_BL); hardclear(fixedbusshunts[FSBL]);
  convert(fixedbusshunts[FSSTATUS], fixedbusshunts_status); hardclear(fixedbusshunts[FSSTATUS]);
  {
    vector<int> BusShunt = indexin(fixedbusshunts_I, N_Bus);
    for(int fbsh=0; fbsh<fixedbusshunts_I.size(); fbsh++) {
      assert(BusShunt[fbsh]>=0);
      if(fixedbusshunts_status[fbsh]==1) {
	N_Gsh[BusShunt[fbsh]] += fixedbusshunts_GL[fbsh] / MVAbase;
	N_Bsh[BusShunt[fbsh]] += fixedbusshunts_BL[fbsh] / MVAbase;
      }
    }
  }

  // non-transformer branches
  vector<int> ntbranches_ST;
  convert(ntbranches[NTBST], ntbranches_ST); hardclear(ntbranches[NTBST]);
  L_Line = findall(ntbranches_ST, [](int val) {return val!=0;});
  convert(ntbranches[NTBI], L_From); hardclear(ntbranches[NTBI]);
  convert(ntbranches[NTBJ], L_To);   hardclear(ntbranches[NTBJ]);
  L_From = select(L_From, L_Line);
  L_To   = select(L_To,   L_Line);
  L_CktID = select(ntbranches[NTBCKT], L_Line); hardclear(ntbranches[NTBCKT]);
  for(auto& s: L_CktID) s.erase(remove(s.begin(), s.end(),'\''), s.end());

  {
    vector<double> R, X;
    convert(ntbranches[NTBR], R); hardclear(ntbranches[NTBR]);
    convert(ntbranches[NTBX], X); hardclear(ntbranches[NTBX]);
    X = select(X, L_Line); R = select(R, L_Line);
    int nlines = X.size(); double aux;
    L_G = L_B = vector<double>(nlines);
    for(int i=0; i<nlines; i++) {
      aux = R[i]*R[i]+X[i]*X[i];
      L_G[i] =  R[i]/aux; 
      L_B[i] = -X[i]/aux;
    }
  }
  convert(ntbranches[NTBB], L_Bch);  hardclear(ntbranches[NTBB]);
  L_Bch = select(L_Bch, L_Line);

  convert(ntbranches[NTBRATEA], L_RateBase); hardclear(ntbranches[NTBRATEA]);
  convert(ntbranches[NTBRATEC], L_RateEmer); hardclear(ntbranches[NTBRATEC]);
  L_RateBase = select(L_RateBase, L_Line);
  L_RateEmer = select(L_RateEmer, L_Line);  
  n=L_Line.size(); scale = 1/MVAbase;
  DSCAL(&n, &scale, L_RateBase.data(), &one);
  DSCAL(&n, &scale, L_RateEmer.data(), &one);

  // transformer branches
  convert(tbranches[TBSTAT], T_Transformer); hardclear(tbranches[TBSTAT]);
  T_Transformer = findall(T_Transformer, [](int val) {return val!=0;});
  convert(tbranches[TBI], T_From); hardclear(tbranches[TBI]);
  convert(tbranches[TBJ], T_To);   hardclear(tbranches[TBJ]);
  T_From = select(T_From, T_Transformer);
  T_To   = select(T_To,   T_Transformer);
  T_CktID = select(tbranches[TBCKT], T_Transformer); hardclear(tbranches[TBCKT]);
  for(auto& s: T_CktID) s.erase(remove(s.begin(), s.end(),'\''), s.end());
  convert(tbranches[TBMAG1], T_Gm); hardclear(tbranches[TBMAG1]);
  convert(tbranches[TBMAG2], T_Bm); hardclear(tbranches[TBMAG2]);
  T_Gm = select(T_Gm, T_Transformer); T_Bm = select(T_Bm, T_Transformer); 
  n=T_Transformer.size(); double aux;
  {
    vector<double> R12, X12;
    convert(tbranches[TBR12], R12); hardclear(tbranches[TBR12]);
    convert(tbranches[TBX12], X12); hardclear(tbranches[TBX12]);
    R12 = select(R12, T_Transformer); X12 = select(X12, T_Transformer); assert(n==R12.size());
    T_G = T_B = vector<double>(n);
    for(int i=0; i<n; i++) {
      aux = R12[i]*R12[i] + X12[i]*X12[i];
      T_G[i] = R12[i]/aux; T_B[i] = -X12[i]/aux;
    }
  }
  {
    vector<double> WINDV1, WINDV2;
    convert(tbranches[TBWINDV1], WINDV1); hardclear(tbranches[TBWINDV1]);
    convert(tbranches[TBWINDV2], WINDV2); hardclear(tbranches[TBWINDV2]);
    WINDV1 = select(WINDV1, T_Transformer); WINDV2 = select(WINDV2, T_Transformer); 
    T_Tau = vector<double>(n);
    for(int i=0; i<n; i++) T_Tau[i] = WINDV1[i]/WINDV2[i];
  }
  convert(tbranches[TBANG1], T_Theta); hardclear(tbranches[TBANG1]);
  T_Theta = select(T_Theta, T_Transformer);
  scale = M_PI/180; DSCAL(&n, &scale, T_Theta.data(), &one);

  convert(tbranches[TBRATA1], T_RateBase); hardclear(tbranches[TBRATA1]);
  convert(tbranches[TBRATC1], T_RateEmer); hardclear(tbranches[TBRATC1]);
  T_RateBase = select(T_RateBase, T_Transformer);
  T_RateEmer = select(T_RateEmer, T_Transformer);
  scale = 1/MVAbase;
  DSCAL(&n, &scale, T_RateBase.data(), &one);
  DSCAL(&n, &scale, T_RateEmer.data(), &one);

  // switched shunts
  convert(switchedshunts[SSSTAT], SSh_SShunt); hardclear(switchedshunts[SSSTAT]);
  SSh_SShunt = findall(SSh_SShunt, [](int val) {return val!=0;});
  convert(switchedshunts[SSI], SSh_Bus); hardclear(switchedshunts[SSI]);
  SSh_Bus = select(SSh_Bus, SSh_SShunt);
  convert(switchedshunts[SSBINIT], SSh_B0); hardclear(switchedshunts[SSBINIT]);
  SSh_B0 = select(SSh_B0, SSh_SShunt);
  scale = 1/MVAbase; n=SSh_B0.size(); DSCAL(&n, &scale, SSh_B0.data(), &one);
 
  SSh_Blb = SSh_Bub = vector<double>(n);
  double Blb, Bub;
  for(int ssh=0; ssh<n; ssh++) {
    Blb = Bub = 0;
    for(int i=1; i<=8; i++) {
      aux = stod(switchedshunts[8+2*i][SSh_SShunt[ssh]]) 
	* stod(switchedshunts[9+2*i][SSh_SShunt[ssh]]) 
	/ MVAbase;
      aux < 0 ? Blb+=aux : Bub+= aux; 
    }
    SSh_Blb[ssh] = Blb; SSh_Bub[ssh]=Bub;
  }

  //generators - RAW
  convert(generators[GSTAT], G_Generator); hardclear(generators[GSTAT]);
  G_Generator = findall(G_Generator, [](int val) {return val!=0;});
  convert(generators[GI], G_Bus); hardclear(generators[GI]);
  G_Bus = select(G_Bus, G_Generator);
  convert(generators[GID], G_BusUnitNum); hardclear(generators[GID]);
  convert(generators[GPB], G_Plb); hardclear(generators[GPB]);
  convert(generators[GPT], G_Pub); hardclear(generators[GPT]);
  convert(generators[GQB], G_Qlb); hardclear(generators[GQB]);
  convert(generators[GQT], G_Qub); hardclear(generators[GQT]);
  convert(generators[GPG], G_p0);  hardclear(generators[GPG]);
  convert(generators[GQG], G_q0);  hardclear(generators[GQG]);

  G_Plb = select(G_Plb, G_Generator); G_Pub = select(G_Pub, G_Generator); 
  G_Qlb = select(G_Qlb, G_Generator); G_Qub = select(G_Qub, G_Generator); 
  G_p0  = select(G_p0,  G_Generator); G_q0  = select(G_q0,  G_Generator); 

  n = G_Generator.size(); scale = 1/MVAbase;
  DSCAL(&n, &scale, G_Plb.data(), &one);
  DSCAL(&n, &scale, G_Pub.data(), &one);
  DSCAL(&n, &scale, G_Qlb.data(), &one);
  DSCAL(&n, &scale, G_Qub.data(), &one);
  DSCAL(&n, &scale, G_p0.data(),  &one);
  DSCAL(&n, &scale, G_q0.data(),  &one);

  // generators - ROP

  VVStr generatordsp, activedsptables;
  VInt costcurves_ltbl; VStr costcurves_label; VVDou costcurves_xi; VVDou costcurves_yi;
  if(!readROP(rop, generatordsp, activedsptables, 
	      costcurves_ltbl, costcurves_label, costcurves_xi, costcurves_yi))
    return false;

  vector<string> vBBUN = vector<string>(n); //G[:Bus], ":", G[:BusUnitNum]
  for(int i=0; i<n; i++) {
    vBBUN[i] = to_string(G_Bus[i]) + ":" + to_string(G_BusUnitNum[i]);
  }
  {
    assert(n<=generatordsp[GDBUS].size());
    
    n = generatordsp[GDBUS].size();
    vector<string> vBGEN = vector<string>(n);
    for(int i=0; i<n; i++) {
      trim(generatordsp[GDBUS][i]);
      trim(generatordsp[GDGENID][i]);
      vBGEN[i] = generatordsp[GDBUS][i] + ":" + generatordsp[GDGENID][i];
    }
    auto gdspix = indexin(vBBUN, vBGEN);
    vector<string> gdsptbl = select(generatordsp[GDDSPTBL], gdspix);
    
    for(auto& s:activedsptables[GADSPTBL]) trim(s);
    
    auto gctblidx = indexin(gdsptbl, activedsptables[GADSPTBL]);
    auto gctbl = select(activedsptables[GADSPCTBL], gctblidx);
    vector<int> gctbl_int; convert(gctbl, gctbl_int); hardclear(gctbl);
    gctblidx = indexin(gctbl_int, costcurves_ltbl);
    assert(gctblidx.end() == find(gctblidx.begin(), gctblidx.end(), -1) 
	   && "there seems to be missing cost curves for generators");
    
    n = G_Generator.size(); int dim;
    assert(n == gctblidx.size());
    G_CostPi = G_CostCi = VVDou(n);
    for(int g=0; g<n; g++) {
      G_CostPi[g] = costcurves_xi[gctblidx[g]];
      scale = 1/MVAbase; dim=G_CostPi[g].size();
      DSCAL(&dim, &scale, G_CostPi[g].data(), &one);
      
      G_CostCi[g] = costcurves_yi[gctblidx[g]];
    }

    // fixing infeasible initial solutions fixing bad bounds in cost functions
    string sgen_inf="", sgen_mod="";
    for(int g=0; g<n; g++) {
      if(G_p0[g] < G_Plb[g] || G_p0[g] > G_Pub[g]) {
	G_p0[g] = 0.5*(G_Plb[g] + G_Pub[g]);
	sgen_inf += to_string(G_Generator[g]) + " ";
      }
      if(G_q0[g] < G_Qlb[g] || G_q0[g] > G_Qub[g]) {
	G_q0[g] = 0.5*(G_Qlb[g] + G_Qub[g]);
	sgen_inf += to_string(G_Generator[g]) + " ";
      }
      VDou& xi = G_CostPi[g], yi = G_CostCi[g];
      size_t nn = xi.size();
      assert(nn>=2); 
      assert(yi.size()==nn);
      nn--;
      if(xi[0] > G_Plb[g]) {
	yi[0] = yi[0] + (yi[1] - yi[0])/(xi[1] - xi[0])*(G_Plb[g] - xi[0]);
	xi[0] = G_Plb[g];
	sgen_mod += to_string(G_Generator[g]) + " ";
      }
      if(xi[nn] < G_Pub[g]) {
	yi[nn] = yi[nn] + (yi[nn] - yi[nn-1])/(xi[nn] - xi[nn-1])*(G_Pub[g] - xi[nn]);
	xi[nn] = G_Pub[g];
	sgen_mod += to_string(G_Generator[g]) + " ";
      }
    }
    if(sgen_inf.size()>0) cout << "generators with infeasible starting points: " << sgen_inf << endl;
    if(sgen_mod.size()>0) cout << "generators with with inconsistent cost functions: " << sgen_mod << endl;

    //printvecvec(G_CostPi, "Pi");
    //printvecvec(G_CostCi, "Ci");
    //printvec(G_q0);
  }

  // generators -- INL
  {
    n = G_Generator.size();
    VVStr governorresponse;
    if(!readINL(inl, governorresponse)) return false;
    int ngov = governorresponse[GORI].size();
    assert(n<=ngov);
    assert(ngov == governorresponse[GORID].size());

    vector<string> vGIID(ngov);
    for(int i=0; i<ngov; i++) {
      trim(governorresponse[GORI][i]);
      trim(governorresponse[GORID][i]);
      vGIID[i] = governorresponse[GORI][i] + ":" + governorresponse[GORID][i];
    }
    auto ggovrespix = indexin(vBBUN, vGIID);
    assert(ggovrespix.end() == find(ggovrespix.begin(), ggovrespix.end(), -1) 
	   && "there seems to be missing participation factors for generators");

    convert(select(governorresponse[GORR], ggovrespix), G_alpha);
  }

  // contingencies
  {
    VStr contingencies_label;
    std::vector<ContingencyType> contingencies_type;
    std::vector<Contingency*> contingencies_con;
    if(!readCON(con, contingencies_label, contingencies_type, contingencies_con)) return false;
    
    int ncont = contingencies_type.size();
    assert(contingencies_con.size() == ncont);

    K_ConType = vector<KType>(ncont, kNotInit);
    K_IDout   = vector<int>  (ncont, -1);

    K_Contingency =  vector<int>(ncont);  iota(K_Contingency.begin(), K_Contingency.end(), 0);

    // -> generators
    auto gencon = findall(contingencies_type, [](int val) {return val==cGenerator;});
    int ngencon = gencon.size();

    vector<string> searchstr(ngencon); int idx;
    for(int i=0; i<ngencon; i++) {
      idx = gencon[i];
      assert(contingencies_type[idx]==cGenerator);   
      GeneratorContingency& gcont = dynamic_cast<GeneratorContingency&>(*contingencies_con[idx]);
      searchstr[i] = to_string(gcont.Bus) + ":" + gcont.unit;
    }

    auto gix = indexin(searchstr, vBBUN);
    for(int i=0; i<ngencon; i++) {
      if(gix[i] != -1) {
	K_ConType[gencon[i]] = kGenerator;
	K_IDout[gencon[i]] = G_Generator[gix[i]];
      }
    }

    // -> line and transformers
    auto txcon = findall(contingencies_type, [](int val) {return val==cBranch;});
    int ntxcon=txcon.size(); assert(ntxcon+ngencon==ncont);
    
    searchstr.resize(ntxcon);
    for(int i=0; i<ntxcon; i++) {
      idx = txcon[i];
      assert(contingencies_type[idx]==cBranch);   
      TransmissionContingency& tcont = dynamic_cast<TransmissionContingency&>(*contingencies_con[idx]);
      searchstr[i] = to_string(tcont.FromBus) + ":" + to_string(tcont.ToBus) + ":" + tcont.Ckt;
    }
    vector<string> lstr(L_Line.size()), tstr(T_Transformer.size());
    for(int i=0; i<lstr.size(); i++) 
      lstr[i] = to_string(L_From[i]) + ":" + to_string(L_To[i]) + ":" + L_CktID[i];
    for(int i=0; i<tstr.size(); i++) 
      tstr[i] = to_string(T_From[i]) + ":" + to_string(T_To[i]) + ":" + T_CktID[i];

    auto lix = indexin(searchstr, lstr);
    auto trix= indexin(searchstr, tstr);

    //printvec(searchstr, "searchstr");
    //printvec(lstr, "lstr");
    //printvec(tstr, "tstr");

    for(int i=0; i<ntxcon; i++) {
      if(lix[i]!=-1) {
	assert(trix[i]==-1);
	K_ConType[txcon[i]] = kLine;
	K_IDout[txcon[i]] = L_Line[lix[i]];
      } else if(trix[i]!=-1){
	assert(lix[i]==-1);
	K_ConType[txcon[i]] = kTransformer;
	K_IDout[txcon[i]] = T_Transformer[trix[i]];
      } else assert(false && "something went wrong");
    }
    assert(K_IDout.end() ==  find(K_IDout.begin(), K_IDout.end(), -1));

    for(auto& i: contingencies_con) delete i;

    //printvec(K_ConType, "contype");
    //printvec(K_IDout, "idout");
  } // end of contingencies

  // penalties
  P_Quantities = P_Penalties = VVDou(3);
  P_Quantities[pP] = {2/MVAbase, 50/MVAbase, 1e+22/MVAbase};
  P_Quantities[pQ] = {2/MVAbase, 50/MVAbase, 1e+22/MVAbase};
  P_Quantities[pS] = {2/MVAbase, 50/MVAbase, 1e+22/MVAbase};
  P_Penalties[pP] = {1E3*MVAbase, 5E3*MVAbase, 1E6*MVAbase};
  P_Penalties[pQ] = {1E3*MVAbase, 5E3*MVAbase, 1E6*MVAbase};
  P_Penalties[pS] = {1E3*MVAbase, 5E3*MVAbase, 1E6*MVAbase};

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
	assert(i==12);
	buses[i].push_back(line);
      }
    }
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
	assert(i==13);
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
	assert(i==4);
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
	assert(i==27);
	generators[i].push_back(line);
      }
    }
    //j. generators[:ID] = strip.(string.(generators[:ID]))
    string& s = generators[1].back(); trim(s);
    //also remove quotes
    s.erase(remove(s.begin(), s.end(),'\''), s.end());
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
	assert(i==23);
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
	assert(i==20);
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
	assert(i==23);
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
	assert(i==40);
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
	assert(i==42);
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
	assert(i==25);
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
  bool isGenDispSec=false, isCostCurvesSec=false, isActiveDispSec=false;
  bool loadedGenDispSec=false, loadedCostCurvesSec=false, loadedActiveDispSec=false;
  ret = getline(file, line); assert(ret);
  while(ret) {
    if(isEndOrStartOfSection(line)) {
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      if(line.find("generator dispatch")!=string::npos && !loadedGenDispSec) isGenDispSec=true;
      if(line.find("active power dispatch")!=string::npos && !loadedActiveDispSec) isActiveDispSec=true;
      if(line.find("piece-wise linear cost")!=string::npos && !loadedCostCurvesSec) isCostCurvesSec=true;
    }
    ////////////////////////////////////////////////////////////////////////////////////
    if(isGenDispSec) {
      while(true) {
	ret = getline(file, line); assert(ret);
	if(isEndOrStartOfSection(line)) {
	  isGenDispSec=false;
	  loadedGenDispSec=true;
	  break;
	}
	for(i=0; i<4; i++) {
	  if( (pos = line.find(delimiter)) != string::npos ) {
	    generatordsp[i].push_back(line.substr(0,pos));
	    line.erase(0, pos+delimiter.length());
	  } else {
	    assert(i==3);
	    //trim and remove '\r'
	    line.erase(remove(line.begin(), line.end(),'\r'), line.end());
	    trim(line);
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
      continue;
    }//end if(isGenDispSec)

    ////////////////////////////////////////////////////////////////////////////////////
    if(isActiveDispSec) {
      while(true) {
	ret = getline(file, line); assert(ret);
	if(isEndOrStartOfSection(line)) {
	  isActiveDispSec=false;
	  loadedActiveDispSec=true;
	  break;
	}
	for(i=0; i<7; i++) {
	  if( (pos = line.find(delimiter)) != string::npos ) {
	    activedsptables[i].push_back(line.substr(0,pos));
	    line.erase(0, pos+delimiter.length());
	  } else {
	    assert(i==6);
	    line.erase(remove(line.begin(), line.end(),'\r'), line.end());
	    trim(line);
	    activedsptables[i].push_back(line);
	  }
	}
      }
      log.printf(hovSummary, "loaded active power dispatch data for %d generators\n", activedsptables[0].size());
      continue;
    } //end if(isActiveDispSec)
    
    ////////////////////////////////////////////////////////////////////////////////////
    if(isCostCurvesSec) {
      int npairs, p; 
      while(true) {
	ret = getline(file, line); assert(ret);
	if(isEndOrStartOfSection(line)) {
	  isCostCurvesSec=false;
	  loadedCostCurvesSec=true;
	  break;
	}
	//parse the first line -> 3 items: bus id, label, numpairs
	pos = line.find(delimiter); assert(pos!=string::npos);
	costcurves_ltbl.push_back(atoi(line.substr(0,pos).c_str()));
	line.erase(0, pos+delimiter.length());
	pos = line.find(delimiter); assert(pos!=string::npos);
	costcurves_label.push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());

	assert(line.find(delimiter)==string::npos);

	npairs = atoi(line.c_str());
        costcurves_xi.push_back(vector<double>()); costcurves_yi.push_back(vector<double>());
	for(p=0; p<npairs; p++) {
	  ret = getline(file, line); assert(ret);
	  assert(!isEndOrStartOfSection(line));
	  pos = line.find(delimiter); assert(pos!=string::npos);
	  costcurves_xi.back().push_back(atof(line.substr(0,pos).c_str()));
	  line.erase(0, pos+delimiter.length());
	  assert(line.find(delimiter)==string::npos);
	  costcurves_yi.back().push_back(atof(line.c_str()));

#ifdef DEBUG
	  if(p>0 && costcurves_xi.back().back() <= costcurves_xi.back()[p-1])
	    log.printf(hovWarning, "!!!! nonmonotone linear cost coeff !?!? check this\n");
#endif
	}
      }
      log.printf(hovSummary, "loaded Piece-wise Linear Cost data for %d generators\n", costcurves_ltbl.size());
      continue;
    }// if(isCostCurvesSec)
    
    assert(!isGenDispSec && !isActiveDispSec && !isCostCurvesSec);
    ret = getline(file, line); 
  }
  return true;
}

bool goSCACOPFData::
readINL(const std::string& inl, VVStr& governorresponse)
{
  ifstream file(inl.c_str());
  if(!file.is_open()) {
    log.printf(hovError, "failed to load inl file %s\n", inl.c_str());
    return false;
  }
  int i,n; string delimiter=","; size_t pos; 
  for(i=0; i<7; i++) governorresponse.push_back(vector<string>());

  bool ret; string line; 

  while(true) {
    ret = getline(file, line); assert(ret);
    if(isEndOrStartOfSection(line)) break;

    for(i=0; i<7; i++) {
      if( (pos = line.find(delimiter)) != string::npos ) {
	governorresponse[i].push_back(line.substr(0,pos));
	line.erase(0, pos+delimiter.length());
      } else {
	assert(i==6);
	governorresponse[i].push_back(line);
      }
    }
  } // end of while

  log.printf(hovSummary, "loaded governor response data for %d generators\n", governorresponse[0].size());

  return true;
}

bool goSCACOPFData::readCON(const string& con,
			     VStr& contingencies_label, 
			     std::vector<ContingencyType>& contingencies_type,
			     std::vector<Contingency*>& contingencies_con)
{
  ifstream file(con.c_str());
  if(!file.is_open()) {
    log.printf(hovError, "failed to load con file %s\n", con.c_str());
    return false;
  }
  int i,n; string delimiter=" "; char cDelimiter=delimiter[0];
  size_t pos; bool ret; string line; 

  while(true) {
    ret = getline(file, line); assert(ret);

    if(line.substr(0,3)=="END") break;

    assert(line.substr(0,11)=="CONTINGENCY");
    
    pos = line.find(delimiter); assert(pos != string::npos );
    contingencies_label.push_back(line.substr(pos+1));

    ret = mygetline(file, line); assert(ret);
    VStr tokens = split_skipempty(line, cDelimiter);

    assert(tokens.size()>=6);
    if(tokens[0]=="REMOVE") {
      contingencies_type.push_back(cGenerator);
      assert(tokens.size()==6);

      contingencies_con.push_back(new GeneratorContingency(stoi(tokens[5]), tokens[2]));

    } else if(tokens[0]=="OPEN") {
       contingencies_type.push_back(cBranch);
      assert(tokens.size()==10);
      contingencies_con.push_back(new TransmissionContingency(stoi(tokens[4]), 
							      stoi(tokens[7]), 
							      tokens[9]));
    } else {
      log.printf(hovWarning, "expected REMOVE or OPEN in line=[%s]\n", line.c_str());
      assert(false && "expected REMOVE or OPEN");
    }    

    ret = getline(file, line); assert(ret);
    assert(line.substr(0,3)=="END");

  } // end of while
  log.printf(hovSummary, "loaded  %d contingencies\n", contingencies_con.size());
  return true;
}

void goSCACOPFData::convert(const VStr& src, VInt& dest)
{
  size_t sz = src.size();
  dest.resize(sz);
  for(int i=0; i<sz; i++)
    dest[i] = atoi(src[i].c_str());
}
void goSCACOPFData::convert(const VStr& src, VDou& dest)
{
  size_t sz = src.size();
  dest.resize(sz);
  for(int i=0; i<sz; i++)
    dest[i] = atof(src[i].c_str());
}

}//end namespace

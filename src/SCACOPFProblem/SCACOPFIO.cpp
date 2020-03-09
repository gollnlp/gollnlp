#include "SCACOPFIO.hpp"

#include "SCACOPFUtils.hpp"
#include "goUtils.hpp"

#include <numeric>
#include <typeinfo> 

#include <iostream>

#include <cmath>

using namespace std;

namespace gollnlp {
  //
  // initialization of static members
  //
  vector<int> SCACOPFIO::SSh_Nidx={};
  vector<int> SCACOPFIO::gmap={};
  vector<double> SCACOPFIO::bcsn={};
  bool SCACOPFIO::sol2_write_1st_call=true;
  //
  // methods
  //

  //static
  void SCACOPFIO::
  write_append_solution_block(const double* v_n, const double* theta_n, const double* b_s,
			      const double* p_g, const double* q_g,
			      SCACOPFData& scdata,
			      const std::string& filename,
			      const std::string& fileopenflags, FILE* file_in)
  {
    if(SSh_Nidx.size()==0)
      SSh_Nidx = indexin(scdata.SSh_Bus, scdata.N_Bus);

    assert(SSh_Nidx.size() == scdata.SSh_Bus.size());

    if(bcsn.size()==0)
      bcsn = vector<double>(scdata.N_Bus.size());

    assert(bcsn.size() == scdata.N_Bus.size());

    for(int i=0; i<scdata.N_Bus.size(); i++) 
      bcsn[i] = 0.;

    for(int itssh = 0; itssh<scdata.SSh_SShunt.size(); itssh++) {
      assert(itssh < SSh_Nidx.size());
      //assert(itssh < b_s->n);
      assert(SSh_Nidx[itssh] < scdata.N_Bus.size());
      assert(SSh_Nidx[itssh]>=0);
    
      bcsn[SSh_Nidx[itssh]] += b_s[itssh];
    }

    for(int i=0; i<scdata.N_Bus.size(); i++) 
      bcsn[i] *= scdata.MVAbase;

    FILE* file = NULL;
    if(NULL==file_in) file = fopen(filename.c_str(), fileopenflags.c_str());
    else 
      file=file_in;

    if(NULL==file) {
      printf("[warning] could not open [%s] file for writing (flags '%s')\n", 
	     filename.c_str(), fileopenflags.c_str());
      return;
    }
    //
    // write bus section
    //
    fprintf(file, "--bus section\ni, v(p.u.), theta(deg), bcs(MVAR at v = 1 p.u.)\n");
    for(int n=0; n<scdata.N_Bus.size(); n++) {
      fprintf(file, "%d, %.20f, %.20f, %.20f\n", 
	      scdata.N_Bus[n], v_n[n], 180/M_PI*theta_n[n], bcsn[n]);
    }
    int GID = SCACOPFData::GID;
    int GI  = SCACOPFData::GI; assert(GI==0);
    assert(scdata.generators[GID].size() == scdata.generators[GI].size());
    assert(scdata.generators[GID].size()>=scdata.G_Generator.size());
  
    if(gmap.size()==0) {
      gmap = vector<int>(scdata.generators[GID].size(), -1);
      for(int g=0; g<scdata.G_Generator.size(); g++) {
	assert(g>=0);
	assert(scdata.G_Generator[g]<scdata.generators[GID].size());
      
	gmap[scdata.G_Generator[g]] = g;
      }
    }
    assert(scdata.G_Bus.size() == scdata.G_BusUnitNum.size());
    assert(scdata.generators[GID].size() ==  gmap.size());

    int g;
    //write generator section
    fprintf(file, "--generator section\ni, id, p(MW), q(MW)\n");
    for(int gi=0; gi<scdata.generators[GI].size(); gi++) {
      g = gmap[gi];
      if(-1 == g) {
	fprintf(file, "%s, \'%s\', 0, 0\n", scdata.generators[GI][gi].c_str(), scdata.generators[GID][gi].c_str());
      } else {
	assert(g>=0);
	assert(g<scdata.G_Bus.size());
	//assert(g<p_g->n);
	//assert(g<q_g->n);


	fprintf(file, "%d, \'%s\', %.20f, %.20f\n", 
		scdata.G_Bus[g], scdata.G_BusUnitNum[g].c_str(), 
		scdata.MVAbase*p_g[g], scdata.MVAbase*q_g[g]);
      }
    }

    if(file_in==NULL) {
      fclose(file);
    }
    //printf("solution block written to file %s\n", filename.c_str());
  }


  //static
  bool SCACOPFIO::read_solution1(std::vector<int>& I_n,  std::vector<double>& v_n, 
				 std::vector<double>& theta_n, std::vector<double>& b_n,
				 std::vector<int>& I_g, std::vector<std::string>& ID_g,
				 std::vector<double>& p_g, std::vector<double>& q_g,
				 const std::string& filename)
  {

    I_n.clear(); v_n.clear(); 
    theta_n.clear(); b_n.clear();
    I_g.clear(); ID_g.clear();
    p_g.clear(); q_g.clear();

    ifstream rawfile(filename.c_str());
    if(!rawfile.is_open()) {
      printf("failed to load raw file %s\n", filename.c_str());
      return false;
    }
    string line; bool ret;

    ret = (bool)getline(rawfile, line); assert(ret);
#ifdef DEBUG
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    assert(line.find("--bus section") != string::npos);
#endif
    ret = (bool)getline(rawfile, line); assert(ret);

    // read bus section
    while(true) {
      ret = (bool)getline(rawfile, line); assert(ret);
      if(!ret) break;

      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      if(line.find("--generator section") != string::npos)
	break;
  
      auto tokens = split(line, ','); assert(tokens.size()==4);
      I_n.push_back(atoi(tokens[0].c_str()));
      v_n.push_back(strtod(tokens[1].c_str(), NULL));
      theta_n.push_back(strtod(tokens[2].c_str(), NULL));
      b_n.push_back(strtod(tokens[3].c_str(), NULL));
    }
    // ignoring headers
    ret = (bool)getline(rawfile, line); assert(ret);

    // read generator section
    while(true) {

      ret = (bool)getline(rawfile, line); 
      if(!ret) break;
      if(line.size()==0 || line.substr(0,2)=="--") break;

      vector<string> tokens = split(line, ','); assert(tokens.size()==4);
      I_g.push_back(atoi(tokens[0].c_str()));

      trim(tokens[1]);
      ID_g.push_back(tokens[1]);

      string& s = ID_g.back();
      s.erase(remove(s.begin(), s.end(),'\''), s.end());

      p_g.push_back(strtod(tokens[2].c_str(), NULL));
      q_g.push_back(strtod(tokens[3].c_str(), NULL));
    }

    return true;
  }

  //static
  void SCACOPFIO::read_solution1(OptVariablesBlock** v_n0, OptVariablesBlock** theta_n0, 
				 OptVariablesBlock** b_s0,
				 OptVariablesBlock** p_g0, OptVariablesBlock** q_g0,    
				 SCACOPFData& data,
				 const std::string& filename)
  {
    std::vector<int> I_n; std::vector<double> v_n;
    std::vector<double> theta_n; std::vector<double> b_n;
    std::vector<int> I_g; std::vector<std::string> ID_g;
    std::vector<double> p_g; std::vector<double> q_g;
    if(!read_solution1(I_n, v_n, theta_n, b_n, 
		       I_g, ID_g, p_g, q_g, filename)) {
      printf("[error] failed to get raw data from solution1 file '%s'\n", filename.c_str());
      return;
    }

    auto Nidx = indexin(data.N_Bus, I_n);

    assert(v_n.size()==data.N_Bus.size());
    assert(theta_n.size()==data.N_Bus.size());

    *v_n0 = new OptVariablesBlock(data.N_Bus.size(), var_name("v_n",data), 
				  data.N_Vlb.data(), data.N_Vub.data()); 
    *theta_n0= new OptVariablesBlock(data.N_Bus.size(), var_name("theta_n",data));
    (*v_n0)->set_xref_to_x(); (*theta_n0)->set_xref_to_x();

    for(int i=0; i<(*v_n0)->n; i++) {
      assert(Nidx[i]>=0 && Nidx[i]<v_n.size());
      (*v_n0)->x[i] = v_n[Nidx[i]];
    }
    double piover180 =  M_PI/180.;
    for(int i=0; i<(*theta_n0)->n; i++) {
      assert(Nidx[i]>=0 && Nidx[i]<v_n.size());
      (*theta_n0)->x[i] = theta_n[Nidx[i]] * piover180;
    }

    *b_s0 = new OptVariablesBlock(data.SSh_SShunt.size(), var_name("b_s",data), 
				  data.SSh_Blb.data(), data.SSh_Bub.data());
    (*b_s0)->set_xref_to_x(); 

    auto SSh_Nidx = indexin(data.SSh_Bus, data.N_Bus);

    for(auto& v: b_n) v /= data.MVAbase;

    double bn;
    for(int ssh=0; ssh<(*b_s0)->n; ssh++) {
      assert(SSh_Nidx[ssh]>=0 && SSh_Nidx[ssh]<b_n.size());
      bn = b_n[SSh_Nidx[ssh]];
      if(bn < data.SSh_Blb[ssh])
	(*b_s0)->x[ssh] = data.SSh_Blb[ssh];
      else if(bn>data.SSh_Bub[ssh])
	(*b_s0)->x[ssh] = data.SSh_Bub[ssh];
      else 
	(*b_s0)->x[ssh] = bn;

      b_n[SSh_Nidx[ssh]] -= (*b_s0)->x[ssh];
    }

    double sum=0.; for(double& v: b_n) sum += fabs(v);
    if(sum>1e-4) printf("[warning] there are %g MVAR unassigned to shunts.", data.MVAbase*sum);

    for(auto& v: p_g) v /= data.MVAbase;
    for(auto& v: q_g) v /= data.MVAbase;

    //
    // p_g and q_q
    //
    *p_g0 = new OptVariablesBlock(data.G_Generator.size(), var_name("p_g",data), 
				  data.G_Plb.data(), data.G_Pub.data());
    *q_g0 = new OptVariablesBlock(data.G_Generator.size(), var_name("q_g",data), 
				  data.G_Qlb.data(), data.G_Qub.data());
    (*p_g0)->set_xref_to_x(); (*q_g0)->set_xref_to_x();

    int ng = data.G_Generator.size();
    vector<string> vBBUN = vector<string>(ng); //G[:Bus], ":", G[:BusUnitNum]
    for(int i=0; i<ng; i++) {
      //!vBBUN[i] = to_string(G_Bus[i]) + ":" + to_string(G_BusUnitNum[i]);
      vBBUN[i] = to_string(data.G_Bus[i]) + ":" + data.G_BusUnitNum[i];
    }
    vector<string> vIgIDg = vector<string>(I_g.size());
    for(int i=0; i<I_g.size(); i++) {
      vIgIDg[i] = to_string(I_g[i]) + ":" + ID_g[i];
    }

    auto Gidx = indexin(vBBUN, vIgIDg);
    assert(Gidx.size() == ng);

    for(int g=0; g<ng; g++) {
      assert(Gidx[g]>=0 && Gidx[g]<p_g.size());
      (*p_g0)->x[g] = p_g[Gidx[g]];
      (*q_g0)->x[g] = q_g[Gidx[g]];
    }
  }

  void SCACOPFIO::write_variable_block(OptVariablesBlock* var, SCACOPFData& data, FILE* file)
  {
    assert(NULL != file);
    fprintf(file, "v %s %d\n", var->id.c_str(), var->n);
    for(int i=0; i<var->n; i++) fprintf(file, "%.20f\n", var->x[i]);
  }

  void SCACOPFIO::read_variables_blocks(SCACOPFData& data, 
					std::unordered_map<std::string, OptVariablesBlock*>& map_basecase_vars)
  {
    string filename = "solution_b_pd.txt";
    ifstream file(filename.c_str());
    if(!file.is_open()) {
      printf("[warning] failed to load solution file '%s'\n", filename.c_str());
      return;
    }
    size_t pos; bool ret; string line; 
    ret = (bool)getline(file, line); assert(ret);
    while(true) {
      if(line.size()<3) { assert(false); continue; }
      if(line[0] != 'v') {assert(false); continue; }
      vector<string> tokens = split(line, ' ');
      if(tokens.size() != 3) { assert(false); continue; }
      assert(tokens[0] == "v"); 
      const string& var_name = tokens[1];
      const int var_size = atoi(tokens[2].c_str());
      
      OptVariablesBlock* var = new OptVariablesBlock(var_size, var_name);
      var->set_xref_to_x();
      for(int l=0; l<var_size; l++) {
	ret = (bool)getline(file, line); assert(ret);
	assert(line.size()>0); assert(line[0] != 'v');
	if(ret) var->x[l] = atof(line.c_str());
	else    var->x[l] = 0.;
      }
      map_basecase_vars.insert({var_name, var});

      ret = (bool)getline(file, line); 
      if(!ret) break;
    }
  }

} //end namespace gollnlp

#ifndef GO_SCOPFDATA
#define GO_SCOPFDATA

#include <string>
#include <vector>

#include <iostream>

namespace gollnlp {
  class SCACOPFData {

  public:
    SCACOPFData();

    bool readinstance(const std::string& raw, const std::string& rop, const std::string& inl, const std::string& con);
  private:
    void buildindexsets(bool ommit_K_related=false);
  public:
    // to keep things vectorized, "cut"/"sliced" copies of data are kept. This methods performs these cuts and
    // (hard-)clear vectors not used in the contigencies
    //'nCont' refers to the total # of contingencies considered
    void rebuild_for_conting(int K_id, int nCont);

    //utilities
    int bus_with_largest_gen() const;

    //void compute_largest_pg_loss_contingency();

    // //return true if the at least one bound was tightened; otherwise false
    // //on entry, plb and pub contain the current bounds
    // //on entry, all arrays are G_Generator.size() (data_sc)
    // //bool compute_pg_bounds_for_Kgens(const double* p_g0_in, double* plb, double* pub);


    //Gk    - indexes of all generators 
    //Gkp   - indexes of AGC participating generators
    //Gknop - indexes of ACG non-participating generators
    // all the above sets except 'outidx' if 'ConType' of Kidx is generator
    void get_AGC_participation(int Kidx, std::vector<int>& Gk, std::vector<int>& Gkp, std::vector<int>& Gknop);

    //returns a vector with areas 
    //void get_AGC_areas(std::vector<int>& vAreas);
    //idxs of contingency generators: first vec is of gen idxs in G_Generator, the second is of idxs in K_Contingency
    //void get_Kgen_idxs(std::vector<int>& gen_idxs, std::vector<int>& K_idxs);
    //get a list of AGC generators (idxs in G_Generator) in the specified area
    //void get_AGC_gens_in_area(int area);
    
  public:
    // 0 when used for ACOPF, conting index (1-based) for contingency subproblems
    int id;

    double MVAbase;

    // - buses
    std::vector<int> N_Bus, N_Area;
    std::vector<double> N_Pd, N_Qd, N_Gsh, N_Bsh, N_Vlb, N_Vub, N_EVlb, N_EVub, N_v0, N_theta0;

    // - linesg
    std::vector<int> L_Line, L_From, L_To;
    std::vector<std::string> L_CktID;
    std::vector<double> L_G, L_B, L_Bch, L_RateBase, L_RateEmer;

    // - transformers
    std::vector<int> T_Transformer, T_From, T_To;
    std::vector<std::string> T_CktID;
    std::vector<double> T_Gm, T_Bm, T_G, T_B, T_Tau, T_Theta, T_RateBase, T_RateEmer;

    // - switched shunts
    std::vector<int> SSh_SShunt, SSh_Bus;
    std::vector<double> SSh_Blb, SSh_Bub, SSh_B0;

    // - generators
    std::vector<int> G_Generator, G_Bus;
    std::vector<std::string> G_BusUnitNum;
    std::vector<double> G_Plb, G_Pub, G_Qlb, G_Qub, G_p0, G_q0, G_alpha;
    std::vector<std::vector<double> > G_CostPi, G_CostCi;

    // - contingencies
    enum KType{kNotInit=-1, kGenerator, kLine, kTransformer};
    //[index], [index in G_Gen, L_Line, or T_Trans] 
    std::vector<int> K_Contingency, K_IDout;
    std::vector<KType> K_ConType;
    std::vector<std::string> K_Label;

    inline std::string cont_type_string(const int& k) {
      if(K_ConType[k]==kGenerator) return "Generator";
      if(K_ConType[k]==kLine) return "Line";
      if(K_ConType[k]==kTransformer) return "Transformer";
      return "Error/Unknown";
    }

    //penalties
    enum PenaltyType{pP=0, pQ=1, pS=2};
    std::vector<std::vector<double> > P_Quantities, P_Penalties;
    
    double DELTA;
    double PenaltyWeight; //DELTA for base case, (1-DELTA)/nK for contingencies
    //
    // -- index sets for efficient iteration
    //
    //indexes of L_From and L_To in N_Bus (stored in L_Nidx[0] and L_Nidx[1])
    std::vector<std::vector<int> > L_Nidx;
    //same as above but of T in N
    std::vector<std::vector<int> > T_Nidx;
    //indexin(SSh[:Bus], N[:Bus])
    std::vector<int> SSh_Nidx;
    //indexin(G[:Bus], N[:Bus])
    std::vector<int> G_Nidx;

    //size nbus -> each vector element 'b' contains a (vector of) lines/transformers
    //attached to bus 'b' 
    std::vector<std::vector<int> > Lidxn1, Lidxn2;
    std::vector<std::vector<int> > Tidxn1, Tidxn2;

    //size nbus -> each vector element 'b' contains a (vector of) shunts/generators
    //attached to 'b'
    std::vector<std::vector<int> > SShn, Gn;

    //indexes of the out element (gen, line, transf) in the corresponding
    //G_Generator, L_Line, or T_Transformer vector (base case)
    std::vector<int> K_outidx;

    //"less" public members
    enum Gheader{GI=0,GID,GPG,GQG,GQT,GQB,GVS,GIREG,GMBASE,GZR,GZX,GRT,GXT,
		 GGTAP,GSTAT,GRMPCT, GPT,GPB,GO1,GF1,GO2,GF2,GO3,GF3,GO4,
		 GF4,GWMOD,GWPF}; //generators header in the raw file

    std::vector<std::vector<std::string> > generators;
  protected:
    struct Contingency{
      virtual ~Contingency() {}; 
    };

    struct GeneratorContingency : Contingency {
    public:
      int Bus; std::string unit;
      GeneratorContingency(int B, const std::string& u) : Bus(B), unit(u) {};
      friend std::ostream& operator<<(std::ostream& os, const GeneratorContingency& o)
      {
	os << o.Bus << ":" << o.unit;
	return os;
      }
      virtual ~GeneratorContingency() {};
    };

    struct TransmissionContingency : Contingency {
    public:
      int FromBus; int ToBus; std::string Ckt;
      TransmissionContingency(int F, int T, const std::string& C)
	: FromBus(F), ToBus(T), Ckt(C) {};
      virtual ~TransmissionContingency() {};
    };
    enum ContingencyType{cGenerator, cBranch};

  protected:
    typedef std::vector<std::vector<std::string> > VVStr;
    typedef std::vector<std::vector<int> > VVInt;
    typedef std::vector<std::vector<double> > VVDou;
    typedef std::vector<std::string> VStr;
    typedef std::vector<int> VInt;
    typedef std::vector<double> VDou;
    bool readRAW(const std::string& raw_file, double& MVAbase,
		 VVStr& buses,  VVStr& loads, VVStr& fixedbusshunts,
		 VVStr& generators, VVStr& ntbranches, VVStr& tbranches,
		 VVStr& switchedshunts);
    bool readROP(const std::string& rop_file, VVStr& generatordsp, VVStr& activedsptables, 
		 VInt& costcurves_ltbl, VStr& costcurves_label, VVDou& costcurves_xi, VVDou& costcurves_yi);
    bool readINL(const std::string& inl_file, VVStr& governorresponse);

    //here contingency type = 0 <-generator or 1 <- branch
    bool readCON(const std::string& con_file, 
		 VStr& contingencies_label, 
		 std::vector<ContingencyType>& contingencies_type,
		 std::vector<Contingency*>& contingencies_con);

    void convert(const VStr& src, VInt& dest);
    void convert(const VStr& src, VDou& dest);

  public:
    int my_rank;
  };//end of class

} //end namespace

#endif

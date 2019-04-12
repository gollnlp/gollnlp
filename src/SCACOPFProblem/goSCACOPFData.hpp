#ifndef GO_SCOPFDATA
#define GO_SCOPFDATA

#include <string>
#include <vector>

#include <iostream>

namespace gollnlp {

  class goSCACOPFData {

  public:
    goSCACOPFData();

    bool readinstance(const std::string& raw, const std::string& rop, const std::string& inl, const std::string& con);

    void buildindexsets();
  public:
    double MVAbase;

    // - buses
    std::vector<int> N_Bus, N_Area;
    std::vector<double> N_Pd, N_Qd, N_Gsh, N_Bsh, N_Vlb, N_Vub, N_EVlb, N_EVub, N_v0, N_theta0;

    // - lines
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
    std::vector<int> G_Generator, G_Bus, G_BusUnitNum;
    std::vector<double> G_Plb, G_Pub, G_Qlb, G_Qub, G_p0, G_q0, G_alpha;
    std::vector<std::vector<double> > G_CostPi, G_CostCi;

    // - contingencies
    enum KType{kNotInit=-1, kGenerator, kLine, kTransformer};
    //[index], [index in G_Gen, L_Line, or T_Trans] 
    std::vector<int> K_Contingency, K_IDout;
    std::vector<KType> K_ConType;

    //penalties
    enum PenaltyType{pP=0, pQ=1, pS=2};
    std::vector<std::vector<double> > P_Quantities, P_Penalties;

    //
    // -- index sets for efficient iteration
    //indexes of L_From and L_To in N_Bus (stored in L_Nidx[0] and L_Nidx[1])
    std::vector<std::vector<int> > L_Nidx;
    //same as above but of T in N
    std::vector<std::vector<int> > T_Nidx;
    //indexin(SSh[:Bus], N[:Bus])
    std::vector<int> SSh_Nidx;
    //indexin(G[:Bus], N[:Bus])
    std::vector<int> G_Nidx;
    //same as above but for reverse referencing (from buses to X)
    std::vector<std::vector<int> > Lidxn, Lin;
    std::vector<std::vector<int> > Tidxn, Tin;
    std::vector<std::vector<int> > SShn, Gn;
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
};//end of class
} //end namespace

#endif

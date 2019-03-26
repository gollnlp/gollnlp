#ifndef GO_SCOPFDATA
#define GO_SCOPFDATA

#include <string>
#include <vector>

namespace gollnlp {

  class goSCACOPFData {

  public:
    goSCACOPFData();

    bool readinstance(const std::string& raw, const std::string& rop, const std::string& inl, const std::string& con);

  public:
    double MVAbase;

    //buses
    std::vector<int> N_Bus, N_Area;
    std::vector<double> N_Pd, N_Qd, N_Gsh, N_Bsh, N_Vlb, N_Vub, N_EVlb, N_EVub, N_v0, N_theta0;

    //lines
    std::vector<int> L_Line, L_From, L_To, L_CktID;
    std::vector<double> L_G, L_B, L_Bch, L_RateBase, L_RateEmer;

    //transformers
    std::vector<int> T_Transformer, T_From, T_To, T_CktID;
    std::vector<double> T_Gm, T_Bm, T_G, T_B, T_Tau, T_Theta, T_RateBase, T_RateEmer;

    //switched shunts
    std::vector<int> SSh_SShunt, SSh_bus;
    std::vector<double> SSh_Blb, SSh_Bub, SSh_B0;

    //generators
    std::vector<int> G_Generator, G_bus, G_BusUnitNum;
    std::vector<double> G_Plb, G_Pub, G_Qlb, G_Qub, G_p0, G_q0;
    std::vector<std::vector<double> > G_CostPi, G_CostCi;

    //contingencies
    enum ContingencyType{Generator, Line, Transformer};
    std::vector<int> K_Contingency, K_IDout;
    std::vector<ContingencyType> K_ConType;

    //penalties
    enum PenaltyType{P=0, Q=1, S=2};
    std::vector<std::vector<double> > P_Quantities, P_Penalties;


  protected:
    typedef std::vector<std::vector<std::string> > VVStr;
    typedef std::vector<std::vector<double> > VVDou;
    typedef std::vector<std::string> VStr;
    typedef std::vector<int> VInt;
    bool readRAW(const std::string& raw_file, double& MVAbase,
		 VVStr& buses,  VVStr& loads, VVStr& fixedbusshunts,
		 VVStr& generators, VVStr& ntbranches, VVStr& tbranches,
		 VVStr& switchedshunts);
    bool readROP(const std::string& raw_file, VVStr& generatordsp, VVStr& activedsptables, 
		 VInt& costcurves_ltbl, VStr& costcurves_label, VVDou& costcurves_xi, VVDou& costcurves_yi);
  };
} //end namespace

#endif

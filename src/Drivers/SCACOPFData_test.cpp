#include "goSCACOPFData.hpp"

using namespace gollnlp;

int main()
{

  {
    goSCACOPFData data;
    data.readinstance("../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/Network_01R-10/scenario_1/case.raw",
		      "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/Network_01R-10/case.rop",
		      "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/Network_01R-10/scenario_1/case.con",
		      "../../goinstances/challenge1/Original_Dataset_1-4/Original_Dataset_Real-Time_Edition_1/Network_01R-10/case.inl");
  }
  return 0;
}

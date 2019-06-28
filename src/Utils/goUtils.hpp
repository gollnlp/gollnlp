#ifndef GOLLNLP_UTILS
#define GOLLNLP_UTILS
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <cassert>

namespace gollnlp {
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

static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  
  while(getline(ss, item, delim)) result.push_back (item);
  
  return result;
}

static std::vector<std::string> split_skipempty(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  
  while(getline(ss, item, delim)) {
    if(!item.empty())
      result.push_back(item);
  }
  return result;
}

static inline bool mygetline(std::ifstream& file, std::string& line)
{
  if(!getline(file,line)) return false;
  if(line.size()==0) return true;
  std::string::iterator last = line.end()-1;
  if(*last=='\r') line.erase(last);
  return true;
}
template<class T> inline void printvec(const std::vector<T>& v, const std::string& msg="") 
{ 
  std::cout.precision(6); 
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  typename std::vector<T>::const_iterator it=v.begin();
  for(;it!=v.end(); ++it) std::cout << (*it) << " ";
  std::cout << std::endl;
}

template<class T> inline void printvecvec(const std::vector<std::vector<T> >& v, const std::string& msg="") 
{ 
  std::cout.precision(6); 
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  for(auto& l: v) {
    for(auto& c: l) std::cout << c << " ";
    std::cout << std::endl;
  }
}
template<class T> inline void hardclear(std::vector<T>& in) { std::vector<T>().swap(in); }


//index of 'e' in v; return -1 if not found
template<class T> inline int indexin(const std::vector<T>& v, const T& e)
{
  auto it = std::find(v.begin(), v.end(), e);
  if(it==v.end()) 
    return -1;
  else
    return std::distance(v.begin(), it);
}
// for entries of 'v' that are not present in 'in', the indexes will be set to -1
template<class T> inline std::vector<int> indexin(const std::vector<T>& v, const std::vector<T>& in)
{
  std::vector<int> vIdx(v.size());
  std::iota(vIdx.begin(), vIdx.end(), 0);
  //sort permutation for v
  sort(vIdx.begin(), vIdx.end(), [&](const int& a, const int& b) { return (v[a] < v[b]); } );

  std::vector<int> inIdx(in.size());
  std::iota(inIdx.begin(), inIdx.end(), 0);
  //sort permutation for in
  sort(inIdx.begin(), inIdx.end(), [&](const int& a, const int& b) { return (in[a] < in[b]); } );

  size_t szv=v.size(), szin=in.size();
  std::vector<int> idxs(szv, -1);
  
  for(int iv=0, iin=0; iv<szv && iin<szin;) {
    //cout << iv << "|" << iin << "  " << v[vIdx[iv]] <<"|" << in[inIdx[iin]] << endl;
    if(v[vIdx[iv]]==in[inIdx[iin]]) {
	idxs[vIdx[iv]]=inIdx[iin];
	iv++; 
      } else v[vIdx[iv]]>in[inIdx[iin]] ? iin++: iv++;
  }
  return idxs;
}

// returns the indexes 'i' in 'v', for which 'v[i]' satisfies (unary) predicate
template<class T>
std::vector<int> findall(const std::vector<T>& v, std::function<bool(const int&)> pred)
{
  std::vector<int> ret; int count=0;
  for(auto& it : v) {
    if(pred(it)) ret.push_back(count);
    count++;
  }
  return ret;
}

// erase first elem 'e' from the vector; if e is not in v return false, otherwise true
template<class T> inline bool erase_elem_from(std::vector<T>& v, const T& e) 
{ 
  auto it = std::find(v.begin(), v.end(), e);
  if(it==v.end()) 
    return false;
  v.erase(it);
  return true;
}

//erase elem at index 'i' from the vector
template<class T> inline void erase_idx_from(std::vector<T>& v, const int& idx)
{
  assert(idx>=0);
  v.erase(v.begin()+idx);
}

//return difference A\B
//this function is not optimized for large B
template<class T>
inline std::vector<T> set_diff(const std::vector<T>& A, const std::vector<T>& B)
{

  std::vector<T> diff=A;
  auto idxs_B_in_A = indexin(B,A);
  for(auto idx : idxs_B_in_A) {
    //if(idx>=0) erase_idx_from(diff,idx);
    if(idx>=0) erase_elem_from(diff, A[idx]);
  }
  return diff;
}

template<class T> std::vector<T> selectfrom(const std::vector<T>& v, const std::vector<int>& idx)
{
  std::vector<T> ret;
  for(auto& keep: idx) {
    assert(keep>=0);
    ret.push_back(v[keep]);
  }
  return ret;
}

//j=max(i,j) and returns min(i,j)
inline int uppertr_swap(const int& i, int& j, int& aux) {
  if(i>j) { aux=j; j=i; return aux; } return i;
}  

}//end namespace
#endif

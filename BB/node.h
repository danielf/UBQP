#ifndef NODE_H
#define NODE_H

#include <armadillo>
#include <utility>
#include <vector>

using namespace arma;
using namespace std;
class node {
  public:
    static int getMinNum();
    static int getMaxNum();
    node(mat &Q, colvec &b, vector<int>& dic, vector<int>& val, double already);
    pair<node, node> branch();
    vector<int> val;
    int numVar();
    void newdiag(colvec d);
    bool possible();
    void fix();
  private:
    static int min_num;
    static int max_num;
    static double incumbent;
    vector<int> dic;
    mat Q;
    colvec b;
    vector<double> p0, p1;
    double already;
    bool _fix();
    void do_fix(int var, int value);
};

#endif

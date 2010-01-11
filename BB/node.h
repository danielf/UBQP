#ifndef NODE_H
#define NODE_H

#include <armadillo>
#include <utility>
#include <vector>

using namespace arma;
using namespace std;
class node {
  public:
    int getMinNum() const;
    int getMaxNum() const;
    node(const mat &Q, const colvec &b);
    pair<node, node> branch() const;
    vector<int> val;
    int numVar() const;
    void newdiag(colvec d);
    bool possible() const;
    bool fix();
    double already;
    static double incumbent;
    static int fixed;
    static int nodes;
    void do_fix(int var, int value);
		void do_brute_force();
    mat Q;
    colvec b;
  private:
    node clone() const;
    int min_num;
    int max_num;
    vector<int> dic;
    vector<double> p0, p1;
    bool _fix();
    int n0, n1;
		double lb;
};

#endif

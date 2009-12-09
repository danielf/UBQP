#include "node.h"
#include <math.h>
#include <algorithm>

using namespace arma;
using namespace std;

#define EPS 1.e-4
static void solve_chol(mat& U, colvec& b, colvec& x) {
  // First solve U^Ty = Ly = b
  colvec y;
  for (int i = 0; i < U.n_rows; i++) {
    double temp = b[i];
    for (int j = 0; j < i; j++)
      temp -= y[j]*U.at(j, i);
    y[i] = temp / U.at(i,i);
  }
  // Now solve Ux = y
  x.copy_size(b);
  for (int i = U.n_rows-1; i >= 0; i--) {
    double temp = y[i];
    for (int j = U.n_rows-1; j > i; j--)
      temp -= x[j]*U.at(i, j);
    x[i] = temp / U.at(i,i);
  }
}

static pair<double, double> optimize(mat& U, colvec& QIb, colvec& b, colvec& c, double K) {
  colvec QIc;
  solve_chol(U, c, QIc);
  double A = 0.5*dot(c, QIc);
  double C = -0.5*dot(b, QIb) - K;
  double delta = - 4*A*C;
  double lambda1 = (-sqrt(delta))/(2*A);
  double lambda2 = (+sqrt(delta))/(2*A);
  colvec x1 = QIb + lambda1*QIc;
  colvec x2 = QIb + lambda2*QIc;
  double v1 = dot(c, x1);
  double v2 = dot(c, x2);
  return make_pair(v1, v2);
}

static pair<double, double> inv_optimize(mat& U, colvec& QIb, colvec& b, colvec& c) {
  colvec QIc;
  solve_chol(U, c, QIc);
  double A = 0.5*dot(c, QIc);
  double C1 = (-pow(((1-dot(c, QIb))/dot(c, QIc))*2*A, 2))/(4*A);
  double K1 = -.5*dot(b, QIb) - C1;
  double C2 = (-pow(((dot(c, QIb))/dot(c, QIc))*2*A, 2))/(4*A);
  double K2 = -.5*dot(b, QIb) - C2;
  return make_pair(K2, K1);
}


// Number of variables set to 1
int node::getMinNum() {
  return min_num;
}

int node::getMaxNum() {
  return max_num;
}

int node::min_num(0);
int node::max_num(1000000);

double node::incumbent(0);

int node::numVar() {
  return Q.n_rows;
}

node::node(mat &Q, colvec &b, vector<int>& dic, vector<int>& val, double already) : Q(Q),
                                                                                    b(b),
                                                                                    dic(dic),
                                                                                    val(val),
                                                                                    already(already) {
  p0.clear(); p1.clear();
  for (int i = 0; i < Q.n_rows; i++) {
    p0.push_back(-INFINITY);
    p1.push_back(-INFINITY);
  }
}

bool node::possible() {
  int n0 = 0, n1 = 0;
  if (incumbent == INFINITY) return true;
  for (int i = 0; i < val.size(); i++) {
    if (val[i] == 0) n0++;
    if (val[i] == 1) n1++;
  }
  if (n1 > max_num) return false;
  if (n0 + numVar() < min_num) return false;
  return true;
}

void node::newdiag(colvec d) {
  mat Q2, U;
  colvec eigval, b2, c, QIb;
  
  Q2 = Q + 2*diagmat(d);
  eigval = eig_sym(Q2);
  
  d -= .5*(eigval[0] - 1.e-3)*ones(Q.n_rows);
  Q2 = Q + 2*diagmat(d);
  b2 = b + d;
  chol(U, Q2);
  solve_chol(U, b2, QIb);
  if (incumbent < INFINITY) {
    c = ones(Q.n_rows);
    pair<double, double> resp = optimize(U, QIb, b2, c, incumbent);
    min_num = max(min_num, (int)ceil(resp.first));
    max_num = min(max_num, (int)floor(resp.second));
  }
  c = zeros(Q.n_rows);
  for (int i = 0; i < Q.n_rows; i++) {
    c[i] = 1;
    pair<double, double> resp = inv_optimize(U, QIb, b2, c);
    p0[i] = max(p0[i], resp.first);
    p1[i] = max(p1[i], resp.second);
    c[i] = 0;
  }
}

void node::do_fix(int var, int value) {
  if (value == 1) {
    already -= b[var];
    b -= Q.row(var);
    val[dic[var]] = 1;
  } else val[dic[var]] = 0;
  Q.swap_rows(var, Q.n_rows-1);
  Q.swap_cols(var, Q.n_cols-1);
  Q = Q.submat(0, 0, Q.n_rows-2, Q.n_cols-2);
  b.swap_rows(var, b.n_rows-1);
  b = b.rows(0, b.n_rows-2);
  dic[var] = dic[dic.size()-1];
  dic.pop_back();
  p0[var] = p0[p0.size()-1];
  p0.pop_back();
  p1[var] = p1[p0.size()-1];
  p1.pop_back();
}

bool node::_fix() {
  if (incumbent == INFINITY) return false;
  for (int i = 0; i < Q.n_rows; i++) {
    if (p1[i] > incumbent + EPS || p0[i] > incumbent + EPS) { // Tem que ser 0 ou 1
      if (p1[i] > incumbent + EPS) do_fix(i, 0);
      if (p0[i] > incumbent + EPS) do_fix(i, 1);
      return true;
    }
  }
  return false;
}

void node::fix() {
  while (_fix());
}

pair<node, node> node::branch() {
  pair<node, node> resp = make_pair(node(Q, b, dic, val, already), node(Q, b, dic, val, already));
  int best0, best1;
  best0 = max_element(p0.begin(), p0.end()) - p0.begin();
  best1 = max_element(p1.begin(), p1.end()) - p1.begin();
  if (p0[best0] > p1[best1]) {
    resp.first.do_fix(best0, 1);
    resp.second.do_fix(best0, 0);
  } else {
    resp.first.do_fix(best1, 0);
    resp.second.do_fix(best1, 1);
  }
  return resp;
}

#include "node.h"
#include <math.h>
#include <algorithm>

using namespace arma;
using namespace std;

#define EPS 1.e-1
static void solve_chol(mat& U, colvec& b, colvec& x) {
  // First solve U^Ty = Ly = b
  colvec y;
	y.copy_size(b);
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
int node::getMinNum() const {
  return min_num;
}

int node::getMaxNum() const {
  return max_num;
}

double node::incumbent(0);

int node::numVar() const {
  return val.size() - n0 - n1;
}

node::node(const mat &Q, const colvec &b) : Q(Q), 
                                            b(b) {
  p0.clear(); p1.clear();
  min_num = 0;
  max_num = 1000000;
  already = 0;
  dic.clear();
  val.clear();
	n0 = 0;
	n1 = 0;
  for (int i = 0; i < Q.n_rows; i++) {
    val.push_back(-1);
    dic.push_back(i);
    p0.push_back(-INFINITY);
    p1.push_back(-INFINITY);
  }
	lb = 0.;
}

node node::clone() const {
  node temp(Q, b);
  temp.p0 = p0;
  temp.p1 = p1;
  temp.already = already;
  temp.min_num = min_num;
  temp.max_num = max_num;
  temp.val = val;
  temp.n0 = n0;
  temp.n1 = n1;
  temp.dic = dic;
	return temp;
}

bool node::possible() const {
  for (int i = 0; i < val.size(); i++) if (val[i] == -1) {
    if (p0[i] > incumbent - already + EPS && p1[i] > incumbent - already + EPS) {
			return false;
		}
  }
  if (n1 > max_num) {
		return false;
	}
  if (n1 + numVar() < min_num) {
		return false;
	}
  if (already + lb > incumbent + EPS) {
		return false;
	}
  return true;
}

void node::newdiag(colvec d) {
  mat Q2, U;
  colvec eigval, b2, c, QIb;
	double new_lb;
  
  Q2 = Q + 2*diagmat(d);
  eigval = eig_sym(Q2);
  
  d -= .5*(eigval[0] - 1.e-3)*ones(numVar());
  Q2 = Q + 2*diagmat(d);
  b2 = b + d;
  chol(U, Q2);
  solve_chol(U, b2, QIb);
	
	new_lb = .5*dot(QIb, Q2*QIb) - dot(b2, QIb);

	lb = min(lb, new_lb);

  if (incumbent < INFINITY) {
    c = ones(numVar());
    pair<double, double> resp = optimize(U, QIb, b2, c, incumbent - already);
    min_num = max(min_num, n1 + (int)ceil(resp.first));
    max_num = min(max_num, n1 + (int)floor(resp.second));
  }
  c = zeros(numVar());
  for (int i = 0; i < numVar(); i++) {
    c[i] = 1;
    pair<double, double> resp = inv_optimize(U, QIb, b2, c);
    p0[i] = max(p0[i], resp.first);
    p1[i] = max(p1[i], resp.second);
    c[i] = 0;
  }
}

void node::do_fix(int var, int value) {
	lb = 0;
  if (value == 1) {
    already -= b[var];
    b -= Q.col(var);
    val[dic[var]] = 1;
    n1++;
  } else {
    val[dic[var]] = 0;
    n0++;
  }
  b.swap_rows(var, b.n_rows-1);
  b = b.rows(0, b.n_rows-1-(numVar() > 0));
  Q.swap_rows(var, Q.n_rows-1);
  Q.swap_cols(var, Q.n_cols-1);
  Q = Q.submat(0, 0, Q.n_rows - 1 - (numVar() > 0), Q.n_cols-1-(numVar() > 0));
  dic[var] = dic[dic.size()-1];
  dic.pop_back();
  p0[var] = p0[p0.size()-1];
  p0.pop_back();
  p1[var] = p1[p1.size()-1];
  p1.pop_back();
}

bool node::_fix() {
  if (incumbent == INFINITY) return false;
  for (int i = 0; i < numVar(); i++) {
    if (p1[i] > incumbent - already + EPS || p0[i] > incumbent - already + EPS) { // Tem que ser 0 ou 1
      if (p1[i] > incumbent - already + EPS) do_fix(i, 0);
      else do_fix(i, 1);
      return true;
    }
  }
  return false;
}

bool node::fix() {
	bool ans = false;
  while (_fix()) {
		ans = true;
		fixed++;
	}
	return ans;
}
int node::fixed(0);
int node::nodes(1);
pair<node, node> node::branch() const {
  pair<node, node> resp = make_pair(clone(), clone());
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
	nodes += 2;
  return resp;
}

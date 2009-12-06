#include <armadillo>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>
using namespace std;
using namespace arma;

struct event {
  double time;
  int var;
  int type;
  event(int time, int var, int type): time(time), var(var), type(type) {}
  bool operator<(const event& rhs) const {
    if (time != rhs.time)
      return time < rhs.time;
    return type < rhs.type;
  }
};

void solve(mat& L, mat& U, colvec& b, colvec& x) {
  // First solve Ly = b
  colvec y;
  y.copy_size(b);
  for (int i = 0; i < L.n_rows; i++) {
    double temp = b[i];
    for (int j = 0; j < i; j++)
      temp -= y[j]*L.at(i,j);
    y[i] = temp / L.at(i,i);
  }
  // Now solve Ux = y
  x.copy_size(b);
  for (int i = L.n_rows-1; i >= 0; i--) {
    double temp = y[i];
    for (int j = L.n_rows-1; j > i; j--)
      temp -= x[j]*U.at(i,j);
    x[i] = temp / U.at(i,i);
  }
}

pair<double, double> solve2(mat& L, mat& U, colvec& QIb, colvec& b, colvec& c) {
  colvec QIc;
  solve(L, U, c, QIc);
  double A = 0.5*dot(c, QIc);
  double C1 = (-pow(((1-dot(c, QIb))/dot(c, QIc))*2*A, 2))/(4*A);
  double K1 = -.5*dot(b, QIb) - C1;
  double C2 = (-pow(((dot(c, QIb))/dot(c, QIc))*2*A, 2))/(4*A);
  double K2 = -.5*dot(b, QIb) - C2;
  return make_pair(K2, K1);
}
pair<double, double> solve(mat& L, mat& U, colvec& QIb, colvec& b, colvec& c, double K) {
  colvec QIc;
  solve(L, U, c, QIc);
  double A = 0.5*dot(c, QIc);
  double C = -0.5*dot(b, QIb) - K;
  double delta = - 4*A*C;
  double lambda1 = (- sqrt(delta))/(2*A);
  double lambda2 = (+ sqrt(delta))/(2*A);
  colvec x1 = QIb + lambda1*QIc;
  colvec x2 = QIb + lambda2*QIc;
  double v1 = dot(c, x1);
  double v2 = dot(c, x2);
  return make_pair(v1, v2);
}

int main() {
  double K;
  int n, m;
  scanf("%lf", &K);
  scanf("%d %d", &n, &m);
  mat Q = zeros(n, n);
  colvec bb = zeros(n);
  for (int i = 0; i < m; i++) {
    int a, b, p;
    scanf("%d %d %d", &a, &b, &p); a--; b--;
    if (b < a) swap(a, b);
    if (a == 0) {
      Q.at(a, b) += p;
      Q.at(b, a) += p;
      bb[a] += p;
    } else {
      Q.at(a, b) += 2*p;
      Q.at(b, a) += 2*p;
      bb[a] += p;
      bb[b] += p;
    }
  }
  colvec c = zeros(n);
  colvec p0 = -1000000*ones(n);
  colvec p1 = -1000000*ones(n);
/*  colvec mi = -1000000*ones(n);
  colvec ma = 1000000*ones(n);*/
  colvec d = 1000*ones(n);
  mat L, U;
  colvec QIb;
  double n_min = 1;
  double n_max = n;
  for (int q = 0; q <= 2*n; q++) {
    colvec dd = d;
    mat Q2 = Q + 2*diagmat(dd);
    colvec eigval = eig_sym(Q2);
    dd -= .5*(eigval[0] - 1.e-3)*ones(n);
    Q2 = Q + 2*diagmat(dd);
    Q2.print("Q2 = ");
    colvec b2 = bb + dd;
    if (!chol(U, Q2)) {
      printf("%d: NOT POSITIVE DEFINITE!!\n", q);
      colvec eigval = eig_sym(Q2);
      eigval.print("eigenvalues = ");
      continue;
    }
    L = trans(U);
    c = ones(n);
    solve(L, U, b2, QIb);
    pair<double, double> resp = solve(L, U, QIb, b2, c, K);
    n_min = max(n_min, resp.first);
    n_max = min(n_max, resp.second);
    c = zeros(n);
    for (int i = 0; i < n; i++) {
      c[i] = 1;
      //pair<double, double> resp = solve(L, U, QIb, b2, c, K);
      //mi(i) = max(mi(i), resp.first);
      //ma(i) = min(ma(i), resp.second);
      pair<double, double> resp2 = solve2(L, U, QIb, b2, c);
      p0[i] = max(p0[i], resp2.first);
      p1[i] = max(p1[i], resp2.second);
      //printf("%d %d: %lf %lf %lf %lf\n", q, i, resp.first, resp.second, resp2.first, resp2.second);
      c[i] = 0;
    }
    if (q < n) {
      if (q) d[q-1] = 1000;
      if (q < n) d[q] = 10;
    } else {
      if (q == n) d = 10*ones(n);
      if (q > n) d[q-n-1] = 10;
      if (q < 2*n) d[q-n] = 1000;
    }
  }
  printf("Tem que haver entre %lf e %lf\n", ceil(n_min), floor(n_max));
  vector<event> v;
  for (int i = 0; i < n; i++) {
    printf("%d: ult pra 0:%lf\tult pra 1:%lf\n", i+1, p0[i], p1[i]);
    v.push_back(event(p0[i], i, 0));
    v.push_back(event(p1[i], i, 1));
    v.push_back(event(min(p0[i], p1[i]), i, -1));
  }
  sort(v.begin(), v.end());
  reverse(v.begin(), v.end());
  for (vector<event>::iterator it = v.begin(); it != v.end(); it++) {
    printf("%lf: %d\t%d\n", it->time, it->var, it->type);
  }
  return 0;
}

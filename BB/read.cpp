#include "read.h"

#include <armadillo>
#include <stdio.h>

#include "node.h"

using namespace arma;

node read(const char* file) {
	FILE* arq = fopen(file, "rt");
	int n, m;
	fscanf(arq, "%d %d", &n, &m);
	mat Q = zeros(n,n);
	colvec bb = zeros(n);
	for (int i = 0; i < m; i++) {
		int a, b, p;
		fscanf(arq, "%d %d %d", &a, &b, &p); a--; b--;
		if (b < a) swap(a, b);
		Q.at(a, b) += 2*p;
		Q.at(b, a) += 2*p;
		bb[a] += p;
		bb[b] += p;
	}
	fclose(arq);
	return node(Q, bb);
}

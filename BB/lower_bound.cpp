#include "lower_bound.h"

#include <armadillo>
#include <stdio.h>
#include <sdpa_call.h>

using namespace arma;

#define D(i) (i+1)
#define T (n+1)

double lower_bound(mat& Q, colvec& b, colvec& d) {
	SDPA Problem;
	//Problem.setParameterType(SDPA::PARAMETER_STABLE_BUT_SLOW);
	Problem.setParameterType(SDPA::PARAMETER_DEFAULT);
	int n = Q.n_rows;
	Problem.inputConstraintNumber(n+1);
	Problem.inputBlockNumber(1);
	Problem.inputBlockSize(1, n+1);
	Problem.inputBlockType(1, SDPA::SDP);
	Problem.initializeUpperTriangleSpace();
	// cVect:
	for (int i = 0; i < n; i++) Problem.inputCVec(D(i), 0);
	Problem.inputCVec(T, -1);
	// Principal
	Problem.inputElement(T, 1, 1, 1, -2, true);
	//   Resto da primeira linha:
	for (int i = 0; i < n; i++) {
		Problem.inputElement(D(i), 1, 1, 2+i, 1, true);
		Problem.inputElement(0, 1, 1, 2+i, -b[i], true);
	}
	//    Parte grande:
	for (int i = 0; i < n; i++) Problem.inputElement(D(i), 1, i+2, i+2, 2, true);
	for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)
		Problem.inputElement(0, 1, i+2, j+2, -Q.at(i, j), true);

	Problem.initializeUpperTriangle();

	Problem.initializeSolve();

	Problem.solve();
	double *elements = Problem.getResultXVec();
	double ans = elements[T-1];
	d.copy_size(b);
	for (int i = 0; i < n; i++) d[i] = elements[i];
	Problem.terminate();
	return ans;
}

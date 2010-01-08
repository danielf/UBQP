#include <armadillo>
#include <stdio.h>
#include <sdpa_call.h>

using namespace arma;

#define D(i) (i+1)
#define T (n+1)

#define QUAL 0

int main(int argc, char* argv[]) {
	SDPA Problem;
	Problem.setParameterType(SDPA::PARAMETER_STABLE_BUT_SLOW);
	FILE* arq = fopen(argv[1], "rt");
	int n, m;
	double K2;
	fscanf(arq, "%lf", &K2);
	fscanf(arq, "%d %d", &n, &m);
	Problem.inputConstraintNumber(n+1);
	Problem.inputBlockNumber(1);
	Problem.inputBlockSize(1, n+1);
	Problem.inputBlockType(1, SDPA::SDP);
	Problem.initializeUpperTriangleSpace();
	// cVect:
	for (int i = 0; i < n; i++) Problem.inputCVec(D(i), 0);
	Problem.inputCVec(T, -1);
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

	// Principal
	Problem.inputElement(T, 1, 1, 1, -2, true);
	//   Resto da primeira linha:
	for (int i = 0; i < n; i++) {
		Problem.inputElement(D(i), 1, 1, 2+i, 1, true);
		Problem.inputElement(0, 1, 1, 2+i, -bb[i], true);
	}
	//    Parte grande:
	for (int i = 0; i < n; i++) Problem.inputElement(D(i), 1, i+2, i+2, 2, true);
	for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)
		Problem.inputElement(0, 1, i+2, j+2, -Q.at(i, j), true);

	Problem.initializeUpperTriangle();

	Problem.initializeSolve();

	Problem.solve();
	printf("ObjVal: -%lf\n", Problem.getPrimalObj());
	Problem.terminate();
	printf("Optimum: %lf\n", K2);
	Problem.printComputationTime();
	return 0;
}

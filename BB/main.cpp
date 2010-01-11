#include "node.h"
#include "read.h"
#include "lower_bound.h"

#include <armadillo>
#include <algorithm>
#include <utility>


using namespace arma;
using namespace std;

#define K 20
int phantom = 0;
void process(node& nod) {
	printf("Processing with %d variables (incumbent = %lf)\n", nod.numVar(), node::incumbent);
	do {
		if (nod.numVar() <= K) nod.do_brute_force();
		if (nod.numVar() == 0) {
			return;
		}
		colvec d;
		lower_bound(nod.Q, nod.b, d);
		nod.newdiag(d);
		d = zeros(nod.numVar());
		nod.newdiag(d);
		for (int i = 0; i < nod.numVar(); i++) {
			d[i] = 1000;
			nod.newdiag(d);
			d[i] = 0;
		}
		if (!nod.possible()) {
			phantom++;
			return;
		}
	} while (nod.fix());
	pair<node, node> b = nod.branch();
	process(b.first);
	process(b.second);
}

int main(int argc, char* argv[]) {
	node first = read(argv[1]);
	first.do_fix(0, 0);
	process(first);
	printf("Sol: %lf (fixed: %d\tnodes: %d\tPhantomed: %d)\n", node::incumbent, node::fixed, node::nodes, phantom);
	return 0;
}

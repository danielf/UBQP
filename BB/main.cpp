#include "node.h"
#include "read.h"

#include <armadillo>
#include <algorithm>
#include <utility>

using namespace arma;
using namespace std;

void process(node& nod) {
	printf("Processing with %d variables\n", nod.numVar());
	do {
		if (nod.numVar() == 0) {
			node::incumbent = min(node::incumbent, nod.already);
			return;
		}
		colvec d = zeros(nod.numVar());
		nod.newdiag(d);
		for (int i = 0; i < nod.numVar(); i++) {
			d[i] = 1000;
			nod.newdiag(d);
			d[i] = 0;
		}
	} while (nod.fix());
	if (!nod.possible()) return;
	pair<node, node> b = nod.branch();
	process(b.first);
	process(b.second);
}

int main(int argc, char* argv[]) {
	node first = read(argv[1]);
	
	first.do_fix(0, 0);
	process(first);

	printf("Sol: %lf (fixed: %d\tnodes: %d)\n", node::incumbent, node::fixed, node::nodes);
	return 0;
}

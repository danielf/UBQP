-include /home/daniel/sdp/sdpa.7.3.1.src/etc/make.inc
ICCFLAGS="-O3"

all: main

main: main.o node.o read.o lower_bound.o
	icpc -o main ${ICCFLAGS} main.o node.o read.o lower_bound.o -larmadillo ${SDPA_LIBS}

main.o: main.cpp node.h read.h brute.h lower_bound.h
	icpc -c ${ICCFLAGS} main.cpp

node.o: node.cpp node.h brute.h
	icpc -c ${ICCFLAGS} node.cpp

read.o: read.cpp read.h node.h
	icpc -c ${ICCFLAGS} read.cpp

lower_bound.o: lower_bound.cpp lower_bound.h
	icpc -c -I${SDPA_DIR}/include -I${MUMPS_DIR}/include -o $@ $< ${ICCFLAGS}

clean:
	rm -f *.o main

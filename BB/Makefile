ICCFLAGS="-O3"

all: main

main: main.o node.o read.o
	icc -o main ${ICCFLAGS} main.o node.o read.o -larmadillo

main.o: main.cpp node.h read.h
	icc -c ${ICCFLAGS} main.cpp

node.o: node.cpp node.h
	icc -c ${ICCFLAGS} node.cpp

read.o: read.cpp read.h node.h
	icc -c ${ICCFLAGS} read.cpp

clean:
	rm -f *.o main

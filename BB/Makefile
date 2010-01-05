all: main

main: main.o node.o read.o
	g++ -o main -g main.o node.o read.o -larmadillo

main.o: main.cpp node.h read.h
	g++ -c -g main.cpp

node.o: node.cpp node.h
	g++ -c -g node.cpp

read.o: read.cpp read.h node.h
	g++ -c -g read.cpp

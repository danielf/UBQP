
-include /home/daniel/sdp/sdpa.7.3.1.src/etc/make.inc

SRC = $(wildcard *.cpp)

main: main.o
	${CXX} -o $@ $< ${SDPA_LIBS} -O3

main.o: example.cpp
	${CXX} -c  \
					-I${SDPA_DIR}/include -I${MUMPS_DIR}/include -o $@ $< -O3

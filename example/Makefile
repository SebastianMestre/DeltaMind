
CXXFLAGS= -O2 -I../src

example0: example0.o ../build/libdeltamind.o
	g++ $(CXXFLAGS) -o $@ $^ $(LIBS)

example0.o: example0.cpp

../build/libdeltamind.o:
	cd ../build;make;cd ../example

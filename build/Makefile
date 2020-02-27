
path_to_source=../src
VPATH:=$(path_to_source)

CXX=g++
CXXFLAGS=-O2

OBJS=Network.o Trainer.o math.o

libdeltamind.o: $(OBJS)
	ar rcs $@ $^

Network.o: Network.cpp Network.hpp types.hpp math.hpp
Network.hpp: types.hpp

Trainer.o: Trainer.cpp Trainer.hpp Network.hpp types.hpp math.hpp
Trainer.hpp: types.hpp Network.hpp

types.o: types.cpp types.hpp
types.hpp:

math.o: math.cpp math.hpp types.hpp
math.hpp: types.hpp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm *.o

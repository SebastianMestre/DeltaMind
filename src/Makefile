
CXX=g++
CXXFLAGS= -fsanitize=undefined -g -Wall
LIBS=-lasan -lubsan

deltaMind: main.o Network.o Trainer.o math.o
	g++ -o $@ $^ $(LIBS)

main.o: main.cpp

Network.o: Network.cpp Network.hpp types.hpp math.hpp
Network.hpp: types.hpp

Trainer.o: Trainer.cpp Trainer.hpp Network.hpp types.hpp math.hpp
Trainer.hpp: types.hpp Network.hpp

types.o: types.cpp types.hpp
types.hpp:

math.o: math.cpp math.hpp types.hpp
math.hpp: types.hpp

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm deltaMind *.o

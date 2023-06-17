main: main.cpp game.o util.o
	g++ -O3 -Wall -o main -g main.cpp game.o util.o -lFJML

run: run.cpp game.o util.o
	g++ -O3 -Wall -o run -g run.cpp game.o util.o -lFJML

%.o: %.cpp %.h
	g++ -O3 -Wall -c -g $<

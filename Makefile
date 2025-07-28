

all: classificationTraining.cpp
	g++ -fopenmp -o classificationTraining classificationTraining.cpp -lopennn

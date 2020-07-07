/*
Jennifer Cho + Alex Cadigan
2/6/2019
COMP-481 Final Project
*/

#include <cmath>
#include <iostream>
#include <mpi.h>

using namespace std;

double paraInterval(int N);
double paraIntegral(int intervals);
double paraMonteCarlo(int num_shots);

int main(int argc, char * argv[]) {
	int rank, size, numCalc, i, num_hits = 0, tempNumHits;
	double start, interval, integral, monteCarlo, tempPi, PiInterval = 0, PiIntegral = 0, x, y, dx;
	unsigned int seed;

	MPI_Init(& argc, & argv);
	MPI_Comm_rank(MPI_COMM_WORLD, & rank);
 	MPI_Comm_size(MPI_COMM_WORLD, & size);

	// Gets user input:
	if (rank == 0) {
		printf("Enter the number of calculations to use:\t");
		flush(std::cout);
		scanf("%d", & numCalc);
	}

	// Broadcast to all processors
	MPI_Bcast(& numCalc, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Interval algorithm
	start = MPI_Wtime();
	tempPi = 0;
	for (i = rank + 1; i <= numCalc; i += size) {
		x = (1 / (double) numCalc) * (i - 0.5);
		y = sqrt(1 - pow(x, 2));
		tempPi += 4 * (y / (double) numCalc);
	}
	MPI_Reduce(& tempPi, & PiInterval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	interval = MPI_Wtime() - start;

	// Integral algorithm
	start = MPI_Wtime();
	tempPi = 0;
	dx = 1 / (double) numCalc;
	for (i = rank + 1; i <= numCalc; i += size) {
		x = i * dx;
		y = sqrt(1 - x * x);
		tempPi = tempPi + y * dx;
	}
	tempPi *= 4;
	MPI_Reduce(& tempPi, & PiIntegral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	integral = MPI_Wtime() - start;

	// Monte Carlo algorithm
	start = MPI_Wtime();
	seed = rank;
	for (i = rank + 1; i <= numCalc; i += size) {
		x = (double) rand_r(& seed) / (double) RAND_MAX;
		y = (double) rand_r(& seed) / (double) RAND_MAX;
		if (x * x + y * y <= 1) {
			num_hits ++;
		}
	}
	MPI_Reduce(& num_hits, & tempNumHits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	monteCarlo = MPI_Wtime() - start;

	// Results
	if (rank == 0) {
		printf("T_Intv:\t%f\tE:\t%f\tT_Intg:\t%f\tE:\t%f\tT_Monte:\t%f\tE:\t%f\n", interval, (fabs(M_PI - PiInterval) / M_PI) * 100, integral, (fabs(M_PI - PiIntegral) / M_PI) * 100, monteCarlo, (fabs(M_PI - (4 * (double) tempNumHits / (double) numCalc)) / M_PI) * 100);
	}

	MPI_Finalize();
}
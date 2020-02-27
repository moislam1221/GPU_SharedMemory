#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <utility>

#define PI 3.14159265358979323
#define RELAXED 1

double * jacobiCpu(const double * initX, const double * rhs, int nGrids, int nIters)
{
    double dx = 1.0 / (nGrids - 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);

    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            double leftX = x0[iGrid - 1];
            double rightX = x0[iGrid + 1];
#ifdef RELAXED
            double centerX = x0[iGrid];
            x1[iGrid] = jacobiRelaxed1DPoisson(leftX, centerX, rightX, rhs[iGrid], dx);
#else
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
#endif

        }
        double * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;
}

int jacobiCpuIterationCountResidual(const double * initX, const double * rhs, int nGrids, double TOL)
{
    double dx = 1.0 / (nGrids - 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);

    double residual = 1000000000000.0;
    int iIter = 0;
    while (residual > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            double leftX = x0[iGrid - 1];
            double rightX = x0[iGrid + 1];
#ifdef RELAXED
            double centerX = x0[iGrid];
            x1[iGrid] = jacobiRelaxed1DPoisson(leftX, centerX, rightX, rhs[iGrid], dx);
#else
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
#endif
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
        iIter++;
		residual = residual1DPoisson(x0, rhs, nGrids);
        if (iIter % 1000 == 0) {
			printf("CPU: The residual at step %d is %f\n", iIter, residual);
		}
    }

    int nIters = iIter;
    delete[] x0;
    delete[] x1;
    return nIters;
}

int jacobiCpuIterationCountSolutionError(const double * initX, const double * rhs, int nGrids, double TOL, const double * solution_exact)
{
    double dx = 1.0 / (nGrids - 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);

    double solution_error = 1000000000000.0;
    int iIter = 0;
    while (solution_error > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            double leftX = x0[iGrid - 1];
            double rightX = x0[iGrid + 1];
#ifdef RELAXED
            double centerX = x0[iGrid];
            x1[iGrid] = jacobiRelaxed1DPoisson(leftX, centerX, rightX, rhs[iGrid], dx);
#else
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
#endif
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
        iIter++;
		solution_error = solutionError1DPoisson(x0, solution_exact, nGrids); 
        if (iIter % 1000 == 0) {
			printf("CPU: The solution error at step %d is %f\n", iIter, solution_error);
		}
    }

    int nIters = iIter;
    delete[] x0;
    delete[] x1;
    return nIters;
}

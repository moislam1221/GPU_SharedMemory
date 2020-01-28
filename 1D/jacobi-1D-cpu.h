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

float * jacobiCpu(const float * initX, const float * rhs, int nGrids, int nIters)
{
    float dx = 1.0 / (nGrids - 1);
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    memcpy(x1, initX, sizeof(float) * nGrids);

    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;
}

int jacobiCpuIterationCount(const float * initX, const float * solution_exact, const float * rhs, int nGrids, float TOL)
{
    float dx = 1.0 / (nGrids - 1);
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    memcpy(x1, initX, sizeof(float) * nGrids);

    // float residual = 1000000000000.0;
    float solution_error = 1000000000000.0;
    int iIter = 0;
    // while (residual > TOL) {
    while (solution_error > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            float leftX = x0[iGrid - 1];
            float rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
        iIter++;
		// residual = residual1DPoisson(x0, rhs, nGrids);
		solution_error = solutionError1DPoisson(x0, solution_exact, nGrids); 
        if (iIter % 1000 == 0) {
			// printf("CPU: The residual at step %d is %f\n", iIter, residual);
			printf("CPU: The solution error at step %d is %f\n", iIter, solution_error);
		}
    }

    int nIters = iIter;
    delete[] x1;
    return nIters;
}

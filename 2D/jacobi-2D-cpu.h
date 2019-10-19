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

float * jacobiCpu(const float * initX, const float * rhs, int nxGrids, int nyGrids,  int nIters)
{
    float dx = 1.0 / (nxGrids - 1);
    float dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    float * x0 = new float[nDofs];
    float * x1 = new float[nDofs];
    memcpy(x0, initX, sizeof(float) * nDofs);
    memcpy(x1, initX, sizeof(float) * nDofs);

    int dof;
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
			for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        dof = jGrid * nxGrids + iGrid;
                float leftX = x0[dof - 1];
				float rightX = x0[dof + 1];
                float topX = x0[dof + nxGrids];
                float bottomX = x0[dof - nxGrids];
				x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			}
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;
}

int jacobiCpuIterationCount(const float * initX, const float * rhs, int nxGrids, int nyGrids, float TOL)
{
    float dx = 1.0 / (nxGrids - 1);
    float dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    float * x0 = new float[nDofs];
    float * x1 = new float[nDofs];
    memcpy(x0, initX, sizeof(float) * nDofs);
    memcpy(x1, initX, sizeof(float) * nDofs);

    float residual = 1000000000000.0;
    int iIter = 0;
    int dof;
    while (residual > TOL) {
        for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
			for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        dof = jGrid * nxGrids + iGrid;
        		float leftX = x0[dof - 1];
				float rightX = x0[dof + 1];
                float topX = x0[dof + nxGrids];
                float bottomX = x0[dof - nxGrids];
				x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			}
        }
		float * tmp = x0; x0 = x1; x1 = tmp;
		iIter++;
		residual = residual2DPoisson(x0, rhs, nxGrids, nyGrids);
		if (iIter % 1000 == 0) {
			printf("CPU: The residual at step %d is %f\n", iIter, residual);
		}
    }
    int nIters = iIter;
    delete[] x1;
    return nIters;
}

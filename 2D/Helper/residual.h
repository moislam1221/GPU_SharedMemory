float normFromRow(const float leftX, const float centerX, const float rightX, const float topX, const float bottomX, const float centerRhs, const float dx, const float dy)
{
    return centerRhs + (leftX + rightX) / (dx*dx) + (topX + bottomX) / (dy*dy) - (2.0/(dx*dx) + 2.0/(dy*dy)) * centerX;
}

float residual2DPoisson(const float * solution, const float * rhs, int nxGrids, int nyGrids)
{
    float residual = 0.0;
    float dx = 1.0 / (nxGrids - 1);
    float dy = 1.0 / (nyGrids - 1);
    float leftX, centerX, rightX, topX, bottomX, residualContributionFromRow;

    int dof;
    for (int jGrid = 0; jGrid < nyGrids; jGrid++) {
		for (int iGrid = 0; iGrid < nxGrids; iGrid++) {
			if (iGrid == 0 || iGrid == nxGrids-1 || jGrid == 0 || jGrid == nyGrids-1) {
				residualContributionFromRow = 0.0;
			}
			else {
                dof = iGrid + jGrid * nxGrids;
				leftX = solution[dof - 1];
				centerX = solution[dof];
				rightX = solution[dof + 1];
				topX = solution[dof + nxGrids];
				bottomX = solution[dof - nxGrids];
				residualContributionFromRow = normFromRow(leftX, centerX, rightX, topX, bottomX, rhs[iGrid], dx, dy);
			}
			residual = residual + residualContributionFromRow * residualContributionFromRow;
		}
    }

    residual = sqrt(residual);
    return residual;
}

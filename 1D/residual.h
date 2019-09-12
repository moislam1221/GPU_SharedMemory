float normFromRow(const float leftX, const float centerX, const float rightX, const float centerRhs, const float dx)
{
    return centerRhs + (leftX - 2.0*centerX + rightX) / (dx*dx);
}

float residual1DPoisson(const float * solution, const float * rhs, int nGrids)
{
    float residual = 0.0;
    float dx = 1.0 / (nGrids - 1);
    float leftX, centerX, rightX, residualContributionFromRow;

    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            residualContributionFromRow = 0.0;
        }
        else {
            leftX = solution[iGrid - 1];
            centerX = solution[iGrid];
            rightX = solution[iGrid + 1];
            residualContributionFromRow = normFromRow(leftX, centerX, rightX, rhs[iGrid], dx);
        }
        residual = residual + residualContributionFromRow * residualContributionFromRow;
    }

    residual = sqrt(residual);
    return residual;
}


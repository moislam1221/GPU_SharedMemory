__host__ __device__
double jacobi1DPoisson(const double leftX, const double rightX, const double centerRhs, const double dx)
{
    return (centerRhs*dx*dx + leftX + rightX) / 2.0;
    // return 1.0/(2.0/(dx*dx)) * (centerRhs + (1.0/(dx*dx) *leftX + (1.0/(dx*dx)) * rightX));
}

__host__ __device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, double leftX, double centerX, double rightX, const double centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
}

__host__ __device__
double jacobiRelaxed1DPoisson(const double leftX, const double centerX, const double rightX, const double centerRhs, const double dx)
{
    return 0.99 * (centerRhs*dx*dx + leftX + rightX) / 2.0 + 0.01 * centerX;
    // return 1.0/(2.0/(dx*dx)) * (centerRhs + (1.0/(dx*dx) *leftX + (1.0/(dx*dx)) * rightX));
}

/*__host__ __device__
double jacobi1DPoissonConstantMemory(const double leftX, double rightX, const double centerRhs)
{
    return (centerRhs - (coeffGpu[1] * leftX + coeffGpu[3] * rightX)) / coeffGpu[2];
}
*/

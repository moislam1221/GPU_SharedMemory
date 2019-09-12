__host__ __device__
float jacobi1DPoisson(const float leftX, float rightX, const float centerRhs, const float dx)
{
    return (centerRhs*dx*dx + leftX + rightX) / 2.0;
}

__host__ __device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
}


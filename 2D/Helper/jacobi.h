__host__ __device__
float jacobi2DPoisson(const float leftX, const float rightX, const float topX, const float bottomX, const float centerRhs, const float dx, const float dy)
{
    return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX+bottomX)) / (2.0*(dx*dx + dy*dy));
}

__host__ __device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix,
             const float leftX, const float centerX, const float rightX, const float topX, const float bottomX, const float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
}

__host__ __device__
double jacobi2DPoisson(const double leftX, const double rightX, const double topX, const double bottomX, const double centerRhs, const double dx, const double dy)
{
    return (centerRhs*dx*dx*dy*dy + dy*dy*(leftX + rightX) + dx*dx*(topX+bottomX)) / (2.0*(dx*dx + dy*dy));
}

__host__ __device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix,
             const double leftX, const double centerX, const double rightX, const double topX, const double bottomX, const double centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
}

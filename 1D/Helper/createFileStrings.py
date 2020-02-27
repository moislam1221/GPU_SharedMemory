def fileNameResults(ALGORITHM_TYPE, nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag):
    if (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "TOLREDUCE"
    elif (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "TOL"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "ERRORREDUCE"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "ERROR"

    FILE_NAME = "RESULTS/" + ALGORITHM_TYPE + ".N" + str(nDim) + "." + TOL_TYPE + str(tolerance_value) + ".txt"

    return FILE_NAME

def fileNameOverlapResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag):

    if (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "TOLREDUCE"
    elif (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "TOL"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "ERRORREDUCE"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "ERROR"

    FILE_NAME = "OVERLAP_RESULTS/TITAN_V/" + "N" + str(nDim) + "." + TOL_TYPE + str(tolerance_value) + ".TPB" + str(threadsPerBlock) + ".txt"

    return FILE_NAME

def fileNameSubiterationResults(nDim, threadsPerBlock, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag):

    if (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "TOLREDUCE"
    elif (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "TOL"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "ERRORREDUCE"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "ERROR"

    FILE_NAME = "SUBITERATION_RESULTS/TITAN_V/" + "N" + str(nDim) + "." + TOL_TYPE + str(tolerance_value) + ".TPB" + str(threadsPerBlock) + ".txt"

    return FILE_NAME

def figuresSubfolderName(nDim, residual_convergence_metric_flag, tolerance_value, tolerance_reduction_flag):
    if (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "TOLREDUCE"
    elif (residual_convergence_metric_flag == str(1) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "TOL"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(1)):
        TOL_TYPE = "ERRORREDUCE"
    elif (residual_convergence_metric_flag == str(0) and tolerance_reduction_flag == str(0)):
        TOL_TYPE = "ERROR"

    FOLDER_NAME = "N" + str(nDim) + "." + TOL_TYPE + str(tolerance_value)

    return FOLDER_NAME


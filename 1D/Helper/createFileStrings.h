std::string createFileString(std::string ALGORITHM_TYPE, const int nDim, const int residual_convergence_metric_flag, const double tolerance_value, const int tolerance_reduction_flag, const int relaxation_flag) {

    // CREATE COMPONENTS OF SCRIPT NAMES BASED ON INPUTS
	std::string BASE_NAME;
    if (ALGORITHM_TYPE == "CPU") {
    	BASE_NAME = "RESULTS/CPU.";
    }
    if (ALGORITHM_TYPE == "GPU") {
		BASE_NAME = "RESULTS/GPU.";
    }
    if (ALGORITHM_TYPE == "SHARED") {
        BASE_NAME = "RESULTS/SHARED.";
    }

    std::string N_STRING = "N" + std::to_string(nDim) + ".";
    std::string TOL_TYPE_STRING;

    if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "TOLREDUCE";
    }
    else if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "TOL";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "ERRORREDUCE";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "ERROR";

    }
    std::string TOL_VAL_STRING = std::to_string(tolerance_value);

    std::string RELAXATION_STRING;	
	if (relaxation_flag == 1) {
        RELAXATION_STRING = ".RELAXED";
    }
    else {
        RELAXATION_STRING = "";
    }

    std::string TXT_STRING = ".txt";

    // CREATE CPU/GPU/SHARED STRING NAMES
    std::string FILE_NAME = BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + RELAXATION_STRING + TXT_STRING;

    return FILE_NAME;

}

std::string createFileStringOverlap(const int nDim, const int threadsPerBlock, const int residual_convergence_metric_flag, const double tolerance_value, const int tolerance_reduction_flag, const int relaxation_flag) {

    std::string BASE_NAME = "OVERLAP_RESULTS/TITAN_V/";
    std::string N_STRING = "N" + std::to_string(nDim) + ".";
    std::string TOL_TYPE_STRING;

    if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "TOLREDUCE";
    }
    else if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "TOL";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "ERRORREDUCE";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "ERROR";

    }

    std::string TOL_VAL_STRING = std::to_string(tolerance_value);
    std::string TPB_STRING = ".TPB" + std::to_string(threadsPerBlock);

    std::string RELAXATION_STRING;	
    if (relaxation_flag == 1) {
        RELAXATION_STRING = ".RELAXED";
    }
    else {
        RELAXATION_STRING = "";
    }
    
    std::string TXT_STRING = ".txt";

    // CREATE CPU/GPU/SHARED STRING NAMES
    std::string FILE_NAME = BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + TPB_STRING + RELAXATION_STRING + TXT_STRING;

    return FILE_NAME;
}

std::string createFileStringSubiterations(const int nDim, const int threadsPerBlock, const int residual_convergence_metric_flag, const double tolerance_value, const int tolerance_reduction_flag, const int relaxation_flag) {

    std::string BASE_NAME = "SUBITERATION_RESULTS/TITAN_V/";
    std::string N_STRING = "N" + std::to_string(nDim) + ".";
    std::string TOL_TYPE_STRING;

    if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "TOLREDUCE";
    }
    else if (residual_convergence_metric_flag == 1 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "TOL";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 1) {
        TOL_TYPE_STRING = "ERRORREDUCE";
    }
    if (residual_convergence_metric_flag == 0 && tolerance_reduction_flag == 0) {
        TOL_TYPE_STRING = "ERROR";

    }

    std::string TOL_VAL_STRING = std::to_string(tolerance_value);
    std::string TPB_STRING = ".TPB" + std::to_string(threadsPerBlock);
    
    std::string RELAXATION_STRING;	
    if (relaxation_flag == 1) {
        RELAXATION_STRING = ".RELAXED";
    }
    else {
        RELAXATION_STRING = "";
    }
    
    std::string TXT_STRING = ".txt";

    // CREATE CPU/GPU/SHARED STRING NAMES
    std::string FILE_NAME = BASE_NAME + N_STRING + TOL_TYPE_STRING + TOL_VAL_STRING + TPB_STRING + RELAXATION_STRING + TXT_STRING;

    return FILE_NAME;
}


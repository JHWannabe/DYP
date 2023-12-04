//#include "Eigen/Core"
//#include "Eigen/Dense"
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/eigen.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>
//#include <vector>
//#include <ctime>
//#include <cstdlib>
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//
//__global__ void copyEigenToCudaKernel(double* cudaData, double* eigenData, int size) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx < size) {
//        cudaData[idx] = eigenData[idx];
//    }
//}
//
//void copyEigenToCuda(double* cudaData, double* eigenData, int size) {
//    int threadsPerBlock = 1024;
//    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//    copyEigenToCudaKernel << <blocksPerGrid, threadsPerBlock >> > (cudaData, eigenData, size);
//    cudaDeviceSynchronize();
//}
//
//int main() {
//    // Assuming size is defined
//    int size = 9;
//
//    // Create an Eigen matrix
//    Eigen::MatrixXd eigenMatrix(3, 3);
//    eigenMatrix << 1, 2, 3,
//        4, 5, 6,
//        7, 8, 9;
//
//    // Allocate memory on GPU
//    double* d_cudaMatrix;
//    cudaMalloc((void**)&d_cudaMatrix, sizeof(double) * size);
//
//    // Copy data from Eigen to CUDA
//    copyEigenToCuda(d_cudaMatrix, eigenMatrix.data(), size);
//
//    // Allocate host memory for cv::Mat
//    cv::Mat cvMatrix(3, 3, CV_64F);
//
//    // Copy data from CUDA to cv::Mat
//    cudaMemcpy(cvMatrix.ptr<double>(), d_cudaMatrix, sizeof(double) * size, cudaMemcpyDeviceToHost);
//
//    // Free GPU memory
//    cudaFree(d_cudaMatrix);
//
//    // Display the OpenCV Mat
//    std::cout << "OpenCV Mat:\n" << cvMatrix << std::endl;
//
//    return 0;
//}
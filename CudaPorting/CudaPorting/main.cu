#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace Eigen;
using namespace std;

#define M_IMAGE_COUNT 4

Eigen::MatrixXd LightMatrixPinv(const std::vector<Eigen::Vector4f>& lightMat) {
    int numRows = lightMat.size();
    int numCols = (numRows > 0) ? lightMat[0].size() : 0;

    Eigen::MatrixXd matrix(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
        matrix.row(i) = lightMat[i].cast<double>();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValuesInv = svd.singularValues().unaryExpr([&](double sv) {
        return (sv > 1e-8) ? (1.0 / sv) : 0.0;
        });
    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}

__global__ void copyMatrixToCUDA(int k, const double* imageData, double* d_merged_matrix, int _size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < _size && j < M_IMAGE_COUNT) {
        d_merged_matrix[i + _size * k] = imageData[i] / 255.0;
    }
}

__global__ void matrixMultiplyKernel(const double* a, const double* b, double* c, int m, int n, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < l) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            double a_val = a[i + m * k];
            double b_val = b[j * n + k];
            sum += a_val * b_val;
        }
        c[i + m * j] = sum;
    }
}

__global__ void rowNormClipKernel(const double* matrixData, double* output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double l2_norm_squared = 0.0;
        for (int j = 0; j < cols; ++j) {
            double element = matrixData[i * cols + j];
            l2_norm_squared += element * element;
        }
        output[i] = fmin(fmax(sqrt(l2_norm_squared), 0.0), 1.0);
    }
}

__global__ void flipKernel(const double* albedo, uchar* result, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < cols && j < rows) {
        int index = i + cols * j;
        result[index] = static_cast<uchar>(albedo[index] * 255.0f);
    }
}

__global__ void elementWiseDivisionKernel(const double* inputMatrix, const double* normVector, double* outputMatrix, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        int index = i + rows * j;
        outputMatrix[index] = inputMatrix[index] / normVector[i];
        if (outputMatrix[i * cols + 2] == 0)
            outputMatrix[i * cols + 2] = 1;
    }
}

__global__ void copyKernel(double* A, double* B, double* maxVal, double* minVal, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        B[i * cols + j] = A[i * cols + j] / 255.0;
        *minVal = (B[i * cols + j] < *minVal) ? B[i * cols + j] : *minVal;
        *maxVal = (B[i * cols + j] > *maxVal) ? B[i * cols + j] : *maxVal;
    }
}

__global__ void normalizeKernel(const double* input, uchar3* output, double* maxVal, double* minVal, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        // Normalize the input values and convert to uchar3
        double normalizedValue = 237.0 * (input[i * cols + j] - *minVal) / (*maxVal - *minVal) + 18.0;
        if (normalizedValue > 253.0) {
            normalizedValue = 253.0;
        }
        uchar3 result;
        result.x = static_cast<uchar>(normalizedValue);
        result.y = static_cast<uchar>(normalizedValue);
        result.z = static_cast<uchar>(normalizedValue);
        output[i * cols + j] = result;
    }
}


int main() {
    int threadsPerBlock = 1024;
    dim3 threadsPerBlock2(32, 32);
    cudaStream_t s1, s2, s3, s4;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);

    cv::Mat img0, img1, img2, img3;
    double minVal = 1e9, maxVal = -1e9;
    double* d_merged_matrix, * d_image0Data, * d_image1Data, * d_image2Data, * d_image3Data;
    double* d_lightMatpinv, * d_rho_t;
    double* d_norm_t, * d_rho, * d_transposed_rho, * d_n, * d_min, * d_max;
    uchar* d_result;
    uchar3* d_output;

    time_t _tstart, _tend;
    _tstart = clock();

    std::vector<Eigen::Vector4f> lightMat;
    lightMat.emplace_back(-0.6133723, -0.6133723, 0.6133723, 0.6133723);
    lightMat.emplace_back(-0.613372, 0.613372, 0.613372, -0.613372);
    lightMat.emplace_back(0.49754286, 0.49754286, 0.49754286, 0.49754286);

    Eigen::MatrixXd _lightMatpinv = LightMatrixPinv(lightMat);

    img0 = cv::imread("resize/0_B0.bmp", cv::IMREAD_GRAYSCALE);
    img1 = cv::imread("resize/0_B1.bmp", cv::IMREAD_GRAYSCALE);
    img2 = cv::imread("resize/0_B2.bmp", cv::IMREAD_GRAYSCALE);
    img3 = cv::imread("resize/0_B3.bmp", cv::IMREAD_GRAYSCALE);

    int _rows = img0.rows;
    int _cols = img0.cols;
    int _size = _rows * _cols;

    std::vector<double> image0Data(img0.ptr<uchar>(), img0.ptr<uchar>() + _size);
    std::vector<double> image1Data(img1.ptr<uchar>(), img1.ptr<uchar>() + _size);
    std::vector<double> image2Data(img2.ptr<uchar>(), img2.ptr<uchar>() + _size);
    std::vector<double> image3Data(img3.ptr<uchar>(), img3.ptr<uchar>() + _size);

    cudaMalloc((void**)&d_merged_matrix, M_IMAGE_COUNT * _size * sizeof(double));
    cudaMalloc((void**)&d_lightMatpinv, M_IMAGE_COUNT * 3 * sizeof(double));
    cudaMalloc((void**)&d_output, _rows * _cols * sizeof(uchar3));
    cudaMalloc((void**)&d_image0Data, _size * sizeof(double));
    cudaMalloc((void**)&d_image1Data, _size * sizeof(double));
    cudaMalloc((void**)&d_image2Data, _size * sizeof(double));
    cudaMalloc((void**)&d_image3Data, _size * sizeof(double));
    cudaMalloc((void**)&d_rho_t, 2 * _size * sizeof(double));
    cudaMalloc((void**)&d_rho, 2 * _size * sizeof(double));
    cudaMalloc((void**)&d_norm_t, _size * sizeof(double));
    cudaMalloc((void**)&d_result, _size * sizeof(uchar));
    cudaMalloc((void**)&d_n, _size * sizeof(double));
    cudaMalloc((void**)&d_min, sizeof(double));
    cudaMalloc((void**)&d_max, sizeof(double));

    cudaMemcpyAsync(d_lightMatpinv, _lightMatpinv.data(), M_IMAGE_COUNT * 3 * sizeof(double), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d_max, &maxVal, sizeof(double), cudaMemcpyHostToDevice, s2);
    cudaMemcpyAsync(d_min, &minVal, sizeof(double), cudaMemcpyHostToDevice, s3);
    cudaMemcpyAsync(d_image0Data, image0Data.data(), _size * sizeof(double), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d_image1Data, image1Data.data(), _size * sizeof(double), cudaMemcpyHostToDevice, s2);
    cudaMemcpyAsync(d_image2Data, image2Data.data(), _size * sizeof(double), cudaMemcpyHostToDevice, s3);
    cudaMemcpyAsync(d_image3Data, image3Data.data(), _size * sizeof(double), cudaMemcpyHostToDevice, s4);

    int blocksPerGrid = (_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocksPerGrid1((_size + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (M_IMAGE_COUNT + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    dim3 blocksPerGrid2((_size + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (3 + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    dim3 blocksPerGrid3((_rows + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (_cols + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    dim3 blocksPerGrid4((_cols + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (_rows + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    copyMatrixToCUDA << <blocksPerGrid1, threadsPerBlock2, 0, s1 >> > (0, d_image0Data, d_merged_matrix, _size);
    copyMatrixToCUDA << <blocksPerGrid1, threadsPerBlock2, 0, s2 >> > (1, d_image1Data, d_merged_matrix, _size);
    copyMatrixToCUDA << <blocksPerGrid1, threadsPerBlock2, 0, s3 >> > (2, d_image2Data, d_merged_matrix, _size);
    copyMatrixToCUDA << <blocksPerGrid1, threadsPerBlock2, 0, s4 >> > (3, d_image3Data, d_merged_matrix, _size);

    matrixMultiplyKernel << <blocksPerGrid2, threadsPerBlock2, 0, s1 >> > (d_merged_matrix, d_lightMatpinv, d_rho_t, _size, M_IMAGE_COUNT, 3);

    rowNormClipKernel << <blocksPerGrid, threadsPerBlock, 0, s2 >> > (&d_rho_t[2 * _size], d_norm_t, _size, 1);

    flipKernel << <blocksPerGrid4, threadsPerBlock2, 0, s3 >> > (d_norm_t, d_result, _rows, _cols);

    elementWiseDivisionKernel << <blocksPerGrid2, threadsPerBlock2, 0, s1 >> > (d_rho_t, d_norm_t, d_rho, _size, 3);

    copyKernel << <blocksPerGrid3, threadsPerBlock2, 0, s2 >> > (d_rho, d_n, d_max, d_min, _rows, _cols);
    normalizeKernel << <blocksPerGrid3, threadsPerBlock2, 0, s3 >> > (d_n, d_output, d_max, d_min, _rows, _cols);

    cv::Mat cvMatResult(_rows, _cols, CV_8UC1), _normalmap(_rows, _cols, CV_8UC3);

    cudaMemcpyAsync(cvMatResult.data, d_result, _size * sizeof(uchar), cudaMemcpyDeviceToHost, s1);
    cudaMemcpyAsync(_normalmap.data, d_output, _size * sizeof(uchar3), cudaMemcpyDeviceToHost, s2);

    cv::imwrite("result/0_albedo0.bmp", cvMatResult);
    cv::imwrite("result/0_albedo1.bmp", _normalmap);

    _tend = clock();
    cout << "수행시간 : " << (float)(_tend - _tstart) / 1000 << " s" << endl;

    cudaFree(d_merged_matrix);
    cudaFree(d_image0Data);
    cudaFree(d_image1Data);
    cudaFree(d_image2Data);
    cudaFree(d_image3Data);
    cudaFree(d_lightMatpinv);
    cudaFree(d_rho_t);
    cudaFree(d_norm_t);
    cudaFree(d_result);
    cudaFree(d_rho);
    cudaFree(d_n);
    cudaFree(d_output);
    cudaFree(d_min);
    cudaFree(d_max);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);
    cudaStreamDestroy(s4);

    waitKey(1000);

}
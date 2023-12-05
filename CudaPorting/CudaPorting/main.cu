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

__global__ void normalizeImage(double* image, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        image[idx] /= 255.0;
    }
}

__global__ void matrixMultiplyKernel(const double* a, const double* b, double* c, int m, int n, int l) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < l) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            double a_val = a[row + m * k];
            double b_val = b[col * n + k];
            sum += a_val * b_val;
        }
        c[row + m * col] = sum;
    }
}

__global__ void rowNormKernel(const double* matrixData, double* normsData, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double l2_norm_squared = 0.0;
        for (int col = 0; col < cols; ++col) {
            double element = matrixData[row * cols + col];
            l2_norm_squared += element * element;
        }
        normsData[row] = sqrt(l2_norm_squared);
    }
}

__global__ void clipKernel(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double value = input[idx];
        output[idx] = fmin(fmax(value, 0.0), 1.0);
    }
}

__global__ void reshapeMatrixKernel(const double* vec, double* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        matrix[row * cols + col] = vec[row * cols + col];
    }
}

__global__ void flipKernel(const double* albedo, uchar* result, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        result[index] = static_cast<uchar>(albedo[index] * 255.0f);
    }
}

__global__ void elementWiseDivisionKernel(const double* inputMatrix, const double* normVector, double* outputMatrix, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        int index = i + rows * j;
        outputMatrix[index] = inputMatrix[index] / normVector[i];
    }
}

__global__ void setZerosToOnes(double* _rhoData, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows) {
        if (_rhoData[i * cols + 2] == 0)
            _rhoData[i * cols + 2] = 1;
    }
}

__global__ void transposeMatrixKernel(const double* inputMatrix, double* outputMatrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        if (col == 0) {
            outputMatrix[row] = inputMatrix[row];
        }
    }
}

__global__ void copyKernel(double* A, double* B, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        B[i + rows * j] = A[i * cols + j] / 255.0;
    }
}


int main() {
    int threadsPerBlock1 = 1024;
    dim3 threadsPerBlock2(32, 32);

    cv::Mat img0, img1, img2, img3;
    time_t _tstart, _tend, mid;
    _tstart = clock();

    img0 = cv::imread("resize/0_B0.bmp", cv::IMREAD_GRAYSCALE);
    img1 = cv::imread("resize/0_B1.bmp", cv::IMREAD_GRAYSCALE);
    img2 = cv::imread("resize/0_B2.bmp", cv::IMREAD_GRAYSCALE);
    img3 = cv::imread("resize/0_B3.bmp", cv::IMREAD_GRAYSCALE);

    int _rows = img0.rows;
    int _cols = img0.cols;
    int _size = _rows * _cols;

    std::vector<Eigen::Vector4f> lightMat;
    lightMat.emplace_back(-0.6133723, -0.6133723, 0.6133723, 0.6133723);
    lightMat.emplace_back(-0.613372, 0.613372, 0.613372, -0.613372);
    lightMat.emplace_back(0.49754286, 0.49754286, 0.49754286, 0.49754286);

    Eigen::MatrixXd _lightMatpinv = LightMatrixPinv(lightMat);

    std::vector<double> image0Data(img0.ptr<uchar>(), img0.ptr<uchar>() + _size);
    std::vector<double> image1Data(img1.ptr<uchar>(), img1.ptr<uchar>() + _size);
    std::vector<double> image2Data(img2.ptr<uchar>(), img2.ptr<uchar>() + _size);
    std::vector<double> image3Data(img3.ptr<uchar>(), img3.ptr<uchar>() + _size);

    Eigen::MatrixXd _merged_matrix(_size, 4);
    _merged_matrix.col(0) = Eigen::VectorXd::Map(image0Data.data(), _size);
    _merged_matrix.col(1) = Eigen::VectorXd::Map(image1Data.data(), _size);
    _merged_matrix.col(2) = Eigen::VectorXd::Map(image2Data.data(), _size);
    _merged_matrix.col(3) = Eigen::VectorXd::Map(image3Data.data(), _size);

    Eigen::MatrixXd _rho_t(_size, 3);
    cv::Mat cvMatResult(_rows, _cols, CV_8UC1);
    double* d_lightMatpinv, * d_rho_t = new double[_size * 3];
    double* d_matrix, * d_norm, * d_norm_t;
    double* d_rho, * d_transposed_rho, * d_n;
    uchar* d_result;
    double* d_merged_matrix;

    cudaMalloc((void**)&d_merged_matrix, 4 * _size * sizeof(double));
    cudaMalloc((void**)&d_lightMatpinv, 4 * _size * sizeof(double));
    cudaMalloc((void**)&d_rho_t, 3 * _size * sizeof(double));
    cudaMalloc((void**)&d_matrix, _size * sizeof(double));
    cudaMalloc((void**)&d_norm, _size * sizeof(double));
    cudaMalloc((void**)&d_norm_t, _size * sizeof(double));
    cudaMalloc((void**)&d_result, sizeof(uchar) * _rows * _cols);
    cudaMalloc((void**)&d_rho, 3 * _size * sizeof(double));
    cudaMalloc((void**)&d_transposed_rho, 3 * _size * sizeof(double));
    cudaMalloc((void**)&d_n, _size * sizeof(double));

    cudaMemcpy(d_merged_matrix, _merged_matrix.data(), 4 * _size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lightMatpinv, _lightMatpinv.data(), 4 * 3 * sizeof(double), cudaMemcpyHostToDevice);

    int blocksPerGrid1 = (4 * _size + threadsPerBlock1 - 1) / threadsPerBlock1;
    dim3 blocksPerGrid2((_size + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (3 + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    dim3 blocksPerGrid3((_rows + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (_cols + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    dim3 blocksPerGrid4((_cols + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (_rows + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    normalizeImage << <blocksPerGrid1, threadsPerBlock1 >> > (d_merged_matrix, 4 * _size);

    blocksPerGrid1 = (_size + threadsPerBlock1 - 1) / threadsPerBlock1;

    matrixMultiplyKernel << <blocksPerGrid2, threadsPerBlock2 >> > (d_merged_matrix, d_lightMatpinv, d_rho_t, _size, 4, 3);
    rowNormKernel << <blocksPerGrid1, threadsPerBlock1 >> > (&d_rho_t[2 * _size], d_norm, _size, 1);
    clipKernel << <blocksPerGrid1, threadsPerBlock1 >> > (d_norm, d_norm_t, _size);
    reshapeMatrixKernel << <blocksPerGrid3, threadsPerBlock2 >> > (d_norm_t, d_matrix, _cols, _rows);
    flipKernel << <blocksPerGrid4, threadsPerBlock2 >> > (d_matrix, d_result, _rows, _cols);

    blocksPerGrid1 = ((3 + threadsPerBlock1 - 1) / threadsPerBlock1);

    elementWiseDivisionKernel << <blocksPerGrid2, threadsPerBlock2 >> > (d_rho_t, d_norm_t, d_rho, _size, 3);
    setZerosToOnes << <blocksPerGrid1, threadsPerBlock1 >> > (d_rho, _size, 3);
    transposeMatrixKernel << <blocksPerGrid2, threadsPerBlock2 >> > (d_rho, d_transposed_rho, _size, 3);
    copyKernel << <blocksPerGrid3, threadsPerBlock2 >> > (d_transposed_rho, d_n, _rows, _cols);

    Eigen::MatrixXd col0(_rows, _cols);

    cudaMemcpy(cvMatResult.data, d_result, sizeof(uchar) * _rows * _cols, cudaMemcpyDeviceToHost);
    cudaMemcpy(col0.data(), d_n, _size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_lightMatpinv);
    cudaFree(d_merged_matrix);
    cudaFree(d_norm);
    cudaFree(d_matrix);
    cudaFree(d_result);
    cudaFree(d_rho_t);
    cudaFree(d_norm_t);
    cudaFree(d_rho);
    cudaFree(d_n);
    cudaFree(d_transposed_rho);

    cv::Mat _normalmap, normalmap_cv(_rows, _cols, CV_64FC1);

    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            normalmap_cv.at<double>(i, j) = col0(i, j);
        }
    }

    cv::normalize(cvMatResult, cvMatResult, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(normalmap_cv, _normalmap, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    _tend = clock();
    cout << "수행시간 : " << (float)(_tend - _tstart) / 1000 << " s" << endl;

    cv::imwrite("result/0_albedo0.bmp", cvMatResult);
    cv::imwrite("result/0_albedo1.bmp", _normalmap);

    waitKey(2000);

}
#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <ctime>


using namespace cv;
using namespace Eigen;
using namespace std;

#define M_IMAGE_COUNT 4
#define Debug 1

// 주어진 3차원 벡터 목록으로부터 Eigen 행렬을 생성하는 함수
Eigen::MatrixXd LightMatrix(const std::vector<Eigen::Vector3f>& lightMat) {
    int numRows = lightMat.size();
    int numCols = (numRows > 0) ? lightMat[0].size() : 0;

    Eigen::MatrixXd matrix(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
        matrix.row(i) = lightMat[i].cast<double>();
    }
    return matrix;
}

// 주어진 행렬에 대한 의사 역행렬을 계산하는 함수
Eigen::MatrixXd pinv(const Eigen::MatrixXd& matrix, double tolerance = 1e-8) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValuesInv = svd.singularValues().unaryExpr([&](double sv) {
        return (sv > tolerance) ? (1.0 / sv) : 0.0;
        });
    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}


// 주어진 행렬의 각 행에 대한 L2 노름을 계산하는 함수
Eigen::VectorXd rowNorm(const Eigen::MatrixXd& matrix) {
    Eigen::VectorXd norms(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        Eigen::VectorXd row = matrix.row(i);
        double l2_norm_squared = row.squaredNorm();
        norms(i) = std::sqrt(l2_norm_squared);
    }
    return norms;
}

// 주어진 벡터의 값들을 최소값과 최대값 사이로 클리핑하는 함수
Eigen::VectorXd clip(const Eigen::VectorXd& vector, double min_val, double max_val) {
    return vector.cwiseMax(min_val).cwiseMin(max_val);
}

// 1차원 벡터를 주어진 행과 열 크기로 재구성하는 함수
Eigen::MatrixXd reshapeMatrix(const Eigen::VectorXd& vec, int rows, int cols) {
    Eigen::Map<const Eigen::MatrixXd> map(vec.data(), rows, cols);
    return map;
}


int main() {

    cv::Mat img0, img1, img2, img3;

    if (Debug == 1)
    {
        // Load images
        img0 = cv::imread("B_0.bmp", cv::IMREAD_GRAYSCALE);
        img1 = cv::imread("B_1.bmp", cv::IMREAD_GRAYSCALE);
        img2 = cv::imread("B_2.bmp", cv::IMREAD_GRAYSCALE);
        img3 = cv::imread("B_3.bmp", cv::IMREAD_GRAYSCALE);
    }
    if (Debug == 0)
    {
        img0 = cv::imread("B0.bmp", cv::IMREAD_GRAYSCALE);
        img1 = cv::imread("B1.bmp", cv::IMREAD_GRAYSCALE);
        img2 = cv::imread("B2.bmp", cv::IMREAD_GRAYSCALE);
        img3 = cv::imread("B3.bmp", cv::IMREAD_GRAYSCALE);
    }


    time_t _tstart, _tend;
    double _tTime;
    _tstart = clock();

    int _rows = img0.rows;
    int _cols = img0.cols;
    int _size = _rows * _cols;


    // 이미지 데이터를 1차원 double 벡터로 변환하여 처리
    std::vector<double> image0Data(img0.ptr<uchar>(), img0.ptr<uchar>() + _size);
    std::vector<double> image1Data(img1.ptr<uchar>(), img1.ptr<uchar>() + _size);
    std::vector<double> image2Data(img2.ptr<uchar>(), img2.ptr<uchar>() + _size);
    std::vector<double> image3Data(img3.ptr<uchar>(), img3.ptr<uchar>() + _size);


    // Process lightMat
    // 빛의 방향 벡터를 포함하는 3차원 벡터를 초기화
    std::vector<Eigen::Vector3f> lightMat;
    lightMat.emplace_back(-0.6133723, -0.613372, 0.49754286);
    lightMat.emplace_back(-0.6133723, 0.613372, 0.49754286);
    lightMat.emplace_back(0.6133723, 0.613372, 0.49754286);
    lightMat.emplace_back(0.6133723, -0.613372, 0.49754286);

    // lightMat로부터 행렬을 생성하고, 행렬의 의사 역행렬을 계산
    Eigen::MatrixXd _lightMat = LightMatrix(lightMat);
    Eigen::MatrixXd _lightMatpinv = pinv(_lightMat);

    // Merge image data into a matrix
    // 이미지 데이터를 행렬로 변환하고 정규화
    Eigen::MatrixXd _merged_matrix(M_IMAGE_COUNT, _size);
    _merged_matrix.row(0) = Eigen::VectorXd::Map(image0Data.data(), _size);
    _merged_matrix.row(1) = Eigen::VectorXd::Map(image1Data.data(), _size);
    _merged_matrix.row(2) = Eigen::VectorXd::Map(image2Data.data(), _size);
    _merged_matrix.row(3) = Eigen::VectorXd::Map(image3Data.data(), _size);


    // Normalize
    _merged_matrix /= 255.0;

    // Step04
    Eigen::MatrixXd _rho = _lightMatpinv * _merged_matrix;
    Eigen::MatrixXd _rho_t = _rho.transpose();
    Eigen::VectorXd _norm = rowNorm(_rho_t);
    Eigen::VectorXd _norm_t = clip(_norm, 0.0, 1.0);


    // Reshape and save albedo
    Eigen::MatrixXd albedo = reshapeMatrix(_norm_t, _cols, _rows);
    cv::Mat cvMat;
    cv::Mat cvMatResult;
    cv::eigen2cv(albedo, cvMat);
    cv::flip(cvMat, cvMatResult, 0);
    cv::rotate(cvMatResult, cvMatResult, cv::ROTATE_90_CLOCKWISE);
    cv::normalize(cvMatResult, cvMatResult, 0, 255, cv::NORM_MINMAX, CV_8UC1);


    _rho = _rho_t.array() / _norm_t.array().replicate(1, _rho_t.cols());
    Eigen::MatrixXd normalmap(_rows, _cols * 3);
    for (int i = 0; i < _rho.rows(); ++i) {
        if (_rho(i, 2) == 0) {
            _rho(i, 2) = 1;
        }
    }

    // Transpose _rho
    Eigen::MatrixXd transposed_rho = _rho.transpose();

    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            normalmap(i, j) = transposed_rho(0, i * _cols + j);
            normalmap(i, j + _cols) = transposed_rho(1, i * _cols + j);
            normalmap(i, j + 2 * _cols) = transposed_rho(2, i * _cols + j);
        }
    }

    cv::Mat normalmap_cv(_rows, _cols, CV_64FC1);  // Use CV_64FC1 for double precision

    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            normalmap_cv.at<double>(i, j) = normalmap(i, j);
        }
    }

    // Convert normalmap to float32 and swap channels to RGB
    cv::Mat normalmap_rgb;
    normalmap_cv.convertTo(normalmap_rgb, CV_32FC3);

    // Normalize and convert to 8-bit unsigned int (CV_8UC3)
    cv::Mat _normalmap;
    cv::normalize(normalmap_rgb, _normalmap, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    cv::imwrite("albedo1.bmp", _normalmap);
    _tend = clock();
    _tTime = _tend - _tstart;

    cout << "수행시간 : " << _tTime / 1000 << " s" << endl;

    cv::imwrite("albedo0.bmp", cvMatResult);

    waitKey(10000);
}

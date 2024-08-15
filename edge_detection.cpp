#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

int main() {

    cv::Mat image = cv::imread("C:\\Users\\user\\Desktop\\car1.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }


    cv::Mat edges(image.size(), CV_8U);


    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };


    #pragma omp parallel for collapse(2)
    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            int sumX = 0;
            int sumY = 0;


            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += Gx[i + 1][j + 1] * image.at<uchar>(y + i, x + j);
                    sumY += Gy[i + 1][j + 1] * image.at<uchar>(y + i, x + j);
                }
            }


            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            edges.at<uchar>(y, x) = cv::saturate_cast<uchar>(magnitude);
        }
    }


    cv::imwrite("edges.jpg", edges);

    std::cout << "Edge detection completed and saved as edges.jpg" << std::endl;
    return 0;
}

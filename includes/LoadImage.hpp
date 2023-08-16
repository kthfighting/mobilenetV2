#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

cv::Mat ReadImage(const std::string &image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat rgbimage;
    cv::cvtColor(image, rgbimage, cv::COLOR_BGR2RGB);

    return rgbimage;
}

cv::Mat ResizeImage(const cv::Mat &image, int target_width, int target_height)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_width, target_height));
    return resized_image;
}

cv::Mat NormalizeImage(const cv::Mat &image, const cv::Scalar &mean, const cv::Scalar &std)
{
    cv::Mat normalized_image;
    cv::subtract(image, mean, normalized_image);
    cv::divide(normalized_image, std, normalized_image);

    cv::Vec3f image_pixel = image.at<cv::Vec3f>(0, 0);
    float red_normalized1 = image_pixel[0];   // 정규화된 R 값
    float green_normalized1 = image_pixel[1]; // 정규화된 G 값
    float blue_normalized1 = image_pixel[2];  // 정규화된 B 값
    cv::Vec3f normalized_pixel = normalized_image.at<cv::Vec3f>(0, 0);
    float red_normalized = normalized_pixel[0];   // 정규화된 R 값
    float green_normalized = normalized_pixel[1]; // 정규화된 G 값
    float blue_normalized = normalized_pixel[2];  // 정규화된 B 값
    // int8 로 가져오기
    // normalized_image = normalized_image *128 + 128;
    // cv::Mat final_image;
    // normalized_image.convertTo(final_image, CV_8UC3);

    return normalized_image;
}

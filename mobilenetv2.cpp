#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <numeric>

#include "Load.hpp"
#include "LoadImage.hpp"
#include "Function.hpp"

int main()
{
    std::string folder_path = "/workspace/mobis_ws/mobis/val2017";
    std::string output_filename = "/workspace/mobis_ws/mobis/mobilenetv2_OUTPUT/output.csv";

    int file_count = 0;

    for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    {
        clock_t start, finish;

        start = clock();
        std::string file_name = entry.path().filename().string();
        cv::Mat original_image = ReadImage(folder_path + "/" + file_name);
        std::cout << folder_path + "/" + file_name << std::endl;

        int target_width = 224;
        int target_height = 224;

        cv::Mat resized_image = ResizeImage(original_image, target_width, target_height);

        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.f / 255.f);

        cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
        cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
        cv::Mat normalized_image = NormalizeImage(float_image, mean, std);

        // 이미지 체크용
        std::string output_image_path1 = "/workspace/mobis_ws/mobis/output_origin.jpg";
        cv::imwrite(output_image_path1, original_image);
        std::string output_image_path2 = "/workspace/mobis_ws/mobis/output_normalized.jpg";
        cv::imwrite(output_image_path2, normalized_image * 255);

        // 이미지 INPUT
        std::vector<std::vector<std::vector<float>>> input_image(3, std::vector<std::vector<float>>(normalized_image.rows,std::vector<float>(normalized_image.cols,0)));
        for (int i = 0; i < normalized_image.rows; i++)
        {
            for (int j = 0; j < normalized_image.cols; j++)
            {
                cv::Vec3f &temp_p = normalized_image.at<cv::Vec3f>(i,j);
                input_image[0][i][j] = temp_p[0];
                input_image[1][i][j] = temp_p[1];
                input_image[2][i][j] = temp_p[2];
            }
        }

        // (0) conv2D 수행
        std::string t_weight_file_0 = GenerateFileName(0, 0, -1, "weight.csv");
        std::vector<std::vector<std::vector<std::vector<float>>>> start_mobilenetv2_kernel = LoadFeaKernelFromCSV(t_weight_file_0, 3, 3);
        std::vector<std::vector<std::vector<float>>> start_mobilenetv2 = FeatureConv2D(input_image, start_mobilenetv2_kernel, 2, 1);///////////////////////////////
        std::string t_weight_0 = GenerateFileName(0, 1, -1, "weight.csv");
        std::string t_bias_0 = GenerateFileName(0, 1, -1, "bias.csv");
        std::string t_running_mean_0 = GenerateFileName(0, 1, -1, "running_mean.csv");
        std::string t_running_var_0 = GenerateFileName(0, 1, -1, "running_var.csv");
        std::vector<float> batchnorm2d_weight_0 = LoadFromCSV1D(t_weight_0);
        std::vector<float> batchnorm2d_bias_0 = LoadFromCSV1D(t_bias_0);
        std::vector<float> batchnorm2d_running_mean_0 = LoadFromCSV1D(t_running_mean_0);
        std::vector<float> batchnorm2d_running_var_0 = LoadFromCSV1D(t_running_var_0);
        std::vector<std::vector<std::vector<float>>> start_batch = BatchNorm2D(start_mobilenetv2, batchnorm2d_weight_0, batchnorm2d_bias_0, batchnorm2d_running_mean_0, batchnorm2d_running_var_0);
        std::vector<std::vector<std::vector<float>>> start_output = ReLU6(start_batch);///////////////////////
        // std::cout << "(0) conv2D Done" << std::endl;

        int setting_n = 0;
        // (1) inverted residual 수행
        // (1) - (0)
        std::vector<std::vector<std::vector<float>>> first_inverted_conv2d_0(32, std::vector<std::vector<float>>(start_output[0].size(), std::vector<float>(start_output[0][0].size(), 0)));
        std::string t_inverted_weight_file_0 = GenerateFileName(1, 0, 0, "weight.csv");
        std::vector<std::vector<std::vector<float>>> Conv2D_kernel_0 = LoadKernelFromCSV(t_inverted_weight_file_0,start_output.size(), 3);
        first_inverted_conv2d_0 = DepthwiseConv2D(start_output, Conv2D_kernel_0, 1, 1);////////////////////////////////////////////
        std::string t_weight_1 = GenerateFileName(1, 0, 1, "weight.csv");
        std::string t_bias_1 = GenerateFileName(1, 0, 1, "bias.csv");
        std::string t_running_mean_1 = GenerateFileName(1, 0, 1, "running_mean.csv");
        std::string t_running_var_1 = GenerateFileName(1, 0, 1, "running_var.csv");
        std::vector<float> batchnorm2d_weight_1 = LoadFromCSV1D(t_weight_1);
        std::vector<float> batchnorm2d_bias_1 = LoadFromCSV1D(t_bias_1);
        std::vector<float> batchnorm2d_running_mean_1 = LoadFromCSV1D(t_running_mean_1);
        std::vector<float> batchnorm2d_running_var_1 = LoadFromCSV1D(t_running_var_1);
        std::vector<std::vector<std::vector<float>>> first_inverted_batch_0 = BatchNorm2D(first_inverted_conv2d_0, batchnorm2d_weight_1, batchnorm2d_bias_1, batchnorm2d_running_mean_1, batchnorm2d_running_var_1);
        std::vector<std::vector<std::vector<float>>> first_inverted_batch_output_0 = ReLU6(first_inverted_batch_0);///////////////////////////////
        // (1) - (1)
        std::vector<std::vector<std::vector<float>>> first_inverted_conv2d_1(first_inverted_batch_output_0.size(), std::vector<std::vector<float>>(first_inverted_batch_output_0[0].size(), std::vector<float>(first_inverted_batch_output_0[0][0].size(), 0)));
        std::string t_1_inverted_weight_file_0 = GenerateFileName(1, 1, -1, "weight.csv");
        std::vector<std::vector<std::vector<std::vector<float>>>> Conv2D_kernel_1 = LoadFeaKernelFromCSV(t_1_inverted_weight_file_0,first_inverted_batch_output_0.size(), 1);
        first_inverted_conv2d_1 = PointwiseConv2D(first_inverted_batch_output_0, Conv2D_kernel_1);//////////////////////////////
        std::string t_1_weight_1 = GenerateFileName(1, 2, -1, "weight.csv");
        std::string t_1_bias_1 = GenerateFileName(1, 2, -1, "bias.csv");
        std::string t_1_running_mean_1 = GenerateFileName(1, 2, -1, "running_mean.csv");
        std::string t_1_running_var_1 = GenerateFileName(1, 2, -1, "running_var.csv");
        std::vector<float> batchnorm2d_1_weight_1 = LoadFromCSV1D(t_1_weight_1);
        std::vector<float> batchnorm2d_1_bias_1 = LoadFromCSV1D(t_1_bias_1);
        std::vector<float> batchnorm2d_1_running_mean_1 = LoadFromCSV1D(t_1_running_mean_1);
        std::vector<float> batchnorm2d_1_running_var_1 = LoadFromCSV1D(t_1_running_var_1);
        std::vector<std::vector<std::vector<float>>> first_inverted_batch_1 = BatchNorm2D(first_inverted_conv2d_1, batchnorm2d_1_weight_1, batchnorm2d_1_bias_1, batchnorm2d_1_running_mean_1, batchnorm2d_1_running_var_1);

        std::vector<std::vector<std::vector<float>>> input = first_inverted_batch_1;

        int feature_num = 0;
        std::vector<std::vector<float>> inverted_residual_setting =
                                                    // t, c, n, s            // t: expansion factor  c:output channel 수 n:반복회수
                                                    {{1, 16, 1, 1},
                                                    {6, 24, 2, 2},
                                                    {6, 32, 3, 2},
                                                    {6, 64, 4, 2},
                                                    {6, 96, 3, 1},
                                                    {6, 160, 3, 2},
                                                    {6, 320, 1, 1}};
        size_t rows = inverted_residual_setting.size();
        for (size_t row = 1; row < rows; row++)
        {
            int r = inverted_residual_setting[row][2];
            feature_num += r;
        }

        //(2) ~ (17) 수행
        for (size_t row = 1; row < feature_num + 1; row++)
        {
            {
                auto start_time = std::chrono::high_resolution_clock::now();

                int stride = 1;
                if (row == 1 || row == 3 || row == 6 || row == 10 || row == 13 || row == 16)
                {
                    setting_n++;
                    stride = inverted_residual_setting[setting_n][3];
                }
                std::vector<std::vector<std::vector<float>>> inverted_residual_result =
                    InvertedResidual(input, inverted_residual_setting[setting_n][0], inverted_residual_setting[setting_n][1], stride, 1, row);
                if (row != 1 || row != 3 || row != 6 || row != 10 || row != 13 || row != 16)
                {
                    if (input.size() == inverted_residual_result.size() && stride == 1)
                    {
                        inverted_residual_result = connection(input, inverted_residual_result);
                    }
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
                input = inverted_residual_result;

                // std::cout << inverted_residual_result.size() << "\t row : " << row + 1 << "\tstride : " << stride << std::endl;
            }
        }

        //(18) 수행
        std::vector<std::vector<std::vector<float>>> inverted_residual_result_18(input.size(), std::vector<std::vector<float>>(input[0].size(), std::vector<float>(input[0][0].size(), 0)));
        std::string t_inverted_residual_18_weight_file = GenerateFileName(18, 0, -1, "weight.csv");
        std::vector<std::vector<std::vector<std::vector<float>>>> inverted_residual_18_kernel = LoadFeaKernelFromCSV(t_inverted_residual_18_weight_file,input.size(), 1);
        inverted_residual_result_18 = PointwiseConv2D(input, inverted_residual_18_kernel);

        std::string t_inverted_residual_18_avgpool_weight = GenerateFileName(18, 1, -1, "weight.csv");
        std::string t_inverted_residual_18_avgpool_bias = GenerateFileName(18, 1, -1, "bias.csv");
        std::string t_inverted_residual_18_avgpool_running_mean = GenerateFileName(18, 1, -1, "running_mean.csv");
        std::string t_inverted_residual_18_avgpool_running_var = GenerateFileName(18, 1, -1, "running_var.csv");
        std::vector<float> batchnorm2d_inverted_residual_result_18_weight = LoadFromCSV1D(t_inverted_residual_18_avgpool_weight);
        std::vector<float> batchnorm2d_inverted_residual_result_18_bias = LoadFromCSV1D(t_inverted_residual_18_avgpool_bias);
        std::vector<float> batchnorm2d_inverted_residual_result_18_running_mean = LoadFromCSV1D(t_inverted_residual_18_avgpool_running_mean);
        std::vector<float> batchnorm2d_inverted_residual_result_18_running_var = LoadFromCSV1D(t_inverted_residual_18_avgpool_running_var);
        std::vector<std::vector<std::vector<float>>> inverted_residual_result_18_batch = BatchNorm2D(inverted_residual_result_18, batchnorm2d_inverted_residual_result_18_weight, batchnorm2d_inverted_residual_result_18_bias, batchnorm2d_inverted_residual_result_18_running_mean, batchnorm2d_inverted_residual_result_18_running_var);
        std::vector<std::vector<std::vector<float>>> inverted_residual_result_18_batch_output = ReLU6(inverted_residual_result_18_batch);
        // std::cout << inverted_residual_result_18_batch_output.size() << "\t row : 18 " << std::endl;

        // pooling 수행
        std::vector<std::vector<std::vector<float>>> avgpool_result = avgpool(inverted_residual_result_18_batch_output);

        // classifier
        std::string t_classifier_weight_file = GenerateFileName(-1, 1, -1, "weight.csv");
        std::vector<std::vector<std::vector<float>>> classifier_weight = LoadFromCSV3D(t_classifier_weight_file);
        std::string t_classifier_bias_file = GenerateFileName(-1, 1, -1, "bias.csv");
        std::vector<float> classifier_bias = LoadFromCSV1D(t_classifier_bias_file);

        std::vector<float> output_last = Classifier(avgpool_result, classifier_weight, classifier_bias);

        int output_index = argmax(output_last);

        finish = clock();
        double duration = (double)(finish - start) / CLOCKS_PER_SEC;

        std::cout << "output_index : " << output_index << "\t연산시간 :\t" << duration << std::endl;
        

    }
    
    return 0;
}
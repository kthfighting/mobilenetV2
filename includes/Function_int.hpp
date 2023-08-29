#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <cmath>
#include <ctime>
#include <filesystem>

std::vector<std::vector<std::vector<int8_t>>> FeatureConv2D(const std::vector<std::vector<std::vector<int8_t>>> &input,
                                                    const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &kernel,
                                                    int stride, int padding)
{   

    int input_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int kernel_channels = kernel.size();
    int kernel_depth = kernel[0].size();
    int kernel_height = kernel[0][0].size();
    int kernel_width = kernel[0][0][0].size();

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
    std::vector<std::vector<std::vector<int8_t>>> output(kernel_channels, std::vector<std::vector<int8_t>>(output_height, std::vector<int8_t>(output_width, 0)));

    int number = 0;
    for (int ch = 0; ch < kernel_channels; ch++)
    {
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                for (int x = 0; x < input_channels; x++)
                {
                    for (int y = 0; y < kernel_height; y++)
                    {
                        for (int z = 0; z < kernel_width; z++)
                        {
                            int input_h = i * stride + y - padding;
                            int input_w = j * stride + z - padding;

                            // 패딩 적용
                            if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width)
                            {
                                output[ch][i][j] += input[x][input_h][input_w] * kernel[ch][x][y][z];
                            }
                            number++;
                        }
                    }
                }
            }
        }
    }
    return output;
}


std::vector<std::vector<std::vector<int8_t>>> DepthwiseConv2D(const std::vector<std::vector<std::vector<float>>> &input,
                                                    const std::vector<std::vector<std::vector<int8_t>>> &kernel,
                                                    int stride, int padding)
{
    int input_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int kernel_channels = kernel.size();
    int kernel_height = kernel[0].size();
    int kernel_width = kernel[0][0].size();

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
    std::vector<std::vector<std::vector<int8_t>>> output(kernel_channels, std::vector<std::vector<int8_t>>(output_height, std::vector<int8_t>(output_width, 0)));
    int number;
    for (int ch = 0; ch < kernel_channels; ch++)
    {
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                for (int y = 0; y < kernel_height; y++)
                {
                    for (int z = 0; z < kernel_width; z++)
                    {
                        int input_h = i * stride + y - padding;
                        int input_w = j * stride + z - padding;

                        // 패딩 적용
                        if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width)
                        {
                            output[ch][i][j] += input[ch][input_h][input_w] * kernel[ch][y][z];
                        }
                        number++;
                    }
                }
            }
        }
    }

    return output;
}


std::vector<std::vector<std::vector<int8_t>>> PointwiseConv2D(const std::vector<std::vector<std::vector<float>>> &input,
                                                    const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &kernel)
{   
    int input_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int kernel_channels = kernel.size();
    int kernel_depth = kernel[0].size();
    int kernel_height = kernel[0][0].size();
    int kernel_width = kernel[0][0][0].size();

    int output_height = input_height;
    int output_width = input_width;
    std::vector<std::vector<std::vector<int8_t>>> output(kernel_channels, std::vector<std::vector<int8_t>>(output_height, std::vector<int8_t>(output_width, 0)));

    int number = 0;
    for (int ch = 0; ch < kernel_channels; ch++)
    {
        for (int x = 0; x < input_channels; x++)
        {
            for (int i = 0; i < output_height; i++)
            {
                for (int j = 0; j < output_width; j++)
                {
                    output[ch][i][j] += input[x][i][j] * kernel[ch][x][0][0];
                    
                    number++;   
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<float>>> BatchNorm2D(const std::vector<std::vector<std::vector<int8_t>>> &input,
                                                         const std::vector<int8_t> &weight,
                                                         const std::vector<int8_t> &bias,
                                                         const std::vector<int8_t> &mean,
                                                         const std::vector<int8_t> &variance,
                                                         float eps = 1e-05)
{
    int num_channel = input.size();
    int num_height = input[0].size();
    int num_width = input[0][0].size();

    // 정규화 수행
    std::vector<std::vector<std::vector<float>>> normalized_input(num_channel, std::vector<std::vector<float>>(num_height, std::vector<float>(num_width, 0.0)));
    for (int i = 0; i < num_channel; i++)
    {
        for (int h = 0; h < num_height; h++)
        {
            for (int w = 0; w < num_width; w++)
            {
                normalized_input[i][h][w] = ((input[i][h][w] - mean[i]) / (std::sqrt(variance[i] + eps))) * weight[i] + bias[i];
            }
        }
    }


    return normalized_input;
}

std::vector<std::vector<std::vector<float>>> ReLU6(const std::vector<std::vector<std::vector<float>>> &input)
{
    int depth = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    std::vector<std::vector<std::vector<float>>> output(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

    for (int d = 0; d < depth; d++)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                float x = input[d][i][j];

                if (x > 6.0f)
                {
                    output[d][i][j] = 6.0f;
                }
                else if (x > 0.0f)
                {
                    output[d][i][j] = x;
                }
                else
                {
                    output[d][i][j] = 0.0f;
                }
            }
        }
    }

    return output;
}

std::string GenerateFileName(int n, int x, int y, const std::string &param)
{
    if (n == -1)
        return "/workspace/mobis_ws/mobis/feature_tensor/classifier." + std::to_string(x) + "." + param;

    if (n == 0 || n == 18)
    {
        if (y == -1)
        {
            return "/workspace/mobis_ws/mobis/feature_tensor/features." + std::to_string(n) + "." + std::to_string(x) + "." + param;
        }
        else
        {
            return "/workspace/mobis_ws/mobis/feature_tensor/features." + std::to_string(n) + "." + std::to_string(x) + "." + std::to_string(y) + "." + param;
        }
    }
    if (y == -1)
    {
        return "/workspace/mobis_ws/mobis/feature_tensor/features." + std::to_string(n) + ".conv." + std::to_string(x) + "." + param;
    }
    else
    {
        return "/workspace/mobis_ws/mobis/feature_tensor/features." + std::to_string(n) + ".conv." + std::to_string(x) + "." + std::to_string(y) + "." + param;
    }
}


// Inverted Residual 블록  (2) ~ (17) 해당
std::vector<std::vector<std::vector<float>>> InvertedResidual(const std::vector<std::vector<std::vector<float>>> &input,
                                                              int expansion_factor,
                                                              int out_channels,
                                                              int stride,
                                                              int padding,
                                                              int feature_num)
{
    double total_duration;

    
    feature_num++; // 파일명 맞추기 위함
    int input_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();

    // Expansion 단계 (pointwise 컨볼루션) (0)
    std::vector<std::vector<std::vector<int8_t>>> expanded(input_channels * expansion_factor, std::vector<std::vector<int8_t>>(input_height, std::vector<int8_t>(input_width, 0)));
    std::string weight_file_0_0 = GenerateFileName(feature_num, 0, 0, "weight.csv");
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> expansion_kernel = LoadFeaKernelFromCSV(weight_file_0_0, input_channels, 1);
    auto start_time = std::chrono::high_resolution_clock::now();
    expanded =PointwiseConv2D(input, expansion_kernel);///////////////////////////
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    total_duration += duration;
    std::string weight_0_1 = GenerateFileName(feature_num, 0, 1, "weight.csv");
    std::string bias_0_1 = GenerateFileName(feature_num, 0, 1, "bias.csv");
    std::string running_mean_0_1 = GenerateFileName(feature_num, 0, 1, "running_mean.csv");
    std::string running_var_0_1 = GenerateFileName(feature_num, 0, 1, "running_var.csv");

    std::vector<int8_t> batchnorm2d_weight_0_1 = LoadFromCSV1D(weight_0_1);
    std::vector<int8_t> batchnorm2d_bias_0_1 = LoadFromCSV1D(bias_0_1);
    std::vector<int8_t> batchnorm2d_running_mean_0_1 = LoadFromCSV1D(running_mean_0_1);
    std::vector<int8_t> batchnorm2d_running_var_0_1 = LoadFromCSV1D(running_var_0_1);


    auto start_time1 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<float>>> expanded_norm = BatchNorm2D(expanded, batchnorm2d_weight_0_1, batchnorm2d_bias_0_1, batchnorm2d_running_mean_0_1, batchnorm2d_running_var_0_1);

    std::vector<std::vector<std::vector<float>>> expanded_out = ReLU6(expanded_norm);

    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    total_duration += duration1;
    // std::cout << "\t InvertedResidual_0_size :" << expanded_out.size() << std::endl;

    // Depthwise 컨볼루션           (1)
    std::vector<std::vector<std::vector<int8_t>>> depthwise(input_channels * expansion_factor, std::vector<std::vector<int8_t>>(input_height, std::vector<int8_t>(input_width, 0)));
    std::string weight_file_1_0 = GenerateFileName(feature_num, 1, 0, "weight.csv");
    std::vector<std::vector<std::vector<int8_t>>> depthwise_kernel = LoadKernelFromCSV(weight_file_1_0,expanded_out.size(), 3);
    auto start_time2 = std::chrono::high_resolution_clock::now();

    depthwise = DepthwiseConv2D(expanded_out, depthwise_kernel, stride, padding);///////////////////////////////
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2).count();
    total_duration += duration2;

    std::string weight_1_1 = GenerateFileName(feature_num, 1, 1, "weight.csv");
    std::string bias_1_1 = GenerateFileName(feature_num, 1, 1, "bias.csv");
    std::string running_mean_1_1 = GenerateFileName(feature_num, 1, 1, "running_mean.csv");
    std::string running_var_1_1 = GenerateFileName(feature_num, 1, 1, "running_var.csv");
    std::vector<int8_t> batchnorm2d_weight_1_1 = LoadFromCSV1D(weight_1_1);
    std::vector<int8_t> batchnorm2d_bias_1_1 = LoadFromCSV1D(bias_1_1);
    std::vector<int8_t> batchnorm2d_running_mean_1_1 = LoadFromCSV1D(running_mean_1_1);
    std::vector<int8_t> batchnorm2d_running_var_1_1 = LoadFromCSV1D(running_var_1_1);

    auto start_time3 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<float>>> depthwise_norm = BatchNorm2D(depthwise, batchnorm2d_weight_1_1, batchnorm2d_bias_1_1, batchnorm2d_running_mean_1_1, batchnorm2d_running_var_1_1);
    std::vector<std::vector<std::vector<float>>> depthwise_out = ReLU6(depthwise_norm);
    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3).count();
    total_duration += duration3;
    // std::cout << "\t InvertedResidual_1_size :" << depthwise_out.size() << std::endl;

    // Projection 단계 (Pointwise 컨볼루션)              (2)
    std::vector<std::vector<std::vector<int8_t>>> output(out_channels, std::vector<std::vector<int8_t>>(input_height, std::vector<int8_t>(input_width, 0)));
    std::string weight_file_2 = GenerateFileName(feature_num, 2, -1, "weight.csv");
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> projection_kernel = LoadFeaKernelFromCSV(weight_file_2,depthwise_out.size(), 1);

    auto start_time4 = std::chrono::high_resolution_clock::now();
    output = PointwiseConv2D(depthwise_out, projection_kernel);
    auto end_time4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4 - start_time4).count();
    total_duration += duration4;
    // std::cout << "\t InvertedResidual_2_size :" << output.size() << std::endl;

    // Output 단계              (3)
    std::string weight_3 = GenerateFileName(feature_num, 3, -1, "weight.csv");
    std::string bias_3 = GenerateFileName(feature_num, 3, -1, "bias.csv");
    std::string running_mean_3 = GenerateFileName(feature_num, 3, -1, "running_mean.csv");
    std::string running_var_3 = GenerateFileName(feature_num, 3, -1, "running_var.csv");
    std::vector<int8_t> batchnorm2d_weight_3 = LoadFromCSV1D(weight_3);
    std::vector<int8_t> batchnorm2d_bias_3 = LoadFromCSV1D(bias_3);
    std::vector<int8_t> batchnorm2d_running_mean_3 = LoadFromCSV1D(running_mean_3);
    std::vector<int8_t> batchnorm2d_running_var_3 = LoadFromCSV1D(running_var_3);
    auto start_time5 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<float>>> output_norm = BatchNorm2D(output, batchnorm2d_weight_3, batchnorm2d_bias_3, batchnorm2d_running_mean_3, batchnorm2d_running_var_3);
    auto end_time5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5).count();
    total_duration += duration5;

    std::cout << total_duration <<std::endl;
    return output_norm;
}


std::vector<std::vector<std::vector<float>>> avgpool(const std::vector<std::vector<std::vector<float>>> &input)
{
    int input_rows = input[0][0].size();
    int input_cols = input[0].size();
    int input_channels = input.size();

    std::vector<std::vector<std::vector<float>>> result(1, std::vector<std::vector<float>>(1, std::vector<float>(input_channels, 0.0f)));
    for (int c = 0; c < input_channels; c++)
    {
        float sum = 0.0f;
        for (int i = 0; i < input_rows; i++)
        {
            for (int j = 0; j < input_cols; j++)
            {
                sum += input[c][i][j];
            }
        }
        result[0][0][c] = sum / (input_rows * input_cols);
    }

    return result;
}



std::vector<float> Classifier(const std::vector<std::vector<std::vector<float>>> &input,
                              const std::vector<std::vector<std::vector<float>>> &weights,
                              const std::vector<float> &bias)
{
    std::vector<float> output(weights.size(), 0.0f);

    for (int i = 0; i < weights.size(); i++)
    {
        for (int j = 0; j < weights[i].size(); j++)
        {
            output[i] += input[0][0][j] * weights[i][j][0];
        }
        output[i] += bias[i];
    }

    return output;
}

int argmax(const std::vector<float> &arr)
{
    float max_val = arr[0];
    int max_index = 0;

    for (int i = 1; i < arr.size(); i++)
    {
        if (arr[i] >= max_val)
        {
            max_val = arr[i];
            max_index = i;
        }
    }

    return max_index;
}



std::vector<std::vector<std::vector<float>>> connection(const std::vector<std::vector<std::vector<float>>> &input,
                                                      const std::vector<std::vector<std::vector<float>>> &output)
{
    std::vector<std::vector<std::vector<float>>> result(input.size(), std::vector<std::vector<float>>(input[0].size(), std::vector<float>(input[0][0].size(), 0.0)));

    for (size_t i = 0; i < input.size(); i++)
    {
        for (size_t j = 0; j < input[i].size(); j++)
        {
            for (size_t k = 0; k < input[i][j].size(); k++)
            {
                result[i][j][k] = input[i][j][k] + output[i][j][k];
            }
        }
    }

    return result;
}

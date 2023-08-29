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

std::vector<int8_t> QuantizeWeights(const std::vector<std::vector<std::vector<float>>> &weights, float min_value, float max_value)
{
    std::vector<int8_t> quantized_weights;

    for (const auto &channel : weights)
    {
        for (const auto &row : channel)
        {
            for (float value : row)
            {
                // 실수값을 양자화하여 정수값으로 변환
                int8_t quantized_value = static_cast<int8_t>(round((value - min_value) / (max_value - min_value) * 255.0f - 128.0f));
                quantized_weights.push_back(quantized_value);
            }
        }
    }

    return quantized_weights;
}

std::vector<std::vector<std::vector<std::vector<int8_t>>>> LoadFeaKernelFromCSV(const std::string &filename,
                                                                                int input_size,
                                                                                int kernel_size
                                                                                )
{    /**
     * 1) 특징 Conv2d 때 사용
     * 2) 차원 바뀔 때 사용
    */
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    std::ifstream file_row(filename);
    std::string num_row;
    int ch = 0;
    int row_count = 0;
    
    // output 차원
    while (std::getline(file_row, num_row, '\n'))
    {
        row_count++;
    }

    int output_channels = row_count;
    if (input_size == output_channels)
    {
        input_size = 1;
    }
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line) && ch < output_channels)
    {
        std::istringstream iss(line);
        std::vector<std::vector<std::vector<float>>> channel(input_size, std::vector<std::vector<float>>(kernel_size, std::vector<float> (kernel_size, 0.0f)));
        for (int input_channel = 0; input_channel <input_size; input_channel++)
        {
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    std::string value;
                    std::getline(iss, value, ',');
                    float val = std::stof(value);
                    channel[input_channel][i][j] = val;

                    max_value = std::max(max_value, val);
                    min_value = std::min(min_value, val);


                }
            }
        }
        kernel.push_back(channel);
        ch++;
    }
    file.close();// 양자화 하기 전 가중치를 3차원 벡터 안에 넣었음


    /**
     * @brief 양자화 시작
     * 
     */

    // 양자화한 가중치를 저장할 컨테이너
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> quantized_kernel;

    // 양자화 범위
    int quantization_bits = 8; // int8
    float range = std::pow(2, quantization_bits) - 1;
    float scale = (max_value - min_value) / range;
    int8_t z = static_cast<int8_t>(std::round((max_value*(-128) - min_value*(127))/(max_value-min_value)));

    for (int ch = 0; ch < kernel.size(); ch++)
    {
        std::vector<std::vector<std::vector<int8_t>>> quantized_channel(input_size, std::vector<std::vector<int8_t>>(kernel_size, std::vector<int8_t>(kernel_size, 0)));
        
        for (int input_channel = 0; input_channel < input_size; input_channel++)
        {
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    float float_value = kernel[ch][input_channel][i][j];
                    int8_t quantized_value = static_cast<int8_t>(std::round( float_value / scale) + z);
                    if (quantized_value < -128)
                    {
                        quantized_value = -128;
                    }
                    else if (quantized_value > 127)
                    {
                        quantized_value = 127;
                    }
                    quantized_channel[input_channel][i][j] = quantized_value;
                }
            }
        }
        quantized_kernel.push_back(quantized_channel);
    }





    // return kernel;
    return quantized_kernel;
}
std::vector<std::vector<std::vector<int8_t>>> QuantizeKernel(const std::vector<std::vector<std::vector<float>>> &kernel,
                                                              float &max_value, float &min_value)
{
    std::vector<std::vector<std::vector<int8_t>>> quantized_kernel;

    int quantization_bits = 8;
    float range = std::pow(2, quantization_bits) - 1;
    float scale = (max_value - min_value) / range;
    int8_t z = static_cast<int8_t>(std::round((max_value*(-128) - min_value*(127))/(max_value-min_value)));

    for (const auto &channel : kernel)
    {
        std::vector<std::vector<int8_t>> quantized_channel;
        for (const auto &row : channel)
        {
            std::vector<int8_t> quantized_row;
            for (float value : row)
            {
                int8_t quantized_value = static_cast<int8_t>(std::round( value / scale) + z);
                if (quantized_value < -128)
                {
                    quantized_value = -128;
                }
                else if (quantized_value > 127)
                {
                    quantized_value = 127;
                }
                quantized_row.push_back(quantized_value);
            }
            quantized_channel.push_back(quantized_row);
        }
        quantized_kernel.push_back(quantized_channel);
    }

    return quantized_kernel;
}

std::vector<std::vector<std::vector<int8_t>>> LoadKernelFromCSV(const std::string &filename,
                                                               int input_size,
                                                               int kernel_size)
{
    std::vector<std::vector<std::vector<float>>> kernel;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    std::ifstream file_row(filename);
    std::string num_row;
    int ch = 0;
    int row_count = 0;
    
    // output 차원
    while (std::getline(file_row, num_row, '\n'))
    {
        row_count++;
    }

    int output_channels = row_count;

    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line) && ch < output_channels)
    {
        std::istringstream iss(line);
        std::vector<std::vector<float>> channel(kernel_size, std::vector<float> (kernel_size, 0.0f));
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                std::string value;
                std::getline(iss, value, ',');
                float val = std::stof(value);
                channel[i][j] = val;

                max_value = std::max(max_value, val);
                min_value = std::min(min_value, val);
            }
        }

        kernel.push_back(channel);
        ch++;
    }
    file.close();
    
    std::vector<std::vector<std::vector<int8_t>>> quantized_kernel = QuantizeKernel(kernel, max_value, min_value);

    return quantized_kernel;
}


std::vector<std::vector<std::vector<int8_t>>> LoadFromCSV3D(const std::string &filename)
{
    std::vector<std::vector<std::vector<float>>> kernel;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();  
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<std::vector<float>> channel(1280, std::vector<float>(1, 0.0f));

        for (int i = 0; i < 1280; i++)
        {
            std::string value;
            std::getline(iss, value, ',');
            float val = std::stof(value);
            channel[i][0] = val;

            max_value = std::max(max_value, val);
            min_value = std::min(min_value, val);
        }
        kernel.push_back(channel);
    }

    file.close();

    std::vector<std::vector<std::vector<int8_t>>> quantized_kernel = QuantizeKernel(kernel, max_value, min_value);

    return quantized_kernel;
}

std::vector<std::vector<std::vector<float>>> LoadFromCSV3D_fp(const std::string &filename)
{
    std::vector<std::vector<std::vector<float>>> kernel;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();  
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<std::vector<float>> channel(1280, std::vector<float>(1, 0.0f));

        for (int i = 0; i < 1280; i++)
        {
            std::string value;
            std::getline(iss, value, ',');
            float val = std::stof(value);
            channel[i][0] = val;

            max_value = std::max(max_value, val);
            min_value = std::min(min_value, val);
        }
        kernel.push_back(channel);
    }

    file.close();


    return kernel;
}


std::vector<int8_t> LoadFromCSV1D(const std::string &filename)
{
    std::vector<float> data;
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();
    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string value;

        while (std::getline(iss, value, ','))
        {
            float val = std::stof(value);
            data.push_back(val);
            max_value = std::max(max_value, val);
            min_value = std::min(min_value, val);
        }
    }

    file.close();


    std::vector<int8_t> quantized_data;

    // Quantization
    int quantization_bits = 8; // int8_t
    float range = std::pow(2, quantization_bits) - 1;
    float scale = (max_value - min_value) / range;
    int8_t z = static_cast<int8_t>(std::round((max_value*(-128) - min_value*(127))/(max_value-min_value)));

    std::ifstream file2(filename);

    std::string line2;

    while (std::getline(file2, line2))
    {
        std::istringstream iss(line2);
        std::string value;

        while (std::getline(iss, value, ','))
        {
            float val = std::stof(value);
            int8_t quantized_value = static_cast<int8_t>(std::round( val / scale) + z);
            if (quantized_value < -128)
            {
                quantized_value = -128;
            }
            else if (quantized_value > 127)
            {
                quantized_value = 127;
            }
            quantized_data.push_back(quantized_value);
        }
    }

    file.close();
    return quantized_data;
}
std::vector<float> LoadFromCSV1D_fp(const std::string &filename)
{
    std::vector<float> data;

    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string value;

        while (std::getline(iss, value, ','))
        {
            data.push_back(std::stof(value));
        }
    }

    file.close();
    return data;
}

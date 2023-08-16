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

std::vector<std::vector<std::vector<std::vector<float>>>> LoadFeaKernelFromCSV(const std::string &filename,
                                                               int input_size,
                                                               int kernel_size)
{    /**
     * 1) 특징 Conv2d 때 사용
     * 2) 차원 바뀔 때 사용
    */
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel;

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
                    channel[input_channel][i][j] = std::stof(value);
                }
            }
        }
        kernel.push_back(channel);
        ch++;
    }
    file.close();
    return kernel;
}

std::vector<std::vector<std::vector<float>>> LoadKernelFromCSV(const std::string &filename,
                                                               int input_size,
                                                               int kernel_size)
{
    std::vector<std::vector<std::vector<float>>> kernel;

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
                channel[i][j] = std::stof(value);
            }
        }

        kernel.push_back(channel);
        ch++;
    }
    file.close();
    return kernel;
}


std::vector<std::vector<std::vector<float>>> LoadFromCSV3D(const std::string &filename)
{
    std::vector<std::vector<std::vector<float>>> kernel;

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
            channel[i][0] = std::stof(value);
        }
        kernel.push_back(channel);
    }

    file.close();
    return kernel;
}

std::vector<float> LoadFromCSV1D(const std::string &filename)
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

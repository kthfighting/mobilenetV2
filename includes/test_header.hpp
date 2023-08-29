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

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

namespace fs = std::filesystem;


class visualOdometry
{
    std::string path;
    cv::Mat K;
    cv::Mat P;
    std::vector<cv::Mat> images;
    cv::Ptr<cv::ORB> orb;
    cv::FlannBasedMatcher flann;
    
public:
     
    visualOdometry(std::string dataset_path);
    ~visualOdometry();

    std::vector<cv::Mat> gtPose;

    void setupPaths(std::string& calibPath, std::string& posePath, std::string& imagePath);
    cv::Mat camera_param(std::string& path, cv::Mat& K);
    std::vector<cv::Mat> get_poses(std::string& posePath);
    std::vector<cv::Mat> get_images(std::string& imagesPath);

    cv::Mat CalculateTMatrix(std::vector<cv::Point2f>, std::vector<cv::Point2f>);
    std::vector<cv::Mat> decomposeEssentialMatrix(const cv::Mat& E, const std::vector<cv::Point2f>& q1, const std::vector<cv::Point2f>& q2);
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getImagePointsFromMatches(int index);
    
};


#endif
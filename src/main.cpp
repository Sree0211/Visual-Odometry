#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "visualOdometry.hpp"
#include "visualization.hpp"

void showProgressBar(int progress, int total) {
    const int barWidth = 50;
    float percent = static_cast<float>(progress) / total;
    int barLength = static_cast<int>(barWidth * percent);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < barLength) std::cout << "=";
        else std::cout << " ";
    }

    std::cout << "] " << std::fixed << std::setprecision(1) << (percent * 100.0) << "%\r";
    std::cout.flush();
}

int main()
{
    std::string dataset_path = "/Users/sreenathswaminathan/Desktop/Uni-Docs/Autonomous fahren kurs/CV Project/Visual-Odometry/KITTI_dataset";

    cv::Mat P = cv::Mat(3,4,CV_64F);
    cv::Mat K = cv::Mat(3,3,CV_64F);

    visualOdometry vo(dataset_path);

    // Create Visualization object
    Visualization visualizer(800, 600);

    std::ofstream file("Points.txt");

    std::vector<std::pair<double, double>> gtPathval;
    std::vector<std::pair<double, double>> estimatedPath;

    cv::Mat currentPose;

    for(int i=0; i<51; i++)
    {

        auto gtPose = vo.gtPose[i];

        if(i == 0)
        {
            currentPose = gtPose;
        }
        else
        {
            auto imagePoints = vo.getImagePointsFromMatches(i);
            std::vector<cv::Point2f> q1 = imagePoints.first;
            std::vector<cv::Point2f> q2 = imagePoints.second;        

            cv::Mat T = vo.CalculateTMatrix(q1, q2);

            // Iterate over each element and replace NaN with zero
            T.forEach<double>([&](double &element, const int *position) -> void {
                if (std::isnan(element)) {
                    element = 0.0;
                }
            });

            CV_Assert(currentPose.rows == 4 && currentPose.cols == 4 && T.rows == 4 && T.cols == 4);

            // Calculate the inverse of the matrix T
            cv::Mat inv_transf = T.inv();

            // Multiply cur_pose by the inverse of T
            currentPose = currentPose * inv_transf;
            
        }

        // Taking the Ground truth pose and estimated current pose for visualization
        gtPathval.push_back(std::make_pair(gtPose.at<double>(0,3),gtPose.at<double>(2,3)));
        estimatedPath.push_back(std::make_pair(currentPose.at<double>(0,3),currentPose.at<double>(2,3)));

        showProgressBar(i, 51);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // Simulate some work

    }
    
    // Set poses for visualization
    visualizer.setGTPoses(gtPathval);
    visualizer.setEstimatedPoses(estimatedPath);

    // Run the visualization
    visualizer.runVisualization();

    // file.close();

    return 0;
}
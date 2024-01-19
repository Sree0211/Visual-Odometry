#include "visualOdometry.hpp"

visualOdometry::visualOdometry(std::string dataset_path) : path(dataset_path) 
{   
    
    std::string calibPath, posePath, imagePath;
    setupPaths(calibPath, posePath, imagePath);

    std::cout<<"Camera file path: "<<calibPath<<std::endl;
    std::cout<<"Poses file path: "<<posePath<<std::endl;
    std::cout<<"Images file path: "<<imagePath<<std::endl;

    // Get the camera parameters
    // cv::Mat K;
    this->P = camera_param(calibPath,this->K);

    // Get the pose values
    this->gtPose = get_poses(posePath);
      
    // Load the images
    this->images = get_images(imagePath);

    // // Extract key features and feature descriptors using ORB
    this->orb = cv::ORB::create(2000,1.2,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);

    // Setup FLANN parameters (K-NN can also be alternatively used)
    cv::Ptr<cv::flann::IndexParams>  indexparams = cv::makePtr<cv::flann::KDTreeIndexParams>();
    indexparams->setAlgorithm(6);
    indexparams->setInt("table_number",6);
    indexparams->setInt("key_size",12);
    indexparams->setInt("probe_level",1);

    cv::Ptr<cv::flann::SearchParams> searchparams = cv::makePtr<cv::flann::SearchParams>();
    searchparams->setInt("checks",50);

    // Perform FLANN based matcher
    this->flann = cv::FlannBasedMatcher(indexparams,searchparams);
    
}

visualOdometry::~visualOdometry() {}

void visualOdometry::setupPaths(std::string& calibPath, std::string& posePath, std::string& imagePath) 
{
    fs::path datasetDir(this->path);

    // Check if the dataset directory exists
    if (!fs::exists(datasetDir) || !fs::is_directory(datasetDir)) {
        std::cerr << "Invalid or non-existent dataset directory: " << this->path << std::endl;
        return;
    }

    // Set paths for calibration file, pose file, and image folder
    calibPath = (datasetDir / "calib.txt").string();
    posePath = (datasetDir / "poses.txt").string();
    imagePath = (datasetDir / "image_l/").string();

    // Check if the files and folders exist
    if (!fs::exists(calibPath)) {
        std::cerr << "Calibration file not found: " << calibPath << std::endl;
        calibPath.clear();
    }

    if (!fs::exists(posePath)) {
        std::cerr << "Pose file not found: " << posePath << std::endl;
        posePath.clear();
    }

    if (!fs::exists(imagePath) || !fs::is_directory(imagePath)) {
        std::cerr << "Image folder not found: " << imagePath << std::endl;
        imagePath.clear();
    }
}


cv::Mat visualOdometry::camera_param(std::string& path, cv::Mat& K)
{
    // Function to retrieve Camera intrinsic parameters (K) and the projection matrix (P)

    std::ifstream file(path);
    std::double_t value;
    std::vector<double> values;

    cv::Mat P(3,4,CV_64F);

    if(!file.is_open())
    {
        std::cerr<<"Unable to open the file "<<path<<std::endl;
    }

    std::string sline;

  
    std::getline(file, sline);
    std::istringstream iss(sline);
    
    while(iss >> value)
    {
        values.push_back(value);
    }
    
    if(values.size() != 12)
    {
        std::cerr << "Invalid number of values in the first line of the file." << std::endl;
        file.close();
        return P;
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            P.at<double>(i,j) = values[i * 4 + j];
        }
    }

    // Create cv::Mat K from the first 3x3 part of P
    K = P(cv::Rect(0, 0, 3, 3)).clone();

    file.close();

    return P;
}


 std::vector<cv::Mat> visualOdometry::get_poses(std::string& posePath)
 {
    // Provides us the GT poses

    std::vector<cv::Mat> poses;
    std::ifstream file(posePath);

    if(!file.is_open())
    {
        std::cerr<<" Error opening the file"<< std::endl;
        return poses;
    }

    std::string line;
    while(std::getline(file,line))
    {
        
        std::istringstream iss(line);
        std::vector<double> values;
        double value;

        // Read values from the line
        while (iss >> value) {
            
            values.push_back(value);
        }
        
        if(values.size() != 12)
        {
            std::cerr<<" Error with the poses"<<std::endl;
            file.close();
            return poses;
        }

        cv::Mat T = cv::Mat::eye(4, 4, CV_64F); // Transformation matrix is of shape (4x4), currenty we have only 12 data, later we append a row

        for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            T.at<double>(i,j) = values[i * 4 + j];
        }
        }
        
        poses.push_back(T);
    }
    
    file.close();
    return poses;
 }

 std::vector<cv::Mat> visualOdometry::get_images(std::string& imagesPath)
 {

    std::vector<cv::Mat> images;
    std::string imgName = imagesPath + "*.png";

    std::vector<std::string> filenames;

    cv::glob(imgName,filenames,false);


    for(const auto& file: filenames)
    {
    
        cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);

        if (!image.empty()) {
            images.push_back(image);
        } else {
            std::cerr << "Error loading image: " << file << std::endl;
        }

    }
    return images;
 }


 std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> visualOdometry::getImagePointsFromMatches(int index)
 {  
    // Get the image points from our matches
    std::vector<cv::Point2f> q1, q2;

    cv::Mat des1,des2;
    std::vector<cv::KeyPoint> kp1,kp2;
    orb->detect(this->images[index-1],kp1);
    orb->detect(this->images[index],kp2);

    orb->compute(this->images[index-1],kp1,des1);
    orb->compute(this->images[index],kp2,des2);

    std::vector<std::vector<cv::DMatch>> matches;

    this->flann.knnMatch(des1,des2,matches,2);

    float thres = 0.7;
    std::vector<cv::DMatch> goodMatches;

    for(size_t i=0; i <matches.size();i++)
    {
        if(matches[i].size() == 2 && matches[i][0].distance < thres*matches[i][1].distance)
        {
            goodMatches.push_back(matches[i][0]);
        }
    }

    // Sample Visualization
    // cv::Mat imgMatches;
    // cv::drawMatches(this->images[index-1],kp1,this->images[index],kp2,goodMatches,imgMatches, cv::Scalar_<double>::all(-1),cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // // Display the matches
    // cv::imshow("Matches", imgMatches);
    // cv::waitKey(400);

    for(const auto& match : goodMatches)
    {
        q1.push_back(kp1[match.queryIdx].pt);
        q2.push_back(kp2[match.queryIdx].pt);
    }

    return std::make_pair(q1, q2);
 }

 cv::Mat visualOdometry::CalculateTMatrix(std::vector<cv::Point2f> q1, std::vector<cv::Point2f> q2)
 {
    // Function to calculate transformation matrix based on the match descriptors (good matches, after performing ratio test)
    cv::Mat TransMat = cv::Mat::eye(4,4,CV_64F); // Transformation matrix

    // std::cout << "Correspondences:\n";
    // for (int i = 0; i < q1.size(); ++i) {
    // std::cout << "q1[" << i << "]: " << q1[i] << "\tq2[" << i << "]: " << q2[i] << std::endl;
    // }
    
    cv::Mat E = cv::findEssentialMat(q1,q2,this->K,cv::RANSAC,0.99,1); // Essential matrix
    

    // std::cout<<"E size: "<<E.size()<<std::endl;
    // std::cout << "E:\n" << E << std::endl;
    E.convertTo(E, CV_64F);

    // Perform SVD to extract R and t values from our essential matrix
    cv::Mat R, t;
    std::vector<cv::Mat> rightPair = this->decomposeEssentialMatrix(E,q1,q2);

    R = rightPair[0];
    t = rightPair[1];
    
    R.copyTo(TransMat(cv::Rect(0, 0, 3, 3)));
    t.copyTo(TransMat(cv::Rect(3, 0, 1, 3)));

    return TransMat;
 }

 std::vector<cv::Mat> visualOdometry::decomposeEssentialMatrix(const cv::Mat& E, const std::vector<cv::Point2f>& q1, const std::vector<cv::Point2f>& q2) {
    
    std::vector<cv::Mat> rightPair;

    CV_Assert(E.rows == 3 && E.cols == 3);

    // Perform SVD on the essential matrix
    cv::SVD svd(E,cv::SVD::FULL_UV);

    // Extract the singular values and matrices U, Vt
    cv::Mat singularValues = svd.w;
    cv::Mat U = svd.u;
    cv::Mat Vt = svd.vt;    


    U.convertTo(U, CV_64F);
    Vt.convertTo(Vt, CV_64F);

    if (cv::determinant(U) < 0)
        U.col(2) *= -1;
    if (cv::determinant(Vt) < 0)
        Vt.row(2) *= -1;

    // Construct the rotation matrices
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1); // Skew-Symmetric matrix
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;

    // Construct the translation vectors
    cv::Mat t1 = U.col(2);
    cv::Mat t2 = -U.col(2);

    // While applying SVD we get 2 Rotation matrices based on U,V_t
    std::vector<cv::Mat> pairs = {R1, R2};


    // Lambda function to calculate the number of positive z coordinates
    auto calculatePositiveZCount = [&](const cv::Mat& R, const cv::Mat& t) -> int {
        
        // Calcualte the transformation matrix
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        t.copyTo(T(cv::Rect(3, 0, 1, 3)));

        // Calucalte the projection matrix
        cv::Mat P1 = this->K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat P2 = this->K * T(cv::Rect(0, 0, 4, 3));

        // Triangulate the 3D points
        cv::Mat hom_Q1, hom_Q2;

        // Convert good matches into Matrix form for triangulation
        cv::Mat q1Mat(q1.size(), 2, CV_64F);
        cv::Mat q2Mat(q2.size(), 2, CV_64F);

        // Copy data from vectors to matrices
        for (size_t i = 0; i < q1.size(); ++i) {
            q1Mat.at<double>(i, 0) = q1[i].x;
            q1Mat.at<double>(i, 1) = q1[i].y;

            q2Mat.at<double>(i, 0) = q2[i].x;
            q2Mat.at<double>(i, 1) = q2[i].y;
        }

        cv::triangulatePoints(P1, P2, q1Mat.t(), q2Mat.t(), hom_Q1);


        cv::Mat uhom_Q1 = cv::Mat::zeros(3, hom_Q1.cols, CV_64F);

        for (int i = 0; i < hom_Q1.cols; ++i) {
            double w = hom_Q1.at<double>(3, i);
            uhom_Q1.at<double>(0, i) = hom_Q1.at<double>(0, i) / w;
            uhom_Q1.at<double>(1, i) = hom_Q1.at<double>(1, i) / w;
            uhom_Q1.at<double>(2, i) = hom_Q1.at<double>(2, i) / w;
        }

        // Find the number of points with positive z coordinate in the second camera
        int sumOfPosZQ2 = cv::countNonZero(uhom_Q1.row(2) > 0);

        return sumOfPosZQ2;
    };

    // Select the pair with the most points with positive z coordinate
    int maxPositiveZCount = 0;
    int rightPairIdx = 0;

    for (size_t i = 0; i < pairs.size(); ++i) {
        const cv::Mat& R = pairs[i];
        const cv::Mat& T = (i == 0) ? t1 : t2;

        int positiveZCount = calculatePositiveZCount(R, T);

        if (positiveZCount > maxPositiveZCount) {
            maxPositiveZCount = positiveZCount;
            rightPairIdx = static_cast<int>(i);
        }
    }

    rightPair.push_back(pairs[rightPairIdx]);
    rightPair.push_back((rightPairIdx == 0) ? t1 : t2);

    return rightPair;
}
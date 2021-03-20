
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

struct DataFrameLiDAR {
    int frameID;
    float ttc;
    float xmin;
}sDataFrameLiDAR;

std::vector<DataFrameLiDAR> vDataFrameLiDAR;

struct DataFrameCamera {
    int frameID;
    string detectorType;
    string descriptorType;
    float ttc_median;
    float ttc_mean;
    float dRatioMedian;
    float dRatioMean;
}sDataFrameCamera;

std::vector<DataFrameCamera> vDataFrameCamera;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results
    bool bPlot = false;          // plot to image
    bool bStoreInFrame = true;
    bool bDebug = false;


    /* MAIN LOOP OVER ALL IMAGES */

    vector<string> sDetectors = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> sDescriptors = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    for (auto itDet=sDetectors.begin(); itDet!=sDetectors.end(); itDet++)
        for (auto itDesc=sDescriptors.begin(); itDesc!=sDescriptors.end(); itDesc++)
        {
            dataBuffer.resize(0);
            if(bDebug) cout << "detector " << *itDet << std::endl;
            if(bDebug) cout << "descriptor " << *itDesc << std::endl;
            string detectorType = *itDet;       // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
            string descType = *itDesc;          // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT            
 
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
            {

                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                if(bDebug) cout << "imgFullFilename " << imgFullFilename << endl;

                // load image from file 
                cv::Mat img = cv::imread(imgFullFilename);

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = img;
                dataBuffer.push_back(frame);

                if(dataBuffer.size() > dataBufferSize)
                    dataBuffer.erase(dataBuffer.begin());

                if(bDebug) cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


                /* DETECT & CLASSIFY OBJECTS */

                float confThreshold = 0.2;
                float nmsThreshold = 0.4;        
                detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                            yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

                if(bDebug) cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


                /* CROP LIDAR POINTS */

                // load 3D Lidar points from file
                string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
                std::vector<LidarPoint> lidarPoints;
                loadLidarFromFile(lidarPoints, lidarFullFilename);

                // remove Lidar points based on distance properties
                float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
                cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
            
                (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

                if(bDebug) cout << "#3 : CROP LIDAR POINTS done" << endl;

                /* CLUSTER LIDAR POINT CLOUD */

                // associate Lidar points with camera-based ROI
                float shrinkFactor = 0.20; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
                clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

                // Visualize 3D objects
                bVis = true;
                if(bVis)
                {
                    show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(960, 960), imgNumber.str(), false);
                }
                bVis = false;

                if(bDebug) cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
                
                
                // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
                //continue; // skips directly to the next image without processing what comes beneath

                /* DETECT IMAGE KEYPOINTS */

                // convert current image to grayscale
                cv::Mat imgGray;
                cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints;     // create empty feature list for current image
                

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    if(bDebug)
                        cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;

                if(bDebug)
                    cout << "#5 : DETECT KEYPOINTS done" << endl;


                /* EXTRACT KEYPOINT DESCRIPTORS */

                cv::Mat descriptors;
                
                if (!((itDet->compare("SIFT") == 0 and itDesc->compare("ORB") == 0) or (!(itDet->compare("AKAZE") == 0) and itDesc->compare("AKAZE") == 0)))
                {
                    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descType);
                
                    // push descriptors for current frame to end of data buffer
                    (dataBuffer.end() - 1)->descriptors = descriptors;
                    if(bDebug)
                        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;
                }


                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    //if((itDet->compare("SIFT") == 0 and itDesc->compare("ORB") == 0) or itDesc->compare("AKAZE") == 0)
                    if ((itDet->compare("SIFT") == 0 and itDesc->compare("ORB") == 0) or (!(itDet->compare("AKAZE") == 0) and itDesc->compare("AKAZE") == 0))
                    {
                        DataFrameCamera datFrameC;
                        datFrameC.frameID = imgIndex;
                        datFrameC.detectorType = detectorType;
                        datFrameC.descriptorType = descType;
                        datFrameC.ttc_median = -1;
                        datFrameC.ttc_mean = -1;
                        datFrameC.dRatioMedian = -1;
                        datFrameC.dRatioMean = -1;
                        vDataFrameCamera.push_back(datFrameC);
                        continue;
                    }

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    string descriptorType = descType.compare("SIFT") == 0 ? "DES_HOG" : "DES_BINARY"; // DES_BINARY, DES_HOG
                    //cout << "descriptorType " << descriptorType << endl;
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    if(bDebug)
                        cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    
                    /* TRACK 3D OBJECT BOUNDING BOXES */

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
                    map<int, int> bbBestMatches;
                    matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                    if(bDebug)
                        cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                    /* COMPUTE TTC ON OBJECT IN FRONT */

                    // loop over all BB match pairs
                    for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                    {
                        // find bounding boxes associates with current match
                        BoundingBox *prevBB, *currBB;
                        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                            {
                                currBB = &(*it2);
                            }
                        }

                        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                        {
                            if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                            {
                                prevBB = &(*it2);
                            }
                        }

                        // compute TTC for current match
                        if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                        {
                            //// STUDENT ASSIGNMENT
                            //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                            double ttcLidar, xmin;
                            computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, xmin);
                            //// EOF STUDENT ASSIGNMENT

                            //// STUDENT ASSIGNMENT
                            //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                            //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                            double ttcCamera, ttcCameraMean, medianDistRatio, meanDistRatio;
                            clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches); 

                            computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, 
                            ttcCamera, ttcCameraMean, medianDistRatio, meanDistRatio); //
                            //cout << "#### keypoints " << (dataBuffer.end() - 2)->keypoints.size() << " keypoint curr " << (dataBuffer.end() - 1)->keypoints.size() << endl;
                            //// EOF STUDENT ASSIGNMENT


                            // fill dataFrames
                            if(bStoreInFrame)
                            {
                                DataFrameLiDAR datFrameL;
                                datFrameL.frameID = imgIndex;
                                datFrameL.ttc = ttcLidar;
                                datFrameL.xmin = xmin;
                                vDataFrameLiDAR.push_back(datFrameL);
                            }
                            if(vDataFrameLiDAR.size() == 18)
                                bStoreInFrame = false;
                            

                            DataFrameCamera datFrameC;
                            datFrameC.frameID = imgIndex;
                            datFrameC.detectorType = detectorType;
                            datFrameC.descriptorType = descType;
                            datFrameC.ttc_median = ttcCamera;
                            datFrameC.ttc_mean = ttcCameraMean;
                            datFrameC.dRatioMedian = medianDistRatio;
                            datFrameC.dRatioMean = meanDistRatio;
                            vDataFrameCamera.push_back(datFrameC);


                            bVis = true;
                            if (bVis)
                            {
                                cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                            
                                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                                cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), 
                                cv::Scalar(0, 255, 0), 2);

                                std::vector<cv::KeyPoint> kp_curr;
                                for (auto elem : (dataBuffer.end() - 1)->kptMatches)
                                    if(currBB->roi.contains((dataBuffer.end() - 1)->keypoints[elem.trainIdx].pt))
                                        kp_curr.push_back((dataBuffer.end() - 1)->keypoints[elem.trainIdx]);
                                cv::drawKeypoints(visImg, kp_curr, visImg, cv::Scalar(0,0,255), cv::DrawMatchesFlags(4));
                                
                                //cout << "currBB->kptMatches " << currBB->kptMatches.size() << endl;

                                cv::Mat visImgPrev = (dataBuffer.end() - 2)->cameraImg.clone();
                                showLidarImgOverlay(visImgPrev, prevBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImgPrev);
                                cv::rectangle(visImgPrev, cv::Point(prevBB->roi.x, prevBB->roi.y), cv::Point(prevBB->roi.x + prevBB->roi.width, prevBB->roi.y + prevBB->roi.height), 
                                cv::Scalar(0, 255, 0), 2);
                                std::vector<cv::KeyPoint> kp_prev;
                                for (auto elem : (dataBuffer.end() - 2)->kptMatches)
                                    if(prevBB->roi.contains((dataBuffer.end() - 2)->keypoints[elem.queryIdx].pt))
                                        kp_prev.push_back((dataBuffer.end() - 2)->keypoints[elem.queryIdx]);
                                cv::drawKeypoints(visImgPrev, kp_prev, visImgPrev, cv::Scalar(0,0,255), cv::DrawMatchesFlags(4));

                                //cout << "prevBB->kptMatches " << prevBB->kptMatches.size() << endl;

                                char str[200];
                                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                                putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                                // stitch images
                                cv::Mat matDst(cv::Size(visImg.cols,visImg.rows*2),visImg.type(),cv::Scalar::all(0));
                                cv::Mat matRoi = matDst(cv::Rect(0,0,visImg.cols,visImg.rows));
                                visImgPrev.copyTo(matRoi);
                                matRoi = matDst(cv::Rect(0,visImg.rows,visImg.cols,visImg.rows));
                                visImg.copyTo(matRoi);

                                string windowName = "Final Results : TTC ";
                                cv::namedWindow(windowName, 4);
                                cv::imshow(windowName, matDst);
                                
                                if(bPlot)
                                {
                                    string saveName = "result_" + imgNumber.str() + imgFileType;
                                    cout << "Writing to: " << saveName << endl;
                                    cv::imwrite(saveName, visImg);
                                }


                                cout << "Press key to continue to next frame" << endl << endl<< endl;
                                cv::waitKey(0);
                            }
                            bVis = false;

                        } // eof TTC computation
                    } // eof loop over all BB matches     
                }
            } // eof loop over all images
        }

    // create a name for the file output
    std::string filename = "performance_eval.csv";
    std::fstream outputFile;
    outputFile.open(filename, std::ios::out);


    outputFile << "frame" << "," << "ttcLiDAR" << "," << "xmin" << std::endl;
    for (std::vector<DataFrameLiDAR>::iterator it=vDataFrameLiDAR.begin(); it!=vDataFrameLiDAR.end(); it++)
        outputFile << it->frameID << "," << it->ttc << "," << it->xmin << std::endl;
    outputFile << std::endl << std::endl;


    // write data to csv
    /*outputFile << "detector" << "," << "descriptor" << "," << "frame" << "ttcCameraMedian" << "," << "ttcCameraMean" << "," << "distRatioMedian" << "," << "distRatioMean" << std::endl;
    for (std::vector<DataFrameCamera>::iterator it=vDataFrameCamera.begin(); it!=vDataFrameCamera.end(); it++)
        if((it->detectorType.compare("SIFT") == 0 and it->descriptorType.compare("ORB") == 0) or it->descriptorType.compare("AKAZE") == 0)
            outputFile << it->detectorType << "," << it->descriptorType << "," << it->frameID << "," << "-" << "," << "-" << "," << "-" << "," << "-" << std::endl;
        else
            outputFile << it->detectorType << "," << it->descriptorType << "," << it->frameID << "," << it->ttc_median << "," << it->ttc_mean << "," << it->dRatioMedian << "," << it->dRatioMean << std::endl;
    */

    // write data to csv
    //outputFile << "detector" << "," << "descriptor" << "," << "frame" << "ttcCameraMedian" << "," << "ttcCameraMean" << "," << "distRatioMedian" << "," << "distRatioMean" << std::endl;
    outputFile << "" << "," << "" << "," << 
    "ttcCameraMedian" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" <<
    "ttcCameraMean" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" <<
    "distRatioMedian" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" << "," << "" <<
    "distRatioMean" << std::endl;

    outputFile << "detector" << "," << "frame" << "," << 
    "BRISK" << "," << "BRIEF" << "," << "ORB" << "," << "FREAK" << "," << "AKAZE" << "," << "SIFT" << "," << 
    "BRISK" << "," << "BRIEF" << "," << "ORB" << "," << "FREAK" << "," << "AKAZE" << "," << "SIFT" << "," << 
    "BRISK" << "," << "BRIEF" << "," << "ORB" << "," << "FREAK" << "," << "AKAZE" << "," << "SIFT" << "," << 
    "BRISK" << "," << "BRIEF" << "," << "ORB" << "," << "FREAK" << "," << "AKAZE" << "," << "SIFT" << std::endl;
    for (int i=0;i<sDetectors.size();i++)
    {
        for (int k=0;k<imgEndIndex;k++)
        {
            int idx = (i*imgEndIndex*(sDetectors.size()-1)) + k;
            outputFile << vDataFrameCamera[idx].detectorType << "," << vDataFrameCamera[idx].frameID;
            for (int j=0;j<sDescriptors.size();j++)
                outputFile << "," << vDataFrameCamera[idx + (j*imgEndIndex)].ttc_median;
            for (int j=0;j<sDescriptors.size();j++)
                outputFile << "," << vDataFrameCamera[idx + (j*imgEndIndex)].ttc_mean;
            for (int j=0;j<sDescriptors.size();j++)
                outputFile << "," << vDataFrameCamera[idx + (j*imgEndIndex)].dRatioMedian;
            for (int j=0;j<sDescriptors.size();j++)
                outputFile << "," << vDataFrameCamera[idx + (j*imgEndIndex)].dRatioMean;
            outputFile << std::endl;
        }
    }
        
    /*for (std::vector<DataFrameCamera>::iterator it=vDataFrameCamera.begin(); it!=vDataFrameCamera.end(); it++)
    {   
        if((it->detectorType.compare("SIFT") == 0 and it->descriptorType.compare("ORB") == 0) or it->descriptorType.compare("AKAZE") == 0)
            outputFile << it->detectorType << "," << it->descriptorType << "," << it->frameID << "," << "-" << "," << "-" << "," << "-" << "," << "-" << std::endl;
        else
            outputFile << it->detectorType << "," << it->descriptorType << "," << it->frameID << "," << it->ttc_median << "," << it->ttc_mean << "," << it->dRatioMedian << "," << it->dRatioMean << std::endl;
    }*/
        
    // loop over DataFrames
    //outputFile << detectorType << "," << descType << "," << imgIndex << "," << ttcLidar << "," << ttcCamera << "," << ttcCameraMean << "," << xmin << "," << medianDistRatio << "," << meanDistRatio << std::endl;
    //outputFile << std::endl << std::endl;

    // close the output file
    outputFile.close();

    return 0;
}

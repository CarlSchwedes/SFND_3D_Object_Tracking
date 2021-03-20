
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, std::string imgNumber, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(false)
    {
        string saveName = "result_3DObj_" + imgNumber + ".png";
        cout << "Writing to: " << saveName << endl;
        cv::imwrite(saveName, topviewImg);
    }

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

template<class T>
T getVar(T mean, std::vector<T> X)
{
    T var = 0.0;
    for (auto it=X.begin();it!=X.end();it++)
        var += pow((*it)-mean,2);
    return var / (T)X.size();
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // return keypoint correspondences in kptMatches
    std::vector<cv::DMatch> matches, outlierFreeMatches;
    std::vector<float> dist;
    for (auto itr=kptMatches.begin(); itr!=kptMatches.end(); itr++)
        if (boundingBox.roi.contains(kptsPrev.at(itr->queryIdx).pt) and boundingBox.roi.contains(kptsCurr.at(itr->trainIdx).pt))
        {
            matches.push_back(*itr);
            dist.push_back(sqrt(pow(kptsCurr.at(itr->trainIdx).pt.x - kptsPrev.at(itr->queryIdx).pt.x, 2) + 
                pow(kptsCurr.at(itr->trainIdx).pt.y - kptsPrev.at(itr->queryIdx).pt.y, 2)));
        }

    float mean = std::accumulate(dist.begin(), dist.end(), 0.0)/ dist.size();
    float sigma = sqrt(getVar<float>(mean, dist));

    // remove outlier matches from vector
    for (auto itr=matches.begin(); itr!=matches.end(); itr++) 
        if (abs(mean - cv::norm(kptsCurr.at(itr->trainIdx).pt - kptsPrev.at(itr->queryIdx).pt)) < sigma)
            outlierFreeMatches.push_back(*itr);

    kptMatches = outlierFreeMatches;
    boundingBox.kptMatches = outlierFreeMatches;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, 
    double &TTC, double &ttcCameraMean, double &medianDistRatio, double &meanDistRatio, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        meanDistRatio = NAN;
        medianDistRatio = NAN;
        ttcCameraMean = NAN;
        TTC = NAN;
    }else
    {
        // compute camera-based TTC from distance ratios
        meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

        if(distRatios.size()%2 == 0)
            medianDistRatio = (distRatios[distRatios.size()/2 - 1] + distRatios[distRatios.size()/2]) / 2.0;
        else
            medianDistRatio = distRatios[distRatios.size()/2];

        double dT = 1.0 / frameRate;
        if(meanDistRatio == 1)
            ttcCameraMean = NAN;
        else
            ttcCameraMean = -dT / (1.0 - meanDistRatio);

        if(medianDistRatio == 1)
            TTC = NAN;
        else
            TTC = -dT / (1.0 - medianDistRatio);
    }

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double &xmin)
{
    // auxiliary variables
    double dT = 1.0/frameRate;  // time between two measurements in seconds
    double laneWidth = 4.0;     // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->x > 2 and it->y > -(laneWidth/2.0) and it->y < laneWidth/2.0)
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->x > 2 and it->y > -(laneWidth/2.0) and it->y < laneWidth/2.0)
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    /*cout << "frameRate " << frameRate << endl;
    cout << "dT " << dT << endl;
    cout << "minXPrev " << minXPrev << endl;
    cout << "minXCurr " << minXCurr << endl;
    cout << endl;*/

    xmin = minXCurr;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


template<class T>
std::map<T,T> getBestMatches(std::vector<std::vector<T>> m)
{
    std::map<T,T> bestMatches;
    std::vector<std::vector<T>> max(m.size(), std::vector<T>(3,-1));

    for (int i=0; i<m.size(); i++)
        for (int j=0; j<m[i].size(); j++)
            if(m[i][j] > max[i][0])
                max[i] = {m[i][j], i, j};

    for (auto elem : max)
        bestMatches.insert({elem[1],elem[2]});
    return bestMatches;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // prepare matrix object to keep count for boxID's
    std::vector<std::vector<int>> mat(prevFrame.boundingBoxes.size(), std::vector<int>(currFrame.boundingBoxes.size(), 0));

    // outer loop, using keypoint matches from prev and current frame
    for (auto itr_m=matches.begin(); itr_m!=matches.end();itr_m++)
    {
        // use matches query and train idx to identify bbox matches
        cv::KeyPoint kp_curr = currFrame.keypoints[itr_m->trainIdx];
        cv::KeyPoint kp_prev = prevFrame.keypoints[itr_m->queryIdx];

        vector<int> train_id, query_id;
        // in which of the curr bbox and prev bbox are matches
        for (auto itr_bb=currFrame.boundingBoxes.begin(); itr_bb!=currFrame.boundingBoxes.end(); itr_bb++)
            if (itr_bb->roi.contains(kp_curr.pt))
                train_id.push_back(itr_bb->boxID);

        for (auto itr_bb=prevFrame.boundingBoxes.begin(); itr_bb!=prevFrame.boundingBoxes.end(); itr_bb++)
            if (itr_bb->roi.contains(kp_prev.pt))
                query_id.push_back(itr_bb->boxID);

        for (int qID : query_id)
            for (int tID : train_id)
                mat[qID][tID]++;
    }

    // return tuple of bboxIDs representing the best match pairs
    bbBestMatches = getBestMatches<int>(mat);
}

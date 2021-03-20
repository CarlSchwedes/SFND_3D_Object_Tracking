# SFND 3D Object Tracking

## FP.1 Match 3D Objects:

Implementing ```matchBoundingBoxes``` turned out to be fairly simple and straight forward when reading the final code sample. Nonetheless, this function was actually quite challenging to me where it was difficult to get the inner for-loops working properly in such a way, to assemble several boxIDs from the previous and the current frame. The identified boxIDs are stored in temporal ```vectors```. Those vectors are used for counting the number of matched keypoint pairs in the bounding boxes. The 2D vector ```std::vector<std::vector<int>> mat(prevFrame.boundingBoxes.size(), std::vector<int>(currFrame.boundingBoxes.size(), 0));``` allows to directly count the number of keypoints for every individual pair of bounding boxes where the boxIDs directly correspond to the indices of the matrix. Lastly the maximum column value of each row in the matrix needs to be determined, where ```(qID,tID)``` finally represents the approximated bounding box match pair.

```cpp
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
```


## FP.2 Compute Lidar-based TTC:

After preprocessing the point cloud information with a ```shrinkFactor``` and a region of interest from camera image,

```cpp
// associate Lidar points with camera-based ROI
float shrinkFactor = 0.20; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);
```

computing TTC from LiDAR turns out to be fairly simple. Hereby, points in the lidar cloud which might intersect with the ego-vehicle itself are filtered by an additional constraint: ```it->x > 1.5```.

```cpp
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0/frameRate;  // time between two measurements in seconds
    double laneWidth = 4.0;     // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->x > 1.5 and it->y > -(laneWidth/2) and it->y < laneWidth/2)
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->x > 1.5 and it->y > -(laneWidth/2) and it->y < laneWidth/2)
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes:

Clustering of the keypoint matches from previous and current frame is done by using region of interest (ROI). Mean and standard deviation is used to remove outlier points from the set of match candidates. The distribution of test samples along the mean value is limited to &plusmn;&sigma; (about 68.27% of all the samples are accepted).

```cpp
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
            // compute euclidean distance of keypoint matches
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
```

## FP.4 Compute Camera-based TTC:

The matched keypoint pairs (keypoints from previous and current step) are used to compute TTC on camera only. For minimizing influence of outlier samples, the euclidean distance between matches is computed to achieve a distance ratio. Finally, the median/mean is used to provide an outlier free set of value for computing TTC.

```cpp
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
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

            double minDist = ******100.0******; // min. required distance

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
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    double medianDistRatio;
    if(distRatios.size()%2 == 0)
        medianDistRatio = (distRatios[distRatios.size()/2 - 1] + distRatios[distRatios.size()/2]) / 2.0;
    else
        medianDistRatio = distRatios[int(distRatios.size()/2)];

    double dT = 1.0 / frameRate;
    TTC = -dT / (1.0 - medianDistRatio);
}
```

## FP.5 Performance Evaluation 1:

| frame | ttcLiDAR | xmin  |
|-------|----------|-------|
| 1     | 12.9722  | 7.913 |
| 2     | 12.264   | 7.849 |
| 3     | 13.9161  | 7.793 |
| 4     | 7.11572  | 7.685 |
| 5     | **16.2511**  | 7.638 |
| 6     | 12.4213  | 7.577 |
| 7     | ****34.3404****  | 7.555 |
| 8     | **18.7875**  | 7.515 |
| 9     | 9.17779  | 7.434 |
| 10    | **18.0318**  | 7.393 |
| 11    | 3.83244  | 7.205 |
| 12    | **-10.8537** | 7.272 |
| 13    | 9.22307  | 7.194 |
| 14    | 10.9678  | 7.129 |
| 15    | 8.09422  | 7.042 |
| 16    | 3.17535  | 6.827 |
| 17    | **-9.99424** | 6.896 |
| 18    | **8.30978**  | 6.814 |

**Table 1:** <span style="color:gray">LiDAR TTC results</span>


<table>
<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0004.png" alt="result_3DObj_0004" style="width:400px;"/>
result_3DObj_0004
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0005.png" alt="result_3DObj_0005" style="width:400px;"/>
result_3DObj_0005
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0006.png" alt="result_3DObj_0006" style="width:400px;"/>
result_3DObj_0006
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0007.png" alt="result_3DObj_0007" style="width:400px;"/>
result_3DObj_0007
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0007.png" alt="result_3DObj_0007" style="width:400px;"/>
result_3DObj_0007
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0008.png" alt="result_3DObj_0008" style="width:400px;"/>
result_3DObj_0008
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0011.png" alt="result_3DObj_0011" style="width:400px;"/>
result_3DObj_0011
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0012.png" alt="result_3DObj_0012" style="width:400px;"/>
result_3DObj_0012
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0016.png" alt="result_3DObj_0016" style="width:400px;"/>
result_3DObj_0016
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/result_3DObj_0017.png" alt="result_3DObj_0017" style="width:400px;"/>
result_3DObj_0017
</th>
</tr>
</table>

**Figure 1:** <span style="color:gray">Detection results LiDAR, top-view perspective, showing measurements from preceding vehicle.</span>


The calculation of ttc based on LiDAR frames highly depends on accuracy of detection as well as sensor frame-rate but also on the velocity of ego-vehicle. From Table 1, containing ttc results (in [s]) from LiDAR measurements, it can be seen that the resulting values from ttc are relatively unstable, where the differences from successive frames can be very large.

The selection of images (top-view perspective) show that there are almost no outliers visible. It can be seen that the detected measurement values clearly belong to the rear-side of the preceding vehicle. In some cases there are a few points with a very small offset, a couple of centimeters, to the rest of measurements. Therefore, it turned out that the LiDAR approach is prone to error where small changes in the detection causes large differences in the output.

For computing the TTC, the equation bellow shows that the result indirectly depends on sensor frame-rate as well as the velocity of ego vehicle.

<div style='text-align:center'>
TTC = d<sub>1</sub> * d<sub>T</sub> / (d<sub>2</sub> - d<sub>1</sub>)
</div>

In the following, it is assumed that the sensor frame-rate is static, e.g. 10Hz or 30Hz. From a mathematical perspective, the ttc will always be very small or very large depending on velocity of ego-vehicle, which is directly influencing the difference between d<sub>2</sub> and d<sub>1</sub>, d<sub>s</sub> = |d<sub>2</sub> - d<sub>1</sub>|. In case of distance measurements which are very close to each other, see ```xmin``` from Table 1, the result of d<sub>s</sub> will turn out to be very small which mainly causes the ttc to be significantly larger when compared to values from previous frames. Here it can be seen that the LiDAR approach of calculating a TTC value might not be the optimal solution, at least for very slow velocities. E.g. by travelling on a highway situation, d<sub>s</sub> will be much larger within successive frames and thus the resulting TTC might be more reliable.


## FP.6 Performance Evaluation 2:


The calculation of the TTC value based on camera frames highly depends on quality of keypoint detection. The table given bellow mainly shows the ttc values from the introduced approaches for 2d feature detectors and descriptors. The table contains several cells marked with ```nan```, this is caused by zero-division or a loss of keypoint matches where no matches could be identified. Harris keypoint detector has been seen to heavily suffer from loss of kepoint matches where the thresholds could further be adjusted to improve detection results. In the following, several graphs are represented to visualize the detector/ descriptor results to easier identify 'good' combinations.

Computation of ttc by mean showed much more reliable results if compared to median, even though the median is more robust against outliers. Several combinations of detector/ descriptor showd good results where the variance of ttc values from successive frames is much smaller. AKAZE/ BRIEF, BRISK/ ORB and SIFT/ BRIEF seem to be the best combinations of algorithms where a continuous decrease of the ttc value could be investigated.

1. AKAZE/ BRIEF
2. BRISK/ ORB
3. SIFT/ BRIEF

<div style='text-align:center'>
TTC = -d<sub>T</sub> / (1.0 - distRatio)
</div>

Additionally, to demonstrate numerical instability of TTC computation, in table 2 ```distRatio``` has been plotted. Accordingly to very larger and very small values from TTC it can easily be seen that values of ```distRatio``` which are very close to ```1.0``` causing the above equation to end up in results which are way off the actual timing of the correct TTC value. The ```distRatio``` of the euclidean distance between matched keypoint pairs from previous and current frame often refers to instability of the approach where the final result does not represent the actual TTC.


<table>
<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/shitomasi_median.png" alt="shitomasi_median" style="width:400px;"/>
shitomasi_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/shitomasi_mean.png" alt="shitomasi_mean" style="width:400px;"/>
shitomasi_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/harris_median.png" alt="harris_median" style="width:400px;"/>
harris_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/harris_mean.png" alt="harris_mean" style="width:400px;"/>
harris_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/fast_median.png" alt="fast_median" style="width:400px;"/>
fast_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/fast_mean.png" alt="fast_mean" style="width:400px;"/>
fast_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/brisk_median.png" alt="brisk_median" style="width:400px;"/>
brisk_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/brisk_mean.png" alt="brisk_mean" style="width:400px;"/>
brisk_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/orb_median.png" alt="orb_median" style="width:400px;"/>
orb_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/orb_mean.png" alt="orb_mean" style="width:400px;"/>
orb_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/akaze_median.png" alt="akaze_median" style="width:400px;"/>
akaze_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/akaze_mean.png" alt="akaze_mean" style="width:400px;"/>
akaze_mean
</th>
</tr>

<tr>
<th style='text-align:center;vertical-align:middle'>
<img src="build/sift_median.png" alt="sift_median" style="width:400px;"/>
sift_median
</th>
<th style='text-align:center;vertical-align:middle'>
<img src="build/sift_mean.png" alt="sift_mean" style="width:400px;"/>
sift_mean
</th>
</tr>
</table>

|           |       | ttcCameraMedian |          |          |          |          |          | ttcCameraMean |          |          |          |         |          | distRatioMedian |          |          |          |          |          | distRatioMean |          |          |          |         |          |
|-----------|-------|-----------------|----------|----------|----------|----------|----------|---------------|----------|----------|----------|---------|----------|-----------------|----------|----------|----------|----------|----------|---------------|----------|----------|----------|---------|----------|
| detector  | frame | BRISK           | BRIEF    | ORB      | FREAK    | AKAZE    | SIFT     | BRISK         | BRIEF    | ORB      | FREAK    | AKAZE   | SIFT     | BRISK           | BRIEF    | ORB      | FREAK    | AKAZE    | SIFT     | BRISK         | BRIEF    | ORB      | FREAK    | AKAZE   | SIFT     |
| SHITOMASI | 1     | 22.7005         | 12.5213  | 14.7363  | 12.3397  | -        | 12.1037  | 16.2372       | 16.0841  | 15.5504  | 14.52    | -       | 16.2169  | 1.00441         | 1.00799  | 1.00679  | 1.0081   | -        | 1.00826  | 1.00616       | 1.00622  | 1.00643  | 1.00689  | -       | 1.00617  |
| SHITOMASI | 2     | 11.7376         | 9.39134  | 11.0498  | 14.7722  | -        | 25.9204  | 13.6127       | 15.2758  | 13.1217  | 15.0714  | -       | 15.1536  | 1.00852         | 1.01065  | 1.00905  | 1.00677  | -        | 1.00386  | 1.00735       | 1.00655  | 1.00762  | 1.00664  | -       | 1.0066   |
| SHITOMASI | 3     | **32.8612**         | 10.2864  | 25.3559  | nan      | -        | 7.86762  | 15.3008       | 14.5519  | 16.432   | 10.6569  | -       | 22.3313  | 1.00304         | 1.00972  | 1.00394  | 1        | -        | 1.01271  | 1.00654       | 1.00687  | 1.00609  | 1.00938  | -       | 1.00448  |
| SHITOMASI | 4     | nan             | 14.3359  | **408.25**   | 26.0116  | -        | 4.01835  | 11.2349       | 20.8293  | 12.0025  | 11.3188  | -       | 14.1686  | 1               | 1.00698  | 1.00024  | 1.00384  | -        | 1.02489  | 1.0089        | 1.0048   | 1.00833  | 1.00883  | -       | 1.00706  |
| SHITOMASI | 5     | 13.0747         | 13.4119  | 14.8128  | 8.20244  | -        | 13.0618  | 14.0393       | 12.2586  | 15.0784  | 12.9796  | -       | 14.7934  | 1.00765         | 1.00746  | 1.00675  | 1.01219  | -        | 1.00766  | 1.00712       | 1.00816  | 1.00663  | 1.0077   | -       | 1.00676  |
| SHITOMASI | 6     | nan             | **32.5892**  | 9.03706  | 20.3889  | -        | **32.5892**  | 20.3371       | 14.1762  | 14.0948  | 14.1238  | -       | 14.1762  | 1               | 1.00307  | 1.01107  | 1.0049   | -        | 1.00307  | 1.00492       | 1.00705  | 1.00709  | 1.00708  | -       | 1.00705  |
| SHITOMASI | 7     | 9.85209         | 13.9034  | 7.82904  | nan      | -        | 19.2268  | 10.0323       | 12.125   | 15.6536  | 10.6859  | -       | 13.1264  | 1.01015         | 1.00719  | 1.01277  | 1        | -        | 1.0052   | 1.00997       | 1.00825  | 1.00639  | 1.00936  | -       | 1.00762  |
| SHITOMASI | 8     | 11.2363         | 7.41759  | 7.41759  | 10.4433  | -        | 7.41759  | 13.9239       | 14.0003  | 14.0003  | 11.5561  | -       | 14.0003  | 1.0089          | 1.01348  | 1.01348  | 1.00958  | -        | 1.01348  | 1.00718       | 1.00714  | 1.00714  | 1.00865  | -       | 1.00714  |
| SHITOMASI | 9     | nan             | 7.50574  | 16.7053  | 14.3007  | -        | 5.27312  | 10.7628       | 10.4376  | 11.1198  | 10.4012  | -       | 11.6545  | 1               | 1.01332  | 1.00599  | 1.00699  | -        | 1.01896  | 1.00929       | 1.00958  | 1.00899  | 1.00961  | -       | 1.00858  |
| SHITOMASI | 10    | 13.0553         | nan      | nan      | 10.6403  | -        | nan      | 17.6321       | 16.1942  | 15.6161  | 15.0943  | -       | 15.6161  | 1.00766         | 1        | 1        | 1.0094   | -        | 1        | 1.00567       | 1.00618  | 1.0064   | 1.00663  | -       | 1.0064   |
| SHITOMASI | 11    | 9.26275         | 16.2713  | 14.462   | 11.5524  | -        | 11.6008  | 10.268        | 10.8455  | 10.9466  | 10.6633  | -       | 11.182   | 1.0108          | 1.00615  | 1.00691  | 1.00866  | -        | 1.00862  | 1.00974       | 1.00922  | 1.00914  | 1.00938  | -       | 1.00894  |
| SHITOMASI | 12    | 12.0902         | 10.2663  | 7.17617  | **37.6686**  | -        | 7.17617  | 9.27798       | 10.1342  | 10.4222  | 11.0974  | -       | 10.4222  | 1.00827         | 1.00974  | 1.01393  | 1.00265  | -        | 1.01393  | 1.01078       | 1.00987  | 1.00959  | 1.00901  | -       | 1.00959  |
| SHITOMASI | 13    | 11.4066         | 7.47154  | **33.6545**  | 10.8491  | -        | 11.6219  | 11.458        | 10.7325  | 11.4794  | 14.1091  | -       | 11.5143  | 1.00877         | 1.01338  | 1.00297  | 1.00922  | -        | 1.0086   | 1.00873       | 1.00932  | 1.00871  | 1.00709  | -       | 1.00868  |
| SHITOMASI | 14    | 19.3887         | 10.047   | 7.64007  | 6.18732  | -        | 10.047   | 8.58698       | 9.86649  | 10.4631  | 8.78967  | -       | 9.82881  | 1.00516         | 1.00995  | 1.01309  | 1.01616  | -        | 1.00995  | 1.01165       | 1.01014  | 1.00956  | 1.01138  | -       | 1.01017  |
| SHITOMASI | 15    | nan             | 12.3397  | 20.5432  | 10.597   | -        | 20.5432  | 8.48429       | 12.6768  | 11.1184  | 10.157   | -       | 11.1184  | 1               | 1.0081   | 1.00487  | 1.00944  | -        | 1.00487  | 1.01179       | 1.00789  | 1.00899  | 1.00985  | -       | 1.00899  |
| SHITOMASI | 16    | 14.462          | 14.8464  | 17.9737  | 12.9773  | -        | 14.8464  | 7.15622       | 12.1345  | 11.1155  | 7.83832  | -       | 12.1345  | 1.00691         | 1.00674  | 1.00556  | 1.00771  | -        | 1.00674  | 1.01397       | 1.00824  | 1.009    | 1.01276  | -       | 1.00824  |
| SHITOMASI | 17    | nan             | 8.08158  | nan      | 5.1673   | -        | 13.8862  | 13.1502       | 10.4154  | 12.2958  | 10.0457  | -       | 9.45393  | 1               | 1.01237  | 1        | 1.01935  | -        | 1.0072   | 1.0076        | 1.0096   | 1.00813  | 1.00995  | -       | 1.01058  |
| SHITOMASI | 18    | 16.572          | 10.8764  | 9.87553  | 11.4316  | -        | 8.19347  | 9.49175       | 9.24964  | 8.58353  | 8.88271  | -       | 10.3655  | 1.00603         | 1.00919  | 1.01013  | 1.00875  | -        | 1.0122   | 1.01054       | 1.01081  | 1.01165  | 1.01126  | -       | 1.00965  |
| HARRIS    | 1     | nan             | nan      | nan      | nan      | -        | nan      | nan           | nan      | nan      | nan      | -       | nan      | nan             | nan      | nan      | nan      | -        | nan      | nan           | nan      | nan      | nan      | -       | nan      |
| HARRIS    | 2     | nan             | 10.586   | 10.586   | nan      | -        | 11.0081  | nan           | 10.586   | 10.586   | nan      | -       | 19.9399  | nan             | 1.00945  | 1.00945  | nan      | -        | 1.00908  | nan           | 1.00945  | 1.00945  | nan      | -       | 1.00502  |
| HARRIS    | 3     | nan             | nan      | nan      | nan      | -        | nan      | nan           | nan      | nan      | nan      | -       | -23.9304 | nan             | nan      | nan      | nan      | -        | 1        | nan           | nan      | nan      | nan      | -       | 0.995821 |
| HARRIS    | 4     | 10.9782         | 10.9782  | 10.9782  | 10.2785  | -        | 10.9782  | 10.9745       | 10.9745  | 10.9745  | 10.7922  | -       | 10.9745  | 1.00911         | 1.00911  | 1.00911  | 1.00973  | -        | 1.00911  | 1.00911       | 1.00911  | 1.00911  | 1.00927  | -       | 1.00911  |
| HARRIS    | 5     | **44.9166**         | 13.4549  | 8.0385   | 13.6432  | -        | **44.9166**  | **89.0261**       | 17.4577  | **30.6534**  | **31.9822**  | -       | 16.2484  | 1.00223         | 1.00743  | 1.01244  | 1.00733  | -        | 1.00223  | 1.00112       | 1.00573  | 1.00326  | 1.00313  | -       | 1.00615  |
| HARRIS    | 6     | 11.0746         | 11.0746  | 11.0746  | nan      | -        | -**31.8409** | 11.4102       | 13.2092  | 13.2092  | nan      | -       | **65.7698**  | 1.00903         | 1.00903  | 1.00903  | nan      | -        | 0.996859 | 1.00876       | 1.00757  | 1.00757  | nan      | -       | 1.00152  |
| HARRIS    | 7     | 21.7427         | **54.3866**  | 17.7488  | 11.2837  | -        | nan      | 24.0312       | 20.747   | 18.7137  | 14.6119  | -       | -**357.728** | 1.0046          | 1.00184  | 1.00563  | 1.00886  | -        | 1        | 1.00416       | 1.00482  | 1.00534  | 1.00684  | -       | 0.99972  |
| HARRIS    | 8     | 10.0433         | 13.5704  | 13.5704  | 10.0433  | -        | 10.0433  | 10.6947       | 19.199   | 19.199   | 9.2516   | -       | 10.6947  | 1.00996         | 1.00737  | 1.00737  | 1.00996  | -        | 1.00996  | 1.00935       | 1.00521  | 1.00521  | 1.01081  | -       | 1.00935  |
| HARRIS    | 9     | 15.4179         | **30.8359**  | 15.4179  | nan      | -        | 3.30058  | 21.0852       | 18.8091  | 21.0852  | nan      | -       | 5.61573  | 1.00649         | 1.00324  | 1.00649  | 1        | -        | 1.0303   | 1.00474       | 1.00532  | 1.00474  | 1        | -       | 1.01781  |
| HARRIS    | 10    | nan             | 10.2931  | nan      | nan      | -        | nan      | nan           | 19.7629  | nan      | nan      | -       | nan      | 1               | 1.00972  | nan      | nan      | -        | nan      | 1             | 1.00506  | nan      | nan      | -       | nan      |
| HARRIS    | 11    | nan             | 23.627   | 23.627   | nan      | -        | 23.627   | nan           | 17.1612  | 17.1612  | nan      | -       | 17.1612  | 1               | 1.00423  | 1.00423  | 1        | -        | 1.00423  | 1             | 1.00583  | 1.00583  | 1        | -       | 1.00583  |
| HARRIS    | 12    | nan             | nan      | nan      | nan      | -        | nan      | 19.4386       | nan      | nan      | 19.4386  | -       | nan      | 1               | nan      | nan      | 1        | -        | nan      | 1.00514       | nan      | nan      | 1.00514  | -       | nan      |
| HARRIS    | 13    | nan             | nan      | nan      | nan      | -        | nan      | nan           | **33.9509**  | **33.9509**  | nan      | -       | **33.9509**  | 1               | 1        | 1        | nan      | -        | 1        | 1             | 1.00295  | 1.00295  | nan      | -       | 1.00295  |
| HARRIS    | 14    | 11.5135         | 11.4169  | 6.2196   | 11.5135  | -        | 11.4169  | 10.4167       | 9.57357  | 7.45426  | 10.4167  | -       | 9.57357  | 1.00869         | 1.00876  | 1.01608  | 1.00869  | -        | 1.00876  | 1.0096        | 1.01045  | 1.01342  | 1.0096   | -       | 1.01045  |
| HARRIS    | 15    | -14.7808        | -14.7808 | -14.7808 | nan      | -        | -25.2781 | -**167.626**      | -**167.626** | -**167.626** | nan      | -       | -**82.1515** | 0.993234        | 0.993234 | 0.993234 | 1        | -        | 0.996044 | 0.999403      | 0.999403 | 0.999403 | 1        | -       | 0.998783 |
| HARRIS    | 16    | 12.8883         | 10.6503  | 12.8883  | 7.70697  | -        | 10.6503  | 8.15044       | 8.8341   | 8.15044  | 7.99979  | -       | 8.8341   | 1.00776         | 1.00939  | 1.00776  | 1.01298  | -        | 1.00939  | 1.01227       | 1.01132  | 1.01227  | 1.0125   | -       | 1.01132  |
| HARRIS    | 17    | 12.7384         | 12.6612  | 12.7384  | nan      | -        | 12.7384  | 12.1414       | 12.0945  | 12.1414  | nan      | -       | 12.1414  | 1.00785         | 1.0079   | 1.00785  | nan      | -        | 1.00785  | 1.00824       | 1.00827  | 1.00824  | nan      | -       | 1.00824  |
| HARRIS    | 18    | nan             | nan      | nan      | nan      | -        | nan      | nan           | nan      | nan      | nan      | -       | nan      | 1               | 1        | 1        | nan      | -        | 1        | 1             | 1        | 1        | nan      | -       | 1        |
| FAST      | 1     | 15.7723         | 6.57131  | 6.32977  | 12.2073  | -        | **43.2936**  | 12.669        | 13.7986  | 12.3562  | 11.9494  | -       | 14.095   | 1.00634         | 1.01522  | 1.0158   | 1.00819  | -        | 1.00231  | 1.00789       | 1.00725  | 1.00809  | 1.00837  | -       | 1.00709  |
| FAST      | 2     | 11.0371         | 19.7997  | 13.2125  | 12.7421  | -        | 4.7514   | 13.0929       | 11.039   | 12.5036  | 11.5728  | -       | 10.9911  | 1.00906         | 1.00505  | 1.00757  | 1.00785  | -        | 1.02105  | 1.00764       | 1.00906  | 1.008    | 1.00864  | -       | 1.0091   |
| FAST      | 3     | 11.3245         | 10.1664  | 9.0112   | 5.95042  | -        | 11.8109  | 14.8161       | 23.0905  | 18.9049  | 13.6396  | -       | 20.5217  | 1.00883         | 1.00984  | 1.0111   | 1.01681  | -        | 1.00847  | 1.00675       | 1.00433  | 1.00529  | 1.00733  | -       | 1.00487  |
| FAST      | 4     | 7.24483         | 14.4147  | 24.0734  | 10.2419  | -        | 11.8708  | 12.6739       | 14.7972  | 12.7867  | 16.1025  | -       | 18.0311  | 1.0138          | 1.00694  | 1.00415  | 1.00976  | -        | 1.00842  | 1.00789       | 1.00676  | 1.00782  | 1.00621  | -       | 1.00555  |
| FAST      | 5     | 8.48428         | 9.81573  | 6.93672  | 11.3991  | -        | -14.8986 | 15.2166       | 18.3069  | 13.9632  | 13.3704  | -       | 17.7794  | 1.01179         | 1.01019  | 1.01442  | 1.00877  | -        | 0.993288 | 1.00657       | 1.00546  | 1.00716  | 1.00748  | -       | 1.00562  |
| FAST      | 6     | 6.57696         | 8.92595  | 6.93762  | **38.4284**  | -        | 5.75807  | 10.3758       | 13.594   | 16.2399  | 10.2629  | -       | 12.8624  | 1.0152          | 1.0112   | 1.01441  | 1.0026   | -        | 1.01737  | 1.00964       | 1.00736  | 1.00616  | 1.00974  | -       | 1.00777  |
| FAST      | 7     | 12.5679         | 12.0801  | **79.4451**  | 13.9689  | -        | 7.02973  | 10.7401       | 16.6286  | 16.0007  | 14.262   | -       | 12.1044  | 1.00796         | 1.00828  | 1.00126  | 1.00716  | -        | 1.01423  | 1.00931       | 1.00601  | 1.00625  | 1.00701  | -       | 1.00826  |
| FAST      | 8     | 11.3861         | 6.09399  | **50.8905**  | 11.5135  | -        | 17.9687  | 12.4841       | 15.0963  | 14.8572  | 12.9685  | -       | 14.468   | 1.00878         | 1.01641  | 1.00197  | 1.00869  | -        | 1.00557  | 1.00801       | 1.00662  | 1.00673  | 1.00771  | -       | 1.00691  |
| FAST      | 9     | 9.32507         | 7.88751  | 8.93212  | 10.1204  | -        | 6.05766  | 10.2509       | 10.0667  | 13.7212  | 14.3965  | -       | 10.4001  | 1.01072         | 1.01268  | 1.0112   | 1.00988  | -        | 1.01651  | 1.00976       | 1.00993  | 1.00729  | 1.00695  | -       | 1.00962  |
| FAST      | 10    | 7.65812         | 8.669    | **58.6136**  | **39.4201**  | -        | 6.83371  | 10.5672       | 14.4221  | 14.033   | 13.5384  | -       | 15.5905  | 1.01306         | 1.01154  | 1.00171  | 1.00254  | -        | 1.01463  | 1.00946       | 1.00693  | 1.00713  | 1.00739  | -       | 1.00641  |
| FAST      | 11    | 5.81378         | 9.7083   | 11.5002  | 6.26584  | -        | 7.31529  | 10.6873       | 12.3871  | 12.7214  | 11.3001  | -       | 12.1393  | 1.0172          | 1.0103   | 1.0087   | 1.01596  | -        | 1.01367  | 1.00936       | 1.00807  | 1.00786  | 1.00885  | -       | 1.00824  |
| FAST      | 12    | 6.42171         | 10.0785  | 16.1817  | 12.8714  | -        | 13.1028  | 13.799        | 11.0657  | 10.6648  | 13.9058  | -       | 10.7175  | 1.01557         | 1.00992  | 1.00618  | 1.00777  | -        | 1.00763  | 1.00725       | 1.00904  | 1.00938  | 1.00719  | -       | 1.00933  |
| FAST      | 13    | 5.96257         | 10.9361  | 7.065    | **35.7876**  | -        | 6.32573  | 9.7219        | 11.0525  | 11.5183  | 10.2612  | -       | 9.73776  | 1.01677         | 1.00914  | 1.01415  | 1.00279  | -        | 1.01581  | 1.01029       | 1.00905  | 1.00868  | 1.00975  | -       | 1.01027  |
| FAST      | 14    | 9.62401         | 11.9134  | 12.551   | 5.60543  | -        | 5.64341  | 12.3132       | 10.2732  | 10.1032  | 10.0216  | -       | 10.1666  | 1.01039         | 1.00839  | 1.00797  | 1.01784  | -        | 1.01772  | 1.00812       | 1.00973  | 1.0099   | 1.00998  | -       | 1.00984  |
| FAST      | 15    | 11.1409         | 13.1282  | 12.8108  | 7.60737  | -        | 5.80835  | 9.31024       | 13.3971  | 10.6848  | 10.0344  | -       | 10.6518  | 1.00898         | 1.00762  | 1.00781  | 1.01315  | -        | 1.01722  | 1.01074       | 1.00746  | 1.00936  | 1.00997  | -       | 1.00939  |
| FAST      | 16    | 11.5035         | 14.1529  | 11.057   | 5.87676  | -        | 6.41379  | 10.436        | 10.9867  | 11.102   | 11.1425  | -       | 10.8153  | 1.00869         | 1.00707  | 1.00904  | 1.01702  | -        | 1.01559  | 1.00958       | 1.0091   | 1.00901  | 1.00897  | -       | 1.00925  |
| FAST      | 17    | nan             | 10.5122  | 7.43978  | 6.26528  | -        | 11.1173  | 11.021        | 10.1467  | 10.1923  | 10.7898  | -       | 9.97966  | 1               | 1.00951  | 1.01344  | 1.01596  | -        | 1.009    | 1.00907       | 1.00986  | 1.00981  | 1.00927  | -       | 1.01002  |
| FAST      | 18    | 10.9069         | 11.0562  | 8.20534  | 11.9038  | -        | 9.00937  | 13.3201       | 12.5807  | 14.1535  | 12.0698  | -       | 10.9373  | 1.00917         | 1.00904  | 1.01219  | 1.0084   | -        | 1.0111   | 1.00751       | 1.00795  | 1.00707  | 1.00829  | -       | 1.00914  |
| BRISK     | 1     | 11.4215         | 14.4175  | 15.5606  | 16.4099  | -        | 7.02576  | 15.0688       | 16.1176  | 15.7752  | 13.233   | -       | 15.5391  | 1.00876         | 1.00694  | 1.00643  | 1.00609  | -        | 1.01423  | 1.00664       | 1.0062   | 1.00634  | 1.00756  | -       | 1.00644  |
| BRISK     | 2     | 9.91774         | 23.6219  | 20.9723  | 19.6449  | -        | 14.1625  | 23.5775       | 19.5071  | 22.5204  | 19.6353  | -       | 14.8181  | 1.01008         | 1.00423  | 1.00477  | 1.00509  | -        | 1.00706  | 1.00424       | 1.00513  | 1.00444  | 1.00509  | -       | 1.00675  |
| BRISK     | 3     | **157.265**         | **164.327**  | **54.0693**  | 7.72666  | -        | -**33.4431** | 15.6367       | 13.6654  | 17.7883  | 14.8643  | -       | 21.792   | 1.00064         | 1.00061  | 1.00185  | 1.01294  | -        | 0.99701  | 1.0064        | 1.00732  | 1.00562  | 1.00673  | -       | 1.00459  |
| BRISK     | 4     | 10.4066         | 27.0255  | 9.06663  | 10.7673  | -        | 12.1546  | 17.5197       | 26.468   | 17.8306  | 13.733   | -       | 10.2084  | 1.00961         | 1.0037   | 1.01103  | 1.00929  | -        | 1.00823  | 1.00571       | 1.00378  | 1.00561  | 1.00728  | -       | 1.0098   |
| BRISK     | 5     | 9.38215         | -20.683  | 18.5375  | **95.4709**  | -        | 10.2816  | 28.8281       | 16.1838  | 19.1267  | **35.9089**  | -       | **51.3851**  | 1.01066         | 0.995165 | 1.00539  | 1.00105  | -        | 1.00973  | 1.00347       | 1.00618  | 1.00523  | 1.00278  | -       | 1.00195  |
| BRISK     | 6     | 14.293          | 17.5693  | 12.3421  | 6.54919  | -        | 7.33748  | 17.2233       | 18.6886  | 25.9502  | 17.5102  | -       | 12.1987  | 1.007           | 1.00569  | 1.0081   | 1.01527  | -        | 1.01363  | 1.00581       | 1.00535  | 1.00385  | 1.00571  | -       | 1.0082   |
| BRISK     | 7     | **35.9183**         | 7.62729  | 12.6997  | **62.5623**  | -        | 13.1894  | 20.4903       | 17.3312  | 18.2438  | 19.3661  | -       | 13.4501  | 1.00278         | 1.01311  | 1.00787  | 1.0016   | -        | 1.00758  | 1.00488       | 1.00577  | 1.00548  | 1.00516  | -       | 1.00743  |
| BRISK     | 8     | 12.4782         | 8.60677  | 12.1902  | 12.0382  | -        | 10.1337  | 19.2891       | 23.1146  | 21.2685  | 20.4166  | -       | 16.5304  | 1.00801         | 1.01162  | 1.0082   | 1.00831  | -        | 1.00987  | 1.00518       | 1.00433  | 1.0047   | 1.0049   | -       | 1.00605  |
| BRISK     | 9     | -**445.057**        | 16.7821  | **361.291**  | 7.38008  | -        | 21.7465  | 15.4656       | 17.8055  | 15.1761  | 14.8797  | -       | 19.3977  | 0.999775        | 1.00596  | 1.00028  | 1.01355  | -        | 1.0046   | 1.00647       | 1.00562  | 1.00659  | 1.00672  | -       | 1.00516  |
| BRISK     | 10    | 7.14567         | 7.3717   | 8.95682  | 15.7096  | -        | 22.5818  | 15.3487       | 15.4091  | 14.8336  | 14.875   | -       | 14.9799  | 1.01399         | 1.01357  | 1.01116  | 1.00637  | -        | 1.00443  | 1.00652       | 1.00649  | 1.00674  | 1.00672  | -       | 1.00668  |
| BRISK     | 11    | 5.44513         | -**333.42**  | 6.8047   | 10.8243  | -        | -12.7185 | 14.8109       | 14.2362  | 16.8371  | 13.2586  | -       | 19.5093  | 1.01837         | 0.9997   | 1.0147   | 1.00924  | -        | 0.992137 | 1.00675       | 1.00702  | 1.00594  | 1.00754  | -       | 1.00513  |
| BRISK     | 12    | 6.62139         | **31.4973**  | 11.6513  | 8.1939   | -        | 9.18319  | 12.5187       | 16.396   | 13.9747  | 11.0203  | -       | 9.69965  | 1.0151          | 1.00317  | 1.00858  | 1.0122   | -        | 1.01089  | 1.00799       | 1.0061   | 1.00716  | 1.00907  | -       | 1.01031  |
| BRISK     | 13    | 9.41734         | 10.4799  | 6.90862  | 6.05624  | -        | 14.4738  | 13.3894       | 13.6021  | 15.2055  | 13.1665  | -       | 18.0225  | 1.01062         | 1.00954  | 1.01447  | 1.01651  | -        | 1.00691  | 1.00747       | 1.00735  | 1.00658  | 1.0076   | -       | 1.00555  |
| BRISK     | 14    | 10.3012         | 10.0376  | 10.1452  | 9.04116  | -        | 4.06708  | 14.0436       | 14.2515  | 15.0584  | 15.3402  | -       | 11.9902  | 1.00971         | 1.00996  | 1.00986  | 1.01106  | -        | 1.02459  | 1.00712       | 1.00702  | 1.00664  | 1.00652  | -       | 1.00834  |
| BRISK     | 15    | 7.80643         | 12.558   | 14.4725  | 12.708   | -        | -6.77499 | 13.4556       | 14.1303  | 14.7066  | 15.297   | -       | 14.8098  | 1.01281         | 1.00796  | 1.00691  | 1.00787  | -        | 0.98524  | 1.00743       | 1.00708  | 1.0068   | 1.00654  | -       | 1.00675  |
| BRISK     | 16    | 17.347          | 17.3451  | 9.96907  | **39.9837**  | -        | -9.55659 | 12.6191       | 13.4782  | 12.034   | 9.89103  | -       | 13.229   | 1.00576         | 1.00577  | 1.01003  | 1.0025   | -        | 0.989536 | 1.00792       | 1.00742  | 1.00831  | 1.01011  | -       | 1.00756  |
| BRISK     | 17    | 10.6251         | 6.0577   | 24.2694  | 17.3618  | -        | 17.5934  | 9.13658       | 11.5513  | 10.3993  | 9.41434  | -       | 10.3636  | 1.00941         | 1.01651  | 1.00412  | 1.00576  | -        | 1.00568  | 1.01094       | 1.00866  | 1.00962  | 1.01062  | -       | 1.00965  |
| BRISK     | 18    | 13.4837         | 9.33847  | 20.9327  | 9.32423  | -        | 2.55086  | 12.5193       | 14.352   | 14.3512  | 13.249   | -       | 8.71774  | 1.00742         | 1.01071  | 1.00478  | 1.01072  | -        | 1.0392   | 1.00799       | 1.00697  | 1.00697  | 1.00755  | -       | 1.01147  |
| ORB       | 1     | 5.81342         | 17.3704  | **91.9878**  | 10.943   | -        | 12.5735  | 15.2351       | 18.0216  | 18.8386  | 14.3471  | -       | 18.3192  | 1.0172          | 1.00576  | 1.00109  | 1.00914  | -        | 1.00795  | 1.00656       | 1.00555  | 1.00531  | 1.00697  | -       | 1.00546  |
| ORB       | 2     | 10.6151         | nan      | 15.7315  | 10.126   | -        | 18.1605  | 18.176        | -**390.095** | 22.6769  | 12.7475  | -       | 13.3869  | 1.00942         | 1        | 1.00636  | 1.00988  | -        | 1.00551  | 1.0055        | 0.999744 | 1.00441  | 1.00784  | -       | 1.00747  |
| ORB       | 3     | 9.37508         | 9.43334  | 5.33939  | -12.327  | -        | -**31.3421** | 10.8839       | **75.6375**  | 11.13    | 15.345   | -       | 22.5565  | 1.01067         | 1.0106   | 1.01873  | 0.991888 | -        | 0.996809 | 1.00919       | 1.00132  | 1.00898  | 1.00652  | -       | 1.00443  |
| ORB       | 4     | -12.0972        | 8.66707  | 12.7974  | 22.7365  | -        | -10.6051 | **80.0415**       | 12.3249  | **32.8246**  | 11.5938  | -       | **83.6445**  | 0.991734        | 1.01154  | 1.00781  | 1.0044   | -        | 0.990571 | 1.00125       | 1.00811  | 1.00305  | 1.00863  | -       | 1.0012   |
| ORB       | 5     | 14.3312         | 23.7891  | **35.5369**  | 12.6967  | -        | **1026.03**  | **52.4358**       | 24.8965  | **51.6484**  | -**1134.96** | -       | **31.0934**  | 1.00698         | 1.0042   | 1.00281  | 1.00788  | -        | 1.0001   | 1.00191       | 1.00402  | 1.00194  | 0.999912 | -       | 1.00322  |
| ORB       | 6     | nan             | -**48.0008** | -16.0607 | 6.58843  | -        | 10.3467  | 14.5597       | **124.682**  | 24.6816  | 15.4253  | -       | 22.5841  | 1               | 0.997917 | 0.993774 | 1.01518  | -        | 1.00966  | 1.00687       | 1.0008   | 1.00405  | 1.00648  | -       | 1.00443  |
| ORB       | 7     | 10.1234         | 6.68332  | **186.958**  | nan      | -        | nan      | 15.5039       | **32.7772**  | **34.6812**  | -**231.137** | -       | 18.6165  | 1.00988         | 1.01496  | 1.00053  | 1        | -        | 1        | 1.00645       | 1.00305  | 1.00288  | 0.999567 | -       | 1.00537  |
| ORB       | 8     | 7.45058         | -12.804  | nan      | 4.02766  | -        | nan      | 12.8768       | **30.7627**  | 15.0575  | 9.55137  | -       | 14.3054  | 1.01342         | 0.99219  | 1        | 1.02483  | -        | 1        | 1.00777       | 1.00325  | 1.00664  | 1.01047  | -       | 1.00699  |
| ORB       | 9     | **60.685**          | nan      | 23.013   | 20.0733  | -        | 9.80616  | 22.241        | **152.411**  | **36.1654**  | 16.6074  | -       | 19.1479  | 1.00165         | 1        | 1.00435  | 1.00498  | -        | 1.0102   | 1.0045        | 1.00066  | 1.00277  | 1.00602  | -       | 1.00522  |
| ORB       | 10    | nan             | -10.3117 | 8.58768  | nan      | -        | nan      | **36.3333**       | 16.0411  | **62.3751**  | -**57.3433** | -       | 11.8839  | 1               | 0.990302 | 1.01164  | 1        | -        | 1        | 1.00275       | 1.00623  | 1.0016   | 0.998256 | -       | 1.00841  |
| ORB       | 11    | 5.90423         | 5.78843  | 8.25012  | 9.08103  | -        | 12.0275  | 7.90706       | 20.7835  | 8.25997  | 7.44254  | -       | 9.17999  | 1.01694         | 1.01728  | 1.01212  | 1.01101  | -        | 1.00831  | 1.01265       | 1.00481  | 1.01211  | 1.01344  | -       | 1.01089  |
| ORB       | 12    | -**231.646**        | 13.4844  | -**125.104** | -**184.048** | -        | 20.7639  | **91.1618**       | 21.6408  | **62.2448**  | **31.0825**  | -       | **33.1409**  | 0.999568        | 1.00742  | 0.999201 | 0.999457 | -        | 1.00482  | 1.0011        | 1.00462  | 1.00161  | 1.00322  | -       | 1.00302  |
| ORB       | 13    | 4.58632         | 9.84171  | 25.7124  | 4.98505  | -        | 6.16034  | 10.657        | 9.51636  | **30.5655**  | 5.6848   | -       | 5.6557   | 1.0218          | 1.01016  | 1.00389  | 1.02006  | -        | 1.01623  | 1.00938       | 1.01051  | 1.00327  | 1.01759  | -       | 1.01768  |
| ORB       | 14    | 9.6914          | 7.45235  | -8.08368 | -11.9818 | -        | **126.884**  | 18.7446       | 14.6873  | **42.7517**  | **49.6542**  | -       | 18.2553  | 1.01032         | 1.01342  | 0.987629 | 0.991654 | -        | 1.00079  | 1.00533       | 1.00681  | 1.00234  | 1.00201  | -       | 1.00548  |
| ORB       | 15    | 7.60034         | 6.65875  | nan      | 7.9638   | -        | 7.98886  | 15.0876       | 8.29977  | 21.167   | 9.45089  | -       | 18.6294  | 1.01316         | 1.01502  | 1        | 1.01256  | -        | 1.01252  | 1.00663       | 1.01205  | 1.00472  | 1.01058  | -       | 1.00537  |
| ORB       | 16    | nan             | **33.4086**  | 9.76932  | **43.8291**  | -        | 20.9262  | 21.856        | 11.0615  | 16.6223  | 7.54712  | -       | 7.9569   | 1               | 1.00299  | 1.01024  | 1.00228  | -        | 1.00478  | 1.00458       | 1.00904  | 1.00602  | 1.01325  | -       | 1.01257  |
| ORB       | 17    | 9.69581         | 12.155   | 14.078   | 24.8397  | -        | **667.588**  | 12.8569       | 13.1614  | 14.1859  | 13.1469  | -       | 13.8357  | 1.01031         | 1.00823  | 1.0071   | 1.00403  | -        | 1.00015  | 1.00778       | 1.0076   | 1.00705  | 1.00761  | -       | 1.00723  |
| ORB       | 18    | 9.93941         | 16.8149  | -185341  | 7.62862  | -        | 5.35572  | 20.9593       | 11.2536  | **34.5054**  | 7.684    | -       | 14.3539  | 1.01006         | 1.00595  | 0.999999 | 1.01311  | -        | 1.01867  | 1.00477       | 1.00889  | 1.0029   | 1.01301  | -       | 1.00697  |
| AKAZE     | 1     | 9.72762         | 12.7306  | 12.0133  | 11.8861  | 13.0126  | 11.7404  | 13.3398       | 16.7581  | 13.4329  | 12.3042  | 12.441  | 14.0346  | 1.01028         | 1.00786  | 1.00832  | 1.00841  | 1.00768  | 1.00852  | 1.0075        | 1.00597  | 1.00744  | 1.00813  | 1.00804 | 1.00713  |
| AKAZE     | 2     | 24.9411         | 10.7551  | 17.2602  | 13.4247  | 27.4683  | 26.6396  | 18.7292       | 19.2714  | 17.1675  | 14.7017  | 17.4887 | 17.8869  | 1.00401         | 1.0093   | 1.00579  | 1.00745  | 1.00364  | 1.00375  | 1.00534       | 1.00519  | 1.00582  | 1.0068   | 1.00572 | 1.00559  |
| AKAZE     | 3     | 11.2882         | 11.5301  | 20.7464  | 8.87144  | 13.3263  | 9.17251  | 15.6597       | 16.1938  | 16.5134  | 17.6306  | 12.9507 | 14.4201  | 1.00886         | 1.00867  | 1.00482  | 1.01127  | 1.0075   | 1.0109   | 1.00639       | 1.00618  | 1.00606  | 1.00567  | 1.00772 | 1.00693  |
| AKAZE     | 4     | 13.7251         | 13.0128  | 26.3508  | 9.63016  | -18.1836 | 16.8029  | 16.7944       | 16.2181  | 17.9255  | 13.5912  | 16.4466 | 15.6754  | 1.00729         | 1.00768  | 1.00379  | 1.01038  | 0.994501 | 1.00595  | 1.00595       | 1.00617  | 1.00558  | 1.00736  | 1.00608 | 1.00638  |
| AKAZE     | 5     | 20.3245         | 16.9259  | 13.4628  | 14.5998  | 17.5154  | 13.3097  | 16.9758       | 14.4723  | 17.7219  | 17.3361  | 19.8348 | 16.9952  | 1.00492         | 1.00591  | 1.00743  | 1.00685  | 1.00571  | 1.00751  | 1.00589       | 1.00691  | 1.00564  | 1.00577  | 1.00504 | 1.00588  |
| AKAZE     | 6     | 10.1614         | 14.4621  | 13.4111  | 17.7585  | -**149.35**  | 13.1701  | 21.4913       | 15.8596  | 18.5471  | 19.9018  | 14.5715 | 20.7041  | 1.00984         | 1.00691  | 1.00746  | 1.00563  | 0.99933  | 1.00759  | 1.00465       | 1.00631  | 1.00539  | 1.00502  | 1.00686 | 1.00483  |
| AKAZE     | 7     | 15.7351         | 16.3901  | 23.2023  | 13.1837  | 12.774   | 14.6946  | 19.4369       | 19.2398  | 18.297   | 20.0261  | 16.1145 | 14.5479  | 1.00636         | 1.0061   | 1.00431  | 1.00759  | 1.00783  | 1.00681  | 1.00514       | 1.0052   | 1.00547  | 1.00499  | 1.00621 | 1.00687  |
| AKAZE     | 8     | 27.7939         | 9.41035  | 23.6681  | 15.3462  | 20.7368  | 9.99793  | 17.5219       | 14.5823  | 17.5076  | 17.5088  | 15.4436 | 14.9597  | 1.0036          | 1.01063  | 1.00423  | 1.00652  | 1.00482  | 1.01     | 1.00571       | 1.00686  | 1.00571  | 1.00571  | 1.00648 | 1.00668  |
| AKAZE     | 9     | 12.8613         | 15.5189  | 28.446   | 10.9047  | 12.7325  | 14.4707  | 19.2965       | 16.91    | 18.2241  | 18.1195  | 14.5992 | 14.5576  | 1.00778         | 1.00644  | 1.00352  | 1.00917  | 1.00785  | 1.00691  | 1.00518       | 1.00591  | 1.00549  | 1.00552  | 1.00685 | 1.00687  |
| AKAZE     | 10    | 9.50216         | 9.82145  | 15.0438  | 17.7837  | 9.55726  | 15.2153  | 13.2417       | 11.6587  | 11.8926  | 11.628   | 12.1946 | 11.7403  | 1.01052         | 1.01018  | 1.00665  | 1.00562  | 1.01046  | 1.00657  | 1.00755       | 1.00858  | 1.00841  | 1.0086   | 1.0082  | 1.00852  |
| AKAZE     | 11    | 11.2625         | 15.0493  | 12.9036  | 11.9478  | 16.7015  | 9.97957  | 14.1839       | 14.4462  | 14.161   | 14.868   | 12.6188 | 12.6288  | 1.00888         | 1.00664  | 1.00775  | 1.00837  | 1.00599  | 1.01002  | 1.00705       | 1.00692  | 1.00706  | 1.00673  | 1.00792 | 1.00792  |
| AKAZE     | 12    | 10.863          | **83.1215**  | 14.0959  | **31.9794**  | 8.14015  | 16.25    | 11.2899       | 14.4161  | 14.6457  | 11.6503  | 11.7634 | 11.6997  | 1.00921         | 1.0012   | 1.00709  | 1.00313  | 1.01228  | 1.00615  | 1.00886       | 1.00694  | 1.00683  | 1.00858  | 1.0085  | 1.00855  |
| AKAZE     | 13    | 11.0217         | 10.4859  | 8.89129  | 10.3653  | 8.97756  | 7.93986  | 11.9529       | 12.3871  | 11.7218  | 12.9907  | 12.6063 | 12.3608  | 1.00907         | 1.00954  | 1.01125  | 1.00965  | 1.01114  | 1.01259  | 1.00837       | 1.00807  | 1.00853  | 1.0077   | 1.00793 | 1.00809  |
| AKAZE     | 14    | 12.064          | **77.0783**  | 8.77053  | **41.3259**  | 4.48346  | 8.66629  | 11.7973       | 12.3383  | 11.228   | 12.2645  | 12.3431 | 12.7186  | 1.00829         | 1.0013   | 1.0114   | 1.00242  | 1.0223   | 1.01154  | 1.00848       | 1.0081   | 1.00891  | 1.00815  | 1.0081  | 1.00786  |
| AKAZE     | 15    | 9.18212         | 8.78599  | 8.43504  | 8.65865  | 9.31696  | -**48.3058** | 16.2994       | 13.3086  | 15.1409  | 12.3069  | 12.4078 | 14.4678  | 1.01089         | 1.01138  | 1.01186  | 1.01155  | 1.01073  | 0.99793  | 1.00614       | 1.00751  | 1.0066   | 1.00813  | 1.00806 | 1.00691  |
| AKAZE     | 16    | 10.6194         | 8.26626  | 13.1384  | 9.80815  | 7.92425  | 13.1057  | 11.5009       | 10.3355  | 10.4008  | 11.2705  | 11.0416 | 10.0869  | 1.00942         | 1.0121   | 1.00761  | 1.0102   | 1.01262  | 1.00763  | 1.0087        | 1.00968  | 1.00961  | 1.00887  | 1.00906 | 1.00991  |
| AKAZE     | 17    | 10.3277         | 10.3758  | 11.2054  | 9.2842   | 9.14676  | 9.0761   | 10.4618       | 10.1235  | 9.92638  | 8.88888  | 9.98054 | 10.6007  | 1.00968         | 1.00964  | 1.00892  | 1.01077  | 1.01093  | 1.01102  | 1.00956       | 1.00988  | 1.01007  | 1.01125  | 1.01002 | 1.00943  |
| AKAZE     | 18    | 9.39457         | **32.6281**  | 6.75617  | 9.02213  | 8.92524  | 8.12541  | 10.7161       | 10.5308  | 10.6857  | 10.4778  | 10.0547 | 10.1962  | 1.01064         | 1.00306  | 1.0148   | 1.01108  | 1.0112   | 1.01231  | 1.00933       | 1.0095   | 1.00936  | 1.00954  | 1.00995 | 1.00981  |
| SIFT      | 1     | 19.8964         | 16.0392  | -        | 16.3473  | -        | **76.5811**  | 15.2595       | 11.4557  | -        | 14.9712  | -       | 12.5569  | 1.00503         | 1.00623  | -        | 1.00612  | -        | 1.00131  | 1.00655       | 1.00873  | -        | 1.00668  | -       | 1.00796  |
| SIFT      | 2     | **30.36**           | **1213.01**  | -        | 12.5145  | -        | 22.988   | 14.8187       | 15.6001  | -        | 13.3977  | -       | 13.4954  | 1.00329         | 1.00008  | -        | 1.00799  | -        | 1.00435  | 1.00675       | 1.00641  | -        | 1.00746  | -       | 1.00741  |
| SIFT      | 3     | 19.6669         | **30.1681**  | -        | **58.545**   | -        | 10.5738  | 15.378        | 17.8087  | -        | 17.1421  | -       | 12.4266  | 1.00508         | 1.00331  | -        | 1.00171  | -        | 1.00946  | 1.0065        | 1.00562  | -        | 1.00583  | -       | 1.00805  |
| SIFT      | 4     | 28.6212         | **149.908**  | -        | 27.9009  | -        | 11.8481  | 29.7311       | **31.1067**  | -        | 24.9234  | -       | 21.195   | 1.00349         | 1.00067  | -        | 1.00358  | -        | 1.00844  | 1.00336       | 1.00321  | -        | 1.00401  | -       | 1.00472  |
| SIFT      | 5     | 10.6609         | 16.2496  | -        | 9.83757  | -        | 11.6899  | 15.0796       | 15.4041  | -        | 14.1494  | -       | 12.9599  | 1.00938         | 1.00615  | -        | 1.01017  | -        | 1.00855  | 1.00663       | 1.00649  | -        | 1.00707  | -       | 1.00772  |
| SIFT      | 6     | 25.1901         | 11.1216  | -        | 11.9582  | -        | 6.89383  | 14.7949       | 13.7117  | -        | 12.1999  | -       | 13.103   | 1.00397         | 1.00899  | -        | 1.00836  | -        | 1.01451  | 1.00676       | 1.00729  | -        | 1.0082   | -       | 1.00763  |
| SIFT      | 7     | 18.8502         | 12.989   | -        | 9.28936  | -        | 11.6232  | 16.2915       | 19.2115  | -        | 14.0859  | -       | 14.8011  | 1.0053          | 1.0077   | -        | 1.01076  | -        | 1.0086   | 1.00614       | 1.00521  | -        | 1.0071   | -       | 1.00676  |
| SIFT      | 8     | 11.4226         | 17.7325  | -        | 11.8035  | -        | 13.4331  | 17.1708       | 15.5713  | -        | 16.5357  | -       | 14.1782  | 1.00875         | 1.00564  | -        | 1.00847  | -        | 1.00744  | 1.00582       | 1.00642  | -        | 1.00605  | -       | 1.00705  |
| SIFT      | 9     | 19.7896         | 25.127   | -        | **61.6915**  | -        | 7.39678  | 15.5937       | 14.1819  | -        | 16.2761  | -       | 11.7881  | 1.00505         | 1.00398  | -        | 1.00162  | -        | 1.01352  | 1.00641       | 1.00705  | -        | 1.00614  | -       | 1.00848  |
| SIFT      | 10    | 19.3771         | 8.04735  | -        | 11.3669  | -        | 7.60845  | 12.3001       | 11.4953  | -        | 11.6221  | -       | 11.2311  | 1.00516         | 1.01243  | -        | 1.0088   | -        | 1.01314  | 1.00813       | 1.0087   | -        | 1.0086   | -       | 1.0089   |
| SIFT      | 11    | 25.1524         | 8.34979  | -        | **84.4177**  | -        | 16.8129  | 12.0467       | 11.4166  | -        | 12.2822  | -       | 12.6813  | 1.00398         | 1.01198  | -        | 1.00118  | -        | 1.00595  | 1.0083        | 1.00876  | -        | 1.00814  | -       | 1.00789  |
| SIFT      | 12    | 9.29545         | 9.97056  | -        | 8.78401  | -        | 19.2902  | 10.1806       | 11.417   | -        | 11.9997  | -       | 13.0126  | 1.01076         | 1.01003  | -        | 1.01138  | -        | 1.00518  | 1.00982       | 1.00876  | -        | 1.00833  | -       | 1.00768  |
| SIFT      | 13    | 13.1549         | 8.09655  | -        | 10.7119  | -        | 15.2549  | 10.3728       | 10.3986  | -        | 10.5647  | -       | 9.47544  | 1.0076          | 1.01235  | -        | 1.00934  | -        | 1.00656  | 1.00964       | 1.00962  | -        | 1.00947  | -       | 1.01055  |
| SIFT      | 14    | 16.4496         | 18.6098  | -        | **64.3496**  | -        | 11.8929  | 9.8896        | 10.2413  | -        | 10.4138  | -       | 10.3136  | 1.00608         | 1.00537  | -        | 1.00155  | -        | 1.00841  | 1.01011       | 1.00976  | -        | 1.0096   | -       | 1.0097   |
| SIFT      | 15    | **39.2796**         | 26.2358  | -        | 24.2402  | -        | 8.66227  | 9.28561       | 10.4246  | -        | 10.9299  | -       | 10.289   | 1.00255         | 1.00381  | -        | 1.00413  | -        | 1.01154  | 1.01077       | 1.00959  | -        | 1.00915  | -       | 1.00972  |
| SIFT      | 16    | 8.1937          | 10.3711  | -        | **86.533**   | -        | 9.39695  | 9.30781       | 9.49228  | -        | 9.72189  | -       | 9.52937  | 1.0122          | 1.00964  | -        | 1.00116  | -        | 1.01064  | 1.01074       | 1.01053  | -        | 1.01029  | -       | 1.01049  |
| SIFT      | 17    | 8.16418         | 7.93221  | -        | 13.9016  | -        | 17.4061  | 9.66967       | 9.16792  | -        | 10.117   | -       | 9.03408  | 1.01225         | 1.01261  | -        | 1.00719  | -        | 1.00575  | 1.01034       | 1.01091  | -        | 1.00988  | -       | 1.01107  |
| SIFT      | 18    | 15.9448         | 10.3767  | -        | 6.36584  | -        | 11.6399  | 11.5037       | 9.85728  | -        | 9.7874   | -       | 9.31176  | 1.00627         | 1.00964  | -        | 1.01571  | -        | 1.00859  | 1.00869       | 1.01014  | -        | 1.01022  | -       | 1.01074  |

**Table 2:** <span style="color:gray">Camera TTC results with additional values for ```distRatio```</span>


## Conclusion:

Estimating TTC by using LiDAR has been seen to perform more reliable if compared to camera based approaches. Hereby, LiDAR benefits from accurate distance measurements where camera can only estimate change of distance by comparing keypoint matches from successive frames of the preceding vehicle. Both approaches have not been seen to directly suffer from the influence of outlier samples, moreover, numerical instability has been described in detail, which is mainly causing the TTC to largely differ from ground truth.

LiDAR nor camera might be an adequate solution for computing TTC, where both solutions heavily depend on velocity of the ego-vehicle and therefore on the amount of change between successive frame pairs. For providing appropriate performance evaluations and ground truth data, both approaches might be compared to radar based measurements.

/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * JuMarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#ifndef MARKERDETECTOR_H
#define MARKERDETECTOR_H

#include "basictypes/picoflann.h"
#include "featureextractors/ORBextractor.h"
#include "marker.h"
#include "parsing.h"
#include <numeric>
using namespace std;

class MarkerDetector
{
    // Given an KeyPoint element, it returns the element of the dimension specified such that dim=0 is x and dim=1 is y
    struct PicoFlann_KeyPointAdapter
    {
        inline float operator()(const cv::KeyPoint &elem, int dim) const
        {
            return dim == 0 ? elem.pt.x : elem.pt.y;
        }
    };

  public:
    TimerAvrg Fps;
    cv::Mat TheInputImage, TheInputImageThresholding, TheInputImageGrey;
    vector<Marker> markersDetected;
    SVGData SVGFileData;

    bool objectPoseDetected = false;
    cv::Mat objectRvec, objectTvec;
    int objectPoseInliers = 0;
    int objectPoseVisibleTags = 0;

    // Contours
    vector<vector<cv::Point>> filterContours;
    vector<vector<cv::Point>> allContour;

    int markerbitsData;
    string modelType, modeInfo;

    // Image
    std::vector<cv::KeyPoint> keypoints_InputImage;
    cv::Mat descriptors_InputImage;

    // ORB UCOSLAM
    ucoslam::ORBextractor pDerived;
    ucoslam::FeatParams params = ucoslam::FeatParams(2000, 8, 1.2, 1);

    // Utils variables
    string vumarkerName;
    bool thresHoldingDebug = false;
    bool VuMark, enableTracking;
    int only_detect = false;
    bool modelupdate = true;
    int timesCallsDetect = 1;

    // For Thresholding
    int threshold_value, threshold_type, Adaptive_Threshold_Value, Adaptive_Block_Size;
    int const max_threshold_value = 255;

    // Erosion and Dilation
    int erosion_elem, erosion_size, dilation_elem, dilation_size;
    int const max_elem = 2;
    int const max_kernel_size = 21;

    MarkerDetector(const std::string &type);

    void markerCenter();

    bool detectAndTrack(const cv::Mat &InputImage, CameraParameters &camParams, float markerSizeMeters);

    bool detect(CameraParameters &camParams, float markerSizeMeters);

  private:
    // Image analyze
    void getThresholdedImage();
    void Erosion();
    void Dilation();

    // Detection
    std::vector<float> extractElementsColors(cv::Mat &homography);
    void getCRCandId(cv::Mat &homo, string &CRC, string &ID);
    void createMarker(vector<cv::Point> &filterContour, CameraParameters &camParams, cv::Mat &homography, string idCode,
                      string crcImage, float markerSizeMeters);
    void addMarkerDetected(CameraParameters &camParams, float markerSizeMeters, int idMarker, const cv::Mat &homography);
    void estimateRigidObjectPose(CameraParameters &camParams);
    void updateStableDetections();

    // Tracking
    bool track(CameraParameters &cameraParameters);

    // Utils
    bool checkRepeatMarker(std::vector<cv::Point2f> &corners, int IndexMarkerInVMarker);
    void checkOrientation(std::vector<cv::DMatch> &matches, Model &_model);

    // Matches
    std::vector<cv::DMatch> findMatches(picoflann::KdTreeIndex<2, PicoFlann_KeyPointAdapter> &kdtree,
                                        const vector<cv::Point2f> &proyectedPoint, const int radius, const Marker &marker);

    vector<Marker> lastStableMarkersDetected;
    int missedStableMarkerFrames = 0;
    int maxMissedStableMarkerFrames = 4;
};

#endif // MARKERDETECTOR_H

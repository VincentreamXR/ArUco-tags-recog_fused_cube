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

#ifndef MARKER_H
#define MARKER_H

#include "model.h"

using namespace std;

class Marker
{
  public:
    int id;
    float ssize;
    vector<cv::Point2f> corners;

    cv::Mat Rvec, Tvec;
    cv::Mat homography;

    Model model;
    cv::Rect boundingBox;

    Marker(const string &type);

    /**Draws this marker in the input image*/
    void draw(cv::Mat &in, cv::Scalar color = cv::Scalar(0, 0, 255), int lineWidth = -1, bool writeId = true,
              bool writeInfo = false) const;

    cv::Point2f getCenter() const;
    float getPerimeter() const;
    void calculateExtrinsics(float markerSize, cv::Mat CameraMatrix, SVGData fileData, cv::Mat Distorsion = cv::Mat(),
                             bool setYPerpendicular = true);
    void updateBoundingBox(std::vector<cv::Point2f> &imagePoint);
    void initModel(std::vector<cv::KeyPoint> &keypoints_InputImage, cv::Mat &descriptors_InputImage,
                   CameraParameters &cameraParameters, SVGData &SVGFileData);
    void updateModel(std::vector<cv::DMatch> &matches, cv::Mat &descriptors_InputImage,
                     std::vector<cv::KeyPoint> &keypoints_InputImage, SVGData &SVGFileData);
};

#endif // MARKER_H

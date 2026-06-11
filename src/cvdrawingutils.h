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

#ifndef CVDRAWINGUTILS_H
#define CVDRAWINGUTILS_H

#include "svgData.h"

class CvDrawingUtils
{
  public:
    static void draw3dAxis(cv::Mat &Image, cv::Mat &Rvec, cv::Mat &Tvec, const cv::Mat &cameraMatrix, const cv::Mat &distorsion,
                           const cv::Point2f centerMarker, const int lineSize, const int height);

    static void draw3dPolygon(cv::Mat &InputImage, cv::Mat &Rvec, const cv::Mat &Tvec, const cv::Mat &cameraMatrix,
                              const cv::Mat &distorsion, const SVGData &SVGFileData, const int lineSize, const int height);

    static void drawMarkerBorder(cv::Mat &InputImage, cv::Mat &Rvec, const cv::Mat &Tvec, const cv::Mat &cameraMatrix,
                                 const cv::Mat &distorsion, const SVGData &SVGFileData, int lineSize);

    static void drawMarkerBorderH(cv::Mat &InputImage, const cv::Mat &homography, const SVGData &SVGFileData,
                                  const cv::Point2f &centerMarker, const int lineSize, const int id);
};

#endif // CVDRAWINGUTILS_H

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

#ifndef CAMERAPARAMETERS_H
#define CAMERAPARAMETERS_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

class CameraParameters
{
  public:
    // 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
    cv::Mat CameraMatrix;
    //  distortion matrix
    cv::Mat Distorsion;
    // size of the image
    cv::Size CamSize;

    /**Empty constructor
     */
    CameraParameters();
    /**Creates the object from the info passed
     * @param cameraMatrix 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
     * @param distorsionCoeff 4x1 matrix (k1,k2,p1,p2)
     * @param size image size
     */
    CameraParameters(cv::Mat cameraMatrix, cv::Mat distorsionCoeff, cv::Size size);
    /**Sets the parameters
     * @param cameraMatrix 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
     * @param distorsionCoeff 4x1 matrix (k1,k2,p1,p2)
     * @param size image size
     */
    void setParams(cv::Mat cameraMatrix, cv::Mat distorsionCoeff, cv::Size size);

    /**Indicates whether this object is valid
     */
    bool isValid() const
    {
        return CameraMatrix.rows != 0 && CameraMatrix.cols != 0 && Distorsion.rows != 0 && Distorsion.cols != 0 &&
               CamSize.width != -1 && CamSize.height != -1;
    }
    /**Reads from a YAML file generated with the opencv2.2 calibration utility
     */
    void readFromXMLFile(std::string filePath);

    /**Adjust the parameters to the size of the image indicated
     */
    void resize(cv::Size size);
};

#endif // CAMERAPARAMETERS_H

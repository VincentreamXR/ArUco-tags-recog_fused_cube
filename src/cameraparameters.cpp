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

#include "cameraparameters.h"

CameraParameters::CameraParameters()
{
    CameraMatrix = cv::Mat();
    Distorsion = cv::Mat();
    CamSize = cv::Size(-1, -1);
}
/**Creates the object from the info passed
 * @param cameraMatrix 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
 * @param distorsionCoeff 4x1 matrix (k1,k2,p1,p2)
 * @param size image size
 */
CameraParameters::CameraParameters(cv::Mat cameraMatrix, cv::Mat distorsionCoeff, cv::Size size)
{
    setParams(cameraMatrix, distorsionCoeff, size);
}

/**
 */
void CameraParameters::setParams(cv::Mat cameraMatrix, cv::Mat distorsionCoeff, cv::Size size)
{
    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
        throw cv::Exception(9000, "invalid input cameraMatrix", "CameraParameters::setParams", __FILE__, __LINE__);
    cameraMatrix.convertTo(CameraMatrix, CV_32FC1);
    if (distorsionCoeff.total() < 4 || distorsionCoeff.total() >= 7)
        throw cv::Exception(9000, "invalid input distorsionCoeff", "CameraParameters::setParams", __FILE__, __LINE__);
    cv::Mat auxD;

    distorsionCoeff.convertTo(Distorsion, CV_32FC1);

    //     Distorsion.create(1,4,CV_32FC1);
    //     for (int i=0;i<4;i++)
    //         Distorsion.ptr<float>(0)[i]=auxD.ptr<float>(0)[i];

    CamSize = size;
}

void CameraParameters::readFromXMLFile(string filePath)
{
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("CameraParameters::readFromXMLFile could not open file:" + filePath);
    int w = -1, h = -1;
    cv::Mat MCamera, MDist;

    fs["image_width"] >> w;
    fs["image_height"] >> h;
    fs["distortion_coefficients"] >> MDist;
    fs["camera_matrix"] >> MCamera;

    if (MCamera.cols == 0 || MCamera.rows == 0)
    {
        fs["Camera_Matrix"] >> MCamera;
        if (MCamera.cols == 0 || MCamera.rows == 0)
            throw cv::Exception(9007, "File :" + filePath + " does not contains valid camera matrix",
                                "CameraParameters::readFromXML", __FILE__, __LINE__);
    }

    if (w == -1 || h == 0)
    {
        fs["image_Width"] >> w;
        fs["image_Height"] >> h;
        if (w == -1 || h == 0)
            throw cv::Exception(9007, "File :" + filePath + " does not contains valid camera dimensions",
                                "CameraParameters::readFromXML", __FILE__, __LINE__);
    }
    if (MCamera.type() != CV_32FC1)
        MCamera.convertTo(CameraMatrix, CV_32FC1);
    else
        CameraMatrix = MCamera;

    if (MDist.total() < 4)
    {
        fs["Distortion_Coefficients"] >> MDist;
        if (MDist.total() < 4)
            throw cv::Exception(9007, "File :" + filePath + " does not contains valid distortion_coefficients",
                                "CameraParameters::readFromXML", __FILE__, __LINE__);
    }
    // convert to 32 and get the 4 first elements only
    cv::Mat mdist32;
    MDist.convertTo(mdist32, CV_32FC1);
    //     Distorsion.create(1,4,CV_32FC1);
    //     for (int i=0;i<4;i++)
    //         Distorsion.ptr<float>(0)[i]=mdist32.ptr<float>(0)[i];

    Distorsion.create(1, 5, CV_32FC1);
    for (int i = 0; i < 5; i++)
        Distorsion.ptr<float>(0)[i] = mdist32.ptr<float>(0)[i];
    CamSize.width = w;
    CamSize.height = h;
}

/**Adjust the parameters to the size of the image indicated
 */
void CameraParameters::resize(cv::Size size)
{
    if (!isValid())
        throw cv::Exception(9007, "invalid object", "CameraParameters::resize", __FILE__, __LINE__);
    if (size == CamSize)
        return;
    // now, read the camera size
    // resize the camera parameters to fit this image size
    float AxFactor = float(size.width) / float(CamSize.width);
    float AyFactor = float(size.height) / float(CamSize.height);
    CameraMatrix.at<float>(0, 0) *= AxFactor;
    CameraMatrix.at<float>(0, 2) *= AxFactor;
    CameraMatrix.at<float>(1, 1) *= AyFactor;
    CameraMatrix.at<float>(1, 2) *= AyFactor;
    CamSize = size;
}

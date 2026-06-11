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

#include "cvdrawingutils.h"

void CvDrawingUtils::drawMarkerBorder(cv::Mat &InputImage, cv::Mat &Rvec, const cv::Mat &Tvec, const cv::Mat &cameraMatrix,
                                      const cv::Mat &distorsion, const SVGData &SVGFileData, const int lineSize)
{
    vector<cv::Point3f> objectPoints;
    for (size_t i = 0; i < SVGFileData.border.size(); i++)
        objectPoints.push_back(cv::Point3f(SVGFileData.border[i].x, SVGFileData.border[i].y, 0));

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, Rvec, Tvec, cameraMatrix, distorsion, imagePoints);
    // draw lines of different colours
    for (int i = 0; i < imagePoints.size(); i++)
    {
        if (i < SVGFileData.border.size() - 1)
            cv::line(InputImage, imagePoints[i], imagePoints[i + 1], cv::Scalar(0, 0, 255), lineSize, cv::LINE_8, 0);
        else
            cv::line(InputImage, imagePoints[i], imagePoints[0], cv::Scalar(0, 0, 255), lineSize, cv::LINE_8, 0);

        cv::circle(InputImage, imagePoints[i], lineSize, cv::Scalar(255, 0, 0), -1, cv::LINE_8, 0);
    }

    // cv::Point2f markerCenter = getCenter(imagePoints);
    // cv::circle(InputImage,markerCenter,lineSize,cv::Scalar(0,255,255 ),-1,cv::LINE_8,0);
    // cv::putText(InputImage,"id: "+to_string(markerDetected.id), markerCenter,cv::FONT_HERSHEY_SIMPLEX, lineSize/2,
    // cv::Scalar(0,0,255), lineSize,cv::LINE_8,0);
    /*
    // crearlo en una función para no tenerlo repetido
    // Calcule boundingRect for marker tracked
    Rect rect = boundingRect(imagePoints);
    rect.x = rect.x - 40;
    rect.y = rect.y - 40;
    rect.width = rect.width + 70;
    rect.height = rect.height + 70;

    // Draws boundingRect in input image
    Point pt1, pt2;
    pt1.x = rect.x;
    pt1.y = rect.y;
    pt2.x = rect.x + rect.width;
    pt2.y = rect.y + rect.height;
    rectangle(InputImage, pt1, pt2, cv::Scalar(0,0,255), 1);
    */
}

void CvDrawingUtils::drawMarkerBorderH(cv::Mat &InputImage, const cv::Mat &homography, const SVGData &SVGFileData,
                                       const cv::Point2f &centerMarker, const int lineSize, const int id)
{
    cv::Point2f _centerMarker = homography * centerMarker;

    std::vector<cv::Point2f> imagePoints;
    for (auto &p : SVGFileData.border)
        imagePoints.push_back(homography * p);

    const cv::Scalar redColor(0, 0, 255);
    const cv::Scalar blueColor(255, 0, 0);
    const cv::Scalar greenColor(0, 255, 0);

    // draw lines of different colours
    for (int i = 0; i < imagePoints.size(); i++)
    {
        if (i < SVGFileData.border.size() - 1)
            cv::line(InputImage, imagePoints[i], imagePoints[i + 1], redColor, lineSize, cv::LINE_8, 0);
        else
            cv::line(InputImage, imagePoints[i], imagePoints[0], redColor, lineSize, cv::LINE_8, 0);

        cv::putText(InputImage, to_string(i), imagePoints[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_8,
                    0);
        cv::circle(InputImage, imagePoints[i], lineSize + 2, blueColor, -1, cv::LINE_8, 0);
    }

    cv::circle(InputImage, _centerMarker, lineSize, cv::Scalar(0, 255, 255), -1, cv::LINE_8, 0);
    cv::putText(InputImage, "id: " + to_string(id), _centerMarker, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2,
                cv::LINE_8, 0);
}

void CvDrawingUtils::draw3dPolygon(cv::Mat &InputImage, cv::Mat &Rvec, const cv::Mat &Tvec, const cv::Mat &cameraMatrix,
                                   const cv::Mat &distorsion, const SVGData &SVGFileData, const int lineSize, const int height)
{
    vector<cv::Point3f> points3d;

    for (auto &point : SVGFileData.border)
    {
        points3d.emplace_back(point.x, point.y, 0);
        points3d.emplace_back(point.x, point.y, -height);
    }

    const cv::Scalar blueColor(255, 0, 0);
    const cv::Scalar redColor(0, 0, 255);

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(points3d, Rvec, Tvec, cameraMatrix, distorsion, imagePoints);

    // Draw horizontal of different colours
    for (int i = 0; i < imagePoints.size(); i += 2)
    {
        cv::line(InputImage, imagePoints[i], imagePoints[i + 1], blueColor, lineSize, cv::LINE_8, 0);
    }

    for (int i = 0; i < imagePoints.size(); i++)
    {
        if (i == (imagePoints.size() - 2))
            cv::line(InputImage, imagePoints[i], imagePoints[0], blueColor, lineSize, cv::LINE_8, 0);
        else if (i == imagePoints.size() - 1)
            cv::line(InputImage, imagePoints[i], imagePoints[1], blueColor, lineSize, cv::LINE_8, 0);
        else
            cv::line(InputImage, imagePoints[i], imagePoints[i + 2], blueColor, lineSize, cv::LINE_8, 0);

        if (i % 2 != 0)
            cv::circle(InputImage, imagePoints[i], lineSize, redColor, -1, cv::LINE_8, 0);
    }
}

void CvDrawingUtils::draw3dAxis(cv::Mat &InputImage, cv::Mat &Rvec, cv::Mat &Tvec, const cv::Mat &cameraMatrix,
                                const cv::Mat &distorsion, const cv::Point2f centerMarker, const int lineSize, const int height)
{
    const bool drawInCenterMarker = false;
    vector<cv::Point3f> objectPoints;

    if (drawInCenterMarker)
    {
        objectPoints.push_back(cv::Point3f(centerMarker.x, centerMarker.y, 0));
        objectPoints.push_back(cv::Point3f(height, centerMarker.y, 0));
        objectPoints.push_back(cv::Point3f(centerMarker.x, height, 0));
        objectPoints.push_back(cv::Point3f(centerMarker.x, centerMarker.y, -height));
    }

    else
    {
        objectPoints.push_back(cv::Point3f(0, 0, 0));
        objectPoints.push_back(cv::Point3f(height, 0, 0));
        objectPoints.push_back(cv::Point3f(0, height, 0));
        objectPoints.push_back(cv::Point3f(0, 0, -height));
    }

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, Rvec, Tvec, cameraMatrix, distorsion, imagePoints);

    // draw lines of different colours
    cv::line(InputImage, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255, 255), lineSize);
    cv::line(InputImage, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0, 255), lineSize);
    cv::line(InputImage, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0, 255), lineSize);
    putText(InputImage, "x", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255, 255), 2);
    putText(InputImage, "y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0, 255), 2);
    putText(InputImage, "z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0, 255), 2);
}

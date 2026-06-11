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

#include "marker.h"
// ----------------------------------------------------------------------------
//   Global varibles
// ----------------------------------------------------------------------------
const int sizeModel = 800;
const int marginBoundingBox = 40;

// ----------------------------------------------------------------------------
//   Aditional functions
// ----------------------------------------------------------------------------

inline bool insidePolygon(cv::Point2f &point, std::vector<cv::Point2f> &points)
{
    int i, j, nvert = points.size();
    bool c = false;
    for (i = 0, j = nvert - 1; i < nvert; j = i++)
    {
        if (((points[i].y >= point.y) != (points[j].y >= point.y)) &&
            (point.x <= (points[j].x - points[i].x) * (point.y - points[i].y) / (points[j].y - points[i].y) + points[i].x))
            c = !c;
    }

    return c;
}

bool insideMarker(cv::Point2f &p, cv::Mat &Image, SVGData &SVGFileData)
{
    if (p.x < Image.cols && p.y < Image.rows && p.y > 0 && p.x > 0 && insidePolygon(p, SVGFileData.border))
        return true;
    else
        return false;
}

inline bool insideBoundingBox(cv::Point2f &p, cv::Rect &box)
{
    if (p.y > box.y + box.height || p.y < box.y)
        return false;
    else if (p.x > box.x + box.width || p.x < box.x)
        return false;
    else
        return true;
}

std::vector<cv::Point3f> get3DPoints(vector<cv::Point2f> corners)
{
    vector<cv::Point3f> objpoints3D;
    for (cv::Point2f p : corners)
        objpoints3D.push_back(cv::Point3f(p.x, p.y, 0));

    return objpoints3D;
}

// ----------------------------------------------------------------------------
//   Marker functions
// ----------------------------------------------------------------------------
Marker::Marker(const std::string &type) : model(model2image::Model2ImageTransform::create(type))
{
}

float Marker::getPerimeter() const
{
    float sum = 0;
    for (size_t i = 0; i < this->corners.size(); i++)
        sum += static_cast<float>(cv::norm((corners)[i] - (corners)[(i + 1) % 4]));
    return sum;
}

cv::Point2f Marker::getCenter() const
{
    float doubleArea = 0;
    cv::Point2f p(0, 0);
    cv::Point2f p0 = corners.back();
    for (const cv::Point2f &p1 : corners)
    {
        float a = p0.x * p1.y - p0.y * p1.x; // cross product, (signed) double area of triangle of vertices (origin,p0,p1)
        p += (p0 + p1) * a;
        doubleArea += a;
        p0 = p1;
    }
    return p * (1 / (3 * doubleArea));
}

void Marker::updateModel(std::vector<cv::DMatch> &matches, cv::Mat &descriptors_InputImage,
                         std::vector<cv::KeyPoint> &keypoints_InputImage, SVGData &SVGFileData)
{
    // Erase matches not strong
    for (auto it = model.elements.begin(); it != model.elements.end();)
    {
        it->increaseAliveCounter();
        if (it->mustBeremoved())
            model.elements.erase(it);
        else
            it++;
    }
    cv::KeyPoint auxKeyPoint;
    // Convert keypoint into SVGImage point
    for (int i = 0; i < keypoints_InputImage.size(); i++)
    {
        if (model.elements.size() > sizeModel)
            break;

        // Check if keypoint has been used to matched
        if (find_if(matches.begin(), matches.end(), [i](const cv::DMatch &match) { return match.trainIdx == i; }) !=
            matches.end())
            continue;

        auxKeyPoint = keypoints_InputImage[i];

        // Check if keypoint is inside bounding rect marker
        if (!insideBoundingBox(auxKeyPoint.pt, boundingBox))
            continue;

        // InputImage2SVGImage
        cv::Point2f pointInSVG = model.pose->Image2Model(auxKeyPoint.pt);

        // Check if pointSVG is inside marker polygon
        if (!insidePolygon(pointInSVG, SVGFileData.border))
            continue;

        cv::KeyPoint keypoint = auxKeyPoint;
        keypoint.pt = cv::Point2f(pointInSVG.x, pointInSVG.y);

        // Add point and descriptor in SVGModel element
        model.elements.push_back(ModelElement(pointInSVG, descriptors_InputImage.row(i), keypoint));
    }
}

void Marker::initModel(std::vector<cv::KeyPoint> &keypoints_InputImage, cv::Mat &descriptors_InputImage,
                       CameraParameters &cameraParameters, SVGData &SVGFileData)
{
    model.elements.clear();
    updateBoundingBox(corners);

    // Convert keypoint into SVGImage point
    for (size_t i = 0; i < keypoints_InputImage.size(); i++)
    {
        // Check if keypoint is inside bonding box
        if (!insideBoundingBox(keypoints_InputImage[i].pt, boundingBox))
            continue;

        cv::Point2f pointInSVG = homography.inv() * keypoints_InputImage[i].pt;

        // Check if pointSVG is inside marker polygon
        if (!insidePolygon(pointInSVG, SVGFileData.border))
            continue;

        cv::KeyPoint keypoint = keypoints_InputImage[i];
        keypoint.pt.x = pointInSVG.x;
        keypoint.pt.y = pointInSVG.y;

        // Add point and descriptor in SVGModel element
        model.elements.push_back(ModelElement(pointInSVG, descriptors_InputImage.row(i), keypoint));
    }
    model.pose->setParameters(cameraParameters.CameraMatrix, cameraParameters.Distorsion, Rvec, Tvec, homography);
}

void Marker::updateBoundingBox(std::vector<cv::Point2f> &imagePoint)
{
    // Calcule boundingRect for each marker
    boundingBox = boundingRect(imagePoint);
    boundingBox.x = boundingBox.x - marginBoundingBox;
    boundingBox.y = boundingBox.y - marginBoundingBox;
    boundingBox.width = boundingBox.width + marginBoundingBox * 2;
    boundingBox.height = boundingBox.height + marginBoundingBox * 2;
}

void Marker::draw(cv::Mat &in, cv::Scalar color, int lineWidth, bool writeId, bool writeInfo) const
{
    auto _to_string = [](int i) {
        std::stringstream str;
        str << i;
        return "id: " + str.str();
    };

    if (lineWidth == -1)
        lineWidth = static_cast<int>(std::max(1.f, float(in.cols) / 1000.f));

    for (int i = 0; i < corners.size(); i++)
    {
        if (i < corners.size() - 1)
            cv::line(in, this->corners[i], this->corners[i + 1], color, lineWidth, cv::LINE_8, 0);
        else
            cv::line(in, this->corners[i], this->corners[0], color, lineWidth, cv::LINE_8, 0);

        // cv::putText(in, to_string(i), corners[i],cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0,255 ),
        // lineWidth*2,cv::LINE_8,0);
        cv::circle(in, corners[i], lineWidth * 2, cv::Scalar(255, 0, 0), -1, cv::LINE_8, 0);
    }

    cv::circle(in, getCenter(), 4, cv::Scalar(0, 255, 255), -1, cv::LINE_8, 0);

    if (writeId)
    {
        std::string str;
        if (writeId)
            str += _to_string(id);
        cv::putText(in, str, getCenter(), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), lineWidth * 2, cv::LINE_8, 0);
    }
}

void Marker::calculateExtrinsics(float markerSizeMeters, cv::Mat camMatrix, SVGData fileData, cv::Mat distCoeff,
                                 bool setYPerpendicular)
{
    vector<cv::Point3f> objpoints = get3DPoints(fileData.border);
    cv::Mat raux, taux;
    cv::solvePnP(objpoints, corners, camMatrix, distCoeff, raux, taux);

    raux.convertTo(Rvec, CV_64F);
    taux.convertTo(Tvec, CV_64F);

    ssize = markerSizeMeters;
}

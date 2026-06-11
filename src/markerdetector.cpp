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

#include "markerdetector.h"
#include <array>
#include <cmath>
#include <map>

const int CRCBits = 16;
const int idMarkerFilter = 0;
const vector<int> radius = {30, 5};
const bool printInfo = false;
const int minMatches = 15;
const double percent2UpdateModel = 0.6;
int idMatchesDebug = 0;

static cv::Point3f scalePoint(const cv::Point3f &p, float scale)
{
    return cv::Point3f(p.x * scale, p.y * scale, p.z * scale);
}

static cv::Point3f addPoints(const cv::Point3f &a, const cv::Point3f &b)
{
    return cv::Point3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

static cv::Point3f subPoints(const cv::Point3f &a, const cv::Point3f &b)
{
    return cv::Point3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

static cv::Point3f rotateY(const cv::Point3f &p, float yawRad)
{
    const float c = std::cos(yawRad);
    const float s = std::sin(yawRad);
    return cv::Point3f(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static std::array<cv::Point3f, 4> makeTagCorners(const cv::Point3f &center, const cv::Point3f &right,
                                                 const cv::Point3f &down, float side)
{
    const cv::Point3f halfRight = scalePoint(right, side * 0.5f);
    const cv::Point3f halfDown = scalePoint(down, side * 0.5f);

    return {subPoints(subPoints(center, halfRight), halfDown), addPoints(subPoints(center, halfDown), halfRight),
            addPoints(addPoints(center, halfRight), halfDown), addPoints(subPoints(center, halfRight), halfDown)};
}

static std::map<int, std::array<cv::Point3f, 4>> createRigidObjectTagCorners()
{
    const float side = 0.04f; // 40 mm cube/tag side, expressed in meters.
    const float h = side * 0.5f;
    const float pi = 3.14159265358979323846f;

    // The fused object coordinate system is anchored to the cube that contains tag 17:
    // origin is the center of the upper cube, +Z is the outward normal of tag 17,
    // +Y is up, and +X is tag 17 right. The lower cube tags are auxiliary observations.
    // The lower cube is rotated relative to this anchor. If the fused pose is mirrored around Y,
    // change this sign from -45 degrees to +45 degrees.
    const float lowerYawRad = -45.0f * pi / 180.0f;

    const cv::Point3f down(0, -1, 0);
    const cv::Point3f upperCenter(0, 0, 0);
    const cv::Point3f lowerCenter(0, -side, 0);

    const cv::Point3f ux(1, 0, 0);
    const cv::Point3f uz(0, 0, 1);
    const cv::Point3f lx = rotateY(ux, lowerYawRad);
    const cv::Point3f lz = rotateY(uz, lowerYawRad);

    std::map<int, std::array<cv::Point3f, 4>> tagCorners;

    tagCorners[17] = makeTagCorners(addPoints(upperCenter, scalePoint(uz, h)), ux, down, side);
    tagCorners[18] = makeTagCorners(addPoints(upperCenter, scalePoint(ux, h)), scalePoint(uz, -1), down, side);
    tagCorners[23] = makeTagCorners(subPoints(upperCenter, scalePoint(uz, h)), scalePoint(ux, -1), down, side);
    tagCorners[20] = makeTagCorners(subPoints(upperCenter, scalePoint(ux, h)), uz, down, side);

    tagCorners[27] = makeTagCorners(addPoints(lowerCenter, scalePoint(lz, h)), lx, down, side);
    tagCorners[30] = makeTagCorners(addPoints(lowerCenter, scalePoint(lx, h)), scalePoint(lz, -1), down, side);
    tagCorners[29] = makeTagCorners(subPoints(lowerCenter, scalePoint(lz, h)), scalePoint(lx, -1), down, side);
    tagCorners[24] = makeTagCorners(subPoints(lowerCenter, scalePoint(lx, h)), lz, down, side);

    return tagCorners;
}

static const std::map<int, std::array<cv::Point3f, 4>> &getRigidObjectTagCorners()
{
    static const std::map<int, std::array<cv::Point3f, 4>> tagCorners = createRigidObjectTagCorners();
    return tagCorners;
}

MarkerDetector::MarkerDetector(const string &type)
{
    modelType = type;
}

inline cv::Mat resize(const cv::Mat &in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

inline float getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p)
{
    float intpartX;
    float decameraParametersartX = std::modf(p.x, &intpartX);
    float intpartY;
    float decameraParametersartY = std::modf(p.y, &intpartY);
    cv::Point tl;

    if (decameraParametersartX > 0.5)
    {
        if (decameraParametersartY > 0.5)
            tl = cv::Point(intpartX, intpartY);
        else
            tl = cv::Point(intpartX, intpartY - 1);
    }
    else
    {
        if (decameraParametersartY > 0.5)
            tl = cv::Point(intpartX - 1, intpartY);
        else
            tl = cv::Point(intpartX - 1, intpartY - 1);
    }
    return (1.f - decameraParametersartY) * (1. - decameraParametersartX) * float(im_grey.at<uchar>(tl.y, tl.x)) +
           decameraParametersartX * (1 - decameraParametersartY) * float(im_grey.at<uchar>(tl.y, tl.x + 1)) +
           (1 - decameraParametersartX) * decameraParametersartY * float(im_grey.at<uchar>(tl.y + 1, tl.x)) +
           decameraParametersartX * decameraParametersartY * float(im_grey.at<uchar>(tl.y + 1, tl.x + 1));
}

inline cv::Vec3b getSubpixelSimpleValue(const cv::Mat &imageBGR, const cv::Point2f &p)
{
    int x = p.x + 0.5;
    int y = p.y + 0.5;
    if (x >= imageBGR.cols)
        x = imageBGR.cols - 1;
    if (y >= imageBGR.cols)
        y = imageBGR.rows - 1;
    return imageBGR.at<cv::Vec3b>(y, x);
}

inline uint16_t crc16(uint8_t const *data, size_t size)
{
    uint16_t crc = 0;
    while (size--)
    {
        crc ^= *data++;
        for (unsigned k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ 0xa001 : crc >> 1;
    }
    return crc;
}

inline string uint16ToBinary(uint16_t n)
{
    char buffer[17]; /* 16 bits, plus room for a \0 */
    buffer[16] = '\0';
    for (int i = 15; i >= 0; --i)
    {                              /* convert bits from the end */
        buffer[i] = '0' + (n & 1); /* '0' + bit => '0' or '1' */
        n >>= 1;                   /* make the next bit the 'low bit' */
    }
    return buffer;
}

inline int hammingDistance(const string &a, const string &b)
{
    if (a.size() != b.size())
        return std::numeric_limits<int>::max();

    int distance = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            distance++;
    return distance;
}

inline bool checkPoint(const cv::Point2f &point, const cv::Mat &Image)
{
    if (point.x < Image.cols - 1 && point.y < Image.rows - 1 && point.y > 1 && point.x > 1)
        return true;
    else
        return false;
}

inline cv::Vec3f hexToBgr(const string &hexColor)
{
    string hex = hexColor;
    if (!hex.empty() && hex[0] == '#')
        hex.erase(0, 1);
    while (hex.length() < 6)
        hex += "0";

    float r = stoi(hex.substr(0, 2), nullptr, 16);
    float g = stoi(hex.substr(2, 2), nullptr, 16);
    float b = stoi(hex.substr(4, 2), nullptr, 16);
    return cv::Vec3f(b, g, r);
}

inline float colorDistanceSquared(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a - b;
    return d.dot(d);
}

inline cv::Vec3f bgrToHsv(const cv::Vec3f &bgr)
{
    cv::Mat bgrPixel(1, 1, CV_8UC3);
    bgrPixel.at<cv::Vec3b>(0, 0) = cv::Vec3b(cv::saturate_cast<uchar>(bgr[0]),
                                             cv::saturate_cast<uchar>(bgr[1]),
                                             cv::saturate_cast<uchar>(bgr[2]));

    cv::Mat hsvPixel;
    cv::cvtColor(bgrPixel, hsvPixel, cv::COLOR_BGR2HSV);
    cv::Vec3b hsv = hsvPixel.at<cv::Vec3b>(0, 0);
    return cv::Vec3f(hsv[0], hsv[1], hsv[2]);
}

inline float hsvColorDistanceSquared(const cv::Vec3f &aBgr, const cv::Vec3f &bBgr)
{
    const cv::Vec3f a = bgrToHsv(aBgr);
    const cv::Vec3f b = bgrToHsv(bBgr);

    float dh = std::abs(a[0] - b[0]);
    dh = std::min(dh, 180.0f - dh);
    float ds = a[1] - b[1];
    float dv = a[2] - b[2];

    return (dh * dh * 4.0f) + (ds * ds * 0.25f) + (dv * dv * 0.02f);
}

inline float markerColorDistanceSquared(const cv::Vec3f &sample, const cv::Vec3f &reference, bool useHSV)
{
    if (useHSV)
        return hsvColorDistanceSquared(sample, reference);
    return colorDistanceSquared(sample, reference);
}

inline cv::Vec3f sampleMeanBgr(const cv::Mat &image, const cv::Point2f &point)
{
    int x = cvRound(point.x);
    int y = cvRound(point.y);
    int x0 = std::max(0, x - 2);
    int y0 = std::max(0, y - 2);
    int x1 = std::min(image.cols - 1, x + 2);
    int y1 = std::min(image.rows - 1, y + 2);

    cv::Scalar meanColor = cv::mean(image(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)));
    return cv::Vec3f(meanColor[0], meanColor[1], meanColor[2]);
}

std::vector<std::vector<cv::Point>> contoursFilter(vector<vector<cv::Point>> &allContours, int vertexNumber, bool convexPolygon)
{
    vector<vector<cv::Point>> filterContour;

    for (auto &contour : allContours)
    {
        if (cv::contourArea(contour) > 1000)
        {
            const double perimeter = cv::arcLength(contour, true);
            const std::array<double, 7> epsilonFactors = {0.008, 0.012, 0.016, 0.020, 0.026, 0.032, 0.040};

            for (double factor : epsilonFactors)
            {
                vector<cv::Point> approxcontour;
                cv::approxPolyDP(contour, approxcontour, factor * perimeter, true);

                if (approxcontour.size() != static_cast<size_t>(vertexNumber))
                    continue;

                // If the marker contour designed is convex, only the convex image contours are interesting.
                if (!convexPolygon || isContourConvex(approxcontour))
                {
                    filterContour.push_back(approxcontour);
                    break;
                }
            }
        }
    }
    return filterContour;
}

inline void normalize(std::vector<float> &vec)
{
    float sum = 0;
    for (auto v : vec)
        sum += v;
    float invsum = 1. / sum;
    for (auto &v : vec)
        v *= invsum;
}

inline vector<vector<cv::Point2f>> circularShift(const vector<cv::Point> &a)
{
    vector<vector<cv::Point2f>> b;
    auto addCircularShifts = [&b](const vector<cv::Point> &points) {
        for (auto it = points.begin(); it != points.end(); ++it)
        {
            vector<cv::Point2f> dest(points.size());
            std::rotate_copy(points.begin(), it, points.end(), dest.begin());
            if (!b.empty() && dest == b[0])
                return;
            b.push_back(dest);
        }
    };

    addCircularShifts(a);

    vector<cv::Point> reversed = a;
    std::reverse(reversed.begin(), reversed.end());
    const size_t beforeReverse = b.size();
    addCircularShifts(reversed);
    if (beforeReverse > 0 && b.size() > beforeReverse && b[beforeReverse] == b[0])
    {
        b.resize(beforeReverse);
    }
    return b;
}

inline float average(std::vector<float> &vec, int start, int end, int initFrequency)
{
    float avrg = 0;
    float sum = 0;
    do
    {
        avrg += float(initFrequency) * vec[start % 180];
        sum += vec[start % 180];
        initFrequency++;
        start++;

    } while (initFrequency < end);
    return avrg / sum;
}

inline void ucoslam_FM___computeThreeMaxima(const vector<vector<int>> &histo, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (size_t i = 0; i < histo.size(); i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

inline cv::Mat calculeHomography(vector<cv::Point2f> &initPoints, vector<cv::Point> &destinationContours)
{

    cv::Mat finalHomography;
    float finalError = std::numeric_limits<float>::max();

    if (initPoints.size() < 4 || destinationContours.size() < 4 || initPoints.size() != destinationContours.size())
        return finalHomography;

    // Calcule all possible changes in vector
    vector<vector<cv::Point2f>> allCombination = circularShift(destinationContours);

    // Check all possible rotation
    for (auto combination : allCombination)
    {
        // Calcule homography for each change in vector
        cv::Mat parcialHomography = cv::findHomography(initPoints, combination);
        if (parcialHomography.empty())
            continue;

        parcialHomography.convertTo(parcialHomography, CV_64FC1, 1, 0);

        float errorPoint = 0;

        // Calcule error
        for (int j = 0; j < combination.size(); j++)
        {
            cv::Point2f p = parcialHomography * initPoints[j];
            errorPoint += cv::norm(p - combination[j]);
        }

        // Update general error
        if (errorPoint < finalError)
        {
            finalHomography = parcialHomography;
            finalError = errorPoint / destinationContours.size();
        }
    }

    return finalHomography;
}

inline float variance(std::vector<float> &vec, float mean, int start, int end, int initFrequency)
{
    float res = 0;
    float sum = 0;
    do
    {
        res += vec[start % 180] * (float(initFrequency) - mean) * (float(initFrequency) - mean);
        sum += vec[start % 180];
        start++;
        initFrequency++;

    } while (initFrequency < end);
    return res / sum;
}

float circularAverage(std::vector<float> vect)
{
    std::vector<float> hist(180, 0);
    for (auto e : vect)
        hist[int(e)]++;
    normalize(hist);

    int rot = 0;
    int rotation = 0;
    std::pair<int, float> best(-1, std::numeric_limits<float>::min());

    float GAvrg = 90;
    float SigmaExtra = variance(hist, GAvrg, rot, 180, 0);
    do
    {
        float Avrg1 = average(hist, rot, 90, 0);
        float Sigma1Intra = variance(hist, Avrg1, rot, 90, 0);

        float Avrg2 = average(hist, rot + 90, 180, 90);
        float Sigma2Intra = variance(hist, Avrg2, rot + 90, 180, 90);

        float Goodness = SigmaExtra / (Sigma1Intra + Sigma2Intra);
        if (std::isnan(Goodness))
            Goodness = 0;

        if (Goodness > best.second)
            best = {rotation, Goodness};

        rotation++;
        rot = (180 - rotation);
    } while (rotation < 180);
    return best.first;
}

string getBinaryCode(const int strat, int end, const std::vector<float> &pixelsColors, const int numberBitsData)
{
    string allColorsBits;
    if (pixelsColors.empty())
    {
        for (int j = 0; j < numberBitsData + CRCBits; j++)
            allColorsBits.push_back('0');
    }
    else
    {
        for (int j = 0; j < numberBitsData + CRCBits; j++)
        {
            // Calcule color subpixel
            float pixelValue = pixelsColors[j];

            if (pixelValue < end && pixelValue > strat)
                allColorsBits.push_back('1');
            else
                allColorsBits.push_back('0');
        }
    }

    return allColorsBits;
}

std::vector<float> MarkerDetector::extractElementsColors(cv::Mat &homography)
{
    std::vector<float> pixelsColorsBis;

    cv::Scalar color = cv::Scalar(-1);
    int sampledElements = 0;
    const int requiredElements = markerbitsData + CRCBits;
    for (auto &element : SVGFileData.elements)
    {
        if (sampledElements >= requiredElements)
            break;
        sampledElements++;

        cv::Point2f p = homography * element.center;

        // cv::circle(TheInputImage,p,5,color,-1,cv::LINE_8,0);

        if (checkPoint(p, TheInputImage))
        {
            if (SVGFileData.HSV)
            {
                cv::Mat HSV;
                cv::Mat BGR = TheInputImage(cv::Rect(p.x, p.y, 1, 1));
                cvtColor(BGR, HSV, cv::COLOR_BGR2HSV);
                cv::Vec3b hsv = HSV.at<cv::Vec3b>(0, 0);
                pixelsColorsBis.push_back(hsv.val[0]);
            }
            else
                pixelsColorsBis.push_back(getSubpixelValue(TheInputImageGrey, p));
        }
    }

    return pixelsColorsBis;
}

void MarkerDetector::getCRCandId(cv::Mat &homo, string &CRC, string &ID)
{
    if (!SVGFileData.colorBit0.empty() && !SVGFileData.colorBit1.empty())
    {
        const cv::Vec3f bit0Color = hexToBgr(SVGFileData.colorBit0);
        const cv::Vec3f bit1Color = hexToBgr(SVGFileData.colorBit1);
        string elementsColors;

        int sampledElements = 0;
        const int requiredElements = markerbitsData + CRCBits;
        for (auto &element : SVGFileData.elements)
        {
            if (sampledElements >= requiredElements)
                break;
            sampledElements++;

            cv::Point2f p = homo * element.center;
            if (!checkPoint(p, TheInputImage))
                return;

            cv::Vec3f pixelColor = sampleMeanBgr(TheInputImage, p);
            float dist0 = markerColorDistanceSquared(pixelColor, bit0Color, SVGFileData.HSV);
            float dist1 = markerColorDistanceSquared(pixelColor, bit1Color, SVGFileData.HSV);
            elementsColors.push_back(dist1 < dist0 ? '1' : '0');
        }

        if (elementsColors.size() < static_cast<size_t>(requiredElements))
            return;

        ID = elementsColors.substr(0, markerbitsData);
        CRC = elementsColors.substr(markerbitsData, CRCBits);
        return;
    }

    std::vector<float> pixelsColors = extractElementsColors(homo);
    if (pixelsColors.size() < static_cast<size_t>(markerbitsData + CRCBits))
        return;

    // Calcule idcode and CRC for each contour
    string elementsColors;
    if (SVGFileData.HSV)
    {
        float maxAverageHSV = circularAverage(pixelsColors);
        if (maxAverageHSV == 0)
            maxAverageHSV = 1;
        float minAverageHSV = (180 - maxAverageHSV);
        if (maxAverageHSV < minAverageHSV)
            swap(maxAverageHSV, minAverageHSV);

        elementsColors = getBinaryCode(minAverageHSV, maxAverageHSV, pixelsColors, markerbitsData);

        // Subtract from
        ID = elementsColors.substr(0, markerbitsData);
        CRC = elementsColors.substr(markerbitsData, markerbitsData + CRCBits);
    }
    else
    {
        float colorAverageGray = std::accumulate(pixelsColors.begin(), pixelsColors.end(), 0.0) / pixelsColors.size();
        ;
        elementsColors = getBinaryCode(0, colorAverageGray, pixelsColors, markerbitsData);
        ID = elementsColors.substr(0, markerbitsData);
        CRC = elementsColors.substr(markerbitsData, markerbitsData + CRCBits);
    }
}

inline bool PointInPolygon(const cv::Point2f &point, const std::vector<cv::Point2f> &points)
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

bool CornersInsideMarker(Marker &m1, Marker &m2)
{
    for (auto corner : m2.corners)
    {
        if (PointInPolygon(corner, m1.corners))
            return true;
    }

    return false;
}

bool subMarker(Marker &m, std::vector<Marker> &markersDetected)
{
    for (auto marker : markersDetected)
        if (CornersInsideMarker(m, marker))
            return true;
    return false;
}

inline cv::Point2f markerCornersCenter(const std::vector<cv::Point2f> &corners)
{
    cv::Point2f center(0, 0);
    for (const auto &corner : corners)
        center += corner;
    if (!corners.empty())
        center *= 1.0f / static_cast<float>(corners.size());
    return center;
}

inline bool sameMarkerCandidate(const Marker &a, const Marker &b)
{
    if (a.corners.size() != b.corners.size() || a.corners.empty())
        return false;

    const float centerDistance = cv::norm(markerCornersCenter(a.corners) - markerCornersCenter(b.corners));
    if (centerDistance > 8.0f)
        return false;

    float averageCornerDistance = 0.0f;
    for (size_t i = 0; i < a.corners.size(); i++)
        averageCornerDistance += cv::norm(a.corners[i] - b.corners[i]);
    averageCornerDistance /= static_cast<float>(a.corners.size());

    return averageCornerDistance < 20.0f || centerDistance < 3.0f;
}

bool duplicateMarker(Marker &m, std::vector<Marker> &markersDetected)
{
    for (const auto &marker : markersDetected)
        if (sameMarkerCandidate(m, marker))
            return true;
    return false;
}

void MarkerDetector::addMarkerDetected(CameraParameters &cameraParameters, float markerSizeMeters, int idMarker,
                                       const cv::Mat &homography)
{
    std::vector<cv::Point2f> corners;

    if (homography.empty() || SVGFileData.border.size() < 4)
        return;

    for (cv::Point2f p : SVGFileData.border)
        corners.push_back(homography * p);

    // Corner SubPixel
    const int _winSize = 12;
    const cv::Size winSize = cv::Size(_winSize, _winSize);
    const cv::Size zeroZone = cv::Size(-1, -1);
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);
    std::vector<cv::Point2f> objectPoints;

    if (TheInputImageGrey.cols <= 2 * _winSize || TheInputImageGrey.rows <= 2 * _winSize)
        return;

    cv::Rect2f validCornerRegion(_winSize, _winSize, TheInputImageGrey.cols - 2 * _winSize,
                                 TheInputImageGrey.rows - 2 * _winSize);
    for (const auto &corner : corners)
    {
        if (!validCornerRegion.contains(corner))
            return;
    }

    cv::cornerSubPix(TheInputImageGrey, corners, winSize, zeroZone, criteria);

    for (size_t i = 0; i < corners.size(); i++)
        objectPoints.push_back(SVGFileData.border[i]);

    // Create marker object
    Marker marker(modelType);
    marker.id = idMarker;
    marker.corners = corners;
    marker.homography = findHomography(objectPoints, corners);
    if (marker.homography.empty())
        return;

    if (!cameraParameters.CameraMatrix.empty())
        marker.calculateExtrinsics(markerSizeMeters, cameraParameters.CameraMatrix, SVGFileData, cameraParameters.Distorsion);
    // Check if the marker detected is a submarker or another orientation of the same contour.
    if (!subMarker(marker, markersDetected) && !duplicateMarker(marker, markersDetected))
        markersDetected.push_back(marker);
}

std::vector<cv::Mat> calculeHomographySymmetry(const vector<cv::Point2f> &initPoints,
                                               const vector<cv::Point> &destinationContours)
{
    std::vector<cv::Mat> homographies;

    if (initPoints.size() < 4 || destinationContours.size() < 4 || initPoints.size() != destinationContours.size())
        return homographies;

    // Calcule all possible changes in vector
    vector<vector<cv::Point2f>> allCombination = circularShift(destinationContours);

    for (auto combination : allCombination)
    {
        cv::Mat parcialHomography = cv::findHomography(initPoints, combination);
        if (parcialHomography.empty())
            continue;

        parcialHomography.convertTo(parcialHomography, CV_64FC1, 1, 0);

        homographies.push_back(parcialHomography);
    }

    return homographies;
}

float getHammDescDistance_2(const uint64_t *ptr1, const uint64_t *ptr2, uint64_t descSize)
{
    int n8 = descSize / 8;
    int hamm = 0;
    for (int i = 0; i < n8; i++)
        hamm += std::bitset<64>(ptr1[i] ^ ptr2[i]).count();

    int extra = descSize - n8 * 4;
    if (extra == 0)
        return hamm;

    const uint8_t *uptr1 = (uint8_t *)ptr1 + n8;
    const uint8_t *uptr2 = (uint8_t *)ptr2 + n8;
    for (int i = 0; i < extra; i++)
        hamm += std::bitset<8>(uptr1[i] ^ uptr2[i]).count();

    // finally, the rest
    return hamm;
}

float getHammDescDistance(const cv::Mat &dsc1, const cv::Mat &dsc2)
{
    assert(dsc1.type() == dsc2.type());
    assert(dsc1.rows == dsc2.rows && dsc2.rows == 1);
    assert(dsc1.cols == dsc2.cols);
    assert(dsc1.type() == CV_8UC1);
    return getHammDescDistance_2(dsc1.ptr<uint64_t>(0), dsc2.ptr<uint64_t>(0), dsc1.total());
}

bool insideBoundingBox(cv::Point2f p, cv::Rect box)
{
    if (p.y > box.y + box.height || p.y < box.y)
        return false;
    else if (p.x > box.x + box.width || p.x < box.x)
        return false;
    else
        return true;
}

void filter_ambiguous_train(std::vector<cv::DMatch> &matches)
{
    if (matches.size() == 0)
        return;
    // determine maximum values of queryIdx
    int maxT = -1;
    for (auto m : matches)
        maxT = std::max(maxT, m.trainIdx);

    // now, create the vector with the elements
    vector<int> used(maxT + 1, -1);
    vector<cv::DMatch> best_matches(maxT);
    int idx = 0;
    bool needRemove = false;

    for (auto &match : matches)
    {
        if (used[match.trainIdx] == -1)
        {
            used[match.trainIdx] = idx;
        }
        else
        {
            if (matches[used[match.trainIdx]].distance > match.distance)
            {
                matches[used[match.trainIdx]].trainIdx = -1; // annulate the other match
                used[match.trainIdx] = idx;
            }
            else
            {
                match.trainIdx = -1;
            } // annulate this match
        }
        needRemove = true;
        idx++;
    }
    if (needRemove)
        matches.erase(std::remove_if(matches.begin(), matches.end(),
                                     [](const cv::DMatch &m) { return m.trainIdx == -1 || m.queryIdx == -1; }),
                      matches.end());
}

void MarkerDetector::createMarker(vector<cv::Point> &filterContour, CameraParameters &cameraParameters, cv::Mat &homography,
                                  string idCode, string crcImage, float markerSizeMeters)
{
    Fps.start();
    bool traceBits_Value = false;
    if (VuMark)
    {
        string codeVumark;
        string crcVumark;

        if (vumarkerName == "rubik")
        {
            codeVumark = "1";
            crcVumark = "1100100001010010";
        }
        else if (vumarkerName == "camera")
        {
            codeVumark = "1";
            crcVumark = "0101000110010000";
        }
        else if (vumarkerName == "cordoba")
        {
            codeVumark = "0";
            crcVumark = "1110010111100111";
        }
        else if (vumarkerName == "seabery")
        {
            codeVumark = "1";
            crcVumark = "1111111001101001";
        }
        else if (vumarkerName == "UCO")
        {
            codeVumark = "1";
            crcVumark = "1110101101110110";
        }
        else if (vumarkerName == "building")
        {
            codeVumark = "1";
            crcVumark = "0110111010010101";
        }
        else if (vumarkerName == "JR")
        {
            codeVumark = "1";
            crcVumark = "0110010110110101";
        }
        else if (vumarkerName == "JRBis")
        {
            codeVumark = "1";
            crcVumark = "0110000000011010";
        }

        string codeVumarkInv = codeVumark;
        boost::replace_all(codeVumarkInv, "1", "3");
        boost::replace_all(codeVumarkInv, "0", "1");
        boost::replace_all(codeVumarkInv, "3", "0");

        if (traceBits_Value)
            std::cout << crcImage << endl;

        string crcVumarkInv = crcVumark;
        boost::replace_all(crcVumarkInv, "1", "3");
        boost::replace_all(crcVumarkInv, "0", "1");
        boost::replace_all(crcVumarkInv, "3", "0");

        if ((idCode == codeVumark && crcImage == crcVumark) || (idCode == codeVumarkInv && crcImage == crcVumark))
        {

            // Calcule corner point with homography
            std::vector<cv::Point2f> corners;
            for (int j = 0; j < filterContour.size(); j++)
            {
                cv::Point2f corner = homography * SVGFileData.border[j];
                corners.push_back(corner);
            }

            const int idVumark = 32;
            addMarkerDetected(cameraParameters, markerSizeMeters, idVumark, homography);
        }
    }
    else
    {
        // Calcule Id marker
        int idMarker = std::stoi(idCode, nullptr, 2);
        const uint8_t *p = reinterpret_cast<const uint8_t *>(idCode.c_str());
        // Calcule its CRC
        string crcID = uint16ToBinary(crc16(p, markerbitsData));

        string crcIDD = crcImage;
        boost::replace_all(crcIDD, "1", "3");
        boost::replace_all(crcIDD, "0", "1");
        boost::replace_all(crcIDD, "3", "0");

        string idInv = idCode;
        boost::replace_all(idInv, "1", "3");
        boost::replace_all(idInv, "0", "1");
        boost::replace_all(idInv, "3", "0");

        const uint8_t *pInv = reinterpret_cast<const uint8_t *>(idInv.c_str());
        string crcImageInv = uint16ToBinary(crc16(pInv, markerbitsData));

        // Check valid Marker. A one-bit tolerance improves robustness against a single noisy cell sample.
        const bool validDirectCRC = hammingDistance(crcID, crcImage) <= 1;
        const bool validInvertedCRC = hammingDistance(crcIDD, crcImageInv) <= 1;
        if (validDirectCRC || validInvertedCRC)
        {
            if (validInvertedCRC && !validDirectCRC)
                idMarker = std::stoi(idInv, nullptr, 2);

            // Calcule corner point with homography
            std::vector<cv::Point2f> corners, cornersContour;
            for (int j = 0; j < filterContour.size(); j++)
            {
                cv::Point2f corner = homography * SVGFileData.border[j];
                corners.push_back(corner);
                cornersContour.push_back(cv::Point2f(filterContour[j].x, filterContour[j].y));
            }

            addMarkerDetected(cameraParameters, markerSizeMeters, idMarker, homography);
        }
        // Revisar
        /*
        if(IndexMarkerInVMarker == -1)
        {
            Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
            // Add marker in markersDetected vector
            markersDetected.push_back(marker);
        }

        // This id marker has been detected, choose marker with biggest area
        else
        {
            // Choose marker with biggest area
            bool pointInsideMarker = checkRepeatMarker(corners,IndexMarkerInVMarker);

            if(pointInsideMarker  && cv::contourArea(corners) > cv::contourArea(markersDetected[IndexMarkerInVMarker].corners))
            {
                Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
                // Add marker in markersDetected vector
                markersDetected[IndexMarkerInVMarker] = marker;
            }
            else
            {
                Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
                // Add marker in markersDetected vector
                markersDetected.push_back(marker);
            }
            return;
        }
        */
    }

    Fps.stop();
}

void MarkerDetector::markerCenter()
{
    for (int i = 0; i < markersDetected.size(); i++)
    {
        cv::Point2f center = markersDetected[i].model.pose->getCenterMarker(SVGFileData);

        // RT origin axis to camera
        cv::Mat R1 = markersDetected[i].model.pose->getRvect().clone();
        cv::Mat T1 = markersDetected[i].model.pose->getTvect().clone();

        // RT marker axis to origin axis
        cv::Mat R2 = R1.clone();
        cv::Mat T2 = T1.clone();

        R2.setTo(0);

        T2.at<double>(0, 0) = center.x;
        T2.at<double>(1, 0) = center.y;
        T2.at<double>(2, 0) = 0;

        cv::Mat R3, T3;

        cv::composeRT(R2, T2, R1, T1, R3, T3);
        markersDetected[i].model.pose->setRvect(R3);
        markersDetected[i].model.pose->setTvect(T3);
    }
}

void MarkerDetector::updateStableDetections()
{
    if (!markersDetected.empty())
    {
        lastStableMarkersDetected = markersDetected;
        missedStableMarkerFrames = 0;
        return;
    }

    if (!lastStableMarkersDetected.empty() && missedStableMarkerFrames < maxMissedStableMarkerFrames)
    {
        markersDetected = lastStableMarkersDetected;
        missedStableMarkerFrames++;
        modeInfo = "HOLD";
        return;
    }

    lastStableMarkersDetected.clear();
    missedStableMarkerFrames = 0;
}

void MarkerDetector::estimateRigidObjectPose(CameraParameters &cameraParameters)
{
    objectPoseDetected = false;
    objectPoseInliers = 0;
    objectPoseVisibleTags = 0;
    objectRvec.release();
    objectTvec.release();

    if (cameraParameters.CameraMatrix.empty() || markersDetected.empty())
        return;

    const auto &tagCorners = getRigidObjectTagCorners();
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (const auto &marker : markersDetected)
    {
        auto tagIt = tagCorners.find(marker.id);
        if (tagIt == tagCorners.end() || marker.corners.size() < 4)
            continue;

        objectPoseVisibleTags++;
        for (int i = 0; i < 4; i++)
        {
            objectPoints.push_back(tagIt->second[i]);
            imagePoints.push_back(marker.corners[i]);
        }
    }

    if (objectPoints.size() < 4)
        return;

    bool ok = false;
    std::vector<int> inliers;

    if (objectPoints.size() >= 8)
    {
        ok = cv::solvePnPRansac(objectPoints, imagePoints, cameraParameters.CameraMatrix, cameraParameters.Distorsion,
                                objectRvec, objectTvec, false, 100, 4.0f, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
        objectPoseInliers = static_cast<int>(inliers.size());
    }
    else
    {
        ok = cv::solvePnP(objectPoints, imagePoints, cameraParameters.CameraMatrix, cameraParameters.Distorsion, objectRvec,
                          objectTvec, false, cv::SOLVEPNP_ITERATIVE);
        objectPoseInliers = ok ? static_cast<int>(objectPoints.size()) : 0;
    }

    if (!ok || objectRvec.empty() || objectTvec.empty())
    {
        objectPoseDetected = false;
        objectPoseInliers = 0;
        return;
    }

    objectRvec.convertTo(objectRvec, CV_64FC1);
    objectTvec.convertTo(objectTvec, CV_64FC1);
    objectPoseDetected = true;
}

void MarkerDetector::checkOrientation(std::vector<cv::DMatch> &matches, Model &_model)
{
    vector<vector<int>> rotHist(30);
    for (auto &v : rotHist)
        v.reserve(500);
    const float factor = 1.0f / float(rotHist.size());
    for (size_t midx = 0; midx < matches.size(); midx++)
    {
        const auto &match = matches[midx];
        float rot = _model.elements[match.queryIdx].inputKeyPoint.angle - keypoints_InputImage[match.trainIdx].angle;
        if (rot < 0.0)
            rot += 360.0f;
        size_t bin = round(rot * factor);
        if (bin == rotHist.size())
            bin = 0;
        assert(bin >= 0 && bin < rotHist.size());
        rotHist[bin].push_back(midx);
    }

    int ind1 = -1, ind2 = -1, ind3 = -1;
    ucoslam_FM___computeThreeMaxima(rotHist, ind1, ind2, ind3);
    for (int i = 0; i < int(rotHist.size()); i++)
    {
        if (i == ind1 || i == ind2 || i == ind3)
            continue;
        for (auto midx : rotHist[i])
            matches[midx].queryIdx = matches[midx].trainIdx = -1; // mark as unused
    }
    matches.erase(
        std::remove_if(matches.begin(), matches.end(), [](const cv::DMatch &m) { return m.trainIdx == -1 || m.queryIdx == -1; }),
        matches.end());
}

void MarkerDetector::Erosion()
{
    int erosion_type;
    if (erosion_elem == 0)
    {
        erosion_type = cv::MORPH_RECT;
    }
    else if (erosion_elem == 1)
    {
        erosion_type = cv::MORPH_CROSS;
    }
    else
    {
        erosion_type = cv::MORPH_ELLIPSE;
    }

    cv::Mat element = getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            cv::Point(erosion_size, erosion_size));

    /// Apply the erosion operation
    erode(TheInputImageThresholding, TheInputImageThresholding, element);
}

void MarkerDetector::Dilation()
{
    int dilation_type;
    if (dilation_elem == 0)
    {
        dilation_type = cv::MORPH_RECT;
    }
    else if (dilation_elem == 1)
    {
        dilation_type = cv::MORPH_CROSS;
    }
    else
    {
        dilation_type = cv::MORPH_ELLIPSE;
    }

    cv::Mat element = getStructuringElement(dilation_type, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                            cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    dilate(TheInputImageThresholding, TheInputImageThresholding, element);
}

void MarkerDetector::getThresholdedImage()
{
    if (threshold_type == 0)
    {
        threshold(TheInputImageGrey, TheInputImageThresholding, threshold_value, max_threshold_value, cv::THRESH_BINARY);
    }
    else if (threshold_type == 2)
    {
        threshold(TheInputImageGrey, TheInputImageThresholding, 0, max_threshold_value, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    else
    {
        if (Adaptive_Block_Size % 2 == 0)
            Adaptive_Block_Size++;
        adaptiveThreshold(TheInputImageGrey, TheInputImageThresholding, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          Adaptive_Block_Size, Adaptive_Threshold_Value);
    }

    // cv::imshow("Thresholding", resize(TheInputImageThresholding ,720));
}

/// DETECT AND TRACK
bool MarkerDetector::detectAndTrack(const cv::Mat &InputImage, CameraParameters &cameraParameters, float markerSizeMeters)
{
    TheInputImage = InputImage;

    if (VuMark)
        markerbitsData = 1;

    // Grayscale Image:
    cvtColor(TheInputImage, TheInputImageGrey, cv::COLOR_BGR2GRAY);

    if (!enableTracking)
    {
        modeInfo = "DETECT";
        if (detect(cameraParameters, markerSizeMeters))
        {
            // Enable tracking
            if (!only_detect)
                enableTracking = true;
            timesCallsDetect++;

            if (!only_detect && !cameraParameters.CameraMatrix.empty())
            {
                // Init all models markers
                pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);

                for (auto &marker : markersDetected)
                    marker.initModel(keypoints_InputImage, descriptors_InputImage, cameraParameters, SVGFileData);
            }
        }
    }

    else
    {
        modeInfo = "TRACK";
        if (!track(cameraParameters))
        {
            enableTracking = false;
            // Recall detect
            modeInfo = "DETECT";

            if (detect(cameraParameters, markerSizeMeters))
            {
                // Enable tracking
                if (!only_detect)
                    enableTracking = true;

                timesCallsDetect++;

                if (!only_detect && !cameraParameters.CameraMatrix.empty())
                {
                    // Init all models markers
                    pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);

                    for (auto &marker : markersDetected)
                        marker.initModel(keypoints_InputImage, descriptors_InputImage, cameraParameters, SVGFileData);
                }
            }
        }
        else
        {
            if (only_detect)
                enableTracking = false;
        }
    }
    objectPoseDetected = false;
    objectPoseInliers = 0;
    objectPoseVisibleTags = 0;

    updateStableDetections();

    if (!markersDetected.empty() && !cameraParameters.CameraMatrix.empty())
    {
        markerCenter();
        estimateRigidObjectPose(cameraParameters);
    }

    return true;
}

/// DETECT
bool MarkerDetector::detect(CameraParameters &cameraParameters, float markerSizeMeters)
{
    filterContours.clear();
    markersDetected.clear();

    std::vector<cv::Point2f> corners;

    auto processFilteredContours = [&]() {
        // For each contour checked
        for (auto &contour : filterContours)
        {
            cv::Mat homography;
            if (SVGFileData.symmetry)
            {
                std::vector<cv::Mat> Homographies = calculeHomographySymmetry(SVGFileData.border, contour);

                for (auto &homo : Homographies)
                {
                    string idCode, crcImage;
                    getCRCandId(homo, crcImage, idCode);

                    // Check idCode is not empy
                    if (!idCode.empty())
                    {
                        createMarker(contour, cameraParameters, homo, idCode, crcImage, markerSizeMeters);
                    }
                }
            }
            else
            {
                // Calcule homography for each filtercontours
                homography = calculeHomography(SVGFileData.border, contour);
                if (homography.empty())
                    continue;

                // Calcule id and crc in image
                string idCode, crcImage;
                getCRCandId(homography, crcImage, idCode);
                // Check idCode is not empy
                if (!idCode.empty())
                    createMarker(contour, cameraParameters, homography, idCode, crcImage, markerSizeMeters);
            }
        }
    };

    const int originalThresholdType = threshold_type;
    const int originalAdaptiveBlockSize = Adaptive_Block_Size;
    const int originalAdaptiveThresholdValue = Adaptive_Threshold_Value;

    auto runDetectionPass = [&](int passThresholdType, int passBlockSize, int passThresholdValue, bool invertThreshold) {
        allContour.clear();
        filterContours.clear();

        threshold_type = passThresholdType;
        Adaptive_Block_Size = passBlockSize;
        Adaptive_Threshold_Value = passThresholdValue;

        // Thresholding
        getThresholdedImage();
        if (invertThreshold)
            cv::bitwise_not(TheInputImageThresholding, TheInputImageThresholding);

        // Erosion and Dilation
        Erosion();
        Dilation();

        // Find all contours
        cv::findContours(TheInputImageThresholding, allContour, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // Filter contours
        filterContours = contoursFilter(allContour, SVGFileData.border.size(), SVGFileData.convexPolygon);
        processFilteredContours();
    };

    runDetectionPass(originalThresholdType, originalAdaptiveBlockSize, originalAdaptiveThresholdValue, false);
    if (markersDetected.empty())
        runDetectionPass(1, std::max(39, originalAdaptiveBlockSize + 20), 7, false);
    if (markersDetected.empty())
        runDetectionPass(1, std::max(39, originalAdaptiveBlockSize + 20), 7, true);
    if (markersDetected.empty())
        runDetectionPass(2, originalAdaptiveBlockSize, originalAdaptiveThresholdValue, false);
    if (markersDetected.empty())
        runDetectionPass(2, originalAdaptiveBlockSize, originalAdaptiveThresholdValue, true);

    threshold_type = originalThresholdType;
    Adaptive_Block_Size = originalAdaptiveBlockSize;
    Adaptive_Threshold_Value = originalAdaptiveThresholdValue;

    double checkMs = Fps.getAvrg() * 1000.0;
    if (!std::isfinite(checkMs))
        checkMs = 0.0;
    cout << "\rTime in check element value for all contour = " << checkMs << " milliseconds" << endl;

    if (idMarkerFilter != 0)
    {
        markersDetected.erase(
            std::remove_if(markersDetected.begin(), markersDetected.end(), [](Marker x) { return x.id != idMarkerFilter; }),
            markersDetected.end());
    }

    if (!markersDetected.empty())
    {
        /// cv::destroyWindow("Thresholding");
        return true;
    }

    return false;
}

/// TRACK
bool MarkerDetector::track(CameraParameters &cameraParameters)
{

    vector<cv::DMatch> matches;

    if (thresHoldingDebug)
    {
        getThresholdedImage();
        Erosion();
        Dilation();
    }

    // Debug draw keypoint
    if (printInfo)
    {
        cv::drawKeypoints(TheInputImage, keypoints_InputImage, TheInputImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        cout << "Detected Keypoints " << keypoints_InputImage.size() << endl;
    }

    /// Detect Keypoints in input image
    pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);
    /// Create and build Kdtree
    picoflann::KdTreeIndex<2, PicoFlann_KeyPointAdapter> kdtree;
    kdtree.build(keypoints_InputImage);

    for (auto &marker : markersDetected)
    {
        std::vector<cv::Point3f> pointModel;
        std::vector<cv::Point2f> proyectedPoints;

        for (auto &p : marker.model.elements)
            pointModel.push_back(cv::Point3f(p.SVGPoint.x, p.SVGPoint.y, 0.0f));

        if (printInfo)
            cout << "Point proyected: " << pointModel.size() << endl;

        string nameWindow = "Matches " + to_string(marker.id);

        for (auto &r : radius)
        {
            /// Proyect 3D point SVGModel2InputImage
            proyectedPoints = marker.model.pose->project(pointModel);

            /// Update bounding rectangles
            marker.updateBoundingBox(proyectedPoints);

            /// Find matches SVGModel2InputImage
            matches = findMatches(kdtree, proyectedPoints, r, marker);

            if (printInfo)
                cout << "Matches found: " << matches.size() << endl;

            /// Filter match
            filter_ambiguous_train(matches);
            checkOrientation(matches, marker.model);

            /// Refine pose Rvec, Tvec, homography and calcule inliers matches
            marker.model.pose->optimize(matches, cameraParameters, keypoints_InputImage, marker.model.elements, TheInputImageGrey,
                                        SVGFileData.border);

            if (printInfo)
                cout << "Matches after refinePose: " << matches.size() << endl;

            /// Update model element which are been matched
            for (auto &match : matches)
                marker.model.elements.at(match.queryIdx).setAsMatched(true);
        }
        /// Set minimun matches
        if (matches.size() < minMatches)
            return false;

        /// Check if is necessary to update model
        const double percentPointMatches = matches.size() * 1.0f / proyectedPoints.size() * 1.0f;

        if (percentPointMatches > percent2UpdateModel)
            modelupdate = false;
        else
            modelupdate = true;

        /// Update model
        if (modelupdate)
            marker.updateModel(matches, descriptors_InputImage, keypoints_InputImage, SVGFileData);
    }

    return true;
}

/// FIND MATCHES
std::vector<cv::DMatch> MarkerDetector::findMatches(picoflann::KdTreeIndex<2, PicoFlann_KeyPointAdapter> &kdtree,
                                                    const vector<cv::Point2f> &proyectedPoint, const int _radius,
                                                    const Marker &marker)
{
    std::vector<cv::DMatch> matches;
    const int numNeighbors = 10;
    const int radius = _radius * _radius;
    const float hammingDistance = 0.7; /*hamming distance between two nearest keypoint*/

    for (int j = 0; j < proyectedPoint.size(); j++)
    {
        cv::KeyPoint idMatchesDebugKeypoint;
        idMatchesDebugKeypoint.pt = proyectedPoint[j];
        // Search knn neighbor of proyectedPoint[j] in vector keypoints2f_InputImage
        // .First [uint32_t] is index in keypoints_InputImage and .Second [double] is euclidean distance
        const std::vector<std::pair<uint32_t, double>> keyPairs =
            kdtree.searchKnn(keypoints_InputImage, idMatchesDebugKeypoint, numNeighbors);

        // .First [uint32_t] is index in keypoints_InputImage and .Second [double] is hamming distance
        std::pair<uint32_t, double> firstNearestNeighbor(-1, std::numeric_limits<double>::max());
        std::pair<uint32_t, double> secondNearestNeighbor(-1, std::numeric_limits<double>::max());

        // Assert find nearest point in inputImage
        bool pointFound = false;
        // Search two minimal keypoint
        for (auto &KeyPair : keyPairs)
        {
            // Check if keypoint is inside bounding marker
            if (!insideBoundingBox(keypoints_InputImage[KeyPair.first].pt, marker.boundingBox))
                continue;

            // Check euclidean distance
            if (KeyPair.second < radius)
            {
                // Check Hamming distance
                const float hammDistance =
                    getHammDescDistance(marker.model.elements[j].descriptor, descriptors_InputImage.row(KeyPair.first));

                // Calcule two minimal hamming distance
                if (hammDistance < firstNearestNeighbor.second)
                {
                    // Update secondNearestNeighbor
                    secondNearestNeighbor.second = firstNearestNeighbor.second;
                    secondNearestNeighbor.first = firstNearestNeighbor.first;
                    // Update firstNearestNeighbor
                    firstNearestNeighbor.second = hammDistance;
                    firstNearestNeighbor.first = KeyPair.first;
                }

                else if (hammDistance < secondNearestNeighbor.second && hammDistance != firstNearestNeighbor.second)
                {
                    // Update secondNearestNeighbor
                    secondNearestNeighbor.second = hammDistance;
                    secondNearestNeighbor.first = KeyPair.first; // keyPairs[i].second = Euclidean distance;
                }
            }

            pointFound = true;
        }

        // Compare hamming distance between two nearest keypoint
        if ((firstNearestNeighbor.second + 0.1) / (secondNearestNeighbor.second + 0.1) > hammingDistance || !pointFound)
            continue;

        // Match SVG2InputImage
        matches.push_back(cv::DMatch(j, firstNearestNeighbor.first,
                                     firstNearestNeighbor.second)); // DMatch(int _queryIdx, int _trainIdx, float _distance)
    }
    return matches;
}

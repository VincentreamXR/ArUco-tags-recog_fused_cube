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

#include <boost/algorithm/string/replace.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem/config.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
#include <locale>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <regex>
#include <stdexcept>
#include <vector>

using namespace std;

inline cv::Point2f operator*(const cv::Mat &M, const cv::Point2f &p)
{
    if (M.type() != CV_64F)
        throw std::runtime_error("NO DOUBLE");

    const double *m = M.ptr<double>();
    cv::Point3f res;
    res.x = p.x * m[0] + p.y * m[1] + m[2];
    res.y = p.x * m[3] + p.y * m[4] + m[5];
    res.z = p.x * m[6] + p.y * m[7] + m[8];

    return cv::Point2f(res.x / res.z, res.y / res.z);
}

class CodeElement
{
  public:
    string color;
    cv::Point2f center;

    CodeElement(const string &_color, const cv::Point2f &_center)
    {
        color = _color;
        center = _center;
    }
};

class SVGData
{

  public:
    std::vector<cv::Point2f> border;
    std::vector<CodeElement> elements;
    cv::Point2f center;

    bool convexPolygon, HSV, symmetry;

    string colorBit0, colorBit1;

    void drawBoder(cv::Mat &Image);

    void drawElements(cv::Mat &Image);

    cv::Point2f getCenter();
};

class SVGParse
{

  public:
    static void read(string inputFile, SVGData &markerDesign);
    static void save(string inputFile, string outputDir, int markerId, SVGData &markerDesign);
};

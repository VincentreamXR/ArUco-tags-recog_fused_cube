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

#ifndef MODEL_H
#define MODEL_H

#include "cameraparameters.h"
#include "cvdrawingutils.h"
#include "levmarq.h"
#include <numeric>

using namespace std;
struct TimerAvrg
{
    std::vector<double> times;
    size_t curr = 0, n;
    std::chrono::high_resolution_clock::time_point begin, end;
    TimerAvrg(int _n = 30)
    {
        n = _n;
        times.reserve(n);
    }
    inline void start()
    {
        begin = std::chrono::high_resolution_clock::now();
    }
    inline void stop()
    {
        end = std::chrono::high_resolution_clock::now();
        double duration = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * 1e-6;
        if (times.size() < n)
            times.push_back(duration);
        else
        {
            times[curr] = duration;
            curr++;
            if (curr >= times.size())
                curr = 0;
        }
    }
    double getAvrg()
    {
        double sum = 0;
        for (auto t : times)
            sum += t;
        return sum / double(times.size());
    }
};

// Resize image
inline cv::Mat resize1(const cv::Mat &in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

class ModelElement
{

  public:
    cv::Point2f SVGPoint;
    cv::Mat descriptor;
    uint64_t _nmatched = 0;
    uint64_t nframes_alive = 0;
    cv::KeyPoint inputKeyPoint;

    bool _isStrong = false;

    ModelElement();
    ModelElement(cv::Point2f &SVGPoint, cv::Mat descriptor, cv::KeyPoint &inputKeyPoint);

    void increaseAliveCounter();
    void setAsMatched(bool val);

    bool mustBeremoved();
    bool isStrong() const
    {
        return _isStrong;
    }
    inline void toStream(std::ostream &str) const;
    inline void fromStream(std::istream &str);
};

namespace model2image
{

class RT;
class Homo;

class Model2ImageTransform
{
  public:
    static std::shared_ptr<Model2ImageTransform> create(string type);

    /// Set params
    virtual void setParameters(cv::Mat &cameraMat, cv::Mat &distorsion, cv::Mat &Rvec, cv::Mat &Tvec, cv::Mat &homography) = 0;

    virtual std::vector<cv::Point2f> project(const std::vector<cv::Point3f> &v)
    {
        std::vector<cv::Point2f> out;
        out.reserve(v.size());
        for (const auto &p : v)
            out.push_back(project(p));
        return out;
    }

    /// Proyect point model2inputImage
    virtual cv::Point2f project(const cv::Point3f &p) = 0;

    /// Proyect point inputImage2Model
    virtual cv::Point2f Image2Model(const cv::Point2f &p) = 0;

    /// Refine homography
    virtual cv::Mat refineHomography(const std::vector<cv::DMatch> &matches,
                                     const std::vector<cv::KeyPoint> &keypoints_InputImage,
                                     const std::vector<ModelElement> &elements, const std::vector<cv::Point2f> &markerCorners,
                                     const cv::Mat &inputImage) = 0;

    virtual void optimize(std::vector<cv::DMatch> &matches, CameraParameters &cameraParameters,
                          const std::vector<cv::KeyPoint> &keypoints_InputImage, const std::vector<ModelElement> &elements,
                          const cv::Mat &inputImage, const std::vector<cv::Point2f> &markerCorners) = 0;

    virtual void setRvect(cv::Mat &Rvect) = 0;
    virtual void setTvect(cv::Mat &Tvect) = 0;

    virtual cv::Mat getRvect() = 0;
    virtual cv::Mat getTvect() = 0;
    virtual cv::Mat getHomo() = 0;

    /// Draw information marker
    virtual void draw(cv::Mat &Image, const SVGData &SVGFileData, const int id, const CameraParameters &cameraParameters) = 0;

    virtual cv::Point2f getCenterMarker(const SVGData &SVGFileData) = 0;
};

class RT : public Model2ImageTransform
{
    cv::Mat Rvec, Tvec, homography, cameraMat, distorsion;
    double Rmat[9];

    void setRvect(cv::Mat &Rvect);
    void setTvect(cv::Mat &Tvect);
    cv::Mat getRvect();
    cv::Mat getTvect();

    cv::Mat getHomo();

    cv::Point2f getCenterMarker(const SVGData &SVGFileData);

    /// Set params
    void setParameters(cv::Mat &cameraMat, cv::Mat &distorsion, cv::Mat &Rvec, cv::Mat &Tvec, cv::Mat &homography);

    /// Proyect point model2inputImage
    cv::Point2f project(const cv::Point3f &p);

    /// Proyect point inputImage2Model
    cv::Point2f Image2Model(const cv::Point2f &p);

    /// Refine homography
    cv::Mat refineHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints_InputImage,
                             const std::vector<ModelElement> &elements, const std::vector<cv::Point2f> &markerCorners,
                             const cv::Mat &inputImage);

    void optimize(std::vector<cv::DMatch> &matches, CameraParameters &cameraParameters,
                  const std::vector<cv::KeyPoint> &keypoints_InputImage, const std::vector<ModelElement> &elements,
                  const cv::Mat &inputImage, const std::vector<cv::Point2f> &markerCorners);

    /// Draw information marker
    void draw(cv::Mat &Image, const SVGData &SVGFileData, const int id, const CameraParameters &cameraParameters);
};

class Homo : public Model2ImageTransform
{
    class CmdLineParser
    {
        int argc;
        char **argv;

      public:
        CmdLineParser(int _argc, char **_argv) : argc(_argc), argv(_argv)
        {
        }
        bool operator[](string param)
        {
            int idx = -1;
            for (int i = 0; i < argc && idx == -1; i++)
                if (string(argv[i]) == param)
                    idx = i;
            return (idx != -1);
        }
        string operator()(string param, string defvalue = "-1")
        {
            int idx = -1;
            for (int i = 0; i < argc && idx == -1; i++)
                if (string(argv[i]) == param)
                    idx = i;
            if (idx == -1)
                return defvalue;
            else
                return (argv[idx + 1]);
        }
    };

    cv::Mat homography, Rvec, Tvec;
    TimerAvrg Fps;

    void setRvect(cv::Mat &Rvect);
    void setTvect(cv::Mat &Tvect);
    cv::Mat getRvect();
    cv::Mat getTvect();

    cv::Mat getHomo();

    cv::Point2f getCenterMarker(const SVGData &SVGFileData);

    /// Set params
    void setParameters(cv::Mat &cameraMat, cv::Mat &distorsion, cv::Mat &_Rvec, cv::Mat &_Tvec, cv::Mat &_homography);

    /// Proyect point model2inputImage
    cv::Point2f project(const cv::Point3f &p);

    /// Proyect point inputImage2Model
    cv::Point2f Image2Model(const cv::Point2f &p);

    /// Refine homography Subpixel
    cv::Mat refineHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints_InputImage,
                             const std::vector<ModelElement> &elements, const std::vector<cv::Point2f> &markerCorners,
                             const cv::Mat &inputImage);

    void optimize(std::vector<cv::DMatch> &matches, CameraParameters &cameraParameters,
                  const std::vector<cv::KeyPoint> &keypoints_InputImage, const std::vector<ModelElement> &elements,
                  const cv::Mat &inputImage, const std::vector<cv::Point2f> &markerCorners);

    /// Draw information marker
    void draw(cv::Mat &Image, const SVGData &SVGFileData, const int id, const CameraParameters &cameraParameters);
};

} // namespace model2image

class Model
{

  public:
    vector<ModelElement> elements;
    std::shared_ptr<model2image::Model2ImageTransform> pose;

    Model(std::shared_ptr<model2image::Model2ImageTransform> pose);
};

#endif // MODEL_H

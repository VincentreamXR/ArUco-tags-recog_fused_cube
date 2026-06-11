/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * jumarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#include "cvdrawingutils.h"
#include "markerdetector.h"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

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

TimerAvrg Fps;
//  Threshold Trackbar setting
int threshold_type = 1, Adaptive_Threshold_Value = 11, threshold_value = 200, Adaptive_Block_Size = 39, erosion_elem = 1,
    erosion_size = 0, dilation_elem = 1, dilation_size = 0;
int const max_elem = 2, max_kernel_size = 21, max_threshold_value = 255;
float resizeFactor, sizeMarker = 0.010f;

// Video setting
const cv::Size imageResolution(1280, 720);
const bool saveVideo = false;
const double fpsOutVideo = 25, frameStart = 0;

// General variables
cv::VideoCapture TheVideoCapturer;
cv::VideoWriter outputVideo;
CameraParameters TheCameraParameters;
MarkerDetector MDetector("homography");
SVGData SVGFileData;
cv::Mat TheInputImage, TheInputCopyImage;
string TheInputVideo, cameraParametersFile, markerType = "building";

// Control variables
int idBitMarker, waitTime = 0, showAllContours = 0, autofocusEnabled = 1;
char key = 0;
bool isVideo = false, thresHoldingDebug = false, vumarkEnable = false;
bool useCaptureThread = false;
std::atomic<bool> captureThreadRunning(false);
std::thread captureThread;
std::mutex latestFrameMutex;
cv::Mat latestCaptureFrame;

// Resize image
inline cv::Mat resize(const cv::Mat &in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

inline cv::Mat resizeImageFactor(cv::Mat &in, float resizeFactor)
{
    if (fabs(1 - resizeFactor) < 1e-3)
        return in;
    float nc = float(in.cols) * resizeFactor;
    float nr = float(in.rows) * resizeFactor;
    cv::Mat imres;
    cv::resize(in, imres, cv::Size(nc, nr));
    return imres;
}

void captureLoop()
{
    cv::Mat frame;
    while (captureThreadRunning)
    {
        if (!TheVideoCapturer.read(frame) || frame.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(latestFrameMutex);
            frame.copyTo(latestCaptureFrame);
        }
    }
}

bool getNextFrame(cv::Mat &frame)
{
    if (!useCaptureThread)
    {
        if (!TheVideoCapturer.read(frame) || frame.empty())
            return false;
        if (frame.channels() == 1)
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        else if (frame.channels() == 4)
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        return true;
    }

    for (int i = 0; i < 100; i++)
    {
        {
            std::lock_guard<std::mutex> lock(latestFrameMutex);
            if (!latestCaptureFrame.empty())
            {
                latestCaptureFrame.copyTo(frame);
                if (frame.channels() == 1)
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                else if (frame.channels() == 4)
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return false;
}

void stopCaptureThread()
{
    captureThreadRunning = false;
    if (captureThread.joinable())
        captureThread.join();
}

inline void makeGrayDisplayBackground(cv::Mat &image)
{
    if (image.channels() == 1)
    {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        return;
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
}

// Init params from marker detector
inline void initMarkerDetector(MarkerDetector &md, const SVGData &svgData)
{
    md.erosion_elem = erosion_elem;
    md.erosion_size = erosion_size;
    md.dilation_elem = dilation_elem;
    md.dilation_size = dilation_size;

    md.threshold_value = threshold_value;
    md.threshold_type = threshold_type;
    md.Adaptive_Threshold_Value = Adaptive_Threshold_Value;
    md.Adaptive_Block_Size = Adaptive_Block_Size;

    // Initialite SVG value
    md.SVGFileData = svgData;
    md.markerbitsData = idBitMarker;
    md.enableTracking = false;
    md.vumarkerName = markerType;
    md.only_detect = 1;
    md.thresHoldingDebug = thresHoldingDebug;
    md.VuMark = vumarkEnable;
}

inline void initMarkerDetector(MarkerDetector &md)
{
    initMarkerDetector(md, SVGFileData);
}

inline vector<string> splitSvgFiles(const string &input)
{
    vector<string> files;
    string item;
    std::stringstream stream(input);
    while (std::getline(stream, item, ','))
    {
        if (!item.empty())
            files.push_back(item);
    }
    return files;
}

// Draw drawAllContourFiltered
inline void drawAllContourFiltered(cv::Mat &TheInputImage, const vector<vector<cv::Point>> &filterContour)
{
    int lineWidth = 3;
    // cv::Mat atom_image = cv::Mat::zeros( TheInputImage.rows, TheInputImage.cols, CV_8UC3 );
    for (int i = 0; i < filterContour.size(); i++)
    {
        for (int j = 0; j < filterContour[i].size(); j++)
        {
            if (j < filterContour[i].size() - 1)
                cv::line(TheInputImage, filterContour[i][j], filterContour[i][j + 1], cv::Scalar(0, 255, 0), lineWidth,
                         cv::LINE_8, 0);
            else
                cv::line(TheInputImage, filterContour[i][j], filterContour[i][0], cv::Scalar(0, 255, 0), lineWidth, cv::LINE_8,
                         0);

            cv::putText(TheInputImage, to_string(j), filterContour[i][j], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2,
                        cv::LINE_8, 0);
            cv::circle(TheInputImage, filterContour[i][j], lineWidth + 4, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
        }
    }

    // cv::imshow("argv[1]", resize(atom_image,1080));
}

inline void drawShrunkMarkerBox(cv::Mat &image, const Marker &marker, float scale = 0.85f)
{
    if (marker.corners.size() < 4)
        return;

    cv::Point2f center(0, 0);
    for (const auto &corner : marker.corners)
        center += corner;
    center *= 1.0f / static_cast<float>(marker.corners.size());

    std::vector<cv::Point2f> shrunkCorners;
    for (const auto &corner : marker.corners)
        shrunkCorners.push_back(center + (corner - center) * scale);

    const cv::Scalar blue(255, 0, 0);
    for (size_t i = 0; i < shrunkCorners.size(); i++)
        cv::line(image, shrunkCorners[i], shrunkCorners[(i + 1) % shrunkCorners.size()], blue, 2, cv::LINE_AA);

    cv::circle(image, center, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
    cv::putText(image, to_string(marker.id), center + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.65, blue, 2,
                cv::LINE_AA);
}

inline void drawFusedObjectPose(cv::Mat &image, MarkerDetector &md, const CameraParameters &cameraParameters)
{
    if (!md.objectPoseDetected || md.objectRvec.empty() || md.objectTvec.empty())
        return;

    const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                              {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    const float h = 0.02f;

    auto scalePoint = [](const cv::Point3f &p, float scale) {
        return cv::Point3f(p.x * scale, p.y * scale, p.z * scale);
    };
    auto addPoints = [](const cv::Point3f &a, const cv::Point3f &b) {
        return cv::Point3f(a.x + b.x, a.y + b.y, a.z + b.z);
    };

    auto drawCube = [&](const cv::Point3f &center, const cv::Point3f &xAxis, const cv::Point3f &zAxis) {
        const cv::Point3f yAxis(0, 1, 0);
        std::vector<cv::Point3f> boxPoints;
        for (float ySign : {-1.0f, 1.0f})
        {
            boxPoints.push_back(addPoints(addPoints(addPoints(center, scalePoint(xAxis, -h)), scalePoint(yAxis, ySign * h)),
                                          scalePoint(zAxis, -h)));
            boxPoints.push_back(addPoints(addPoints(addPoints(center, scalePoint(xAxis, h)), scalePoint(yAxis, ySign * h)),
                                          scalePoint(zAxis, -h)));
            boxPoints.push_back(addPoints(addPoints(addPoints(center, scalePoint(xAxis, h)), scalePoint(yAxis, ySign * h)),
                                          scalePoint(zAxis, h)));
            boxPoints.push_back(addPoints(addPoints(addPoints(center, scalePoint(xAxis, -h)), scalePoint(yAxis, ySign * h)),
                                          scalePoint(zAxis, h)));
        }

        std::vector<cv::Point2f> boxImagePoints;
        cv::projectPoints(boxPoints, md.objectRvec, md.objectTvec, cameraParameters.CameraMatrix,
                          cameraParameters.Distorsion, boxImagePoints);

        for (const auto &edge : edges)
            cv::line(image, boxImagePoints[edge[0]], boxImagePoints[edge[1]], cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
    };

    drawCube(cv::Point3f(0, 0, 0), cv::Point3f(1, 0, 0), cv::Point3f(0, 0, 1));

    std::vector<cv::Point3f> axisPoints = {cv::Point3f(0, 0, 0), cv::Point3f(0.04f, 0, 0), cv::Point3f(0, 0.04f, 0),
                                           cv::Point3f(0, 0, 0.04f)};
    std::vector<cv::Point2f> axisImagePoints;
    cv::projectPoints(axisPoints, md.objectRvec, md.objectTvec, cameraParameters.CameraMatrix, cameraParameters.Distorsion,
                      axisImagePoints);

    cv::line(image, axisImagePoints[0], axisImagePoints[1], cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
    cv::line(image, axisImagePoints[0], axisImagePoints[2], cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
    cv::line(image, axisImagePoints[0], axisImagePoints[3], cv::Scalar(255, 0, 0), 4, cv::LINE_AA);
    cv::putText(image, "X", axisImagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "Y", axisImagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Z", axisImagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
}

inline void drawTransparentRectangle(cv::Mat &image, const cv::Rect &rect, const cv::Scalar &color, double alpha, int thickness)
{
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, rect, color, thickness, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image);
}

inline void drawTransparentText(cv::Mat &image, const string &text, const cv::Point &origin, double fontScale,
                                const cv::Scalar &color, int thickness, double alpha)
{
    cv::Mat overlay = image.clone();
    cv::putText(overlay, text, origin, cv::FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image);
}

inline double matValue(const cv::Mat &mat, int index)
{
    if (mat.empty())
        return 0.0;

    const int row = mat.rows == 1 ? 0 : index;
    const int col = mat.rows == 1 ? index : 0;
    if (mat.depth() == CV_32F)
        return static_cast<double>(mat.at<float>(row, col));
    return mat.at<double>(row, col);
}

inline string formatVec3(const cv::Mat &vec, int precision = 3)
{
    if (vec.empty() || vec.total() < 3)
        return "n/a";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision)
        << matValue(vec, 0) << ", " << matValue(vec, 1) << ", " << matValue(vec, 2);
    return oss.str();
}

inline void drawDetectionInfoPanel(cv::Mat &image, int markerCount, const string &idsText, bool objectPoseDetected,
                                   int objectPoseVisibleTags, int objectPoseInliers, double detectionMs, double fps,
                                   const cv::Mat &objectRvec, const cv::Mat &objectTvec, bool hasCalibration)
{
    const cv::Rect panel(12, 12, 760, 204);
    drawTransparentRectangle(image, panel, cv::Scalar(0, 0, 0), 0.35, -1);
    drawTransparentRectangle(image, panel, cv::Scalar(0, 255, 255), 0.55, 2);

    drawTransparentText(image, "JuMarker detection", cv::Point(24, 38), 0.72, cv::Scalar(0, 255, 255), 2, 0.75);
    drawTransparentText(image,
                        "markers: " + to_string(markerCount) + "  detect: " + to_string((int)detectionMs) +
                            " ms  fps: " + to_string((int)fps),
                        cv::Point(24, 66), 0.62, cv::Scalar(255, 255, 255), 2, 0.72);
    drawTransparentText(image,
                        "resolution: " + to_string(image.cols) + "x" + to_string(image.rows) + "  only_detect: " +
                            string(MDetector.only_detect ? "on" : "off"),
                        cv::Point(24, 94), 0.62, cv::Scalar(255, 255, 255), 2, 0.72);

    drawTransparentText(image, idsText, cv::Point(24, 122), 0.62, cv::Scalar(255, 255, 255), 2, 0.72);
    drawTransparentText(image,
                        "object: " + string(objectPoseDetected ? "on" : "off") + "  tags: " +
                            to_string(objectPoseVisibleTags) + "  inliers: " + to_string(objectPoseInliers),
                        cv::Point(24, 150), 0.62, cv::Scalar(255, 255, 255), 2, 0.72);

    if (!hasCalibration)
    {
        drawTransparentText(image, "pose: no calibration file (-c/--calibration required for XYZ)",
                            cv::Point(24, 178), 0.58, cv::Scalar(0, 200, 255), 2, 0.72);
        return;
    }

    drawTransparentText(image, "tvec XYZ(m): " + formatVec3(objectTvec, 3), cv::Point(24, 178), 0.58,
                        cv::Scalar(255, 255, 255), 2, 0.72);
    drawTransparentText(image, "rvec XYZ(rad): " + formatVec3(objectRvec, 3), cv::Point(24, 202), 0.58,
                        cv::Scalar(255, 255, 255), 2, 0.72);
}

void cvTackBarEvents(int pos, void *)
{
    (void)(pos);

    initMarkerDetector(MDetector);

    if (!getNextFrame(TheInputImage))
        return;

    TheInputImage.copyTo(TheInputCopyImage);

    /// Check resize factor
    TheInputCopyImage = resizeImageFactor(TheInputCopyImage, resizeFactor);

    /// Detect and track marker
    Fps.start();
    MDetector.detectAndTrack(TheInputImage, TheCameraParameters, sizeMarker);
    Fps.stop();

    /// Draw informarker
    for (auto marker : MDetector.markersDetected)
        drawShrunkMarkerBox(TheInputCopyImage, marker);
    drawFusedObjectPose(TheInputCopyImage, MDetector, TheCameraParameters);

    if (showAllContours)
        drawAllContourFiltered(TheInputCopyImage, MDetector.filterContours);

    if (MDetector.thresHoldingDebug)
        cv::imshow("Thresholding", resize(MDetector.TheInputImageThresholding, 1080));
    cv::imshow("InputImage", resize(TheInputCopyImage, 1080));
}

// Create trackbar in Thresholding Image
inline void createTrackbar()
{
    const string nameWindow = "Thresholding";
    /// Thresholding Trackbar
    cv::createTrackbar("Thresholding Type", nameWindow, &threshold_type, 1, cvTackBarEvents);
    cv::createTrackbar("Threshold Value", nameWindow, &threshold_value, max_threshold_value, cvTackBarEvents);
    cv::createTrackbar("Adaptative Threshold Value", nameWindow, &Adaptive_Threshold_Value, 40, cvTackBarEvents);
    cv::createTrackbar("Adaptative BlockSice", nameWindow, &Adaptive_Block_Size, 40, cvTackBarEvents);
    cv::createTrackbar("Show all contours", nameWindow, &showAllContours, 1, cvTackBarEvents);

    /// Erosion Trackbar
    cv::createTrackbar("Element: erotion \n 0: Rect \n 1: Cross \n 2: Ellipse", nameWindow, &erosion_elem, max_elem,
                       cvTackBarEvents);
    cv::createTrackbar("Kernel erotion size:\n 2n +1", nameWindow, &erosion_size, max_kernel_size, cvTackBarEvents);

    /// Dilation Trackbar
    cv::createTrackbar("Element: dilation \n 0: Rect \n 1: Cross \n 2: Ellipse", "Thresholding", &dilation_elem, max_elem,
                       cvTackBarEvents);
    cv::createTrackbar("Kernel dilation size:\n 2n +1", nameWindow, &dilation_size, max_kernel_size, cvTackBarEvents);
}

// Show information in the inputImage
void showInformation()
{
    // cv::putText(TheInputCopyImage, "RESOLUTION: [ " + to_string(TheInputCopyImage.cols) + " x "
    // +to_string(TheInputCopyImage.rows) + " ]" ,cv::Point(TheInputCopyImage.cols - 300,50),cv::FONT_HERSHEY_SIMPLEX,
    // 0.6f,cv::Scalar(125,255,255),2);
    cv::putText(TheInputCopyImage, "FPS: " + to_string(1. / Fps.getAvrg()), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6f,
                cv::Scalar(125, 255, 255), 2);
    cv::putText(TheInputCopyImage, "MODE: " + MDetector.modeInfo, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6f,
                cv::Scalar(125, 255, 255), 2);
    // cv::putText(TheInputCopyImage, "FRAME: " +
    // to_string((int)TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES)),cv::Point(20,90),cv::FONT_HERSHEY_SIMPLEX,
    // 0.6f,cv::Scalar(125,255,255),2); cv::putText(TheInputCopyImage, "TIMES CALLS DETECTED: " +
    // to_string(MDetector.timesCallsDetect),cv::Point(20,130),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(125,255,255),2);
    // cv::putText(TheInputCopyImage, "Path video file: " + TheInputVideo,cv::Point(20,TheInputCopyImage.rows -
    // 50),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(255,255,255),2); cv::putText(TheInputCopyImage, "Path calibration file: " +
    // cameraParametersFile,cv::Point(20,TheInputCopyImage.rows - 80),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(255,255,255),2);
}

// ----------------------------------------------------------------------------
//   Main
// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 5 || cml["-h"])
        {
            cerr << "Usage: [path svg file] [bits used for check id marker] "
                    "[-c Camera parameters] [-v [in_image|video] ]  [-t VumarkerType] [-vm|--vumark] [-rf 0.X]"
                 << endl;
            cerr << endl;
        }

        /// Calcule bits used for identificate id in marker
        idBitMarker = std::stoi(argv[2]);

        vector<string> svgFilePaths = splitSvgFiles(argv[1]);
        if (svgFilePaths.empty())
            throw std::runtime_error("No marker SVG files were provided");

        vector<SVGData> svgDataList;
        vector<string> svgLabels;
        for (const auto &svgFilePath : svgFilePaths)
        {
            SVGData svgData;
            SVGParse::read(svgFilePath, svgData);
            svgDataList.push_back(svgData);

            std::size_t slash = svgFilePath.find_last_of("/\\");
            string label = slash == string::npos ? svgFilePath : svgFilePath.substr(slash + 1);
            std::size_t dot = label.find_last_of(".");
            if (dot != string::npos)
                label = label.substr(0, dot);
            svgLabels.push_back(label);
        }
        SVGFileData = svgDataList[0];

        /// Read input informations
        if (cml["-v"])
            TheInputVideo = cml("-v");
        if (cml["-c"])
        {
            TheCameraParameters.readFromXMLFile(cml("-c"));
            cameraParametersFile = cml("-c");
        }
        if (cml["-t"])
            markerType = cml("-t");
        if (cml["-vm"] || cml["--vumark"])
            vumarkEnable = true;
        if (cml["--autofocus"])
            autofocusEnabled = std::stoi(cml("--autofocus")) != 0 ? 1 : 0;
        if (cml["-af"])
            autofocusEnabled = std::stoi(cml("-af")) != 0 ? 1 : 0;

        /// Open video capture
        if (TheInputVideo.find("live") != string::npos)
        {
            int vIdx = 0;
            char cad[100];
            if (TheInputVideo.find(":") != string::npos)
            {
                std::replace(TheInputVideo.begin(), TheInputVideo.end(), ':', ' ');
                sscanf(TheInputVideo.c_str(), "%s %d", cad, &vIdx);
            }
            cout << "Opening camera index " << vIdx << endl;

            TheVideoCapturer.open(vIdx);
            TheVideoCapturer.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            TheVideoCapturer.set(cv::CAP_PROP_CONVERT_RGB, 1);
            TheVideoCapturer.set(cv::CAP_PROP_AUTOFOCUS, autofocusEnabled);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_WIDTH, imageResolution.width);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_HEIGHT, imageResolution.height);
            cout << "Requested camera resolution " << imageResolution.width << "x" << imageResolution.height
                 << " autofocus=" << autofocusEnabled << " color=BGR" << endl;
            waitTime = 10;
            isVideo = true;
            useCaptureThread = true;
            captureThreadRunning = true;
            captureThread = std::thread(captureLoop);
        }
        else
        {
            TheVideoCapturer.open(TheInputVideo);
            if (TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT) >= 2)
                isVideo = true;
            if (cml["-skip"])
                TheVideoCapturer.set(cv::CAP_PROP_POS_FRAMES, stoi(cml("-skip")));
        }

        /// Instanciate TheInputImage
        if (!getNextFrame(TheInputImage))
            throw std::runtime_error("Could not read initial video frame");

        /// Init value for MDetector
        initMarkerDetector(MDetector);
        vector<std::unique_ptr<MarkerDetector>> markerDetectors;
        for (const auto &svgData : svgDataList)
        {
            std::unique_ptr<MarkerDetector> detector(new MarkerDetector("homography"));
            initMarkerDetector(*detector, svgData);
            markerDetectors.push_back(std::move(detector));
        }

        /// Check resize factor
        resizeFactor = stof(cml("-rf", "1"));
        if (resizeFactor != 1)
            TheInputImage = resizeImageFactor(TheInputImage, resizeFactor);
        if (!TheCameraParameters.CameraMatrix.empty())
            TheCameraParameters.resize(TheInputImage.size());

        /// Setting Video Saved
        if (saveVideo)
        {
            TheVideoCapturer.set(cv::CAP_PROP_AUTOFOCUS, autofocusEnabled);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_WIDTH, imageResolution.width);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_HEIGHT, imageResolution.height);

            /// Start video in X frame
            if (frameStart != 0)
                TheVideoCapturer.set(cv::CAP_PROP_POS_FRAMES, frameStart);
            outputVideo = cv::VideoWriter("video/outputVideo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsOutVideo,
                                          cv::Size(TheInputImage.cols, TheInputImage.rows));
        }

        // ----------------------------------------------------------------------------
        //   Method starts
        // ----------------------------------------------------------------------------
        char key = 0;
        do
        {
            /// Start capture video
            if (!getNextFrame(TheInputImage))
                break;

            /// Create a new image to draw
            TheInputImage.copyTo(TheInputCopyImage);

            /// Check resize factor
            TheInputCopyImage = resizeImageFactor(TheInputCopyImage, resizeFactor);

            /// Detect and track marker
            Fps.start();
            int markerCount = 0;
            bool objectPoseDetected = false;
            int objectPoseVisibleTags = 0;
            int objectPoseInliers = 0;
            cv::Mat objectRvec, objectTvec;
            string idsText = "ids:";
            for (size_t detectorIndex = 0; detectorIndex < markerDetectors.size(); detectorIndex++)
            {
                MarkerDetector &detector = *markerDetectors[detectorIndex];
                detector.detectAndTrack(TheInputCopyImage, TheCameraParameters, sizeMarker);
                markerCount += detector.markersDetected.size();
                objectPoseDetected = objectPoseDetected || detector.objectPoseDetected;
                objectPoseVisibleTags += detector.objectPoseVisibleTags;
                objectPoseInliers += detector.objectPoseInliers;
                if (detector.objectPoseDetected && objectTvec.empty())
                {
                    objectRvec = detector.objectRvec.clone();
                    objectTvec = detector.objectTvec.clone();
                }

                if (!detector.markersDetected.empty())
                {
                    idsText += " " + svgLabels[detectorIndex] + ":";
                    for (auto marker : detector.markersDetected)
                        idsText += to_string(marker.id) + ",";
                    if (idsText.back() == ',')
                        idsText.pop_back();
                }
            }
            Fps.stop();

            /// Draw marker information
            for (auto &detector : markerDetectors)
            {
                for (auto marker : detector->markersDetected)
                    drawShrunkMarkerBox(TheInputCopyImage, marker);
                drawFusedObjectPose(TheInputCopyImage, *detector, TheCameraParameters);
            }

            if (showAllContours)
                drawAllContourFiltered(TheInputCopyImage, markerDetectors[0]->filterContours);

            std::cout << "Frame:" << TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES) << "/"
                      << TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT);
            std::cout << " Time detection: " << Fps.getAvrg() * 1000
                      << " milliseconds nmarkers: " << markerCount
                      << " image resolution=" << TheInputCopyImage.size();
            if (!TheCameraParameters.CameraMatrix.empty() && objectPoseDetected)
            {
                std::cout << " tvec_xyz_m=[" << formatVec3(objectTvec, 4) << "]"
                          << " rvec_xyz_rad=[" << formatVec3(objectRvec, 4) << "]";
            }
            std::cout << endl;

            double detectionMs = Fps.getAvrg() * 1000.0;
            double fps = detectionMs > 0.0 ? 1000.0 / detectionMs : 0.0;
            drawDetectionInfoPanel(TheInputCopyImage, markerCount, idsText, objectPoseDetected, objectPoseVisibleTags,
                                   objectPoseInliers, detectionMs, fps, objectRvec, objectTvec,
                                   !TheCameraParameters.CameraMatrix.empty());

            /// Show information
            showInformation();

            if (saveVideo)
                outputVideo.write(TheInputCopyImage);

            /// Debug thresHoldImage
            if (thresHoldingDebug)
            {
                cv::namedWindow("Thresholding", 1);
                createTrackbar();
                cv::imshow("Thresholding", resize(MDetector.TheInputImageThresholding, 1080));
            }

            /// Show image
            cv::imshow("InputImage", resize(TheInputCopyImage, 1080));

            key = cv::waitKey(waitTime);

            if (key == 's')
                waitTime = waitTime == 0 ? 10 : 0;

            if (key == 'd')
                MDetector.only_detect = !MDetector.only_detect;

        } while (key != 27); //&& TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES) < TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT));

        stopCaptureThread();
        outputVideo.release();
    }
    catch (std::exception &ex)
    {
        stopCaptureThread();
        cout << "Exception :" << ex.what() << endl;
    }
}

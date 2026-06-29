#include <cmath>
#include <mutex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#if __has_include(<opencv2/objdetect/aruco_detector.hpp>)
#include <opencv2/objdetect/aruco_detector.hpp>
#define MRB_HAS_ARUCO_DETECTOR 1
#elif __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#define MRB_HAS_LEGACY_ARUCO 1
#else
#error "OpenCV ArUco headers were not found"
#endif

#if defined(_WIN32)
#define MRB_EXPORT extern "C" __declspec(dllexport)
#else
#define MRB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
constexpr int kOk = 0;
constexpr int kPoseFound = 1;
constexpr int kNoMarker = 0;
constexpr int kInvalidArgument = -1;
constexpr int kNotInitialized = -2;
constexpr int kUnsupportedDictionary = -3;
constexpr int kSolvePnpFailed = -4;
constexpr int kOpenCvException = -5;
constexpr int kUnknownException = -6;

std::mutex g_mutex;
bool g_initialized = false;
int g_marker_id = -1;
float g_marker_length_meters = 0.0f;
#if MRB_HAS_ARUCO_DETECTOR
cv::aruco::Dictionary g_dictionary;
cv::aruco::DetectorParameters g_detector_parameters;
#else
cv::Ptr<cv::aruco::Dictionary> g_dictionary;
cv::Ptr<cv::aruco::DetectorParameters> g_detector_parameters;
#endif

bool IsFinitePositive(float value)
{
    return std::isfinite(value) && value > 0.0f;
}

bool IsSupportedDictionary(int dictionary_id)
{
    return dictionary_id >= cv::aruco::DICT_4X4_50
        && dictionary_id <= cv::aruco::DICT_ARUCO_ORIGINAL;
}

std::vector<cv::Point3f> MakeMarkerObjectPoints(float marker_length_meters)
{
    const float half = marker_length_meters * 0.5f;
    return {
        cv::Point3f(-half, half, 0.0f),
        cv::Point3f(half, half, 0.0f),
        cv::Point3f(half, -half, 0.0f),
        cv::Point3f(-half, -half, 0.0f),
    };
}

float ComputeMeanReprojectionError(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs)
{
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected);

    double total = 0.0;
    for (size_t i = 0; i < image_points.size() && i < projected.size(); ++i)
    {
        const double dx = static_cast<double>(image_points[i].x) - projected[i].x;
        const double dy = static_cast<double>(image_points[i].y) - projected[i].y;
        total += std::sqrt(dx * dx + dy * dy);
    }

    return image_points.empty() ? 0.0f : static_cast<float>(total / image_points.size());
}
}

MRB_EXPORT int MrbAruco_Init(int dictionary_id, int marker_id, float marker_length_meters)
{
    if (marker_id < 0 || !IsFinitePositive(marker_length_meters))
    {
        return kInvalidArgument;
    }

    if (!IsSupportedDictionary(dictionary_id))
    {
        return kUnsupportedDictionary;
    }

    try
    {
        std::lock_guard<std::mutex> lock(g_mutex);
#if MRB_HAS_ARUCO_DETECTOR
        g_dictionary = cv::aruco::getPredefinedDictionary(dictionary_id);
        g_detector_parameters = cv::aruco::DetectorParameters();
#else
        g_dictionary = cv::aruco::getPredefinedDictionary(
            static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(dictionary_id));
        g_detector_parameters = cv::aruco::DetectorParameters::create();
#endif
        g_marker_id = marker_id;
        g_marker_length_meters = marker_length_meters;
        g_initialized = true;
        return kOk;
    }
    catch (const cv::Exception&)
    {
        return kOpenCvException;
    }
    catch (...)
    {
        return kUnknownException;
    }
}

MRB_EXPORT int MrbAruco_DetectGray(
    const unsigned char* gray,
    int width,
    int height,
    int row_stride,
    float fx,
    float fy,
    float cx,
    float cy,
    float* out_rvec3,
    float* out_tvec3,
    float* out_mean_error,
    int* out_detected_id)
{
    if (gray == nullptr
        || width <= 0
        || height <= 0
        || row_stride < width
        || !IsFinitePositive(fx)
        || !IsFinitePositive(fy)
        || !std::isfinite(cx)
        || !std::isfinite(cy)
        || out_rvec3 == nullptr
        || out_tvec3 == nullptr
        || out_mean_error == nullptr
        || out_detected_id == nullptr)
    {
        return kInvalidArgument;
    }

#if MRB_HAS_ARUCO_DETECTOR
    cv::aruco::Dictionary dictionary;
    cv::aruco::DetectorParameters detector_parameters;
#else
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detector_parameters;
#endif
    int marker_id = -1;
    float marker_length_meters = 0.0f;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized)
        {
            return kNotInitialized;
        }

        dictionary = g_dictionary;
        detector_parameters = g_detector_parameters;
        marker_id = g_marker_id;
        marker_length_meters = g_marker_length_meters;
    }

    try
    {
        cv::Mat gray_image(
            height,
            width,
            CV_8UC1,
            const_cast<unsigned char*>(gray),
            static_cast<size_t>(row_stride));

        std::vector<std::vector<cv::Point2f>> marker_corners;
        std::vector<int> marker_ids;
#if MRB_HAS_ARUCO_DETECTOR
        cv::aruco::ArucoDetector detector(dictionary, detector_parameters);
        detector.detectMarkers(gray_image, marker_corners, marker_ids);
#else
        cv::aruco::detectMarkers(gray_image, dictionary, marker_corners, marker_ids, detector_parameters);
#endif

        int match_index = -1;
        for (size_t i = 0; i < marker_ids.size(); ++i)
        {
            if (marker_ids[i] == marker_id)
            {
                match_index = static_cast<int>(i);
                break;
            }
        }

        if (match_index < 0)
        {
            return kNoMarker;
        }

        const std::vector<cv::Point3f> object_points = MakeMarkerObjectPoints(marker_length_meters);
        const std::vector<cv::Point2f>& image_points = marker_corners[match_index];
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3)
            << fx, 0.0, cx,
               0.0, fy, cy,
               0.0, 0.0, 1.0);
        cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
        cv::Mat rvec;
        cv::Mat tvec;

        const bool solved = cv::solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE);

        if (!solved)
        {
            return kSolvePnpFailed;
        }

        out_rvec3[0] = static_cast<float>(rvec.at<double>(0));
        out_rvec3[1] = static_cast<float>(rvec.at<double>(1));
        out_rvec3[2] = static_cast<float>(rvec.at<double>(2));
        out_tvec3[0] = static_cast<float>(tvec.at<double>(0));
        out_tvec3[1] = static_cast<float>(tvec.at<double>(1));
        out_tvec3[2] = static_cast<float>(tvec.at<double>(2));
        out_mean_error[0] = ComputeMeanReprojectionError(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs);
        out_detected_id[0] = marker_id;
        return kPoseFound;
    }
    catch (const cv::Exception&)
    {
        return kOpenCvException;
    }
    catch (...)
    {
        return kUnknownException;
    }
}

MRB_EXPORT void MrbAruco_Shutdown()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_initialized = false;
    g_marker_id = -1;
    g_marker_length_meters = 0.0f;
}

# MRBadminton ArUco Native Plugin

This project builds `libmrbadminton_aruco.so`, the Android native plugin used by Unity to detect one configured ArUco marker from a grayscale camera frame.

## Required Inputs

- Android NDK with CMake toolchain.
- OpenCV Android SDK or Android OpenCV build that includes `core`, `calib3d`, and `objdetect` with ArUco support.

The Unity project does not currently contain a ready-to-package OpenCV Android SDK. Provide `OpenCV_DIR` as the path to `OpenCV-android-sdk/sdk/native/jni` or an equivalent Android build directory that contains `OpenCVConfig.cmake`.

## Configure

```bash
cmake -S NativePlugins/MRBadmintonAruco \
  -B /tmp/mrbadminton_aruco_android_arm64 \
  -DOpenCV_DIR=/path/to/OpenCV-android-sdk/sdk/native/jni \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_TOOLCHAIN_FILE=/home/zsyy/Android/android-ndk-r20b/build/cmake/android.toolchain.cmake
```

## Build

```bash
cmake --build /tmp/mrbadminton_aruco_android_arm64 --config Release
```

Copy the output shared object to:

```text
Assets/Plugins/Android/arm64-v8a/libmrbadminton_aruco.so
```

If OpenCV is linked dynamically, also package the required OpenCV `.so` files in the same ABI folder or through the Android Gradle project.

## ABI

```c
int MrbAruco_Init(int dictionary_id, int marker_id, float marker_length_meters);

int MrbAruco_DetectGray(
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
    int* out_detected_id);

void MrbAruco_Shutdown();
```

`MrbAruco_DetectGray` returns `1` for a valid pose, `0` for no configured marker, and a negative value for invalid arguments, missing initialization, unsupported dictionary, solvePnP failure, or OpenCV exceptions.

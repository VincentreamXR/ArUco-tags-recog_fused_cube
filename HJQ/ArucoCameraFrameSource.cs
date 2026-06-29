using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public struct ArucoGrayFrame
{
    public int width;
    public int height;
    public int rowStride;
    public byte[] grayBytes;
    public float fx;
    public float fy;
    public float cx;
    public float cy;
    public double timestampSeconds;
}

public sealed class ArucoCameraFrameSource : MonoBehaviour
{
    public const string CameraUnavailableStatus = "camera unavailable";
    public const string IntrinsicsUnavailableStatus = "intrinsics unavailable";

    [SerializeField] private ARCameraManager cameraManager;

    private ArucoGrayFrame latestFrame;
    private byte[] latestGrayBytes;
    private bool hasFrame;
    private bool subscribed;
    private string lastStatus = CameraUnavailableStatus;

    public string LastStatus => lastStatus;

    private void Awake()
    {
        ResolveCameraManager();
    }

    private void OnEnable()
    {
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    public bool TryGetLatestGrayFrame(out ArucoGrayFrame frame, out string status)
    {
        if (cameraManager == null)
        {
            ResolveCameraManager();
        }

        if (cameraManager == null)
        {
            frame = default;
            status = CameraUnavailableStatus;
            return false;
        }

        if (!hasFrame)
        {
            frame = default;
            status = lastStatus;
            return false;
        }

        frame = latestFrame;
        status = "frame";
        return true;
    }

    private void Subscribe()
    {
        if (subscribed)
        {
            return;
        }

        if (cameraManager == null)
        {
            ResolveCameraManager();
        }

        if (cameraManager == null)
        {
            lastStatus = CameraUnavailableStatus;
            return;
        }

        cameraManager.frameReceived += OnFrameReceived;
        subscribed = true;
    }

    private void Unsubscribe()
    {
        if (!subscribed || cameraManager == null)
        {
            subscribed = false;
            return;
        }

        cameraManager.frameReceived -= OnFrameReceived;
        subscribed = false;
    }

    private void ResolveCameraManager()
    {
        if (cameraManager != null)
        {
            return;
        }

        cameraManager = FindFirstObjectByType<ARCameraManager>();
        if (cameraManager == null)
        {
            lastStatus = CameraUnavailableStatus;
        }
    }

    private void OnFrameReceived(ARCameraFrameEventArgs args)
    {
        if (cameraManager == null)
        {
            lastStatus = CameraUnavailableStatus;
            return;
        }

        if (!cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            lastStatus = CameraUnavailableStatus;
            return;
        }

        try
        {
            if (!cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intrinsics))
            {
                lastStatus = IntrinsicsUnavailableStatus;
                return;
            }

            CopyYPlane(image, intrinsics);
            lastStatus = "frame";
            hasFrame = true;
        }
        finally
        {
            image.Dispose();
        }
    }

    private void CopyYPlane(XRCpuImage image, XRCameraIntrinsics intrinsics)
    {
        XRCpuImage.Plane yPlane = image.GetPlane(0);
        int width = image.width;
        int height = image.height;

        if (yPlane.pixelStride == 1)
        {
            int requiredLength = yPlane.rowStride * height;
            EnsureBuffer(requiredLength);
            for (int row = 0; row < height; row++)
            {
                int sourceOffset = row * yPlane.rowStride;
                int copyLength = Mathf.Min(yPlane.rowStride, yPlane.data.Length - sourceOffset);
                if (copyLength > 0)
                {
                    for (int i = 0; i < copyLength; i++)
                    {
                        latestGrayBytes[sourceOffset + i] = yPlane.data[sourceOffset + i];
                    }
                }
            }

            latestFrame.rowStride = yPlane.rowStride;
        }
        else
        {
            int requiredLength = width * height;
            EnsureBuffer(requiredLength);
            for (int row = 0; row < height; row++)
            {
                int sourceRow = row * yPlane.rowStride;
                int targetRow = row * width;
                for (int col = 0; col < width; col++)
                {
                    int sourceIndex = sourceRow + col * yPlane.pixelStride;
                    latestGrayBytes[targetRow + col] = sourceIndex < yPlane.data.Length ? yPlane.data[sourceIndex] : (byte)0;
                }
            }

            latestFrame.rowStride = width;
        }

        latestFrame.width = width;
        latestFrame.height = height;
        latestFrame.grayBytes = latestGrayBytes;
        latestFrame.fx = intrinsics.focalLength.x;
        latestFrame.fy = intrinsics.focalLength.y;
        latestFrame.cx = intrinsics.principalPoint.x;
        latestFrame.cy = intrinsics.principalPoint.y;
        latestFrame.timestampSeconds = Time.realtimeSinceStartupAsDouble;
    }

    private void EnsureBuffer(int requiredLength)
    {
        if (latestGrayBytes == null || latestGrayBytes.Length != requiredLength)
        {
            latestGrayBytes = new byte[requiredLength];
        }
    }
}

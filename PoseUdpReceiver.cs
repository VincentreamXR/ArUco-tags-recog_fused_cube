using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public sealed class PoseUdpReceiver : MonoBehaviour
{
    private enum PoseSpace
    {
        TargetParentLocal,
        ReferenceLocalToWorld,
        World
    }

    [Header("UDP")]
    [SerializeField] private int listenPort = 5055;

    [Header("Target")]
    [SerializeField] private Transform trackedRoot;
    [SerializeField] private bool hideWhenInvalid = false;
    [SerializeField] private bool applyPose = true;

    [Header("Coordinate Conversion")]
    [SerializeField] private PoseSpace incomingPoseSpace = PoseSpace.ReferenceLocalToWorld;
    [SerializeField] private Transform poseReference;
    [SerializeField] private float positionScale = 1.0f;
    [SerializeField] private Vector3 markerLocalPositionOffset = Vector3.zero;
    [SerializeField] private Vector3 positionOffset = Vector3.zero;
    [SerializeField] private Vector3 eulerOffsetDegrees = Vector3.zero;

    [Header("Smoothing")]
    [SerializeField] private bool smoothPose = true;
    [SerializeField] private float positionLerpSpeed = 30.0f;
    [SerializeField] private float rotationSlerpSpeed = 30.0f;
    [SerializeField] private float staleTimeoutSeconds = 0.0f;

    [Header("Debug")]
    [SerializeField] private bool showDebugOverlay = true;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;
    private readonly object poseLock = new object();
    private PosePacket latestPose;
    private bool hasPose;
    private long latestPoseTicks;
    private bool hasAppliedPose;
    private bool warnedMissingReference;
    private Vector3 lastConvertedPosition;
    private Quaternion lastConvertedRotation = Quaternion.identity;
    private float lastPoseAgeSeconds;

    [Serializable]
    private sealed class PosePacket
    {
        public bool valid;
        public double timestamp;
        public float[] rvec;
        public float[] tvec;
        public int[] used_ids;
        public float mean_error;
        public string pose_state;
    }

    private void OnEnable()
    {
        lock (poseLock)
        {
            latestPose = null;
            latestPoseTicks = 0;
            hasPose = false;
        }

        hasAppliedPose = false;
        warnedMissingReference = false;
        lastPoseAgeSeconds = 0.0f;

        udpClient = new UdpClient(listenPort);
        running = true;
        receiveThread = new Thread(ReceiveLoop)
        {
            IsBackground = true,
            Name = "ArUco pose UDP receiver"
        };
        receiveThread.Start();
    }

    private void OnDisable()
    {
        running = false;
        udpClient?.Close();
        udpClient = null;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(100);
        }
        receiveThread = null;
    }

    private void LateUpdate()
    {
        if (!applyPose || trackedRoot == null)
        {
            return;
        }

        PosePacket pose;
        long poseTicks;
        lock (poseLock)
        {
            if (!hasPose)
            {
                return;
            }
            pose = latestPose;
            poseTicks = latestPoseTicks;
        }

        if (pose == null || !pose.valid || pose.rvec == null || pose.tvec == null || pose.rvec.Length < 3 || pose.tvec.Length < 3)
        {
            if (hideWhenInvalid)
            {
                trackedRoot.gameObject.SetActive(false);
            }
            return;
        }

        lastPoseAgeSeconds = poseTicks == 0
            ? 0.0f
            : (float)((System.Diagnostics.Stopwatch.GetTimestamp() - poseTicks) / (double)System.Diagnostics.Stopwatch.Frequency);

        if (staleTimeoutSeconds > 0.0f && lastPoseAgeSeconds > staleTimeoutSeconds)
        {
            if (hideWhenInvalid)
            {
                trackedRoot.gameObject.SetActive(false);
            }
            return;
        }

        if (hideWhenInvalid && !trackedRoot.gameObject.activeSelf)
        {
            trackedRoot.gameObject.SetActive(true);
        }

        Vector3 referenceLocalPosition = ConvertOpenCvPositionToUnity(pose.tvec);
        Quaternion markerLocalRotation = ConvertOpenCvRodriguesToUnity(pose.rvec);
        Quaternion referenceLocalRotation = markerLocalRotation * Quaternion.Euler(eulerOffsetDegrees);
        referenceLocalPosition += markerLocalRotation * markerLocalPositionOffset;
        referenceLocalPosition += positionOffset;

        ApplyConvertedPose(referenceLocalPosition, referenceLocalRotation);
    }

    private void ReceiveLoop()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remote);
                string json = System.Text.Encoding.UTF8.GetString(data);
                PosePacket pose = JsonUtility.FromJson<PosePacket>(json);
                lock (poseLock)
                {
                    latestPose = pose;
                    latestPoseTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                    hasPose = true;
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (SocketException)
            {
                if (!running)
                {
                    break;
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("Invalid ArUco pose UDP packet: " + exception.Message);
            }
        }
    }

    private Vector3 ConvertOpenCvPositionToUnity(float[] tvec)
    {
        return new Vector3(tvec[0], -tvec[1], tvec[2]) * positionScale;
    }

    private void ApplyConvertedPose(Vector3 referenceLocalPosition, Quaternion referenceLocalRotation)
    {
        lastConvertedPosition = referenceLocalPosition;
        lastConvertedRotation = referenceLocalRotation;

        switch (incomingPoseSpace)
        {
            case PoseSpace.TargetParentLocal:
                ApplyTargetLocalPose(referenceLocalPosition, referenceLocalRotation);
                break;
            case PoseSpace.World:
                ApplyTargetWorldPose(referenceLocalPosition, referenceLocalRotation);
                break;
            default:
                Transform reference = ResolvePoseReference();
                if (reference == null)
                {
                    ApplyTargetLocalPose(referenceLocalPosition, referenceLocalRotation);
                    return;
                }

                ApplyTargetWorldPose(
                    reference.TransformPoint(referenceLocalPosition),
                    reference.rotation * referenceLocalRotation);
                break;
        }
    }

    private Transform ResolvePoseReference()
    {
        if (poseReference != null)
        {
            return poseReference;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            return mainCamera.transform;
        }

        if (!warnedMissingReference)
        {
            warnedMissingReference = true;
            Debug.LogWarning("PoseUdpReceiver has no poseReference and no Main Camera; falling back to target parent local pose.");
        }

        return null;
    }

    private void ApplyTargetLocalPose(Vector3 targetPosition, Quaternion targetRotation)
    {
        if (smoothPose && hasAppliedPose)
        {
            float positionAlpha = GetFrameAlpha(positionLerpSpeed);
            float rotationAlpha = GetFrameAlpha(rotationSlerpSpeed);
            targetPosition = Vector3.Lerp(trackedRoot.localPosition, targetPosition, positionAlpha);
            targetRotation = Quaternion.Slerp(trackedRoot.localRotation, targetRotation, rotationAlpha);
        }

        trackedRoot.localPosition = targetPosition;
        trackedRoot.localRotation = targetRotation;
        hasAppliedPose = true;
    }

    private void ApplyTargetWorldPose(Vector3 targetPosition, Quaternion targetRotation)
    {
        if (smoothPose && hasAppliedPose)
        {
            float positionAlpha = GetFrameAlpha(positionLerpSpeed);
            float rotationAlpha = GetFrameAlpha(rotationSlerpSpeed);
            targetPosition = Vector3.Lerp(trackedRoot.position, targetPosition, positionAlpha);
            targetRotation = Quaternion.Slerp(trackedRoot.rotation, targetRotation, rotationAlpha);
        }

        trackedRoot.SetPositionAndRotation(targetPosition, targetRotation);
        hasAppliedPose = true;
    }

    private static float GetFrameAlpha(float speed)
    {
        if (speed <= 0.0f)
        {
            return 1.0f;
        }

        return 1.0f - Mathf.Exp(-speed * Time.deltaTime);
    }

    private static Quaternion ConvertOpenCvRodriguesToUnity(float[] rvec)
    {
        Matrix4x4 openCvRotation = RodriguesToMatrix(rvec[0], rvec[1], rvec[2]);
        Matrix4x4 unityRotation = FlipY(openCvRotation);
        return QuaternionFromMatrix(unityRotation);
    }

    private static Matrix4x4 RodriguesToMatrix(float rx, float ry, float rz)
    {
        float theta = Mathf.Sqrt(rx * rx + ry * ry + rz * rz);
        Matrix4x4 matrix = Matrix4x4.identity;
        if (theta < 1e-6f)
        {
            return matrix;
        }

        float x = rx / theta;
        float y = ry / theta;
        float z = rz / theta;
        float c = Mathf.Cos(theta);
        float s = Mathf.Sin(theta);
        float oneMinusC = 1.0f - c;

        matrix.m00 = c + x * x * oneMinusC;
        matrix.m01 = x * y * oneMinusC - z * s;
        matrix.m02 = x * z * oneMinusC + y * s;
        matrix.m10 = y * x * oneMinusC + z * s;
        matrix.m11 = c + y * y * oneMinusC;
        matrix.m12 = y * z * oneMinusC - x * s;
        matrix.m20 = z * x * oneMinusC - y * s;
        matrix.m21 = z * y * oneMinusC + x * s;
        matrix.m22 = c + z * z * oneMinusC;
        return matrix;
    }

    private static Matrix4x4 FlipY(Matrix4x4 source)
    {
        Matrix4x4 result = Matrix4x4.identity;
        result.m00 = source.m00;
        result.m01 = -source.m01;
        result.m02 = source.m02;
        result.m10 = -source.m10;
        result.m11 = source.m11;
        result.m12 = -source.m12;
        result.m20 = source.m20;
        result.m21 = -source.m21;
        result.m22 = source.m22;
        return result;
    }

    private static Quaternion QuaternionFromMatrix(Matrix4x4 matrix)
    {
        Vector3 forward = new Vector3(matrix.m02, matrix.m12, matrix.m22);
        Vector3 upwards = new Vector3(matrix.m01, matrix.m11, matrix.m21);
        if (forward.sqrMagnitude < 1e-8f || upwards.sqrMagnitude < 1e-8f)
        {
            return Quaternion.identity;
        }
        return Quaternion.LookRotation(forward, upwards);
    }

    private void OnGUI()
    {
        if (!showDebugOverlay)
        {
            return;
        }

        PosePacket pose;
        lock (poseLock)
        {
            pose = latestPose;
        }

        string state = pose == null
            ? "waiting"
            : string.Format(
                CultureInfo.InvariantCulture,
                "{0} ids={1} err={2:0.00}",
                pose.pose_state,
                pose.used_ids == null ? 0 : pose.used_ids.Length,
                pose.mean_error);
        GUI.Label(new Rect(8, 8, 420, 24), "ArUco UDP pose: " + state);

        Transform reference = incomingPoseSpace == PoseSpace.ReferenceLocalToWorld
            ? ResolvePoseReference()
            : null;
        string referenceName = reference == null ? "none" : reference.name;

        GUI.Label(
            new Rect(8, 32, 760, 24),
            string.Format(
                CultureInfo.InvariantCulture,
                "space={0} ref={1} age={2:0.000}s pos={3} rot={4}",
                incomingPoseSpace,
                referenceName,
                lastPoseAgeSeconds,
                lastConvertedPosition.ToString("F3"),
                lastConvertedRotation.eulerAngles.ToString("F1")));
    }
}

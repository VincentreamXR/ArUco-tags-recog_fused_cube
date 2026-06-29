using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public struct NativeArucoResult
{
    public bool valid;
    public int markerId;
    public float meanError;
    public Vector3 rvec;
    public Vector3 tvec;
}

public static class NativeArucoPlugin
{
    public const string PluginMissingStatus = "native plugin missing";
    public const string NoMarkerStatus = "no marker";
    public const string InvalidFrameStatus = "invalid frame";
    public const string NotInitializedStatus = "native plugin not initialized";

    private const string PluginName = "mrbadminton_aruco";

    private static readonly Dictionary<string, int> DictionaryIds = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
    {
        { "DICT_4X4_50", 0 },
        { "DICT_4X4_100", 1 },
        { "DICT_4X4_250", 2 },
        { "DICT_4X4_1000", 3 },
        { "DICT_5X5_50", 4 },
        { "DICT_5X5_100", 5 },
        { "DICT_5X5_250", 6 },
        { "DICT_5X5_1000", 7 },
        { "DICT_6X6_50", 8 },
        { "DICT_6X6_100", 9 },
        { "DICT_6X6_250", 10 },
        { "DICT_6X6_1000", 11 },
        { "DICT_7X7_50", 12 },
        { "DICT_7X7_100", 13 },
        { "DICT_7X7_250", 14 },
        { "DICT_7X7_1000", 15 },
        { "DICT_ARUCO_ORIGINAL", 16 }
    };

#if UNITY_ANDROID && !UNITY_EDITOR
    private static bool initialized;

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int MrbAruco_Init(int dictionaryId, int markerId, float markerLengthMeters);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int MrbAruco_DetectGray(
        byte[] gray,
        int width,
        int height,
        int rowStride,
        float fx,
        float fy,
        float cx,
        float cy,
        [Out] float[] outRvec3,
        [Out] float[] outTvec3,
        [Out] float[] outMeanError,
        [Out] int[] outDetectedId);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void MrbAruco_Shutdown();
#endif

    public static bool TryGetDictionaryId(string dictionaryName, out int dictionaryId)
    {
        dictionaryId = -1;
        if (string.IsNullOrWhiteSpace(dictionaryName))
        {
            return false;
        }

        return DictionaryIds.TryGetValue(dictionaryName.Trim(), out dictionaryId);
    }

    public static bool Configure(
        string dictionaryName,
        int markerId,
        float markerLengthMeters,
        out string status)
    {
        status = string.Empty;
        if (!TryGetDictionaryId(dictionaryName, out int dictionaryId))
        {
            status = "unsupported dictionary";
            return false;
        }

        if (markerId < 0)
        {
            status = "invalid marker id";
            return false;
        }

        if (markerLengthMeters <= 0.0f || float.IsNaN(markerLengthMeters) || float.IsInfinity(markerLengthMeters))
        {
            status = "invalid marker length";
            return false;
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            int code = MrbAruco_Init(dictionaryId, markerId, markerLengthMeters);
            initialized = code >= 0;
            status = code >= 0 ? "native ready" : "native init error " + code.ToString();
            return initialized;
        }
        catch (DllNotFoundException)
        {
            initialized = false;
            status = PluginMissingStatus;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            initialized = false;
            status = PluginMissingStatus;
            return false;
        }
#else
        status = PluginMissingStatus;
        return false;
#endif
    }

    public static bool Detect(ArucoGrayFrame frame, out NativeArucoResult result, out string status)
    {
        result = default;
        status = string.Empty;

        if (!IsUsableFrame(frame))
        {
            status = InvalidFrameStatus;
            return false;
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!initialized)
        {
            status = NotInitializedStatus;
            return false;
        }

        var rvec = new float[3];
        var tvec = new float[3];
        var meanError = new float[1];
        var detectedId = new int[1];

        try
        {
            int code = MrbAruco_DetectGray(
                frame.grayBytes,
                frame.width,
                frame.height,
                frame.rowStride,
                frame.fx,
                frame.fy,
                frame.cx,
                frame.cy,
                rvec,
                tvec,
                meanError,
                detectedId);

            if (code == 1)
            {
                result = new NativeArucoResult
                {
                    valid = true,
                    markerId = detectedId[0],
                    meanError = meanError[0],
                    rvec = new Vector3(rvec[0], rvec[1], rvec[2]),
                    tvec = new Vector3(tvec[0], tvec[1], tvec[2])
                };
                status = "track";
                return true;
            }

            status = code == 0 ? NoMarkerStatus : "native detect error " + code.ToString();
            return false;
        }
        catch (DllNotFoundException)
        {
            initialized = false;
            status = PluginMissingStatus;
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            initialized = false;
            status = PluginMissingStatus;
            return false;
        }
#else
        status = PluginMissingStatus;
        return false;
#endif
    }

    public static bool TryCreatePosePacket(NativeArucoResult result, out ArucoPosePacket packet)
    {
        packet = null;
        if (!result.valid || !IsFinite(result.rvec) || !IsFinite(result.tvec))
        {
            return false;
        }

        packet = new ArucoPosePacket
        {
            valid = true,
            timestamp = Time.realtimeSinceStartupAsDouble,
            rvec = new[] { result.rvec.x, result.rvec.y, result.rvec.z },
            tvec = new[] { result.tvec.x, result.tvec.y, result.tvec.z },
            used_ids = new[] { result.markerId },
            mean_error = result.meanError,
            pose_state = "native"
        };
        return true;
    }

    public static void Shutdown()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!initialized)
        {
            return;
        }

        MrbAruco_Shutdown();
        initialized = false;
#endif
    }

    private static bool IsUsableFrame(ArucoGrayFrame frame)
    {
        return frame.width > 0
            && frame.height > 0
            && frame.rowStride >= frame.width
            && frame.grayBytes != null
            && frame.grayBytes.Length >= frame.rowStride * frame.height
            && IsFinite(frame.fx)
            && IsFinite(frame.fy)
            && IsFinite(frame.cx)
            && IsFinite(frame.cy)
            && frame.fx > 0.0f
            && frame.fy > 0.0f;
    }

    private static bool IsFinite(Vector3 value)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }
}

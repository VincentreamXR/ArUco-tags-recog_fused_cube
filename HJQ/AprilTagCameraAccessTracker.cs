using TMPro;
using UnityEngine;

public class AprilTagCameraAccessTracker : MonoBehaviour
{
    [SerializeField] private TextMeshPro infoText;
    [SerializeField] private bool showDisabledMessage = true;

    private void Start()
    {
        if (infoText == null)
        {
            infoText = GameObject.Find("InfoText")?.GetComponent<TextMeshPro>();
        }

        if (showDisabledMessage && infoText != null)
        {
            infoText.text = "AprilTag tracker disabled\nArUco UDP driver active";
        }

        Debug.Log("[ARUCO_PADDLE] AprilTagCameraAccessTracker is disabled; ArUco UDP driver replaces Android AprilTag tracking.");
    }
}

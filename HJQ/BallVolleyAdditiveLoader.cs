using UnityEngine;
using UnityEngine.SceneManagement;

public class BallVolleyAdditiveLoader : MonoBehaviour
{
    [SerializeField] private string scenePath = "Assets/Scenes/BallVolleyScene.unity";
    [SerializeField] private string rootObjectName = "BallVolleyRoot";
    [SerializeField] private string playerPaddleName = "PaddleB";
    [SerializeField] private string targetObjectName = "PaddleTarget";
    [SerializeField] private string floorReferenceName = "CourtSurface";
    [SerializeField] private string launcherObjectName = "LauncherRig";
    [SerializeField] private float launcherDistanceFromHead = 3.0f;
    [SerializeField] private float floorWorldY = 0.0f;
    [SerializeField] private float cameraResolveTimeoutSeconds = 4.0f;
    [SerializeField] private bool placePlayerPaddleStandbyWhenUntracked = true;
    [SerializeField] private float playerPaddleStandbyDistanceFromHead = 0.75f;
    [SerializeField] private float playerPaddleStandbyHeightBelowHead = 0.28f;
    [SerializeField] private bool waitForAarTrackingBeforeFirstServe = true;
    [SerializeField] private float firstServeDelayAfterTracking = 0.35f;
    [SerializeField] private bool useDebugCubePaddles = true;
    [SerializeField] private Vector3 debugPaddleCubeScale = new Vector3(0.15f, 0.15f, 0.15f);
    [SerializeField] private Vector3 debugPlayerTagCubeScale = new Vector3(0.05f, 0.05f, 0.05f);
    [SerializeField] private float debugPlayerTagFaceSize = 0.026f;
    [SerializeField] private float debugPlayerTagFaceThickness = 0.002f;
    [SerializeField] private bool createDebugBall = false;
    [SerializeField] private float debugBallDiameter = 0.22f;

    private void Start()
    {
        StartCoroutine(LoadAndBindScene());
    }

    private System.Collections.IEnumerator LoadAndBindScene()
    {
        Scene loadedScene = GetLoadedScene(scenePath);
        if (!loadedScene.IsValid())
        {
            Debug.Log("[BALL_VOLLEY] loading additive scene: " + scenePath);
            AsyncOperation operation = SceneManager.LoadSceneAsync(scenePath, LoadSceneMode.Additive);
            if (operation == null)
            {
                Debug.LogError("[BALL_VOLLEY] LoadSceneAsync returned null. Check Build Settings path: " + scenePath);
                yield break;
            }

            while (!operation.isDone)
            {
                yield return null;
            }

            loadedScene = GetLoadedScene(scenePath);
        }

        if (!loadedScene.IsValid())
        {
            Debug.LogError("[BALL_VOLLEY] additive scene is still invalid after load: " + scenePath);
            yield break;
        }

        SetSceneAutoServe(loadedScene, false);
        yield return BindSceneWhenCameraReady(loadedScene);
    }

    private System.Collections.IEnumerator BindSceneWhenCameraReady(Scene loadedScene)
    {
        Camera xrCamera = null;
        float deadline = Time.realtimeSinceStartup + cameraResolveTimeoutSeconds;
        while (xrCamera == null && Time.realtimeSinceStartup < deadline)
        {
            xrCamera = ResolveMainCamera();
            if (xrCamera == null)
            {
                yield return null;
            }
        }

        Transform root = FindInScene(loadedScene, rootObjectName);
        Transform paddleB = FindInScene(loadedScene, playerPaddleName);
        Transform target = FindInScene(loadedScene, targetObjectName);
        Transform floorReference = FindInScene(loadedScene, floorReferenceName) ?? FindInScene(loadedScene, "Floor");
        Transform launcher = FindInScene(loadedScene, launcherObjectName);

        if (root == null)
        {
            Debug.LogError("[BALL_VOLLEY] missing root object: " + rootObjectName);
            yield break;
        }

        root.gameObject.SetActive(false);
        SetActiveIfNotNull(target, false);
        SetActiveIfNotNull(paddleB, false);

        if (xrCamera != null)
        {
            PlaceRootOnFloorFromLauncher(root, launcher, floorReference, xrCamera.transform);
        }

        DetachWorldDrivenPlayerPaddle(root, target, paddleB);
        ReparentIfNeeded(FindInScene(loadedScene, "PaddleA"), root);

        if (xrCamera != null)
        {
            PlacePlayerPaddleStandby(target, paddleB, xrCamera.transform);
        }

        if (useDebugCubePaddles)
        {
            EnsurePaddleCubeVisible(paddleB, "PaddleB", new Color(0.1f, 0.85f, 1.0f, 1.0f), debugPlayerTagCubeScale, true);
            EnsurePaddleCubeVisible(FindInScene(loadedScene, "PaddleA"), "PaddleA", new Color(1.0f, 0.35f, 0.12f, 1.0f), debugPaddleCubeScale, false);
        }
        else
        {
            EnsureRacketModelVisible(paddleB, "PaddleB");
            EnsureRacketModelVisible(FindInScene(loadedScene, "PaddleA"), "PaddleA");
        }

        EnsureDebugBall(loadedScene, launcher, target, xrCamera);
        AarPaddleTargetDriver aarDriver = BindAarDriver(loadedScene, xrCamera, target, paddleB);
        SetActiveIfNotNull(target, true);
        SetActiveIfNotNull(paddleB, true);
        root.gameObject.SetActive(true);
        yield return EnableAutoServeWhenReady(loadedScene, aarDriver, xrCamera != null);

        Debug.LogFormat(
            "[BALL_VOLLEY] scene ready. camera={0} root={1} paddleB={2} target={3} paddleA={4}",
            xrCamera != null ? xrCamera.name : "NULL",
            root.position,
            paddleB != null ? paddleB.position.ToString("F3") : "NULL",
            target != null ? target.position.ToString("F3") : "NULL",
            FindInScene(loadedScene, "PaddleA") != null ? "YES" : "NO");
    }

    private static void SetActiveIfNotNull(Transform target, bool active)
    {
        if (target != null)
        {
            target.gameObject.SetActive(active);
        }
    }

    private System.Collections.IEnumerator EnableAutoServeWhenReady(Scene scene, AarPaddleTargetDriver aarDriver, bool hasCamera)
    {
        if (!hasCamera)
        {
            SetSceneAutoServe(scene, true, 1.0f);
            yield break;
        }

        if (waitForAarTrackingBeforeFirstServe && aarDriver != null)
        {
            while (!aarDriver.HasRecentTracking)
            {
                yield return null;
            }
        }

        SetSceneAutoServe(scene, true, firstServeDelayAfterTracking);
    }

    private static void SetSceneAutoServe(Scene scene, bool enabled, float restartDelay = -1f)
    {
        Transform launcher = FindInScene(scene, "LauncherRig");
        BallServeLauncher serveLauncher = launcher != null
            ? launcher.GetComponent<BallServeLauncher>()
            : FindObjectOfType<BallServeLauncher>();

        if (serveLauncher == null)
        {
            return;
        }

        // 瀛愬満鏅繕娌℃寜 XR 鐩告満鎽嗕綅鍓嶅厛鏆傚仠鍙戠悆锛岄伩鍏嶉鎵圭悆鍦ㄦ棫鍧愭爣涓敓鎴愬悗绔嬪埢鍒ゅ畾 miss銆?
        serveLauncher.SetAutoServe(enabled, restartDelay);
    }

    private static Scene GetLoadedScene(string path)
    {
        for (int i = 0; i < SceneManager.sceneCount; i++)
        {
            Scene scene = SceneManager.GetSceneAt(i);
            if (scene.path == path)
            {
                return scene;
            }
        }

        return default;
    }

    private static Camera ResolveMainCamera()
    {
        if (Camera.main != null)
        {
            return Camera.main;
        }

        Camera[] cameras = FindObjectsOfType<Camera>();
        for (int i = 0; i < cameras.Length; i++)
        {
            if (cameras[i] != null && cameras[i].isActiveAndEnabled)
            {
                return cameras[i];
            }
        }

        return null;
    }

    private static Transform FindInScene(Scene scene, string objectName)
    {
        if (!scene.IsValid() || string.IsNullOrEmpty(objectName))
        {
            return null;
        }

        GameObject[] roots = scene.GetRootGameObjects();
        for (int i = 0; i < roots.Length; i++)
        {
            Transform found = FindChildRecursive(roots[i].transform, objectName);
            if (found != null)
            {
                return found;
            }
        }

        return null;
    }

    private static Transform FindChildRecursive(Transform current, string objectName)
    {
        if (current.name == objectName)
        {
            return current;
        }

        for (int i = 0; i < current.childCount; i++)
        {
            Transform found = FindChildRecursive(current.GetChild(i), objectName);
            if (found != null)
            {
                return found;
            }
        }

        return null;
    }

    private static void ReparentIfNeeded(Transform child, Transform root)
    {
        if (child == null || root == null || child == root || child.IsChildOf(root))
        {
            return;
        }

        child.SetParent(root, true);
    }

    private static void DetachWorldDrivenPlayerPaddle(Transform root, Transform target, Transform paddleB)
    {
        DetachFromRootPreservingWorld(target, root, "PaddleTarget");
        DetachFromRootPreservingWorld(paddleB, root, "PaddleB");
    }

    private static void DetachFromRootPreservingWorld(Transform child, Transform root, string objectName)
    {
        if (child == null || root == null || child == root || !child.IsChildOf(root))
        {
            return;
        }

        child.SetParent(null, true);
        Debug.Log("[BALL_VOLLEY] detached world-driven object from floor root: " + objectName);
    }

    private void PlaceRootOnFloorFromLauncher(Transform root, Transform launcher, Transform floorReference, Transform head)
    {
        Vector3 forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 0.0001f)
        {
            forward = Vector3.forward;
        }

        Vector3 rootPosition = root.position;
        Vector3 desiredLauncherPosition = head.position + forward.normalized * launcherDistanceFromHead;

        // A 鐞冩媿鍜屽彂鐞冩満淇濇寔鍦ㄧ悆鍦?鍦版澘鍙傝€冪郴涓紱B 鐞冩媿鍙敱 AAR 鐨?world pose 椹卞姩銆?
        root.rotation = Quaternion.identity;
        rootPosition.x = launcher != null
            ? desiredLauncherPosition.x - launcher.localPosition.x
            : desiredLauncherPosition.x;
        rootPosition.z = launcher != null
            ? desiredLauncherPosition.z - launcher.localPosition.z
            : desiredLauncherPosition.z;
        rootPosition.y = floorReference != null
            ? floorWorldY - floorReference.localPosition.y
            : floorWorldY;

        root.position = rootPosition;
    }

    private void PlacePlayerPaddleStandby(Transform target, Transform paddleB, Transform head)
    {
        if (!placePlayerPaddleStandbyWhenUntracked || target == null || head == null)
        {
            return;
        }

        Vector3 forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 0.0001f)
        {
            forward = Vector3.forward;
        }

        forward.Normalize();
        Vector3 standbyPosition = head.position + forward * playerPaddleStandbyDistanceFromHead;
        standbyPosition.y = Mathf.Max(floorWorldY + 1.0f, head.position.y - playerPaddleStandbyHeightBelowHead);
        Quaternion standbyRotation = Quaternion.LookRotation(forward, Vector3.up);

        target.SetPositionAndRotation(standbyPosition, standbyRotation);

        if (paddleB == null)
        {
            return;
        }

        paddleB.SetPositionAndRotation(standbyPosition, standbyRotation);
        if (paddleB.TryGetComponent<Rigidbody>(out var body))
        {
            body.position = standbyPosition;
            body.rotation = standbyRotation;
            body.velocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
        }
    }

    private static AarPaddleTargetDriver BindAarDriver(Scene scene, Camera xrCamera, Transform target, Transform paddleB)
    {
        Transform targetTransform = target != null ? target : FindInScene(scene, "PaddleTarget");
        AarPaddleTargetDriver driver = targetTransform != null
            ? targetTransform.GetComponent<AarPaddleTargetDriver>()
            : FindObjectOfType<AarPaddleTargetDriver>();

        if (driver == null)
        {
            Debug.LogWarning("[BALL_VOLLEY] AarPaddleTargetDriver not found.");
            return null;
        }

        driver.BindRuntimeReferences(
            xrCamera != null ? xrCamera.transform : null,
            targetTransform,
            paddleB,
            GameObject.Find("InfoText")?.GetComponent<TMPro.TextMeshPro>());
        return driver;
    }

    private void EnsurePaddleCubeVisible(Transform paddle, string paddleName, Color color, Vector3 cubeScale, bool showTagFace)
    {
        if (paddle == null)
        {
            Debug.LogWarning("[BALL_VOLLEY] " + paddleName + " not found. Cannot draw debug cube.");
            return;
        }

        paddle.gameObject.SetActive(true);

        Renderer[] existingRenderers = paddle.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < existingRenderers.Length; i++)
        {
            if (existingRenderers[i] != null)
            {
                existingRenderers[i].enabled = false;
            }
        }

        Transform cube = FindChildRecursive(paddle, "DebugPaddleCube");
        if (cube == null)
        {
            GameObject cubeObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cubeObject.name = "DebugPaddleCube";
            cubeObject.transform.SetParent(paddle, false);
            cube = cubeObject.transform;
        }

        cube.gameObject.SetActive(true);
        cube.localPosition = Vector3.zero;
        cube.localRotation = Quaternion.identity;
        cube.localScale = cubeScale;

        Renderer cubeRenderer = cube.GetComponent<Renderer>();
        if (cubeRenderer != null)
        {
            cubeRenderer.enabled = true;
            cubeRenderer.sharedMaterial = CreateDebugMaterial(paddleName + "_DebugCubeMaterial", color);
        }

        BoxCollider boxCollider = cube.GetComponent<BoxCollider>();
        if (boxCollider == null)
        {
            boxCollider = cube.gameObject.AddComponent<BoxCollider>();
        }

        boxCollider.enabled = true;
        EnsureTagFaceMarker(paddle, paddleName, cubeScale, showTagFace);
        Debug.Log("[BALL_VOLLEY] " + paddleName + " debug cube visible.");
    }

    private void EnsureTagFaceMarker(Transform paddle, string paddleName, Vector3 cubeScale, bool visible)
    {
        Transform face = FindChildRecursive(paddle, "DebugTagFace");
        if (!visible)
        {
            if (face != null)
            {
                face.gameObject.SetActive(false);
            }

            return;
        }

        if (face == null)
        {
            GameObject faceObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            faceObject.name = "DebugTagFace";
            faceObject.transform.SetParent(paddle, false);
            face = faceObject.transform;

            Collider collider = faceObject.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
        }

        float faceSize = Mathf.Clamp(debugPlayerTagFaceSize, 0.001f, Mathf.Min(cubeScale.x, cubeScale.y));
        float faceThickness = Mathf.Clamp(debugPlayerTagFaceThickness, 0.0005f, Mathf.Max(0.0005f, cubeScale.z * 0.2f));

        face.gameObject.SetActive(true);
        face.localPosition = new Vector3(0.0f, 0.0f, -cubeScale.z * 0.51f);
        face.localRotation = Quaternion.identity;
        face.localScale = new Vector3(faceSize, faceSize, faceThickness);

        Renderer faceRenderer = face.GetComponent<Renderer>();
        if (faceRenderer != null)
        {
            faceRenderer.enabled = true;
            faceRenderer.sharedMaterial = CreateDebugMaterial(paddleName + "_DebugTagFaceMaterial", new Color(0.02f, 0.02f, 0.02f, 1.0f));
        }
    }

    private void EnsureDebugBall(Scene scene, Transform launcher, Transform target, Camera xrCamera)
    {
        if (!createDebugBall)
        {
            Transform existingBall = FindInScene(scene, "DebugVolleyBall");
            if (existingBall != null)
            {
                Destroy(existingBall.gameObject);
            }

            return;
        }

        Transform ball = FindInScene(scene, "DebugVolleyBall");
        if (ball == null)
        {
            GameObject ballObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            ballObject.name = "DebugVolleyBall";
            SceneManager.MoveGameObjectToScene(ballObject, scene);
            ball = ballObject.transform;

            Collider collider = ballObject.GetComponent<Collider>();
            if (collider != null)
            {
                collider.enabled = false;
                Destroy(collider);
            }
        }

        Vector3 ballPosition;
        if (launcher != null && target != null)
        {
            ballPosition = Vector3.Lerp(launcher.position, target.position, 0.42f);
            ballPosition.y = Mathf.Max(ballPosition.y, floorWorldY + 1.15f);
        }
        else if (xrCamera != null)
        {
            Vector3 forward = Vector3.ProjectOnPlane(xrCamera.transform.forward, Vector3.up);
            if (forward.sqrMagnitude < 0.0001f)
            {
                forward = Vector3.forward;
            }

            ballPosition = xrCamera.transform.position + forward.normalized * 1.25f;
            ballPosition.y = Mathf.Max(floorWorldY + 1.15f, xrCamera.transform.position.y - 0.25f);
        }
        else
        {
            ballPosition = new Vector3(0.0f, 1.25f, 1.2f);
        }

        ball.SetPositionAndRotation(ballPosition, Quaternion.identity);
        ball.localScale = Vector3.one * debugBallDiameter;
        ball.gameObject.SetActive(true);

        Renderer renderer = ball.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.enabled = true;
            renderer.sharedMaterial = CreateDebugMaterial("DebugVolleyBallMaterial", new Color(1.0f, 0.85f, 0.05f, 1.0f));
        }

        Debug.Log("[BALL_VOLLEY] debug ball visible at " + ball.position.ToString("F3"));
    }

    private static Material CreateDebugMaterial(string materialName, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }

        return new Material(shader)
        {
            name = materialName,
            color = color
        };
    }

    private static void EnsureRacketModelVisible(Transform paddle, string paddleName)
    {
        if (paddle == null)
        {
            return;
        }

        paddle.gameObject.SetActive(true);
        Transform racketModel = FindChildRecursive(paddle, "PaddleModel");
        if (racketModel == null)
        {
            Debug.LogWarning("[BALL_VOLLEY] " + paddleName + " is missing PaddleModel from tennis_racket.glb.");
            return;
        }

        racketModel.gameObject.SetActive(true);

        Renderer[] renderers = racketModel.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i] != null)
            {
                renderers[i].gameObject.SetActive(true);
                renderers[i].enabled = true;
            }
        }

        if (renderers.Length == 0)
        {
            Debug.LogWarning("[BALL_VOLLEY] " + paddleName + " PaddleModel has no Renderer. Check tennis_racket.glb import.");
        }
    }
}

using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;

public static class CodexAndroidBuild
{
    public static void BuildApk()
    {
        string outputDirectory = Path.Combine(Environment.CurrentDirectory, "Builds", "CodexAndroid");
        Directory.CreateDirectory(outputDirectory);

        string outputPath = Path.Combine(outputDirectory, PlayerSettings.productName + ".apk");
        string[] scenes = EditorBuildSettings.scenes
            .Where(scene => scene.enabled)
            .Select(scene => scene.path)
            .ToArray();

        if (scenes.Length == 0)
        {
            throw new InvalidOperationException("No enabled scenes found in Build Settings.");
        }

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = outputPath,
            target = BuildTarget.Android,
            options = BuildOptions.None
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;
        if (summary.result != BuildResult.Succeeded)
        {
            throw new InvalidOperationException(
                $"Android build failed: result={summary.result}, errors={summary.totalErrors}, warnings={summary.totalWarnings}");
        }

        UnityEngine.Debug.Log($"Android build succeeded: {outputPath}");
    }
}

using System.IO;
using UnityEditor.Android;

public sealed class CodexAndroidGradlePostprocessor : IPostGenerateGradleAndroidProject
{
    public int callbackOrder => 10000;

    public void OnPostGenerateGradleAndroidProject(string unityLibraryPath)
    {
        string gradleRoot = Directory.GetParent(unityLibraryPath)?.FullName;
        if (string.IsNullOrEmpty(gradleRoot))
        {
            return;
        }

        RemoveLine(Path.Combine(gradleRoot, "launcher", "build.gradle"), "apply plugin: 'kotlin-android'");
        RemoveLine(Path.Combine(gradleRoot, "build.gradle"), "id 'org.jetbrains.kotlin.jvm' version '1.6.10' apply false");
        RemoveLine(Path.Combine(gradleRoot, "build.gradle"), "id 'org.jetbrains.kotlin.android' version '1.6.10' apply false");
    }

    private static void RemoveLine(string path, string lineToRemove)
    {
        if (!File.Exists(path))
        {
            return;
        }

        string text = File.ReadAllText(path);
        string updated = text
            .Replace(lineToRemove + "\r\n", string.Empty)
            .Replace(lineToRemove + "\n", string.Empty);

        if (updated != text)
        {
            File.WriteAllText(path, updated);
        }
    }
}

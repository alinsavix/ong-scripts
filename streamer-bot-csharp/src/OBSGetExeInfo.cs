using System;
using System.Diagnostics;
using System.Text.RegularExpressions;

#if EXTERNAL_EDITOR
public class OBSGetExeInfo : CPHInlineBase
#else
public class CPHInline
#endif
{
    public bool Execute()
    {
        Process[] processes = Process.GetProcessesByName("obs64");
        if (processes.Length == 0)
        {
            CPH.LogWarn("OBSGetExeInfo: obs64.exe is not running.");
            return false;
        }

        if (processes.Length > 1)
        {
            CPH.LogWarn($"OBSGetExeInfo: Multiple obs64.exe processes found ({processes.Length}).");
            return false;
        }

        string exePath = processes[0].MainModule.FileName;
        CPH.LogInfo($"OBSGetExeInfo: Found obs64.exe at '{exePath}'");

        // Expected path structure: <drive>\...\OBS <version>\bin\64bit\obs64.exe
        Match pathMatch = Regex.Match(exePath,
            @"^(?<topLevel>.+\\[^\\]*?(?<version>\d+\.\d+\.\d+(?:-[^\\]+)?)[^\\]*)\\bin\\64bit\\obs64\.exe$",
            RegexOptions.IgnoreCase);
        if (!pathMatch.Success)
        {
            CPH.LogWarn($"OBSGetExeInfo: Unexpected path structure '{exePath}', expected '...\\<name with version>\\bin\\64bit\\obs64.exe'.");
            return false;
        }

        string topLevelPath = pathMatch.Groups["topLevel"].Value;
        string version = pathMatch.Groups["version"].Value;

        if (string.IsNullOrEmpty(version))
        {
            CPH.LogWarn($"OBSGetExeInfo: Could not parse version from path '{exePath}'.");
        }

        CPH.SetArgument("obsExePath", exePath);
        CPH.SetArgument("obsTopLevelPath", topLevelPath);
        CPH.SetArgument("obsVersion", version);

        CPH.LogInfo($"OBSGetExeInfo: topLevel='{topLevelPath}', evePath='{exePath}', version='{version}'");
        return true;
    }
}

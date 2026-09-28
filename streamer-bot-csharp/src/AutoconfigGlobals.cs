using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Threading;

#if EXTERNAL_EDITOR
public class AutoconfigGlobals : CPHInlineBase
#else
public class CPHInline
#endif
{
    private static readonly Regex SecretNamePattern =
        new Regex("apikey|password|secret", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public bool Execute()
    {
        if (!CPH.TryGetArg("configTarget", out string configTarget))
        {
            CPH.LogError("AutoconfigGlobals: Failed to get 'configTarget' argument.");
            return false;
        }

        if (string.IsNullOrWhiteSpace(configTarget))
        {
            CPH.LogError("AutoconfigGlobals: 'configTarget' argument is empty.");
            return false;
        }

        CPH.LogInfo($"AutoconfigGlobals: Switching configuration to '{configTarget}'");

        var allGlobals = CPH.GetGlobalVarValues();
        if (allGlobals == null || allGlobals.Count == 0)
        {
            CPH.LogWarn("AutoconfigGlobals: No global variables found.");
            return true;
        }

        string test_suffix = "." + configTarget.ToLower();
        int copiedCount = 0;
        var changedNames = new List<string>();

        var currentValues = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        foreach (var globalVar in allGlobals)
        {
            currentValues[globalVar.VariableName] = globalVar.Value;
        }

        foreach (var globalVar in allGlobals)
        {
            string varName = globalVar.VariableName;
            string lowerVarName = varName.ToLower();
            // CPH.LogInfo($"Checking global {varName}");

            if (lowerVarName.EndsWith(test_suffix))
            {
                // CPH.LogInfo("Global matched test suffix");
                // Extract the base name (without the user-specific suffix)
                string baseName = varName.Substring(0, varName.Length - test_suffix.Length);

                // Use the Value property directly from the GlobalVariableValue object
                object value = globalVar.Value;
                object? oldValue;
                if (!currentValues.TryGetValue(baseName, out oldValue))
                {
                    oldValue = null;
                }

                bool unchanged = Equals(oldValue, value);
                bool isSecret = SecretNamePattern.IsMatch(varName);

                CPH.SetGlobalVar(baseName, value, true);

                CPH.LogInfo(
                    $"AutoconfigGlobals: Copied '{varName}' -> '{baseName}': " +
                    $"current='{FormatValue(oldValue, isSecret)}', new='{FormatValue(value, isSecret)}'" +
                    (unchanged ? " (unchanged)" : "")
                );
                copiedCount++;

                if (!unchanged)
                {
                    changedNames.Add(baseName);
                }
            }
        }

        if (copiedCount > 0)
        {
            CPH.LogInfo($"AutoconfigGlobals: Successfully copied {copiedCount} configuration variable(s) for '{configTarget}'");

            if (changedNames.Count > 0)
            {
                CPH.LogInfo($"AutoconfigGlobals: Waiting 15 seconds before re-checking {changedNames.Count} changed variable(s)");
                Thread.Sleep(15000);

                var refreshed = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                var refreshedGlobals = CPH.GetGlobalVarValues();
                if (refreshedGlobals != null)
                {
                    foreach (var globalVar in refreshedGlobals)
                    {
                        refreshed[globalVar.VariableName] = globalVar.Value;
                    }
                }

                foreach (string name in changedNames)
                {
                    object? currentValue;
                    if (!refreshed.TryGetValue(name, out currentValue))
                    {
                        currentValue = null;
                    }

                    CPH.LogInfo(
                        $"AutoconfigGlobals: RECHECK '{name}': " +
                        $"value='{FormatValue(currentValue, SecretNamePattern.IsMatch(name))}'"
                    );
                }
            }

            return true;
        } else {
            CPH.LogWarn($"AutoconfigGlobals: No configuration variables found for target '{configTarget}'");

            CPH.ShowToastNotification(
                "AUTOCONFIG ERROR",
                "Failed to automatically configure for this host " +
                $"(target: '{configTarget}'), many stream features will be broken.",
                "streamer.bot", "icons/ongPanic.png"
            );
            return true;
        }
    }

    private static string FormatValue(object? value, bool isSecret)
    {
        if (value == null)
        {
            return "<null>";
        }

        string text = value.ToString();
        if (!isSecret)
        {
            return text;
        }

        if (text.Length <= 6)
        {
            return new string('*', text.Length);
        }

        return text.Substring(0, 6) + new string('*', text.Length - 6);
    }
}

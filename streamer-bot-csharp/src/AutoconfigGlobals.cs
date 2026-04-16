using System;
using System.Collections.Generic;

#if EXTERNAL_EDITOR
public class AutoconfigGlobals : CPHInlineBase
#else
public class CPHInline
#endif
{
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
                CPH.SetGlobalVar(baseName, value, true);

                CPH.LogInfo($"AutoconfigGlobals: Copied '{varName}' -> '{baseName}'");
                copiedCount++;
            }
        }

        if (copiedCount > 0)
        {
            CPH.LogInfo($"AutoconfigGlobals: Successfully copied {copiedCount} configuration variable(s) for '{configTarget}'");
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
}

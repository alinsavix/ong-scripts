using System;

#if EXTERNAL_EDITOR
public class OBSWaitMediaDisabled : CPHInlineBase
#else
public class CPHInline
#endif
{
    // FIXME: This snippet always returns true, so that the enclosing action
    // will continue running even if OBS isn't there or responding. This seems
    // like the best options, so that actions can actually clean up and such
    // rather than just... stopping. Opinions welcome.
    public bool Execute()
    {
        if (!CPH.TryGetArg("sceneName", out string sceneName))
        {
            CPH.LogError("WaitMediaDisabled: Failed to get 'sceneName' argument, not waiting.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        if (string.IsNullOrWhiteSpace(sceneName))
        {
            CPH.LogError("WaitMediaDisabled: 'sceneName' argument is empty, not waiting.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        if (!CPH.TryGetArg("sourceName", out string sourceName))
        {
            CPH.LogError("WaitMediaDisabled: Failed to get 'sourceName' argument, not waiting.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        if (string.IsNullOrWhiteSpace(sourceName))
        {
            CPH.LogError("WaitMediaDisabled: 'sourceName' argument is empty, not waiting.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        int connectionIdx = 0;
        if (CPH.TryGetArg("obsConnection", out string obsConnectionName))
        {
            connectionIdx = CPH.ObsGetConnectionByName(obsConnectionName);
        }
        else
        {
            obsConnectionName = "(Default)";
        }

        if (!CPH.ObsIsConnected(connectionIdx))
        {
            CPH.LogError($"WaitMediaDisabled: OBS connection '{obsConnectionName}' is not connected or does not exist.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        // FIXME: Should we try/catch on this to catch invalid numbers?
        int check_interval = 250;
        if (CPH.TryGetArg<int>("checkIntervalMs", out int arg_interval))
        {
            if (arg_interval >= 50)
            {
                check_interval = arg_interval;
            }
        }

        // FIXME: Should we try/catch on this to catch invalid numbers?
        int max_wait = 300 * 1000; // 5 minutes
        if (CPH.TryGetArg<int>("maxWaitMs", out int arg_maxwait))
        {
            if (arg_maxwait >= 1000)
            {
                max_wait = arg_maxwait;
            }
        }


        // For whatever reason, if ObsGetSceneItemProperties fails, it throws
        // an exception, unlike most other streamer.bot OBS functions.
        try
        {
            string props = CPH.ObsGetSceneItemProperties(sceneName, sourceName, connectionIdx);
            // CPH.LogInfo($"WaitMediaDisabled: Props for '{sceneName}::{sourceName}': {props}");
        }
        catch
        {
            CPH.LogError($"WaitMediaDisabled: Couldn't get props for '{sceneName}::{sourceName}', assuming it doesn't exist, not waiting.");
            CPH.SetArgument("waitedTime", -1);
            return true;
        }

        // Wait for the source to become invisible/disabled
        CPH.LogInfo($"WaitMediaDisabled: Waiting on '{sceneName}::{sourceName}', check interval {check_interval}ms, max wait {max_wait}ms");

        // actually do the waiting
        bool isVisible = true;
        int wait_time = 0;
        int message_interval = (1000 / check_interval) * check_interval;

        while (isVisible)
        {
            isVisible = CPH.ObsIsSourceVisible(sceneName, sourceName, connectionIdx);

            if (isVisible)
            {
                wait_time += check_interval;
                if (wait_time >= max_wait)
                {
                    CPH.LogWarn($"WaitMediaDisabled: Timeout waiting for '{sourceName}' to be disabled after {max_wait}ms");
                    CPH.SetArgument("waitedTime", -1);
                    return true;
                }

                if (wait_time % message_interval == 0) // Log every 5 seconds
                {
                    CPH.LogInfo($"WaitMediaDisabled: Still waiting for '{sourceName}' to be disabled (waited {wait_time}ms so far)");
                }
                CPH.Wait(check_interval); // Wait before checking again
            }
        }

        CPH.LogInfo($"WaitMediaDisabled: Source '{sourceName}' went disabled after {wait_time}ms");
        CPH.SetArgument("waitedTime", wait_time);
        return true;
    }
}

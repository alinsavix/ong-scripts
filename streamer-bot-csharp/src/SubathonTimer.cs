using Streamer.bot.Plugin.Interface;
using Streamer.bot.Plugin.Interface.Model;
using Streamer.bot.Plugin.Interface.Enums;
using Streamer.bot.Common.Events;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Timers;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Reflection;
using System.Text;
using Newtonsoft.Json;

#if EXTERNAL_EDITOR
public class SubathonTimer : CPHInlineBase
#else
public class CPHInline
#endif
{
    const string LOG_HEADER = "Subathon Log: ";
    const string PAUSE_SYMBOL = "⏸";
    const string PLAY_SYMBOL = "⏵";
    const string TIME_SEPARATOR = ":";
    const string ZERO_TIME = "00:00:00";

    // Default Global Values
    const bool DEFAULT_DISPLAY_DAYS = false;
    const double DEFAULT_HYPE_TRAIN_INCREMENT = 0.2;
    const string DEFAULT_HYPE_TRAIN_MULTIPLIER_TYPE = "linear";
    const string DEFAULT_HYPE_TRAIN_NAME = "Twitch Hype Train";
    const int DEFAULT_HYPE_TRAIN_MULTIPLIER_MAX = 10;
    const string DEFAULT_INITIAL_DURATION = "2h";
    const string DEFAULT_MAX_DURATION = "7d";
    const int DEFAULT_OBS_CONNECTION = 0;
    const bool DEFAULT_SEND_ALL_PLATFORMS = true;
    const bool DEFAULT_SEND_AS_BOT = true;
    const string DEFAULT_SEND_TO_PLATFORM = "twitch";
    const string DEFAULT_SCENE = "[NS] Subathon";
    const string DEFAULT_TS = "[TS] Subathon Time";
    const bool DEFAULT_USE_TIME_UNITS = false;
    const string DEFAULT_TS_DAYS = "[TS] Subathon Time Days";
    const string DEFAULT_TS_HOURS = "[TS] Subathon Time Hours";
    const string DEFAULT_TS_MINUTES = "[TS] Subathon Time Minutes";
    const string DEFAULT_TS_SECONDS = "[TS] Subathon Time Seconds";
    const string DEFAULT_TS_PLAY_PAUSE = "[TS] Subathon Time Play Pause";
    const string SUBATHON_NAME = "Subathon";
    private System.Timers.Timer countdownTimer;
    private long subathonTimeRemaining = -1;     // Time left on timer
    private long subathonTimeAddedTotal = 0;   // Total elapsed time after adding time
    private long subathonHypeTrainTimeAdded;
    private long subathonHypeTrainExtraTimeAdded;
    private long subathonElapsedTime;
    private string subathonTimeLeftFile;
    private string subathonTimeAddedInTimeFile;
    private string subathonElapsedTimeFile;
    private string countdownString;
    private string countdownStringCap;
    private bool timerOn;
    private bool limitReached;
    private bool messageOnce;
    private bool newSubathonConfirm;
    private bool subathonCancelConfirm;
    private string commandSource;
    private bool sendAsBot = true;
    private bool sendAllPlatforms = true;
    private int obsConnection;
    private Platform currentPlatform;
    private HashSet<Platform> platformList = new HashSet<Platform>();

    public void Init()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        countdownTimer = new System.Timers.Timer(1000);
        countdownTimer.Elapsed += OnTimedEvent;
        countdownTimer.AutoReset = true;
        countdownTimer.Enabled = true;
        countdownTimer.Stop();

        obsConnection = GetObsConnection();

        GetDirectory();      // Set File Directory
        RegisterTriggers();  // Set Custom triggers
        SetDefaultGlobals(); // Set default global variables
    }

    public void Dispose()
    {
        CPH.UnsetGlobalVar("subathonMaxDurationGlobal");
        CPH.UnsetGlobalVar("subathonObsConnectionGlobal");
        countdownTimer.Dispose();
    }

    public bool Execute()
    {
        return true;
    }

    private void RegisterTriggers()
    {
        string[] category = { "Subathon" };
        CPH.RegisterCustomTrigger("Subathon Started", "subathon_started", category);
        CPH.RegisterCustomTrigger("Subathon Paused", "subathon_paused", category);
        CPH.RegisterCustomTrigger("Subathon Resumed", "subathon_resumed", category);
        CPH.RegisterCustomTrigger("Subathon Cancelled", "subathon_cancelled", category);
        CPH.RegisterCustomTrigger("Subathon Time Added", "subathon_time_added", category);
        CPH.RegisterCustomTrigger("Subathon Limit Reached", "subathon_limit_reached", category);
        CPH.RegisterCustomTrigger("Subathon Completed", "subathon_completed", category);
    }

    private void SetDefaultGlobals()
    {
        // Set default global variables
        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonName")))
            CPH.SetGlobalVar("subathonName", SUBATHON_NAME);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonRunning")))
            CPH.SetGlobalVar("subathonRunning", false);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonHypeTrain")))
            CPH.SetGlobalVar("subathonHypeTrain", false);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonDisplayDays")))
            CPH.SetGlobalVar("subathonDisplayDays", DEFAULT_DISPLAY_DAYS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonHypeTrainLevelUpIncrement")))
            CPH.SetGlobalVar("subathonHypeTrainLevelUpIncrement", DEFAULT_HYPE_TRAIN_INCREMENT);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonHypeTrainMultiplier")))
            CPH.SetGlobalVar("subathonHypeTrainMultiplier", DEFAULT_HYPE_TRAIN_MULTIPLIER_TYPE);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonHypeTrainName")))
            CPH.SetGlobalVar("subathonHypeTrainName", DEFAULT_HYPE_TRAIN_NAME);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonHypeTrainMaxMultiplier")))
            CPH.SetGlobalVar("subathonHypeTrainMaxMultiplier", DEFAULT_HYPE_TRAIN_MULTIPLIER_MAX);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonInitialDuration")))
            CPH.SetGlobalVar("subathonInitialDuration", DEFAULT_INITIAL_DURATION);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonMaxDuration")))
            CPH.SetGlobalVar("subathonMaxDuration", DEFAULT_MAX_DURATION);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonObsConnection")))
            CPH.SetGlobalVar("subathonObsConnection", DEFAULT_OBS_CONNECTION);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSendAllPlatforms")))
            CPH.SetGlobalVar("subathonSendAllPlatforms", DEFAULT_SEND_ALL_PLATFORMS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSendAsBot")))
            CPH.SetGlobalVar("subathonSendAsBot", DEFAULT_SEND_AS_BOT);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSendMessageTo")))
            CPH.SetGlobalVar("subathonSendMessageTo", DEFAULT_SEND_TO_PLATFORM);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonScene")))
            CPH.SetGlobalVar("subathonScene", DEFAULT_SCENE);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSource")))
            CPH.SetGlobalVar("subathonSource", DEFAULT_TS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonIndividualTimeUnits")))
            CPH.SetGlobalVar("subathonIndividualTimeUnits", DEFAULT_USE_TIME_UNITS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSourceDays")))
            CPH.SetGlobalVar("subathonSourceDays", DEFAULT_TS_DAYS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSourceHours")))
            CPH.SetGlobalVar("subathonSourceHours", DEFAULT_TS_HOURS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSourceMinutes")))
            CPH.SetGlobalVar("subathonSourceMinutes", DEFAULT_TS_MINUTES);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSourceSeconds")))
            CPH.SetGlobalVar("subathonSourceSeconds", DEFAULT_TS_SECONDS);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonSourcePlayPause")))
            CPH.SetGlobalVar("subathonSourcePlayPause", DEFAULT_TS_PLAY_PAUSE);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonLimitReachedDateTime")))
            CPH.SetGlobalVar("subathonLimitReachedDateTime", string.Empty);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonLimitReached12")))
            CPH.SetGlobalVar("subathonLimitReached12", string.Empty);

        if (string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonLimitReached24")))
            CPH.SetGlobalVar("subathonLimitReached24", string.Empty);

    }

    private void TriggerEventStarted()
    {
        CPH.TriggerCodeEvent("subathon_started", CreateArgsDictionary());
    }

    private void TriggerEventPaused()
    {
        CPH.TriggerCodeEvent("subathon_paused", CreateArgsDictionary() );
    }

    private void TriggerEventResumed()
    {
        CPH.TriggerCodeEvent("subathon_resumed", CreateArgsDictionary() );
    }

    private void TriggerEventCancelled()
    {
        SubathonClear();
        CPH.TriggerCodeEvent("subathon_cancelled", CreateArgsDictionary());
    }

    private void TriggerEventTimeAdded(Dictionary<string, object> newArgs)
    {
        CPH.TriggerCodeEvent("subathon_time_added", newArgs );
    }

    private void TriggerEventLimitReached()
    {
        CPH.TriggerCodeEvent("subathon_limit_reached", CreateArgsDictionary() );
    }

    private void TriggerEventCompleted()
    {
        SubathonClear();
        CPH.TriggerCodeEvent("subathon_completed", CreateArgsDictionary() );
    }

    public bool StartSubathon()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        string subathonName = CPH.GetGlobalVar<string>("subathonName") ?? SUBATHON_NAME;
        obsConnection = GetObsConnection();
        platformList = SetMessagePlatform(platformList);

        // Check if timer is currently running
        if (timerOn)
        {
            string errorMessage = $"Error: {subathonName} timer is currently running.";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }

        // Check if the subathon backup exists
        if (string.IsNullOrEmpty(File.ReadAllText(subathonTimeLeftFile)))
        {
            LogVerbose($"{methodName}Starting new {subathonName}");
            StartSubathonTimer();
            return true;
        }

        if (!newSubathonConfirm)
        {
            if (!CPH.TryGetArg("command", out string command))
                command = "!subathonStart";
            string errorMessage = $"Error: A previous {subathonName} timer exists. To overwrite the previous {subathonName} timer, run {command} again. Otherwise use !subathonResume";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            newSubathonConfirm = true;
            return false;
        }
        else
        {
            LogVerbose($"{methodName}Overwrite confirmed. Starting new {subathonName} timer");
            // If backup doesn't exist, start a new countdown
            newSubathonConfirm = false;
            StartSubathonTimer();
            return true;
        }
    }

    public bool ResumeSubathon()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        obsConnection = GetObsConnection();
        limitReached = false;
        platformList = SetMessagePlatform(platformList);
        string subathonName = CPH.GetGlobalVar<string>("subathonName") ?? SUBATHON_NAME;

        // Check if timer is currently running
        if (timerOn)
        {
            string errorMessage = $"Error: {subathonName} timer is currently running";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }

        try
        {
            if (!TryRestoreBackup(out subathonTimeRemaining, out subathonTimeAddedTotal, out subathonElapsedTime) &&
                 subathonTimeRemaining < 0)
            {
                LogError($"{methodName}Unable to parse backup files or subathonTimeRemaining is bad '{ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(subathonTimeRemaining))}'");
                return false;
            }

            TriggerEventResumed();
            MessageSetArguments();

            // Check if limit has been reached
            if (!string.IsNullOrEmpty(CPH.GetGlobalVar<string>("subathonLimitReachedDateTime") ?? null))
            {
                limitReached = true;
                messageOnce = true;
            }

            LogVerbose($"{methodName}Resuming existing {subathonName}");
            BackupWriteToFile();
            StartTimer();
            return true;
        }
        catch (Exception ex)
        {
            LogError($"{methodName}Exception: {ex.Message}");
            return false;
        }
    }

    public bool SubathonGoalProgress()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        long subathonTimeLimit = GetSubathonLimit();
        BackupWriteToFile();
        double subathonPercent = (double)subathonTimeAddedTotal / subathonTimeLimit * 100;
        LogVerbose($"{methodName}subathonTimeAddedInSeconds '{subathonTimeAddedTotal}' subathonTimeLimitInSeconds '{subathonTimeLimit}' subathonPercent '{subathonPercent}'");
        string subathonPercentString = subathonPercent.ToString("F2");
        CPH.SetArgument("subathonTimeAddedLong", GetTimerStringLong(subathonTimeAddedTotal));
        CPH.SetArgument("subathonTimeAddedShort", GetTimerStringShort(subathonTimeAddedTotal));
        CPH.SetArgument("goalPercent", subathonPercentString);
        CPH.SetArgument("subathonTimeAddedTotalInSeconds", subathonTimeAddedTotal);
        CPH.SetArgument("subathonTimeLimitInSeconds", subathonTimeLimit);
        LogVerbose($"{methodName}subathonPercentString '{subathonPercentString}'");

        MessageSetArguments();
        return true;
    }

    public bool PauseSubathon()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        platformList = SetMessagePlatform(platformList);

        if (!timerOn)
        {
            string errorMessage = $"Error: Timer is not currently running";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }

        // Backup remaining time to file
        BackupWriteToFile();
        StopSubathon($"{PAUSE_SYMBOL}{GetTimerStringShort(subathonTimeRemaining)}");
        CreateArgsDictionary();
        TriggerEventPaused();
        MessageSetArguments();
        return true;
    }

    public bool CancelSubathon()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        platformList = SetMessagePlatform(platformList);
        string subathonName = CPH.GetGlobalVar<string>("subathonName") ?? SUBATHON_NAME;
        if (!CPH.TryGetArg("command", out string command))
            command = "!subathonCancel";

        if (!timerOn)
        {
            string errorMessage = $"Error: Timer is not currently running";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }
        else
        {
            // Ask for confirmation
            if (!subathonCancelConfirm)
            {
                string errorMessage = $"Are you sure you want to cancel the {subathonName} timer? Type {command} again to cancel the {subathonName} timer";
                LogError($"{methodName}{errorMessage}");
                SendMessage(sendAllPlatforms, platformList, errorMessage, false);
                subathonCancelConfirm = true;
                return false;
            }
            // Cancel confirmed
            else
            {
                CPH.SetGlobalVar("subathonRunning", false);
                MessageSetArguments();
                string initialDuration = CPH.GetGlobalVar<string>("subathonInitialDuration") ?? DEFAULT_INITIAL_DURATION;
                TryParseTimeString(initialDuration, out TimeSpan initialTime);
                StopSubathon(GetTimerStringShort((long)initialTime.TotalSeconds));
                UpdateOBSSourceSeparateUnits((long)initialTime.TotalSeconds);
                TriggerEventCancelled();
                CreateArgsDictionary();
                return true;
            }
        }
    }

    private bool TryGetTwitchSubTier(out TimeSpan parsedTime, out bool addTime)
    {
        parsedTime = TimeSpan.Zero;
        addTime = true;
        if (!CPH.TryGetArg("tier", out string tier))
            return false;
        int i = 1;
        var match = Regex.Match(tier.Trim().ToLower(), @"^tier\s+(\d+)$");
        if (match.Success)
            i = int.Parse(match.Groups[1].Value);
        if (!CPH.TryGetArg($"timeToAddTier{i}", out string timeToAddString))
            return false;

        // Determine the sign based on the presence of the "-" at the beginning of timeToAddString
        addTime = !timeToAddString.StartsWith("-");

        // If negative, strip the "-" from the string
        if (!addTime)
            timeToAddString = timeToAddString.TrimStart('-');

        return TryParseTimeString(timeToAddString, out parsedTime);
    }

    public bool TwitchSubAddTime()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!TryGetTwitchSubTier(out TimeSpan timeToAddSpan, out bool addTime))
        {
            if (!CPH.TryGetArg("timeToAdd", out string timeToAddString))
            {
                LogError($"{methodName}%timeToAdd% doesn't exist!");
                return false;
            }
            if (!TryParseTimeString(timeToAddString, out timeToAddSpan))
            {
                LogError($"{methodName}Unable to parse %timeToAdd%");
                return false;
            }
        }
        LogVerbose($"{methodName}Adding time '{ConvertTimeSpanToTimeString(timeToAddSpan)}'");
        long secondsToAdd = Convert.ToInt64(timeToAddSpan.TotalSeconds); // Convert TimeSpan into seconds
        if (!addTime)
            secondsToAdd = -secondsToAdd; // If negative time, change seconds to negative
        if (!CPH.TryGetArg("monthsGifted", out long monthsGifted))
            monthsGifted = -1;
        if (!CPH.TryGetArg("gifts", out long gifts))
            gifts = -1;
        if (monthsGifted > 0)
            secondsToAdd *= monthsGifted;
        if (gifts > 0)
            secondsToAdd *= gifts;
        return AddTime(secondsToAdd, addTime);
    }

    public bool MoneyAddTime()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!CPH.TryGetArg("amountReceived", out string amountReceivedString))
        {
            LogError($"{methodName}%amountReceived% doesn't exist!");
            return false;
        }
        LogVerbose($"{methodName}amountReceivedString after import '{amountReceivedString}'");

        // Get the current culture's decimal separator
        string decimalSeparator = CultureInfo.CurrentCulture.NumberFormat.NumberDecimalSeparator;

        // Check if the amountReceivedString uses a different decimal separator
        if (decimalSeparator == "." && amountReceivedString.Contains(","))
            amountReceivedString = amountReceivedString.Replace(',', decimalSeparator[0]); // Replace commas with the current culture's decimal separator
        else if (decimalSeparator == "," && amountReceivedString.Contains("."))
            amountReceivedString = amountReceivedString.Replace('.', decimalSeparator[0]); // Replace periods with the current culture's decimal separator
        LogVerbose($"{methodName}amountReceivedString before parse '{amountReceivedString}'");

        if (!double.TryParse(amountReceivedString, NumberStyles.Any, CultureInfo.CurrentCulture, out double amountReceived))
        {
            // Failed to parse amountReceived. Handle the error or provide feedback to the user.
            LogError($"{methodName}Failed to parse amountReceived.");
            return false;
        }
        LogVerbose($"{methodName}amountReceived '{amountReceived}'");

        if (!CPH.TryGetArg("timeToAdd", out string timeToAddString))
        {
            LogError($"{methodName}%timeToAdd% doesn't exist!");
            return false;
        }

        if (!TryParseTimeString(timeToAddString, out TimeSpan timeToAddSpan))
        {
            LogError($"{methodName}Unable to parse %timeToAdd%");
            return false;
        }

        long timeToAdd = (long)timeToAddSpan.TotalSeconds;

        if (!CPH.TryGetArg("moneyDivide", out double moneyDivide))
        {
            LogError($"{methodName}{moneyDivide}%moneyDivide% doesn't exist!");
            return false;
        }
        long moneyHundred = Convert.ToInt64(Math.Floor(amountReceived / moneyDivide));
        long secondsToAdd = moneyHundred * timeToAdd;
        return AddTime(secondsToAdd, true);
    }

    public bool CheckElapsed()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!timerOn)
        {
            string errorMessage = $"Error: Timer is not currently running";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }

        long timeElapsed = subathonElapsedTime;
        countdownString = GetTimerStringLong(timeElapsed);
        MessageSetArguments();
        return true;
    }

    public bool CheckRemaining()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!timerOn)
        {
            string errorMessage = $"Error: Timer is not currently running";
            LogError($"{methodName}{errorMessage}");
            SendMessage(sendAllPlatforms, platformList, errorMessage);
            return false;
        }

        long timeLeft = subathonTimeRemaining;
        countdownString = GetTimerStringLong(timeLeft);
        MessageSetArguments();
        return true;
    }

    public bool SendMessageToChat()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!CPH.TryGetArg("message", out string message))
            message = null;
        if (string.IsNullOrEmpty(message))
        {
            LogError($"{methodName}%message% doesn't exist");
            return false;
        }
        return SendMessage(sendAllPlatforms, platformList, message, sendAsBot);
    }

    private void StartSubathonTimer()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        limitReached = false;
        messageOnce = false;
        subathonCancelConfirm = false;
        subathonElapsedTime = 0;

        string initialDurationString = CPH.GetGlobalVar<string>("subathonInitialDuration");
        // Calculate the time remaining in seconds
        if (!TryParseTimeString(initialDurationString, out TimeSpan initialDuration))
        {
            LogError($"{methodName}Unable to parse %initialSubathonDuration%");
            return;
        }

        string maxDurationString = CPH.GetGlobalVar<string>("subathonMaxDuration");
        // Calculate the total length of the subathon in seconds
        if (!TryParseTimeString(maxDurationString, out TimeSpan maxDuration))
        {
            LogError($"{methodName}Unable to parse ~subathonMaxDuration~");
            return;
        }

        subathonTimeRemaining = (long)initialDuration.TotalSeconds;
        subathonTimeAddedTotal = subathonTimeRemaining; // This is used to calculate when the subathon limit has been reached.
        TriggerEventStarted();
        BackupWriteToFile();
        StartTimer();
    }

    private void StartTimer()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";

        // Start timer
        _ = subathonTimeRemaining - 1;
        countdownTimer.Start();
        timerOn = true;
        PausePlay();
        CPH.SetGlobalVar("subathonRunning", true);
        newSubathonConfirm = false;
        subathonCancelConfirm = false;
        LogVerbose($"{methodName}Timer started");
    }

    private void StopSubathon(string message)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";

        // Stop timer
        countdownTimer.Stop();
        timerOn = false;
        PausePlay();
        subathonCancelConfirm = false;
        UpdateOBSSourceMethod(message);
        LogVerbose($"{methodName}Timer stopped");
    }

    public bool UpdateHypeTrainMultiplier()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        GetHypeTrainMultiplier(out _);
        return true;
    }

    public bool HypeTrainEnded()
    {
        long regularTimeAdded = subathonHypeTrainTimeAdded;
        long extraTimeAdded = subathonHypeTrainExtraTimeAdded;
        long totalTimeAdded = regularTimeAdded + extraTimeAdded;
        CPH.SetArgument("regularTimeAddedShort", GetTimerStringShort(regularTimeAdded));
        CPH.SetArgument("regularTimeAddedLong", GetTimerStringLong(regularTimeAdded));
        CPH.SetArgument("extraTimeAddedShort", GetTimerStringShort(extraTimeAdded));
        CPH.SetArgument("extraTimeAddedLong", GetTimerStringLong(extraTimeAdded));
        CPH.SetArgument("totalTimeAddedShort", GetTimerStringShort(totalTimeAdded));
        CPH.SetArgument("totalTimeAddedLong", GetTimerStringLong(totalTimeAdded));
        CPH.SetGlobalVar("subathonLastHypeTrainRegularTimeAdded", GetTimerStringLong(subathonHypeTrainTimeAdded));
        CPH.SetGlobalVar("subathonLastHypeTrainExtraTimeAdded", GetTimerStringLong(subathonHypeTrainExtraTimeAdded));
        CPH.SetGlobalVar("subathonLastHypeTrainTotalTimeAdded", GetTimerStringLong(totalTimeAdded));
        return true;
    }

    private void GetHypeTrainMultiplier(out double hypeTrainMultiplier)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        string hypeTrainMultiplierType = CPH.GetGlobalVar<string>("subathonHypeTrainMultiplier").Trim().ToLower();
        hypeTrainMultiplier = 2.0; // Default multiplier
        double multiplierMax = CPH.GetGlobalVar<double?>("subathonHypeTrainMaxMultiplier") ?? 100;
        string multiplierTypeString = "straight";

        if (double.TryParse(hypeTrainMultiplierType, out double parsedMultiplier))
            hypeTrainMultiplier = parsedMultiplier; // If the string can be parsed into a double, use it as the multiplier
        else
        {
            int hypeTrainLevel = CPH.GetGlobalVar<int?>("subathonHypeTrainLevel") ?? 1;
            if (hypeTrainMultiplierType.Equals("line") || hypeTrainMultiplierType.Equals("linear"))
            {
                // Calculate multiplier using linear function
                multiplierTypeString = "linear";
                double hypeTrainMultiplierIncrement = CPH.GetGlobalVar<double?>("subathonHypeTrainLevelUpIncrement") ?? 0.1;
                LogVerbose($"{methodName}Linear: hypeTrainMultiplierIncrement: '{hypeTrainMultiplierIncrement}'");
                hypeTrainMultiplier = Math.Round(1.1 + (hypeTrainMultiplierIncrement * (hypeTrainLevel - 1)), 2);
            }
            else if (hypeTrainMultiplierType.Equals("exp") || hypeTrainMultiplierType.Equals("exponential"))
            {
                // Calculate multiplier using exponential function
                multiplierTypeString = "exponential";
                double exponent = 1.1; // Adjust this as needed
                hypeTrainMultiplier = Math.Round(Math.Pow(exponent, hypeTrainLevel), 2);
            }
            else if (hypeTrainMultiplierType.Equals("sig") || hypeTrainMultiplierType.Equals("sigmoidal"))
            {
                // Calculate multiplier using sigmoidal function
                multiplierTypeString = "sigmoidal";
                double L = hypeTrainLevel; // Maximum value (adjust as needed)
                double k = 0.2; // Growth rate (adjust as needed)
                double x0 = hypeTrainLevel / 2; // Midpoint of the sigmoid (adjust as needed)
                hypeTrainMultiplier = Math.Round(1 + (L / (1 + Math.Exp(-k * (hypeTrainLevel - x0)))), 2);
            }

            // Apply the maximum multiplier constraint
            hypeTrainMultiplier = Math.Min(hypeTrainMultiplier, (double)multiplierMax);
        }
        LogVerbose($"{methodName}Multiplier Type: '{multiplierTypeString}' Current multiplier: '{hypeTrainMultiplier}'");
        CPH.SetGlobalVar("subathonHypeTrainMultiplierCurrent", hypeTrainMultiplier);
    }

    // This method gets any previously set Membership names for configuration
    public bool GetYouTubeMembershipTiers()
    {
        // Run through for loop to try to get up to six YouTube Membership Tiers
        for (int i = 1; i < 7; i++)
        {
            string tierName = CPH.GetGlobalVar<string>($"subathonYouTubeMemberLvl{i}");
            if (string.IsNullOrEmpty(tierName))
            {
                CPH.SetArgument($"subathonYouTubeMemberLvl{i}Args", $"Input your Membership Lvl {i} name here");
                continue;
            }
            CPH.SetArgument($"subathonYouTubeMemberLvl{i}Args", tierName);
        }
        return true;
    }

    // Set from Streamer.bot action
    public bool SetYouTubeMembershipTiers()
    {
        var youTubeMembershipTiers = new Dictionary<string, int>();
        for (int i = 1; i < 7; i++)
        {
            if (!CPH.TryGetArg($"membershipLvl{i}", out string tierName) ||
                 tierName == $"Input your Membership Lvl {i} name here"
            )
                continue;

            youTubeMembershipTiers.Add(tierName, i);
            CPH.SetGlobalVar($"subathonYouTubeMemberLvl{i}", tierName);
        }
        LogVerbose($"{LOG_HEADER}Configured YouTube Membership Tier Levels:{JsonConvert.SerializeObject(youTubeMembershipTiers, Formatting.None)}");
        return true;
    }

    private bool TryGetYouTubeMembershipTiers(out Dictionary<string, int> youTubeMembershipTiers)
    {
        youTubeMembershipTiers = new Dictionary<string, int>();
        for (int i = 1; i < 7; i++)
        {
            string tierName = CPH.GetGlobalVar<string>($"subathonYouTubeMemberLvl{i}") ?? null;
            if (string.IsNullOrEmpty(tierName))
            {
                continue;
            }
            youTubeMembershipTiers.Add(tierName, i);
        }
        return youTubeMembershipTiers.Count > 0;
    }

    // Tries to match a YouTube membership tier with the given input.
    private bool TryMatchYouTubeTier(string input, out int level)
    {
        level = 0;
        if (!TryGetYouTubeMembershipTiers(out Dictionary<string, int> youTubeMembershipTiers))
            return false;
        return youTubeMembershipTiers.TryGetValue(input, out level);
    }

    public bool TwitchSubLabel()
    {
        string label = "";
        switch (CPH.GetEventType())
        {
            case EventType.TwitchSub:
                if (!CPH.TryGetArg("subLabel", out label))
                    label = "Twitch Subscription! ";
                break;
            case EventType.TwitchReSub:
                if (!CPH.TryGetArg("resubLabel", out label))
                    label = "Twitch Resubscription! ";
                break;
            case EventType.TwitchGiftSub:
                if (!CPH.TryGetArg("giftLabel", out label))
                    label = "Twitch Gift Subscription! ";
                break;
            case EventType.TwitchGiftBomb:
                if (!CPH.TryGetArg("giftBombLabel", out label))
                    label = "Twitch Multiple Gift Subs! ";
                break;
        }
        CPH.SetArgument("message", label);
        return true;
    }

    public bool YouTubeMembershipLabel()
    {
        string label = "";
        switch (CPH.GetEventType())
        {
            case EventType.YouTubeNewSponsor:
                if (!CPH.TryGetArg("sponsorLabel", out label))
                    label = "YouTube New Sponsor! ";
                break;
            case EventType.YouTubeMemberMileStone:
                if (!CPH.TryGetArg("milestoneLabel", out label))
                    label = "YouTube Membership Milestone! ";
                break;
            case EventType.YouTubeGiftMembershipReceived:
                if (!CPH.TryGetArg("giftReceivedLabel", out label))
                    label = "YouTube Gift Membership Received! ";
                break;
            case EventType.YouTubeMembershipGift:
                if (!CPH.TryGetArg("giftLabel", out label))
                    label = "YouTube Membership Gift! ";
                break;
        }
        CPH.SetArgument("message", label);
        return true;
    }

    public bool YouTubeMembershipAddTime()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        string levelName = null;
        int count = 1;
        switch (CPH.GetEventType())
        {
            case EventType.YouTubeNewSponsor:
            case EventType.YouTubeMemberMileStone:
                if (!CPH.TryGetArg("levelName", out levelName))
                {
                    LogError($"{methodName}%levelName% doesn't exist!");
                    return false;
                }
                break;
            case EventType.YouTubeGiftMembershipReceived:
                if (!CPH.TryGetArg("tier", out levelName))
                {
                    LogError($"{methodName}%tier% doesn't exist!");
                    return false;
                }
                break;
            case EventType.YouTubeMembershipGift:
                if (!CPH.TryGetArg("tier", out levelName))
                {
                    LogError($"{methodName}%tier% doesn't exist!");
                    return false;
                }
                if (!CPH.TryGetArg("count", out count))
                {
                    LogError($"{methodName}%count% doesn't exist!");
                    return false;
                }
                break;
        }

        // Attempt to match levelName with YouTube Tier dictionary
        if (string.IsNullOrEmpty(levelName) || !TryMatchYouTubeTier(levelName, out int level))
        {
            LogError($"{methodName}Unable to match level name '{levelName}' with existing tiers");
            return false;
        }

        // Get timeToAdd based on the level
        if (!CPH.TryGetArg($"timeToAddYouTubeLvl{level}", out string timeToAddString))
        {
            LogError($"{methodName}%timeToAddYouTubeLvl{level}% doesn't exist!");
            return false;
        }

        // Attempt to parse time string
        if (!TryParseTimeString(timeToAddString, out TimeSpan timeToAddSpan))
        {
            LogError($"{methodName}Unable to parse 'timeToAddYouTubeLvl{level}'");
            return false;
        }

        long timeToAddSeconds = (long)timeToAddSpan.TotalSeconds;
        if (count > 1)
            timeToAddSeconds *= count;

        return AddTime(timeToAddSeconds, true);
    }

    private bool AddTime(long secondsToAdd, bool addTime)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";

        if (subathonTimeRemaining < 0)
        {
            if (!TryRestoreBackup(out subathonTimeRemaining, out subathonTimeAddedTotal, out subathonElapsedTime))
            {
                LogError($"{methodName}Unable to parse backup or nothing to add time to '{GetTimerStringShort(subathonTimeRemaining)}'");
                return false;
            }
            if (subathonTimeRemaining < 0)
            {
                LogError($"{methodName}subathonTimeRemaining is bad '{GetTimerStringShort(subathonTimeRemaining)}'");
                return false;
            }
        }

        subathonCancelConfirm = false;
        long originalSecondsToAdd = 0;
        if ( !addTime &&
             -secondsToAdd > subathonTimeRemaining )
        {
            string errorMessage = "You cannot remove more time than there is remaining";
            LogError($"{methodName}{errorMessage}");
            SendMessage(true, platformList, errorMessage);
            return false;
        }

        long hypeTrainSecondsAdded = 0;
        bool hypeTrain = CPH.GetGlobalVar<bool?>("subathonHypeTrain") ?? false;
        if (hypeTrain && addTime)
        {
            GetHypeTrainMultiplier(out double hypeTrainMultiplier);
            long timeToAddMultiplied = (long) Math.Round(secondsToAdd * hypeTrainMultiplier);
            originalSecondsToAdd = secondsToAdd;
            subathonHypeTrainTimeAdded += originalSecondsToAdd;
            hypeTrainSecondsAdded = timeToAddMultiplied - secondsToAdd;
            subathonHypeTrainExtraTimeAdded += hypeTrainSecondsAdded;

            secondsToAdd = timeToAddMultiplied;
        }

        // Determine if time limit has been reached
        long subathonTimeLimit = GetSubathonLimit();
        if ((subathonTimeLimit - (subathonTimeAddedTotal + secondsToAdd)) > 0)
        {
            subathonTimeAddedTotal += secondsToAdd;
            subathonTimeRemaining = subathonTimeRemaining + secondsToAdd;
        }
        else
        {
            subathonTimeRemaining = subathonTimeRemaining + (subathonTimeLimit - subathonTimeAddedTotal);
            secondsToAdd = subathonTimeLimit - subathonTimeAddedTotal;
            subathonTimeAddedTotal = subathonTimeLimit;
            limitReached = true;
        }

        var newArgs = CreateArgsDictionary();
        if (!CPH.TryGetArg("message", out string message))
            message = "";
        newArgs.Add("message", message);
        if (addTime)
            newArgs.Add("timeVerb", "added to");
        else
        {
            secondsToAdd = -secondsToAdd;
            newArgs.Add("timeVerb", "removed from");
        }

        string timeToAddShort = GetTimerStringShort(secondsToAdd);
        string timeToAddLong = GetTimerStringLong(secondsToAdd);
        if (limitReached && messageOnce)
        {
            timeToAddShort = GetTimerStringShort(0);
            timeToAddLong = "No time";
        }

        if ( hypeTrain &&
             addTime &&
             originalSecondsToAdd > 0 &&
             hypeTrainSecondsAdded > 0 )
        {
            newArgs.Add("timeToAddShort", GetTimerStringShort(originalSecondsToAdd));
            newArgs.Add("timeToAddLong", GetTimerStringLong(originalSecondsToAdd));
            newArgs.Add("hypeTrainAddedShort", GetTimerStringShort(hypeTrainSecondsAdded));
            newArgs.Add("hypeTrainAddedLong", GetTimerStringLong(hypeTrainSecondsAdded));
        }
        else
        {
            newArgs.Add("timeToAddShort", timeToAddShort);
            newArgs.Add("timeToAddLong", timeToAddLong);
            newArgs.Add("hypeTrainAddedShort", "");
            newArgs.Add("hypeTrainAddedLong", "");
        }

        if (limitReached && !messageOnce)
        {
            TriggerEventLimitReached();
            CPH.SetGlobalVar("subathonLimitReachedDateTime", DateTime.Now);
            CPH.SetGlobalVar("subathonLimitReached12", DateTime.Now.ToString("yyyy-MM-dd hh:mm:ss tt"));
            CPH.SetGlobalVar("subathonLimitReached24", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
            messageOnce = true;
        }

        if (!timerOn)
        {
            UpdateOBSSourceMethod($"{PAUSE_SYMBOL}{GetTimerStringShort(subathonTimeRemaining)}");
        }
        TriggerEventTimeAdded(newArgs);
        BackupWriteToFile();
        return true;
    }

    private void OnTimedEvent(Object source, ElapsedEventArgs e)
    {
        subathonTimeRemaining--;
        subathonElapsedTime++;
        long timeLeft = subathonTimeRemaining;
        countdownString = GetTimerStringShort(timeLeft);
        if (subathonTimeRemaining % 300 == 0)
            BackupWriteToFile();
        if (subathonTimeRemaining == 0)
        {
            CPH.SetGlobalVar("subathonRunning", false);
            StopSubathon("Complete!");
            CreateArgsDictionary();
            TriggerEventCompleted();
        }
        else
        {
            UpdateOBSSourceMethod(countdownString);
            if (CPH.GetGlobalVar<bool?>("subathonIndividualTimeUnits") ?? false)
                UpdateOBSSourceSeparateUnits(subathonTimeRemaining);
            else
                UpdateOBSSourceSeparateUnits(0);
        }
    }

    public bool UpdateOBSSource()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!CPH.TryGetArg("obsInput", out string obsInput))
        {
            LogError($"{methodName}%obsInput% doesn't exist!");
            return false;
        }
        TryParseTimeString(obsInput, out TimeSpan parsedTime);

        UpdateOBSSourceMethod(GetTimerStringShort((long)parsedTime.TotalSeconds));
        UpdateOBSSourceSeparateUnits((long)parsedTime.TotalSeconds);
        return true;
    }

    private void UpdateOBSSourceMethod(string message)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        if (!CPH.ObsIsConnected(obsConnection))
            return;

        CPH.ObsSetGdiText(
            CPH.GetGlobalVar<string>("subathonScene") ?? DEFAULT_SCENE,
            CPH.GetGlobalVar<string>("subathonSource") ?? DEFAULT_TS,
            message,
            obsConnection
        );
    }

    private void UpdateOBSSourceSeparateUnits(long timeLeft)
    {
        TimeSpan timeSpanLeft = TimeSpan.FromSeconds(timeLeft);
        string subathonScene = CPH.GetGlobalVar<string>("subathonScene") ?? DEFAULT_SCENE;
        string subathonSourceDays = CPH.GetGlobalVar<string>("subathonSourceDays") ?? DEFAULT_TS_DAYS;
        string subathonSourceHours = CPH.GetGlobalVar<string>("subathonSourceHours") ?? DEFAULT_TS_HOURS;
        string subathonSourceMinutes = CPH.GetGlobalVar<string>("subathonSourceMinutes") ?? DEFAULT_TS_MINUTES;
        string subathonSourceSeconds = CPH.GetGlobalVar<string>("subathonSourceSeconds") ?? DEFAULT_TS_SECONDS;

        bool displayDays = CPH.GetGlobalVar<bool>("subathonDisplayDays");
        string days = "";
        string hours = "";
        if (displayDays)
        {
            if (timeSpanLeft.Days > 0)
                days = $"{timeSpanLeft.Days}"; // Days
            if (timeSpanLeft.Hours > 0)
                hours = $"{timeSpanLeft.Hours:D2}"; // Hours
        }
        else
            if (timeSpanLeft.Hours > 0)
                hours = $"{timeSpanLeft.Days * 24 + timeSpanLeft.Hours:D2}";
        CPH.ObsSetGdiText(subathonScene, subathonSourceDays, days, obsConnection);
        CPH.ObsSetGdiText(subathonScene, subathonSourceHours, hours, obsConnection);
        CPH.ObsSetGdiText(subathonScene, subathonSourceMinutes, $"{timeSpanLeft.Minutes:D2}", obsConnection); // Minutes
        CPH.ObsSetGdiText(subathonScene, subathonSourceSeconds, $"{timeSpanLeft.Seconds:D2}", obsConnection); // Seconds
    }

    private void PausePlay()
    {
        string subathonScene = CPH.GetGlobalVar<string>("subathonScene") ?? DEFAULT_SCENE;
        string subathonSourcePlayPause = CPH.GetGlobalVar<string>("subathonSourcePlayPause") ?? DEFAULT_TS_PLAY_PAUSE;
        if (timerOn)
            CPH.ObsSetGdiText(subathonScene, subathonSourcePlayPause, PLAY_SYMBOL, obsConnection); // Seconds
        else
            CPH.ObsSetGdiText(subathonScene, subathonSourcePlayPause, PAUSE_SYMBOL, obsConnection); // Seconds
    }

    private int GetObsConnection()
    {
        string obsConnectionString = CPH.GetGlobalVar<string>("subathonObsConnection") ?? CPH.GetGlobalVar<string>("subathonObsConnectionGlobal");
        if (string.IsNullOrEmpty(obsConnectionString))
            return 0;
        int.TryParse(obsConnectionString, out int currentObsConnection);
        return currentObsConnection;
    }

    private void GetDirectory()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        string secondsLeftBackupDirectory = AppDomain.CurrentDomain.BaseDirectory;
        secondsLeftBackupDirectory = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "..", "subathonTimerBackup"));
        // secondsLeftBackupDirectory = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "data"));
        LogVerbose($"{methodName}secondsLeftBackupDirectory: '{secondsLeftBackupDirectory}'");
        string subathonTimeLeftFileOld = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "subathonSecondsLeft.txt"));
        string subathonTimeAddedInTimeFileOld = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "subathonTimeAddedInSeconds.txt"));
        string subathonElapsedTimeFileOld = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "subathonElapsedSeconds.txt"));
        subathonTimeLeftFile = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "remainingTime.txt"));
        subathonTimeAddedInTimeFile = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "addedTime.txt"));
        subathonElapsedTimeFile = Path.GetFullPath(Path.Combine(secondsLeftBackupDirectory, "elapsedTime.txt"));
        if (File.Exists(subathonTimeLeftFileOld))
            RenameExistingBackups(subathonTimeLeftFileOld, subathonTimeLeftFile);
        if (File.Exists(subathonTimeAddedInTimeFileOld))
            RenameExistingBackups(subathonTimeAddedInTimeFileOld, subathonTimeAddedInTimeFile);
        if (File.Exists(subathonElapsedTimeFileOld))
            RenameExistingBackups(subathonElapsedTimeFileOld, subathonElapsedTimeFile);

        // Check if the directory exists, create it if it doesn't
        if (!Directory.Exists(secondsLeftBackupDirectory))
        {
            Directory.CreateDirectory(secondsLeftBackupDirectory);
            LogVerbose($"{methodName}Created the directory '{secondsLeftBackupDirectory}'");
        }
        if (!File.Exists(subathonTimeLeftFile))
        {
            File.Create(subathonTimeLeftFile).Dispose();
            LogVerbose($"{methodName}Created the file '{subathonTimeLeftFile}'");
        }
        if (!File.Exists(subathonTimeAddedInTimeFile))
        {
            File.Create(subathonTimeAddedInTimeFile).Dispose();
            LogVerbose($"{methodName}Created the file '{subathonTimeAddedInTimeFile}'");
        }
        if (!File.Exists(subathonElapsedTimeFile))
        {
            File.Create(subathonElapsedTimeFile).Dispose();
            LogVerbose($"{methodName}Created the file '{subathonElapsedTimeFile}'");
        }
    }

    private void RenameExistingBackups(string oldFilePath, string newFilePath)
    {
        File.Move(oldFilePath, newFilePath);
    }

    private void SubathonClear()
    {
        File.WriteAllText(subathonTimeLeftFile, "");
        File.WriteAllText(subathonElapsedTimeFile, "");
        File.WriteAllText(subathonTimeAddedInTimeFile, "");
        CPH.UnsetGlobalVar("subathonTimeAdded");
        CPH.UnsetGlobalVar("subathonTimeElapsed");
        CPH.UnsetGlobalVar("subathonTimeRemaining");
    }

    private long GetSubathonLimit()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";

        // Calculate the total length of the subathon in seconds
        string subathonMaxDuration = CPH.GetGlobalVar<string>("subathonMaxDuration") ?? CPH.GetGlobalVar<string>("subathonMaxDurationGlobal");
        if (string.IsNullOrEmpty(subathonMaxDuration))
        {
            LogError($"{methodName}~subathonMaxDuration~ and ~subathonMaxDurationGlobal~ is null!");
            return -1;
        }
        if (!TryParseTimeString(subathonMaxDuration, out TimeSpan maxDuration))
        {
            LogError($"{methodName}Unable to parse ~subathonMaxDuration~");
            return -1;
        }
        return (long)maxDuration.TotalSeconds;
    }

    private bool TryRestoreBackup(out long restoredSubathonSecondsRemaining, out long restoredSubathonTimeAddedInSeconds, out long restoredSubathonElapsedSeconds)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        restoredSubathonSecondsRemaining = -1;
        restoredSubathonTimeAddedInSeconds = -1;
        restoredSubathonElapsedSeconds = -1;
        TimeSpan remainingTimeSpan = TimeSpan.Zero;
        TimeSpan timeAddedTimeSpan = TimeSpan.Zero;
        TimeSpan elapsedTimeSpan = TimeSpan.Zero;

        try
        {
            GetDirectory();
            // Attempt to restore via backup file
            if (!TryParseTimeString(CPH.GetGlobalVar<string>("subathonTimeRemaining"), out remainingTimeSpan))
                if (!TryParseTimeString(File.ReadAllText(subathonTimeLeftFile), out remainingTimeSpan))
                    return false;
            if (!TryParseTimeString(CPH.GetGlobalVar<string>("subathonTimeAdded"), out timeAddedTimeSpan))
                if (!TryParseTimeString(File.ReadAllText(subathonTimeAddedInTimeFile), out timeAddedTimeSpan))
                    return false;
            if (!TryParseTimeString(CPH.GetGlobalVar<string>("subathonTimeElapsed"), out elapsedTimeSpan))
                if (!TryParseTimeString(File.ReadAllText(subathonElapsedTimeFile), out elapsedTimeSpan))
                    return false;

            // Check if remainingTimeSpans are zero
            if (remainingTimeSpan == TimeSpan.Zero && timeAddedTimeSpan == TimeSpan.Zero && elapsedTimeSpan == TimeSpan.Zero)
                return false;

            restoredSubathonSecondsRemaining = (long)remainingTimeSpan.TotalSeconds + 1; // Resuming seconds left from backup
            restoredSubathonTimeAddedInSeconds = (long)timeAddedTimeSpan.TotalSeconds; // Recalling seconds left with time added
            restoredSubathonElapsedSeconds = (long)elapsedTimeSpan.TotalSeconds; // Recall subathon elapsed
            LogVerbose($"restoredSubathonSecondsRemaining '{ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(restoredSubathonSecondsRemaining))}'");
            LogVerbose($"restoredSubathonTimeAddedInSeconds '{ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(restoredSubathonTimeAddedInSeconds))}'");
            LogVerbose($"restoredSubathonElapsedSeconds '{ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(restoredSubathonElapsedSeconds))}'");
            return true;
        }
        catch (Exception ex)
        {
            LogError($"{methodName}{ex.Message}");
            return false;
        }
    }

    private void BackupWriteToFile()
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        CPH.Wait(100);
        string remainingTimeString = ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(subathonTimeRemaining));
        string timeAddedTimeString = ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(subathonTimeAddedTotal));
        string elapsedTimeString = ConvertTimeSpanToTimeString(TimeSpan.FromSeconds(subathonElapsedTime));

        File.WriteAllText(subathonTimeLeftFile, remainingTimeString);
        CPH.SetGlobalVar("subathonTimeRemaining", remainingTimeString);

        File.WriteAllText(subathonTimeAddedInTimeFile, timeAddedTimeString);
        CPH.SetGlobalVar("subathonTimeAdded", timeAddedTimeString);

        File.WriteAllText(subathonElapsedTimeFile, elapsedTimeString);
        CPH.SetGlobalVar("subathonTimeElapsed", elapsedTimeString);

        LogVerbose($"{methodName}Backup created remaining: '{remainingTimeString}'");
        LogVerbose($"{methodName}Backup created timeAdded: '{timeAddedTimeString}'");
        LogVerbose($"{methodName}Backup created elapsed: '{elapsedTimeString}'");
    }

    private Dictionary<string, object> CreateArgsDictionary()
    {
        // Set arguments to use in custom message
        long subathonTimeLimit = GetSubathonLimit();
        return new Dictionary<string, object>
        {
            { "countdownElapsedShort", GetTimerStringShort(subathonElapsedTime) },
            { "countdownElapsedLong", GetTimerStringLong(subathonElapsedTime) },
            { "countDownElapsedTimeString", ConvertTimeSpanToTimeString( TimeSpan.FromSeconds(subathonElapsedTime) ) },
            { "countdownRemainingShort", GetTimerStringShort(subathonTimeRemaining) },
            { "countdownRemainingLong", GetTimerStringLong(subathonTimeRemaining) },
            { "countDownRemainingTimeString", ConvertTimeSpanToTimeString( TimeSpan.FromSeconds(subathonTimeRemaining) ) },
            { "addedTimeString", ConvertTimeSpanToTimeString( TimeSpan.FromSeconds(subathonTimeAddedTotal) ) },
            { "maxDurationShort", GetTimerStringShort(subathonTimeLimit) },
            { "maxDurationLong", GetTimerStringLong(subathonTimeLimit) },
            { "maxDurationTimeString", ConvertTimeSpanToTimeString( TimeSpan.FromSeconds(subathonTimeLimit)) },
            { "goalPercent", $"{((double)subathonTimeAddedTotal / subathonTimeLimit * 100).ToString("F2")}"}
        };
    }

    private void MessageSetArguments()
    {
        long subathonTimeLimit = GetSubathonLimit();
        // Set arguments to use in custom message
        CPH.SetArgument("countdownElapsedShort", GetTimerStringShort(subathonElapsedTime));
        CPH.SetArgument("countdownElapsedLong", GetTimerStringLong(subathonElapsedTime));

        CPH.SetArgument("countdownRemainingShort", GetTimerStringShort(subathonTimeRemaining));
        CPH.SetArgument("countdownRemainingLong", GetTimerStringLong(subathonTimeRemaining));

        CPH.SetArgument("maxDurationShort", GetTimerStringShort(subathonTimeLimit));
        CPH.SetArgument("maxDurationLong", GetTimerStringLong(subathonTimeLimit));
    }

    private string GetTimerStringShort(long timeLeft)
    {
        TimeSpan time = TimeSpan.FromSeconds(timeLeft);
        bool displayDays = CPH.GetGlobalVar<bool>("subathonDisplayDays");
        StringBuilder sb = new StringBuilder();
        if (displayDays && time.Days > 0)
            sb.Append($"{time.Days}.{time.Hours:D2}{TIME_SEPARATOR}");
        else
            sb.Append($"{time.Days * 24 + time.Hours:D2}{TIME_SEPARATOR}");

        string minutesStr = time.Minutes == 0 ? "00" : $"{time.Minutes:D2}";
        string secondsStr = time.Seconds == 0 ? "00" : $"{time.Seconds:D2}";
        sb.Append($"{minutesStr}{TIME_SEPARATOR}{secondsStr}");

        if (time == TimeSpan.Zero)
            return ZERO_TIME;

        return sb.ToString();
    }

    private string GetTimerStringLong(long timeLeft)
    {
        TimeSpan time = TimeSpan.FromSeconds(timeLeft);
        bool displayDays = CPH.GetGlobalVar<bool>("subathonDisplayDays");
        StringBuilder sb = new StringBuilder();
        string s = "";
        int totalHours = time.Hours + time.Days * 24;
        List<string> timeIncrements = new List<string>();
        if (displayDays && time.Days > 0)
        {
            s = time.Days > 1 ? "s" : "";
            timeIncrements.Add($"{time.Days} day{s}");
            totalHours -= time.Days * 24;
        }
        if (totalHours > 0)
        {
            s = totalHours > 1 ? "s" : "";
            timeIncrements.Add($"{totalHours} hour{s}");
        }
        if (time.Minutes > 0)
        {
            s = time.Minutes > 1 ? "s" : "";
            timeIncrements.Add($"{time.Minutes} minute{s}");
        }
        if (time.Seconds > 0)
        {
            s = time.Seconds > 1 ? "s" : "";
            timeIncrements.Add($"{time.Seconds} second{s}");
        }

        if (time == TimeSpan.Zero)
            return "No time";

        if (timeIncrements.Count > 0)
        {
            for (int i = 0; i < timeIncrements.Count; i++)
            {
                if (timeIncrements.Count > 1 && i == timeIncrements.Count - 1)
                    sb.Append("and ");
                sb.Append(timeIncrements[i]);
                if (timeIncrements.Count > 2 && i < timeIncrements.Count)
                    sb.Append(", ");
                else
                    sb.Append(" ");
            }
        }

        return sb.ToString().Trim().TrimEnd(',');
    }

    private string ConvertTimeSpanToTimeString(TimeSpan timeSpan)
    {
        StringBuilder output = new StringBuilder();

        if (timeSpan.Days > 0)
            output.Append($"{timeSpan.Days}d");

        if (timeSpan.Hours > 0)
            output.Append($"{timeSpan.Hours}h");

        if (timeSpan.Minutes > 0)
            output.Append($"{timeSpan.Minutes}m");

        if (timeSpan.Seconds > 0)
            output.Append($"{timeSpan.Seconds}s");

        if (timeSpan == TimeSpan.Zero)
            return "0s";

        return output.ToString();
    }

    private static bool TryParseTimeString(string input, out TimeSpan parsedTime)
    {
        parsedTime = TimeSpan.Zero;
        if (string.IsNullOrEmpty(input))
            return false;

        // Remove all whitespaces from the input string
        input = Regex.Replace(input, @"\s+", "");

        // Define the pattern for the input string to match a sequence of digits
        string pattern = @"^(-)?\d+$"; // Modified to allow an optional negative sign at the beginning
        Match match = Regex.Match(input, pattern); // Match the pattern using Regex
        int seconds;
        if (match.Success)
        {
            // If the input is a sequence of digits, parse it as seconds
            seconds = int.Parse(input);

            // Construct the TimeSpan with the parsed seconds
            parsedTime = TimeSpan.FromSeconds(seconds);
            return true;
        }

        // Define the pattern for the input string to match days, hours, minutes, and seconds
        pattern = @"^(-)?(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$"; // Modified to allow an optional negative sign at the beginning

        // Match the pattern using Regex
        match = Regex.Match(input, pattern, RegexOptions.IgnoreCase);
        if (!match.Success)
            return false;

        // Extract values from the matched groups
        int sign = match.Groups[1].Success && match.Groups[1].Value == "-" ? -1 : 1; // Extract sign
        int days = match.Groups[2].Success ? int.Parse(match.Groups[2].Value) : 0;
        int hours = match.Groups[3].Success ? int.Parse(match.Groups[3].Value) : 0;
        int minutes = match.Groups[4].Success ? int.Parse(match.Groups[4].Value) : 0;
        seconds = match.Groups[5].Success ? int.Parse(match.Groups[5].Value) : 0;

        // Adjust values based on sign
        days *= sign;
        hours *= sign;
        minutes *= sign;
        seconds *= sign;

        // Construct the TimeSpan with the extracted values
        parsedTime = new TimeSpan(days, hours, minutes, seconds);
        return true;
    }

    private HashSet<Platform> SetMessagePlatform(HashSet<Platform> platformList)
    {
        string methodName = $"{MethodBase.GetCurrentMethod().Name}: ";
        EventSource eventSource = CPH.GetSource();
        EventType eventType = CPH.GetEventType();
        Platform currentPlatform;
        platformList = ParsePlatforms(platformList);

        switch (eventSource)
        {
            case EventSource.Twitch:
                platformList.Add(Platform.Twitch);
                break;

            case EventSource.YouTube:
                platformList.Add(Platform.YouTube);
                break;

            case EventSource.Trovo:
                platformList.Add(Platform.Trovo);
                break;

            default:
                break;
        }

        switch (eventType)
        {
            case EventType.CommandTriggered:
                CPH.TryGetArg("commandSource", out string commandSource);
                if (Enum.TryParse(commandSource, true, out currentPlatform))
                {
                    platformList.Add(currentPlatform);
                    LogVerbose($"{methodName}'{currentPlatform}' added to platformList");
                    LogVerbose($"{methodName}platformList:{JsonConvert.SerializeObject(platformList, Formatting.None)}");
                }
                LogVerbose($"commandSource '{commandSource}'");
                break;
        }
        sendAllPlatforms = CPH.GetGlobalVar<bool>("subathonSendAllPlatforms");
        sendAsBot = CPH.GetGlobalVar<bool>("subathonSendAsBot");
        return platformList;
    }

    public bool SendMessage(bool sendAllPlatforms, HashSet<Platform> platformList, string message, bool sendAsBot = true)
    {
        if (sendAllPlatforms)
        {
            bool success = false;
            foreach (Platform platform in platformList)
            {
                success = SendMessageChunksToPlatform(platform, message, sendAsBot);
            }
            return success;
        }
        return SendMessageChunksToPlatform(platformList.FirstOrDefault(), message, sendAsBot);
    }

    private bool SendMessageChunksToPlatform(Platform currentPlatform, string message, bool sendAsBot = true)
    {
        bool success = false;
        int maxChunkSize = GetMaxChunkSize(currentPlatform, out int delay);
        List<string> chunks = SplitMessageIntoChunksByWords(message, maxChunkSize);

        foreach (string chunk in chunks)
        {
            success = SendMessageToPlatform(currentPlatform, chunk, sendAsBot);
            CPH.Wait(delay);
        }
        return success;
    }

    private static int GetMaxChunkSize(Platform currentPlatform, out int delay)
    {
        switch (currentPlatform)
        {
            case Platform.Twitch:
                delay = 1000;
                return 500;
            case Platform.Trovo:
                delay = 1000;
                return 300;
            default: // YouTube
                delay = 2000;
                return 200;
        }
    }

    private List<string> SplitMessageIntoChunksByWords(string message, int maxChunkSize)
    {
        List<string> chunks = new List<string>();
        string[] words = message.Split(' ');

        StringBuilder currentChunk = new StringBuilder();
        foreach (string word in words)
        {
            if (currentChunk.Length + word.Length + 1 <= maxChunkSize)
            {
                if (currentChunk.Length > 0)
                {
                    currentChunk.Append(' '); // Add space between words
                }
                currentChunk.Append(word);
            }
            else
            {
                chunks.Add(currentChunk.ToString());
                currentChunk = new StringBuilder(word);
            }
        }

        if (currentChunk.Length > 0)
            chunks.Add(currentChunk.ToString());

        return chunks;
    }

    private HashSet<Platform> ParsePlatforms(HashSet<Platform> newPlatformList)
    {
        string sendToPlatforms = CPH.GetGlobalVar<string>("subathonSendMessageTo");
        string[] platformArray = Regex.Split(sendToPlatforms, @"[^a-zA-Z0-9]+");

        foreach (string platformString in platformArray)
        {
            string trimmedPlatform = platformString.Trim();
            if (string.IsNullOrEmpty(trimmedPlatform))
                return null;
            Enum.TryParse(trimmedPlatform, true, out Platform platform);
            newPlatformList.Add(platform);
        }
        return newPlatformList;
    }

    // Send a message based on the specified command source
    private bool SendMessageToPlatform(Platform currentPlatform, string message, bool sendAsBot = true)
    {
        bool messageSent = false;
        switch (currentPlatform)
        {
            case Platform.YouTube:
                CPH.SendYouTubeMessage(message, sendAsBot);
                messageSent = true;
                break;
            case Platform.Twitch:
                CPH.SendMessage(message, sendAsBot);
                messageSent = true;
                break;
            case Platform.Trovo:
                CPH.SendTrovoMessage(message, sendAsBot);
                messageSent = true;
                break;
            default:
                break;
        }
        return messageSent;
    }

    private void LogError(string errorMessage)
    {
        CPH.LogError($"{LOG_HEADER}{errorMessage}");
    }

    private void LogVerbose(string verboseMessage)
    {
        CPH.LogVerbose($"{LOG_HEADER}{verboseMessage}");
    }
}

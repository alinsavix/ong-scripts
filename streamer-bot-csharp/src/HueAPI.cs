// This is a streamer.bot plugin controlling Philips Hue lights via Q42.HueApi.
//
// Registration of a new bridge is messy and really should be reimplemented.
//
// Useful functions:
//   - turnOnHueID / turnOffHueID - turns on/off a light with given hue.id
//   - changeLightColor - changes light hue.id, in hue.transitionTime ms.
//     Sets to (hue.hexColor, hue.saturation, hue.brightness)
//     If hue.saturation or hue.brightness is -1, they will be calculated
//   - changeGroupColor - change color of group, same args as changeLightColor
//   - transitionLightColorPerceptual / transitionGroupColorPerceptual
//     Perceptual color transition using OKLCH. Given hue.startColor, hue.endColor,
//     hue.durationSeconds, and hue.deltaERate, calculates the intermediate color
//     reachable at the given rate and sends a transition command.
//   - listAllLights - prints all available lights and groups to the log

// using Streamer.bot.Plugin.Interface;
// using Streamer.bot.Plugin.Interface.Enums;
// using Streamer.bot.Plugin.Interface.Model;
// using Streamer.bot.Common.Events;
using System;
using System.Threading.Tasks;
using System.Linq;

// ReSharper disable InconsistentNaming
// ReSharper disable CheckNamespace
#pragma warning disable CS0114
#pragma warning disable IDE1006

using Q42.HueApi;
using Q42.HueApi.ColorConverters;
using Q42.HueApi.ColorConverters.HSB;

/**
 * [API] HUE
 */
// ReSharper disable once UnusedType.Global

#if EXTERNAL_EDITOR
public class HueApi : CPHInlineBase
#else
public class CPHInline
#endif
{
    private const string HUE_BRIDGE_APP_KEY_PROPERTY = "config.hue.apikey";

    private string hueHubIpAddress;
    private string hueHubAppKey;

    private LocalHueClient hueClient;

    private void init()
    {
        hueHubIpAddress = getProperty("hue.bridge.ip", "127.0.0.1");
        hueHubAppKey = CPH.GetGlobalVar<string>(HUE_BRIDGE_APP_KEY_PROPERTY, true);
        if (hueHubAppKey is null or "")
        {
            DEBUG(() => "HUE BRIDGE APP KEY NOT FOUND, REGISTERING NEW CLIENT");
            registerNewApplication().GetAwaiter().GetResult();
        }
        else
        {
            DEBUG(() => "HUE BRIDGE APP KEY FOUND, CREATING CLIENT");
            try
            {
                hueClient = new LocalHueClient(hueHubIpAddress);
                hueClient.Initialize(hueHubAppKey);
            }
            catch (Exception e)
            {
                ERROR(() => $"Cannot initialize client for {hueHubIpAddress}: {e.Message}" +
                    (e.InnerException != null ? $"\nStackTrace: {e.InnerException.Message}" : "") + $"\n{e.StackTrace}"
                );
            }
        }
        printAllLights().GetAwaiter().GetResult();
    }


    public bool turnOnHueID()
    {
        var id = getProperty("hue.id", "UNKNOWN");

        runAsync(() => updateHueState(id, true));
        return true;
    }


    public bool turnOffHueID()
    {
        var id = getProperty("hue.id", "UNKNOWN");

        runAsync(() => updateHueState(id, false));
        return true;
    }


    public bool changeLightColor()
    {
        var id = getProperty("hue.id", "UNKNOWN");
        var color = getProperty("hue.hexColor", "#FF0000");
        var saturation = getProperty("hue.saturation", -1);
        var brightness = getProperty("hue.brightness", -1);
        var transitionTime = getProperty("hue.transitionTime", 400);

        runAsync(() => updateLightColor(id, color, saturation, brightness, transitionTime));
        return true;
    }

    public bool changeGroupColor()
    {
        var id = getProperty("hue.id", "UNKNOWN");
        var color = getProperty("hue.hexColor", "#FF0000");
        var saturation = getProperty("hue.saturation", -1);
        var brightness = getProperty("hue.brightness", -1);
        var transitionTime = getProperty("hue.transitionTime", 400);

        runAsync(() => updateGroupColor(id, color, saturation, brightness, transitionTime));
        return true;
    }

    public bool transitionLightColorPerceptual()
    {
        var id = getProperty("hue.id", "UNKNOWN");
        var startColor = getProperty("hue.startColor", "#FF0000");
        var endColor = getProperty("hue.endColor", "#0000FF");
        var durationSeconds = getProperty("hue.durationSeconds", 5.0);
        var deltaERate = getProperty("hue.deltaERate", 0.1);

        var (targetHex, transitionTimeMs) = CalculatePerceptualTransition(
            startColor, endColor, durationSeconds, deltaERate);

        INFO(() => $"Perceptual light transition:" +
             $"\n  Source color (requested):      {startColor}" +
             $"\n  Destination color (requested):  {endColor}" +
             $"\n  Delta-E rate:                   {deltaERate}/s" +
             $"\n  Transition time (requested):    {durationSeconds}s" +
             $"\n  End color (calculated):         {targetHex}" +
             $"\n  Transition time (calculated):   {transitionTimeMs}ms");

        CPH.SetArgument("hue.calculatedEndColor", targetHex);
        CPH.SetArgument("hue.calculatedTransitionTimeMs", transitionTimeMs);
        runAsync(() => updateLightColor(id, targetHex, -1, -1, transitionTimeMs));
        return true;
    }

    public bool transitionGroupColorPerceptual()
    {
        var id = getProperty("hue.id", "UNKNOWN");
        var startColor = getProperty("hue.startColor", "#FF0000");
        var endColor = getProperty("hue.endColor", "#0000FF");
        var durationSeconds = getProperty("hue.durationSeconds", 5.0);
        var deltaERate = getProperty("hue.deltaERate", 0.1);

        var (targetHex, transitionTimeMs) = CalculatePerceptualTransition(
            startColor, endColor, durationSeconds, deltaERate);

        INFO(() => $"Perceptual group transition:" +
             $"\n  Source color (requested):      {startColor}" +
             $"\n  Destination color (requested):  {endColor}" +
             $"\n  Delta-E rate:                   {deltaERate}/s" +
             $"\n  Transition time (requested):    {durationSeconds}s" +
             $"\n  End color (calculated):         {targetHex}" +
             $"\n  Transition time (calculated):   {transitionTimeMs}ms");

        CPH.SetArgument("hue.calculatedEndColor", targetHex);
        CPH.SetArgument("hue.calculatedTransitionTimeMs", transitionTimeMs);
        runAsync(() => updateGroupColor(id, targetHex, -1, -1, transitionTimeMs));
        return true;
    }

    public bool calculatePerceptualTransitionTime()
    {
        var startColor = getProperty("hue.startColor", "#FF0000");
        var endColor = getProperty("hue.endColor", "#0000FF");
        var deltaERate = getProperty("hue.deltaERate", 0.1);

        var totalTimeMs = CalculateTotalTransitionTimeMs(startColor, endColor, deltaERate);

        INFO(() => $"Transition time calculation:" +
             $" src {startColor}," +
             $" dest {endColor}," +
             $" delta-E rate {deltaERate}/s," +
             $" transition time: {totalTimeMs}ms");

        CPH.SetArgument("hue.calculatedTransitionTimeMs", totalTimeMs);
        return true;
    }

    public bool listAllLights()
    {
        runAsync(() => printAllLights());
        return true;
    }

    //----------------------------------------------------------------
    // HUE API METHODS
    //----------------------------------------------------------------
    private void runAsync(Func<Task> action)
    {
        Task.Run(async () =>
        {
            try { await action(); }
            catch (Exception e) { ERROR(() => $"Async operation failed: {e.Message}\n{e.StackTrace}"); }
        });
    }

    private LightCommand buildColorCommand(string hexColor, int saturation, int brightness, int transitionTimeMs = 400)
    {
        var hsb = new RGBColor(hexColor).GetHSB();
        var command = new LightCommand();
        command.TurnOn();
        command.Hue = hsb.Hue;
        command.Saturation = saturation != -1 ? (byte)saturation : (byte)hsb.Saturation;
        command.Brightness = brightness != -1 ? (byte)brightness : (byte)hsb.Brightness;
        command.TransitionTime = TimeSpan.FromMilliseconds(transitionTimeMs);
        return command;
    }

    private async Task updateLightColor(string hueId, string hexColor, int saturation, int brightness, int transitionTimeMs = 400)
    {
        try
        {
            var light = await hueClient.GetLightAsync(hueId);
            if (light != null)
            {
                DEBUG(() => "Updating color for: " + light.Name);
                var command = buildColorCommand(hexColor, saturation, brightness, transitionTimeMs);
                INFO(() => $"Setting light values - Hue: {command.Hue}, Saturation: {command.Saturation}, Brightness: {command.Brightness}");
                await hueClient.SendCommandAsync(command, new[] { light.Id });
                INFO(() => "Color updated to hex:" + hexColor + ", for: " + light.Name);
            }
        }
        catch (Exception e)
        {
            INFO(() => "Cannot change color for: {hue.id: " + hueId + "}: " + e.Message);
        }
    }

    private async Task updateGroupColor(string groupId, string hexColor, int saturation, int brightness, int transitionTimeMs = 400)
    {
        try
        {
            var group = await hueClient.GetGroupAsync(groupId);
            if (group != null)
            {
                DEBUG(() => "Updating color for group: " + group.Name);
                var command = buildColorCommand(hexColor, saturation, brightness, transitionTimeMs);
                INFO(() => $"Setting group values for group {groupId}: Hue {command.Hue}/{hexColor}, Time {transitionTimeMs}ms, Saturation: {command.Saturation}, Brightness: {command.Brightness}");
                await hueClient.SendGroupCommandAsync(command, group.Id);
                INFO(() => "Color updated to hex:" + hexColor + ", for group: " + group.Name);
            }
        }
        catch (Exception e)
        {
            INFO(() => "Cannot change color for: {group.id: " + groupId + "}: " + e.Message);
        }
    }

    private async Task updateHueState(string hueId, bool isOn)
    {
        try
        {
            var device = await hueClient.GetLightAsync(hueId);
            if (device != null)
            {
                DEBUG(() => "Updating state for: " + device.Name);

                var command = new LightCommand();
                if (isOn)
                {
                    command.TurnOn();
                }
                else
                {
                    command.TurnOff();
                }


                await hueClient.SendCommandAsync(command, new[] { device.Id });
                INFO(() => "State updated for: " + device.Name);
            }
        }
        catch (Exception e)
        {
            INFO(() => "Cannot change state for: {hue.id: " + hueId + "}: " + e.Message);
        }
    }

    private async Task printAllLights()
    {
        try
        {
            var lights = await hueClient.GetLightsAsync();
            INFO(() => "");
            INFO(() => "LISTING ALL AVAILABLE HUE LIGHTS");
            INFO(() => "================================");
            foreach (var light in lights)
            {
                INFO(() => "HUE ID    : " + light.Id);
                INFO(() => "HUE NAME  : " + light.Name);
                INFO(() => "HUE TYPE  : " + light.Type);
                INFO(() => "HUE MODEL : " + light.ModelId);
                INFO(() => "HUE STATE : " + (light.State.On ? "ON" : "OFF"));
                INFO(() => "=====================");
            }

            var groups = await hueClient.GetGroupsAsync();
            INFO(() => "");
            INFO(() => "LISTING ALL AVAILABLE HUE GROUPS");
            INFO(() => "================================");
            foreach (var group in groups)
            {
                INFO(() => "HUE GROUP ID    : " + group.Id);
                INFO(() => "HUE GROUP NAME  : " + group.Name);
                INFO(() => "HUE TYPE  : " + group.Type);
                INFO(() => "HUE CLASS : " + group.Class);

                group.Lights.ForEach(lightId =>
                {
                    var light = lights.FirstOrDefault(l => l.Id == lightId);
                    if (light != null)
                    {
                        INFO(() => "  (id: " + light.Id + ") " + light.Name);
                    }
                });
                INFO(() => "=====================");
            }
        }
        catch (Exception e)
        {
            ERROR(() => $"Cannot get lights: {e.Message}" +
                (e.InnerException != null ? $"\nStackTrace: {e.InnerException.Message}" : "") + $"\n{e.StackTrace}"
            );
        }
    }

    private async Task registerNewApplication()
    {
        try
        {
            DEBUG(() => "REGISTERING NEW APPLICATION");

            hueClient = new LocalHueClient(hueHubIpAddress);
            var registrationResult = await hueClient.RegisterAsync("SBotHueApplication", "StreamerBot", false);
            if (registrationResult == null)
            {
                DEBUG(() => "CANNOT CONNECT TO HUE BRIDGE");
                return;
            }

            hueHubAppKey = registrationResult.Username;
            CPH.SetGlobalVar(HUE_BRIDGE_APP_KEY_PROPERTY, hueHubAppKey, true);

            DEBUG(() => "REGISTERING OF APPLICATION COMPLETED");
        }
        catch (Exception e)
        {
            INFO(() => "Cannot register SBot: " + e.Message);
        }
    }

    //----------------------------------------------------------------
    // OKLCH PERCEPTUAL TRANSITION
    //----------------------------------------------------------------
    private (string targetHex, int transitionTimeMs) CalculatePerceptualTransition(
        string startHex, string endHex, double durationSeconds, double deltaERate)
    {
        if (deltaERate <= 0 || durationSeconds <= 0)
        {
            WARN(() => $"Invalid transition params: deltaERate={deltaERate}, duration={durationSeconds}s. Using instant transition.");
            return (endHex, 0);
        }

        var startLab = HexToOklab(startHex);
        var endLab = HexToOklab(endHex);
        var totalDeltaE = DeltaEOklab(startLab, endLab);

        if (totalDeltaE < 1e-9)
        {
            DEBUG(() => "Start and end colors are identical, no transition needed.");
            return (endHex, 0);
        }

        var neededSeconds = totalDeltaE / deltaERate;

        if (neededSeconds <= durationSeconds)
        {
            DEBUG(() => $"Full transition in {neededSeconds:F2}s (requested {durationSeconds:F2}s). totalDeltaE={totalDeltaE:F4}");
            return (endHex, (int)(neededSeconds * 1000));
        }

        var t = durationSeconds / neededSeconds;
        var targetHex = InterpolateHexColor(startHex, endHex, t);

        DEBUG(() => $"Partial transition t={t:F4} in {durationSeconds:F2}s. totalDeltaE={totalDeltaE:F4}, needed={neededSeconds:F2}s");
        return (targetHex, (int)(durationSeconds * 1000));
    }

    private int CalculateTotalTransitionTimeMs(string startHex, string endHex, double deltaERate)
    {
        if (deltaERate <= 0)
        {
            WARN(() => $"Invalid deltaERate={deltaERate}. Returning 0.");
            return 0;
        }

        var startLab = HexToOklab(startHex);
        var endLab = HexToOklab(endHex);
        var totalDeltaE = DeltaEOklab(startLab, endLab);

        if (totalDeltaE < 1e-9)
        {
            DEBUG(() => "Start and end colors are identical, no transition needed.");
            return 0;
        }

        return (int)(totalDeltaE / deltaERate * 1000);
    }

    //----------------------------------------------------------------
    // OKLCH COLOR MATH
    //----------------------------------------------------------------
    private static double SrgbToLinear(double c)
    {
        return c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
    }

    private static double LinearToSrgb(double c)
    {
        return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.Pow(c, 1.0 / 2.4) - 0.055;
    }

    private static (double r, double g, double b) HexToLinearRgb(string hex)
    {
        hex = hex.TrimStart('#');
        var ri = Convert.ToInt32(hex.Substring(0, 2), 16);
        var gi = Convert.ToInt32(hex.Substring(2, 2), 16);
        var bi = Convert.ToInt32(hex.Substring(4, 2), 16);
        return (SrgbToLinear(ri / 255.0), SrgbToLinear(gi / 255.0), SrgbToLinear(bi / 255.0));
    }

    private static string LinearRgbToHex(double r, double g, double b)
    {
        var ri = (int)Math.Round(Math.Min(1, Math.Max(0, LinearToSrgb(r))) * 255);
        var gi = (int)Math.Round(Math.Min(1, Math.Max(0, LinearToSrgb(g))) * 255);
        var bi = (int)Math.Round(Math.Min(1, Math.Max(0, LinearToSrgb(b))) * 255);
        return $"#{ri:X2}{gi:X2}{bi:X2}";
    }

    private static (double L, double a, double b) LinearRgbToOklab(double r, double g, double b)
    {
        // RGB to LMS (Ottosson's matrix)
        var l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        var m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        var s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        // Cube root (sign-preserving for safety)
        var lp = Cbrt(l);
        var mp = Cbrt(m);
        var sp = Cbrt(s);

        // LMS' to Lab
        return (
            0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
            1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
            0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp
        );
    }

    private static (double r, double g, double b) OklabToLinearRgb(double L, double a, double b)
    {
        // Lab to LMS' (inverse of LMS' to Lab)
        var lp = L + 0.3963377774 * a + 0.2158037573 * b;
        var mp = L - 0.1055613458 * a - 0.0638541728 * b;
        var sp = L - 0.0894841775 * a - 1.2914855480 * b;

        // Cube each to undo cube root
        var l = lp * lp * lp;
        var m = mp * mp * mp;
        var s = sp * sp * sp;

        // LMS to RGB (inverse of RGB to LMS)
        return (
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        );
    }

    private static (double L, double C, double H) OklabToOklch(double L, double a, double b)
    {
        var C = Math.Sqrt(a * a + b * b);
        var H = Math.Atan2(b, a);
        if (H < 0) H += 2 * Math.PI;
        return (L, C, H);
    }

    private static (double L, double a, double b) OklchToOklab(double L, double C, double H)
    {
        return (L, C * Math.Cos(H), C * Math.Sin(H));
    }

    private static (double L, double a, double b) HexToOklab(string hex)
    {
        var (r, g, b) = HexToLinearRgb(hex);
        return LinearRgbToOklab(r, g, b);
    }

    private static double DeltaEOklab(
        (double L, double a, double b) lab1,
        (double L, double a, double b) lab2)
    {
        var dL = lab1.L - lab2.L;
        var da = lab1.a - lab2.a;
        var db = lab1.b - lab2.b;
        return Math.Sqrt(dL * dL + da * da + db * db);
    }

    private static (double L, double C, double H) LerpOklch(
        (double L, double C, double H) from,
        (double L, double C, double H) to,
        double t)
    {
        var L = from.L + (to.L - from.L) * t;
        var C = from.C + (to.C - from.C) * t;

        // Shortest-arc hue interpolation
        var dH = to.H - from.H;
        if (dH > Math.PI) dH -= 2 * Math.PI;
        if (dH < -Math.PI) dH += 2 * Math.PI;
        var H = from.H + dH * t;
        if (H < 0) H += 2 * Math.PI;
        if (H >= 2 * Math.PI) H -= 2 * Math.PI;

        return (L, C, H);
    }

    private static string InterpolateHexColor(string startHex, string endHex, double t)
    {
        var (sr, sg, sb) = HexToLinearRgb(startHex);
        var startLab = LinearRgbToOklab(sr, sg, sb);
        var startLch = OklabToOklch(startLab.L, startLab.a, startLab.b);

        var (er, eg, eb) = HexToLinearRgb(endHex);
        var endLab = LinearRgbToOklab(er, eg, eb);
        var endLch = OklabToOklch(endLab.L, endLab.a, endLab.b);

        var lerpLch = LerpOklch(startLch, endLch, t);
        var (la, aa, ba) = OklchToOklab(lerpLch.L, lerpLch.C, lerpLch.H);
        var (r, g, b) = OklabToLinearRgb(la, aa, ba);

        return LinearRgbToHex(r, g, b);
    }

    private static double Cbrt(double x)
    {
        return x >= 0 ? Math.Pow(x, 1.0 / 3.0) : -Math.Pow(-x, 1.0 / 3.0);
    }

    //----------------------------------------------------------------
    // DEFAULT METHODS AND SETUP
    //----------------------------------------------------------------
    private bool isDebugEnabled;
    private bool isInitialized;
    private string widgetActionName = "TEMPLATE";

    // ReSharper disable once UnusedMember.Global
    public bool Execute()
    {
        setUp();
        return true;
    }

    private void setUp()
    {
        if (isInitialized)
        {
            return;
        }
        widgetActionName = getProperty("actionName", "TEMPLATE");
        isDebugEnabled = getProperty("hue.debug", false);

        INFO(() => "INITIAL SETUP");
        init();

        isInitialized = true;
    }

    private T getProperty<T>(string key, T defaultValue)
    {
        var result = CPH.TryGetArg(key, out T value);
        DEBUG(() => "{key: " + key + ", value: " + value + ", default: " + defaultValue + "}");

        return result ?
            !value.Equals("") ?
                value
                : defaultValue
            : defaultValue;
    }

    private void DEBUG(Func<string> getMessage)
    {
        if (!isDebugEnabled)
        {
            return;
        }

        CPH.LogInfo("DEBUG: " + widgetActionName + " :: " + getMessage());
    }

    private void INFO(Func<string> getMessage)
    {
        CPH.LogInfo("INFO : " + widgetActionName + " :: " + getMessage());
    }

    private void WARN(Func<string> getMessage)
    {
        CPH.LogWarn("WARN : " + widgetActionName + " :: " + getMessage());
    }

    private void ERROR(Func<string> getMessage)
    {
        CPH.LogError("ERROR: " + widgetActionName + " :: " + getMessage());
    }
}

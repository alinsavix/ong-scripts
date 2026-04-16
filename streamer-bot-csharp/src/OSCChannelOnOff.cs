using System;
using SharpOSC;
using System;

#if EXTERNAL_EDITOR
public class OSCChannelOnOff : CPHInlineBase
#else
public class CPHInline
#endif
{
    public bool Execute()
    {

        if (!CPH.TryGetArg("oscIP", out string oscIP))
        {
            CPH.LogError("OSCChannelOnOff: Failed to get oscIP argument.");
            return false;
        }

        if (!CPH.TryGetArg("oscChannels", out string oscChannels))
        {
            CPH.LogError("OSCChannelOnOff: Failed to get oscChannels argument.");
            return false;
        }

        if (!CPH.TryGetArg("oscOnOff", out int oscOnOff))
        {
            CPH.LogError("OSCChannelOnOff: Failed to get oscOnOff argument.");
            return false;
        }

        var sender = new SharpOSC.UDPSender(oscIP, 10024);

        string[] channels = oscChannels.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string channelStr in channels)
        {
            if (int.TryParse(channelStr.Trim(), out int channelNum))
            {
                string channelFormatted = channelNum.ToString("D2"); // 2-digit string
                var message = new SharpOSC.OscMessage($"/ch/{channelFormatted}/mix/on", oscOnOff);
                sender.Send(message);
            }
            else
            {
                CPH.LogWarn($"OSCChannelOnOff: Invalid channel number '{channelStr.Trim()}' - skipping.");
            }
        }

        return true;
    }
}

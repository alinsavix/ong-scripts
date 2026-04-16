using System;
using System.Globalization;

public class CPHInline
{
    public bool Execute()
    {
        string htmlhexcolor = args["hexColor"].ToString();

        htmlhexcolor = htmlhexcolor.TrimStart('#');

        // string input1 = args["alphaPercent"].ToString();
        string input1 = "100";
        decimal alphaPercent = 0m;
        if (decimal.TryParse(input1, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out alphaPercent))
        {
            decimal decimalValue = Decimal.Round(alphaPercent * 255 / 100);
            int decimalToInt = Decimal.ToInt32(decimalValue);
            string hexValue = Convert.ToString(decimalToInt, 16);
            string[] split = new string[htmlhexcolor.Length / 2 + (htmlhexcolor.Length % 2 == 0 ? 0 : 1)];
            for (int i = 0; i < split.Length; i++)
            {
                split[i] = htmlhexcolor.Substring(i * 2, i * 2 + 2 > htmlhexcolor.Length ? 1 : 2);
            }

            string R = split[0];
            string G = split[1];
            string B = split[2];
            string A = hexValue.ToString().ToUpper();
            string abgrHexColor = (A + B + G + R);
            long obsColor = Convert.ToInt64(abgrHexColor, 16);
            string abgrColor = obsColor.ToString();
            CPH.SetArgument("abgrColor", abgrColor);

            return true;
        }
        else
        {
            // handle invalid input1
            return false;
        }
    }
}

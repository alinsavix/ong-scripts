using System;

#if EXTERNAL_EDITOR
public class PickRecenterColor : CPHInlineBase
#else
public class CPHInline
#endif
{
    public bool Execute()
    {
        if (!CPH.TryGetArg("recenter_colors", out string colorChoices))
        {
            colorChoices = "#4d8dff";
        }

        if (!CPH.TryGetArg("recenter_index", out int colorIndex))
        {
            colorIndex = 0;
        }

        string[] colors = colorChoices.Split(',');
        for (int i = 0; i < colors.Length; i++)
            colors[i] = colors[i].Trim();

        string selectedColor = colors[colorIndex % colors.Length];
        CPH.SetArgument("selected_color", selectedColor);

        return true;
    }
}

using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

#if EXTERNAL_EDITOR
public class ParseJson : CPHInlineBase
#else
public class CPHInline
#endif
{
    public bool Execute()
    {
        if (!CPH.TryGetArg("json", out string json))
        {
            CPH.LogWarn($"Parse JSON :: Missing argument 'json', returning...");
            return true;
        }

        if (!CPH.TryGetArg("prefix", out string prefix))
        {
            prefix = "json";
        }

        JToken token = JToken.Parse(json);
        Dictionary<string, object> dict = new Dictionary<string, object>();
        JTokenToDict(dict, token, prefix);

        foreach (KeyValuePair<string, object> item in dict)
        {
            CPH.SetArgument(item.Key, item.Value);
        }

        return true;
    }

    private static void JTokenToDict(Dictionary<string, object> dict, JToken token, string prefix)
    {
        switch (token.Type)
        {
            case JTokenType.Object:
                foreach (JProperty property in token.Children<JProperty>())
                {
                    string newPrefix = string.IsNullOrEmpty(prefix) ? property.Name : $"{prefix}.{property.Name}";
                    JTokenToDict(dict, property.Value, newPrefix);
                }
                break;
            case JTokenType.Array:
                int index = 0;
                foreach (JToken arrayItem in token.Children())
                {
                    string newPrefix = $"{prefix}[{index}]";
                    JTokenToDict(dict, arrayItem, newPrefix);
                    index++;
                }
                break;
            default:
                if (!string.IsNullOrEmpty(prefix))
                {
                    dict[prefix] = ((JValue)token).Value;
                }
                break;
        }
    }
}

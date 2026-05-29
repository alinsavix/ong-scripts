using System;
using System.IO;

#if EXTERNAL_EDITOR
public class SbGetInfo : CPHInlineBase
#else
public class CPHInline
#endif
{
	public bool Execute()
	{
        string version = CPH.GetVersion();
        CPH.SetArgument("botVersion", version);

        string topLevelPath = AppDomain.CurrentDomain.BaseDirectory;
        CPH.SetArgument("botTopLevelPath", topLevelPath);

        CPH.LogInfo($"SbGetInfo: topLevel='{topLevelPath}', version='{version}'");

        return true;
	}
}

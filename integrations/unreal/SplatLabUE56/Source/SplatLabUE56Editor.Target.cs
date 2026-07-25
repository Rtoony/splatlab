using UnrealBuildTool;
using System.Collections.Generic;

public class SplatLabUE56EditorTarget : TargetRules
{
    public SplatLabUE56EditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_6;
        ExtraModuleNames.Add("SplatLabUE56");
    }
}

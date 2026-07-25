using UnrealBuildTool;
using System.Collections.Generic;

public class SplatLabUE56Target : TargetRules
{
    public SplatLabUE56Target(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_6;
        ExtraModuleNames.Add("SplatLabUE56");
    }
}

// Real tooltips (delay, positioning, aria) replacing native title= attributes
// on high-traffic controls. Wrap the app once in <TooltipProvider>.
import { Tooltip as RadixTooltip } from "radix-ui";
import { type ReactNode } from "react";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <RadixTooltip.Provider delayDuration={350}>{children}</RadixTooltip.Provider>;
}

export function Tooltip({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          sideOffset={6}
          className="z-[60] max-w-xs rounded-lg border border-white/10 bg-[#0a0f1a] px-2.5 py-1.5 text-xs leading-snug text-zinc-200 shadow-xl"
        >
          {label}
          <RadixTooltip.Arrow className="fill-[#0a0f1a]" />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

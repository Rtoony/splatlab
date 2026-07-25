// Mode-tab strip on Radix Tabs (keyboard nav + aria). Visual style matches
// the scene-regen stage rail: pill triggers, cyan active state.
import { Tabs as RadixTabs } from "radix-ui";
import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Tabs({
  value,
  onValueChange,
  className,
  children,
}: {
  value: string;
  onValueChange: (value: string) => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <RadixTabs.Root value={value} onValueChange={onValueChange} className={className}>
      {children}
    </RadixTabs.Root>
  );
}

export function TabsList({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <RadixTabs.List className={cn("flex items-center gap-1 overflow-x-auto", className)}>{children}</RadixTabs.List>
  );
}

export function TabsTrigger({
  value,
  className,
  children,
}: {
  value: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <RadixTabs.Trigger
      value={value}
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-bold transition",
        "border-white/10 bg-white/[0.03] text-zinc-400 hover:text-zinc-100",
        "data-[state=active]:border-cyan-300/45 data-[state=active]:bg-cyan-300/15 data-[state=active]:text-cyan-100",
        className,
      )}
    >
      {children}
    </RadixTabs.Trigger>
  );
}

export function TabsContent({
  value,
  className,
  children,
}: {
  value: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <RadixTabs.Content value={value} className={cn("focus:outline-none", className)}>
      {children}
    </RadixTabs.Content>
  );
}

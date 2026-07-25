// Modal shell on Radix Dialog: focus trap, Escape, aria wiring for free —
// styled to match the existing hand-rolled fixed-inset modals it replaces.
// Two layouts: "panel" (centered card, site-sections style) and "screen"
// (near-fullscreen workspace, scene-regen / geo-locate style).
import { Dialog as RadixDialog } from "radix-ui";
import { type ReactNode } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";

export function Dialog({
  open,
  onOpenChange,
  title,
  layout = "panel",
  hideClose = false,
  className,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Accessible name for the dialog. Rendered visually only if you include it in children. */
  title: string;
  layout?: "panel" | "screen";
  hideClose?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
        <RadixDialog.Content
          className={cn(
            "fixed z-50 border border-white/10 bg-surface text-zinc-100 shadow-2xl focus:outline-none",
            layout === "screen"
              ? "inset-2 flex min-w-0 flex-col overflow-hidden rounded-2xl sm:inset-4"
              : "left-1/2 top-1/2 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-2xl",
            className,
          )}
        >
          <RadixDialog.Title className="sr-only">{title}</RadixDialog.Title>
          {!hideClose && (
            <RadixDialog.Close
              aria-label="Close"
              className="absolute right-3 top-3 z-10 rounded-lg p-1.5 text-zinc-400 hover:bg-white/10 hover:text-zinc-100"
            >
              <X className="h-4 w-4" />
            </RadixDialog.Close>
          )}
          {children}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
}

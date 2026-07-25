// Overflow / action menus on Radix DropdownMenu (positioning, keyboard nav,
// aria). Styled like the existing DownloadMenu popover.
import { DropdownMenu as RadixDropdownMenu } from "radix-ui";
import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export function DropdownMenu({ trigger, children }: { trigger: ReactNode; children: ReactNode }) {
  return (
    <RadixDropdownMenu.Root>
      <RadixDropdownMenu.Trigger asChild>{trigger}</RadixDropdownMenu.Trigger>
      <RadixDropdownMenu.Portal>
        <RadixDropdownMenu.Content
          sideOffset={4}
          align="end"
          className="z-[60] min-w-48 overflow-hidden rounded-xl border border-white/10 bg-[#0a0f1a] py-1 shadow-2xl"
        >
          {children}
        </RadixDropdownMenu.Content>
      </RadixDropdownMenu.Portal>
    </RadixDropdownMenu.Root>
  );
}

export function DropdownItem({
  onSelect,
  className,
  children,
}: {
  onSelect?: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <RadixDropdownMenu.Item
      onSelect={onSelect}
      className={cn(
        "cursor-pointer px-3 py-2 text-sm text-zinc-200 outline-none data-[highlighted]:bg-white/5",
        className,
      )}
    >
      {children}
    </RadixDropdownMenu.Item>
  );
}

export function DropdownSeparator() {
  return <RadixDropdownMenu.Separator className="my-1 h-px bg-white/10" />;
}

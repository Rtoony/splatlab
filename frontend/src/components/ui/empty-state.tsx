import { type ReactNode } from "react";
import { cn } from "@/lib/cn";

export function EmptyState({
  icon,
  title,
  hint,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  hint?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-6 py-12 text-center", className)}>
      {icon && <div className="mb-1 text-zinc-600">{icon}</div>}
      <p className="text-sm font-semibold text-zinc-300">{title}</p>
      {hint && <p className="max-w-sm text-xs leading-relaxed text-zinc-500">{hint}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

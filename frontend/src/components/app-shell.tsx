// App shell: the slim top nav shared by Scenes / New Capture / Feedback.
// The viewer (/view/:jobId) stays fullscreen with its own header — it is a
// workspace, not a browsing page. Access is tunnel-gated, so there is no
// user session to show or log out of.
import { type ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { DropdownItem, DropdownMenu } from "@/components/ui";
import { MessageSquareText, MoreHorizontal, Plus } from "lucide-react";
import { cn } from "@/lib/cn";

export function AppShell({ children }: { children: ReactNode }) {
  const [location, navigate] = useLocation();
  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-40 border-b border-white/10 bg-surface/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1880px] items-center gap-3 px-4 sm:px-6 xl:px-10">
          <Link href="/" className="flex shrink-0 items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="h-7 w-7 rounded-lg" />
            <span className="display text-lg font-black tracking-tight text-white">SplatLab</span>
          </Link>
          <div className="ml-3 flex items-center gap-1">
            <Link
              href="/"
              className={cn(
                "rounded-full px-3 py-1.5 text-xs font-bold transition",
                location === "/" ? "bg-white/10 text-zinc-100" : "text-zinc-400 hover:text-zinc-100",
              )}
            >
              Scenes
            </Link>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Link
              href="/new"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition-colors",
                location === "/new"
                  ? "bg-white/10 text-zinc-100"
                  : "bg-accent text-accent-ink hover:bg-accent-hover",
              )}
            >
              <Plus className="h-3.5 w-3.5" /> New capture
            </Link>
            <DropdownMenu
              trigger={
                <button
                  type="button"
                  aria-label="More"
                  className="rounded-xl border border-white/15 bg-white/5 p-2 text-zinc-300 hover:bg-white/10"
                >
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              }
            >
              <DropdownItem onSelect={() => navigate("/feedback")}>
                <span className="inline-flex items-center gap-2">
                  <MessageSquareText className="h-3.5 w-3.5" /> Feedback
                </span>
              </DropdownItem>
            </DropdownMenu>
          </div>
        </div>
      </nav>
      <main>{children}</main>
    </div>
  );
}

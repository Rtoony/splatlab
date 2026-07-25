// Route-level error boundary: a viewer/page crash renders a recover card
// instead of a white screen. State is reset by the Reload action.
import { Component, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 px-6 text-center">
        <AlertTriangle className="h-8 w-8 text-amber-300" />
        <p className="text-sm font-semibold text-zinc-200">Something broke rendering this page.</p>
        <p className="max-w-md break-all font-mono text-xs text-zinc-500">{String(this.state.error)}</p>
        <div className="mt-2 flex items-center gap-3">
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-accent-ink hover:bg-accent-hover"
          >
            Reload
          </button>
          <a href="/" className="text-sm text-cyan-300 hover:underline">
            Back to Scenes
          </a>
        </div>
        <p className="mt-1 text-[11px] text-zinc-600">If this keeps happening, file it via Feedback (⋯ menu).</p>
      </div>
    );
  }
}

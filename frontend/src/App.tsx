import { lazy, Suspense } from "react";
import { Route, Switch } from "wouter";
import { AppShell } from "@/components/app-shell";
import { ErrorBoundary } from "@/components/error-boundary";
import { FeedbackWidget } from "@/components/feedback-widget";
import { Compass } from "lucide-react";

const ScenesPage = lazy(() => import("@/pages/scenes"));
const NewCapturePage = lazy(() => import("@/pages/new-capture"));
const SplatViewPage = lazy(() => import("@/pages/splat-view"));
const FeedbackPage = lazy(() => import("@/pages/feedback"));
const SparkTestPage = lazy(() => import("@/pages/spark-test"));

function Loading() {
  return (
    <div className="flex h-screen items-center justify-center text-xs uppercase tracking-[0.3em] text-zinc-600">
      Loading SplatLab…
    </div>
  );
}

function NotFound() {
  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center gap-3 px-6 text-center">
      <Compass className="h-9 w-9 text-zinc-600" />
      <p className="display text-xl font-black text-white">Nothing at this address</p>
      <p className="max-w-sm text-sm text-zinc-500">
        The page you're after doesn't exist — it may have moved when SplatLab's layout was reorganized.
      </p>
      <a
        href="/"
        className="mt-1 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-accent-ink hover:bg-accent-hover"
      >
        Back to Scenes
      </a>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<Loading />}>
        <Switch>
          {/* Fullscreen workspace — no app shell. */}
          <Route path="/view/:jobId">
            <SplatViewPage />
          </Route>
          <Route path="/spark-test">
            <SparkTestPage />
          </Route>
          <Route path="/feedback">
            <AppShell>
              <FeedbackPage />
            </AppShell>
          </Route>
          <Route path="/new">
            <AppShell>
              <NewCapturePage />
            </AppShell>
          </Route>
          <Route path="/">
            <AppShell>
              <ScenesPage />
            </AppShell>
          </Route>
          <Route>
            <AppShell>
              <NotFound />
            </AppShell>
          </Route>
        </Switch>
      </Suspense>
      <FeedbackWidget />
    </ErrorBoundary>
  );
}

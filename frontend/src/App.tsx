import { lazy, Suspense } from "react";
import { Route, Switch } from "wouter";
import { FeedbackWidget } from "@/components/feedback-widget";

const SplatLabPage = lazy(() => import("@/pages/splat"));
const SplatViewPage = lazy(() => import("@/pages/splat-view"));
const FeedbackPage = lazy(() => import("@/pages/feedback"));

function Loading() {
  return (
    <div className="flex h-screen items-center justify-center text-xs uppercase tracking-[0.3em] text-zinc-600">
      Loading Splat Lab…
    </div>
  );
}

export default function App() {
  return (
    <>
      <Suspense fallback={<Loading />}>
        <Switch>
          <Route path="/feedback">
            <FeedbackPage />
          </Route>
          <Route path="/view/:jobId">
            <SplatViewPage />
          </Route>
          <Route path="/">
            <SplatLabPage />
          </Route>
          <Route>
            <div className="flex h-screen flex-col items-center justify-center gap-3 text-zinc-400">
              <p>Page not found.</p>
              <a href="/" className="text-cyan-300 hover:underline">
                Back to Splat Lab
              </a>
            </div>
          </Route>
        </Switch>
      </Suspense>
      <FeedbackWidget />
    </>
  );
}

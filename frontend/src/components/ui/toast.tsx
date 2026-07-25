// App-wide toast provider replacing the page-local setTimeout toast divs.
// Hand-rolled on purpose (~80 lines): matches the existing visual language
// exactly and needs no positioning library. Stacks bottom-center, auto-
// dismisses, pauses nothing — these are short status pings, not alerts.
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/cn";

type ToastTone = "info" | "success" | "error";
type ToastItem = { id: number; message: string; tone: ToastTone };

const ToastContext = createContext<(message: string, tone?: ToastTone) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

const TONE_STYLE: Record<ToastTone, string> = {
  info: "border-cyan-300/30 bg-cyan-950/90 text-cyan-100",
  success: "border-emerald-300/30 bg-emerald-950/90 text-emerald-100",
  error: "border-red-300/30 bg-red-950/90 text-red-100",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const push = useCallback((message: string, tone: ToastTone = "info") => {
    const id = nextId.current++;
    setToasts((prev) => [...prev.slice(-3), { id, message, tone }]);
    window.setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4200);
  }, []);

  return (
    <ToastContext.Provider value={push}>
      {children}
      {toasts.length > 0 && (
        <div className="pointer-events-none fixed inset-x-0 bottom-5 z-[70] flex flex-col items-center gap-2 px-4">
          {toasts.map((t) => (
            <div
              key={t.id}
              role="status"
              className={cn(
                "pointer-events-auto max-w-md rounded-xl border px-4 py-2.5 text-sm font-medium shadow-2xl backdrop-blur",
                TONE_STYLE[t.tone],
              )}
            >
              {t.message}
            </div>
          ))}
        </div>
      )}
    </ToastContext.Provider>
  );
}

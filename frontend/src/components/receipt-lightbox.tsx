// Shared full-screen image lightbox — extracted from scene-regen.tsx so both
// it and site-sections.tsx (and any future receipt-image panel) use the same
// component instead of duplicating it.
import { ChevronLeft, ChevronRight, X } from "lucide-react";

export default function ReceiptLightbox({
  src,
  onClose,
  onPrev,
  onNext,
}: {
  src: string;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/85 p-6"
      onClick={onClose}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
      role="presentation"
    >
      <button type="button" onClick={onClose} className="absolute right-5 top-5 rounded-full bg-black/40 p-2 text-zinc-200 hover:bg-black/70" title="Close (Esc)">
        <X className="h-5 w-5" />
      </button>
      {onPrev && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onPrev();
          }}
          className="absolute left-5 rounded-full bg-black/40 p-2 text-zinc-200 hover:bg-black/70"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
      )}
      <img src={src} alt="" className="max-h-[85vh] max-w-[85vw] rounded-xl object-contain" onClick={(e) => e.stopPropagation()} />
      {onNext && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
          className="absolute right-5 rounded-full bg-black/40 p-2 text-zinc-200 hover:bg-black/70"
        >
          <ChevronRight className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}

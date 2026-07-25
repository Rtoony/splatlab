// Design-system barrel: `@/components/ui` keeps resolving here after the
// ui.tsx → ui/ folder split, so existing imports stay untouched.
export { Badge, Button, Card, Input, SectionLabel } from "./primitives";
export { Dialog } from "./dialog";
export { Tooltip, TooltipProvider } from "./tooltip";
export { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";
export { DropdownItem, DropdownMenu, DropdownSeparator } from "./dropdown-menu";
export { ToastProvider, useToast } from "./toast";
export { Skeleton } from "./skeleton";
export { EmptyState } from "./empty-state";

export const FEEDBACK_TYPES = ["Comment", "Idea", "Bug", "UX", "Copy", "Data", "Performance", "Request", "Question"] as const;
export const FEEDBACK_PRIORITIES = ["Low", "Medium", "High", "Critical"] as const;
export const FEEDBACK_STATUSES = [
  "New",
  "Triaged",
  "Planned",
  "In Progress",
  "Needs Info",
  "Ready to Test",
  "Fixed",
  "Accepted",
  "Closed",
  "Won't Fix",
  "Archived",
] as const;

export type FeedbackType = (typeof FEEDBACK_TYPES)[number];
export type FeedbackPriority = (typeof FEEDBACK_PRIORITIES)[number];
export type FeedbackStatus = (typeof FEEDBACK_STATUSES)[number];

export type FeedbackQueueKey = "active" | "codex" | "user" | "terminal" | "all";

export const FEEDBACK_QUEUES: Record<FeedbackQueueKey, { label: string; statuses: FeedbackStatus[] | null }> = {
  active: {
    label: "Active Loop",
    statuses: ["New", "Triaged", "Planned", "In Progress", "Needs Info", "Ready to Test", "Fixed"],
  },
  codex: { label: "Codex Queue", statuses: ["New", "Triaged", "Planned", "In Progress"] },
  user: { label: "Ready For User/Test", statuses: ["Needs Info", "Ready to Test", "Fixed"] },
  terminal: { label: "Accepted/Closed/Archived", statuses: ["Accepted", "Closed", "Won't Fix", "Archived"] },
  all: { label: "All", statuses: null },
};

export interface FeedbackAttachment {
  id: string | number;
  filename?: string;
  original_name?: string;
  name?: string;
  content_type?: string;
  mime_type?: string;
  size_bytes?: number;
  url?: string;
  download_url?: string;
  created_at?: string;
}

export interface FeedbackComment {
  id: string | number;
  body: string;
  created_by?: string;
  author?: string;
  created_at?: string;
}

export interface FeedbackContextSnapshot {
  feedback_context_version: 1;
  captured_at: string;
  route: {
    url: string;
    path: string;
    search?: string;
    tab?: string | null;
  };
  ui_state: {
    component_label?: string | null;
    scroll_x: number;
    scroll_y: number;
    splatlab?: unknown;
  };
  active_element?: Record<string, unknown> | null;
  last_click?: Record<string, unknown> | null;
  browser: {
    user_agent: string;
    language: string;
    timezone: string;
    viewport: { width: number; height: number };
    screen: { width: number; height: number; device_pixel_ratio: number };
  };
  recent_js_errors: Array<Record<string, unknown>>;
  recent_failed_api_calls: Array<Record<string, unknown>>;
  app_context?: Record<string, unknown> | null;
}

export interface FeedbackItem {
  id: string | number;
  title: string;
  body?: string;
  details?: string;
  feedback_type?: FeedbackType | string;
  type?: FeedbackType | string;
  priority: FeedbackPriority | string;
  status: FeedbackStatus | string;
  page_url?: string;
  page_path?: string;
  page_tab?: string | null;
  component_label?: string | null;
  tags?: string[] | string;
  tags_json?: string[] | string;
  context_json?: FeedbackContextSnapshot | string | null;
  resolution_notes?: string;
  resolution_metadata_json?: Record<string, unknown> | string | null;
  created_by?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string | null;
  attachments?: FeedbackAttachment[];
  comments?: FeedbackComment[];
  attachments_count?: number;
  comments_count?: number;
}

export interface CreateFeedbackPayload {
  title: string;
  body: string;
  feedback_type: FeedbackType;
  priority: FeedbackPriority;
  status: FeedbackStatus;
  page_url: string;
  page_path: string;
  page_tab?: string | null;
  component_label?: string | null;
  context_json: FeedbackContextSnapshot;
  tags_json?: string[];
}

export type UpdateFeedbackPayload = Partial<
  Pick<
    FeedbackItem,
    "title" | "body" | "status" | "priority" | "feedback_type" | "tags_json" | "resolution_notes"
  >
>;

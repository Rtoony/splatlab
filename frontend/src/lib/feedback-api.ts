import { apiRequest } from "@/lib/api";
import type { CreateFeedbackPayload, FeedbackComment, FeedbackItem, UpdateFeedbackPayload } from "@/lib/feedback-contracts";

type FeedbackListResponse = FeedbackItem[] | { items?: FeedbackItem[]; feedback?: FeedbackItem[]; results?: FeedbackItem[] };

export async function listFeedback(queue = "all"): Promise<FeedbackItem[]> {
  const data = await apiRequest<FeedbackListResponse>(`/api/feedback?queue=${encodeURIComponent(queue)}`);
  return Array.isArray(data) ? data : data.items ?? data.feedback ?? data.results ?? [];
}

export function getFeedback(id: string | number): Promise<FeedbackItem> {
  return apiRequest<FeedbackItem>(`/api/feedback/${encodeURIComponent(String(id))}`);
}

export function createFeedback(payload: CreateFeedbackPayload): Promise<FeedbackItem> {
  return apiRequest<FeedbackItem>("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateFeedback(id: string | number, payload: UpdateFeedbackPayload): Promise<FeedbackItem> {
  return apiRequest<FeedbackItem>(`/api/feedback/${encodeURIComponent(String(id))}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function addFeedbackComment(id: string | number, body: string): Promise<FeedbackComment> {
  return apiRequest<FeedbackItem>(`/api/feedback/${encodeURIComponent(String(id))}/comments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ body }),
  });
}

export function uploadFeedbackAttachment(id: string | number, file: File): Promise<unknown> {
  const form = new FormData();
  form.append("files", file);
  return apiRequest<unknown>(`/api/feedback/${encodeURIComponent(String(id))}/attachments`, {
    method: "POST",
    body: form,
  });
}

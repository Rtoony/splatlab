export function captureSummary(receipt) {
  const views = receipt.validation;
  if (!Array.isArray(views) || !views.length || new Set(views.map(view => view.image)).size !== views.length) {
    throw new Error("Capture validation views must be nonempty and unique");
  }
  const matches = views.map(view => /^(?:[^/]+\/)?lens-[01]-(frame-\d{6})-(?:left|centre|right)\.png$/.exec(view.image));
  const timestampCount = matches.every(Boolean) ? new Set(matches.map(match => match[1])).size : null;
  const label = timestampCount === null
    ? `${views.length} validation views; independent timestamp grouping unavailable.`
    : `${views.length} validation crops at ${timestampCount} appearance-reserved timestamps; crops overlap.`;
  return {viewCount: views.length, timestampCount, label};
}

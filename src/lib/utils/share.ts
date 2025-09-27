/**
 * Share content using Web Share API with fallback to clipboard
 * Returns true if content was successfully shared or copied
 */
export async function shareContent(options: {
  title?: string;
  text?: string;
  url?: string;
}): Promise<boolean> {
  // Check if Web Share API is available and we have shareable content
  const canShare = 'share' in navigator &&
    (options.url || options.text) &&
    // Web Share requires HTTPS or localhost
    (window.location.protocol === 'https:' || window.location.hostname === 'localhost');

  if (canShare) {
    try {
      // Filter out undefined values for Web Share API
      const shareData: ShareData = {};
      if (options.title) shareData.title = options.title;
      if (options.text) shareData.text = options.text;
      if (options.url) shareData.url = options.url;

      // Attempt native share
      await navigator.share(shareData);
      return true; // User completed share action (even if cancelled)
    } catch (error) {
      // User cancelled or share failed, fall back to clipboard
      if (error instanceof Error && error.name === 'AbortError') {
        // User cancelled share - this is not a failure
        return false;
      }
      // Other error, try clipboard fallback
      console.warn('Web Share API failed, falling back to clipboard:', error);
    }
  }

  // Fallback to clipboard
  try {
    // Combine text and URL for clipboard
    const clipboardText = options.text && options.url
      ? `${options.text}\n${options.url}`
      : options.text || options.url || '';

    await navigator.clipboard.writeText(clipboardText);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
}

/**
 * Check if Web Share API is available
 */
export function canNativeShare(): boolean {
  return 'share' in navigator &&
    (window.location.protocol === 'https:' || window.location.hostname === 'localhost');
}
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

/**
 * Panel deep links (§5.3): `?panel=<id>` scrolls the target panel into view and briefly
 * highlights it, so a rebuttal can cite an exact panel (e.g.
 * `#/ablation?panel=negative-result&lang=ar`). `copyLink` puts the current URL with the
 * panel id on the clipboard.
 */
export function usePanelAnchor<T extends HTMLElement>(id?: string) {
  const ref = useRef<T>(null);
  const [params] = useSearchParams();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!id || params.get("panel") !== id || !ref.current) return;
    const el = ref.current;
    const timer = setTimeout(() => {
      el.scrollIntoView({ block: "start", behavior: "smooth" });
      el.classList.add("panel-highlight");
      setTimeout(() => el.classList.remove("panel-highlight"), 2400);
    }, 150);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const copyLink = async () => {
    if (!id) return;
    const url = new URL(window.location.href);
    const [path, query] = url.hash.replace(/^#/, "").split("?");
    const q = new URLSearchParams(query ?? "");
    q.set("panel", id);
    url.hash = `${path}?${q.toString()}`;
    try {
      await navigator.clipboard.writeText(url.toString());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard denied — the address bar still has the base URL.
    }
  };

  return { ref, copyLink, copied };
}

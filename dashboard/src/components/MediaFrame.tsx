import { useState } from "react";
import { useTranslation } from "react-i18next";

import { dataUrl, type MediaItem } from "../lib/data";
import { fmtBytes } from "../lib/format";

interface Props {
  item: MediaItem;
  caption: string;
}

/**
 * Media frame (§12): fixed light-tone matte (GIFs keep their baked light background — §9.2),
 * poster-first with click-to-play. The rollout GIFs run 5–69 MB, so they are never fetched
 * until explicitly requested; this also satisfies `prefers-reduced-motion` (§11.1).
 * Missing media (fresh clone / deploy without --copy-media) degrades to a labeled
 * placeholder instead of a broken-image icon.
 */
export function MediaFrame({ item, caption }: Props) {
  const { t } = useTranslation();
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);
  const poster = item.poster ? dataUrl(item.poster) : undefined;
  const src = dataUrl(item.src);
  const isVideo = item.src.endsWith(".mp4");

  return (
    <figure className="media-frame" style={{ margin: 0 }}>
      <div className="matte">
        {failed ? (
          <div
            role="img"
            aria-label={caption}
            style={{
              aspectRatio: "16 / 10",
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#7a7873",
              fontSize: 13,
              padding: "var(--sp-4)",
              textAlign: "center",
            }}
          >
            {t("replays.mediaUnavailable")}
          </div>
        ) : playing ? (
          isVideo ? (
            <video
              src={src}
              poster={poster}
              autoPlay
              muted
              loop
              playsInline
              controls
              aria-label={caption}
              onError={() => setFailed(true)}
              style={{ width: "100%", height: "auto", display: "block" }}
            />
          ) : (
            <img src={src} alt={caption} onError={() => setFailed(true)} />
          )
        ) : (
          <>
            {poster ? (
              <img src={poster} alt={caption} loading="lazy" onError={() => setFailed(true)} />
            ) : (
              <div style={{ aspectRatio: "16 / 9", width: "100%" }} />
            )}
            <button
              className="play-overlay"
              onClick={() => setPlaying(true)}
              aria-label={`${t("actions.play")} — ${caption}`}
            >
              <span className="play-circle" aria-hidden>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5.5v13l11-6.5-11-6.5Z" />
                </svg>
              </span>
              <span className="size-note" aria-hidden>
                {fmtBytes(item.bytes)}
              </span>
            </button>
          </>
        )}
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

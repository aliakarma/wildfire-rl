import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

/** Skeleton shimmer with the 200ms appearance delay (§8.5); fixed height, no layout shift. */
export function Skeleton({ height = 320 }: { height?: number }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => setVisible(true), 200);
    return () => clearTimeout(id);
  }, []);
  return (
    <div
      className={visible ? "skeleton" : undefined}
      style={{ height, borderRadius: "var(--radius-card)" }}
      aria-hidden
    />
  );
}

/** Localized error card with retry (§8.5). */
export function LoadError({ retry }: { retry: () => void }) {
  const { t } = useTranslation();
  return (
    <div className="error-card" role="alert">
      <p>{t("common.loadError")}</p>
      <button
        onClick={retry}
        style={{
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-control)",
          padding: "4px 12px",
        }}
      >
        {t("actions.retry")}
      </button>
    </div>
  );
}

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { useData, type Meta } from "../lib/data";

/** Provenance footer on every page (§2.2, §5.2): fingerprint, commit, build date. */
export function Footer() {
  const { t } = useTranslation();
  const { data: meta } = useData<Meta>("meta.json");
  const [copied, setCopied] = useState(false);

  const copyFingerprint = async () => {
    if (!meta) return;
    try {
      await navigator.clipboard.writeText(meta.fingerprint);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard permission denied — the fingerprint is still selectable text.
    }
  };

  return (
    <footer className="footer">
      <div className="inner">
        <span>{t("footer.provenance")}</span>
        {meta && (
          <>
            <span>
              {t("footer.fingerprint")} <code>{meta.fingerprint.slice(0, 8)}…</code>{" "}
              <button className="copy-btn" onClick={copyFingerprint}>
                {copied ? t("actions.copied") : t("actions.copy")}
              </button>
            </span>
            <span>
              {t("footer.commit")} <code>{meta.commit.slice(0, 7)}</code>
            </span>
            <span>
              {t("footer.built")} <code>{meta.built}</code>
            </span>
          </>
        )}
        <Link to="/reproducibility">{t("footer.downloads")}</Link>
        <span>{t("footer.license")}</span>
      </div>
    </footer>
  );
}

import { useTranslation } from "react-i18next";

import { fmtP, sigStars } from "../lib/format";

interface Props {
  p: number;
  showP?: boolean;
}

/**
 * Significance badge (§12): stars or "n.s." in neutral ink (never green/red);
 * the monospace p-value is always available on hover and to screen readers.
 */
export function SigBadge({ p, showP = false }: Props) {
  const { t } = useTranslation();
  const stars = sigStars(p);
  const label = stars === "n.s." ? t("stats.ns") : t("stats.sig");
  return (
    <span className="sig-badge" title={fmtP(p)} aria-label={`${label}, ${fmtP(p)}`}>
      <span aria-hidden>{stars}</span>
      {showP && <span className="p">{fmtP(p)}</span>}
    </span>
  );
}

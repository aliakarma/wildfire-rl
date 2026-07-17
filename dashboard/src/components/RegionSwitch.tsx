import { useTranslation } from "react-i18next";

import type { RegionChoice } from "../lib/data";
import { useRegionChoice } from "../lib/urlState";

/** Global region segmented control (§5.2); persists across pages via the URL. */
export function RegionSwitch({ allowBoth = true }: { allowBoth?: boolean }) {
  const { t } = useTranslation();
  const [region, setRegion] = useRegionChoice();
  const options: { value: RegionChoice; label: string }[] = [
    { value: "saudi", label: t("region.saudi") },
    { value: "california", label: t("region.california") },
    ...(allowBoth ? [{ value: "both" as RegionChoice, label: t("region.both") }] : []),
  ];
  const effective = !allowBoth && region === "both" ? "saudi" : region;
  return (
    <div className="context-bar">
      <span style={{ fontSize: 13, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
        {t("region.label")}
      </span>
      <div className="segmented" role="group" aria-label={t("region.label")}>
        {options.map((o) => (
          <button
            key={o.value}
            aria-pressed={effective === o.value}
            onClick={() => setRegion(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

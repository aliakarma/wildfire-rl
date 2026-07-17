import { useTranslation } from "react-i18next";

import { METHOD_TOKEN } from "../theme/echartsTheme";

interface Props {
  policy: string;
  /** Ablation variants render in HierComm orange at reduced opacity (§7.3). */
  dimmed?: boolean;
  labelOverride?: string;
}

/** Method chip (§12): 10px color dot + localized name; "ours" adds an accent outline. */
export function MethodChip({ policy, dimmed, labelOverride }: Props) {
  const { t } = useTranslation();
  const isOurs = policy === "hiercomm_heur" && !labelOverride;
  const tokenName = METHOD_TOKEN[policy];
  const label = labelOverride ?? t(`policy.${policy}`);
  // Latin method names keep lang="en" inside Arabic sentences (§13).
  const latin = /^[\x20-\x7E]*$/.test(label);
  return (
    <span className={`method-chip${isOurs ? " ours" : ""}`}>
      {tokenName ? (
        <span
          className="dot"
          style={{ background: `var(${tokenName})`, opacity: dimmed ? 0.55 : 1 }}
          aria-hidden
        />
      ) : (
        <span className="dot dashed" aria-hidden />
      )}
      <span lang={latin ? "en" : undefined}>{label}</span>
    </span>
  );
}

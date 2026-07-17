import i18n from "../i18n";

/** Locale-aware number formatting. Western digits are pinned in Arabic (§10.2). */
export function fmt(n: number, opts?: Intl.NumberFormatOptions): string {
  const locale = i18n.language === "ar" ? "ar-SA-u-nu-latn" : "en-US";
  return new Intl.NumberFormat(locale, opts).format(n);
}

/** Render-time rounding rules (§4.2): WEL 2 decimals, ISR 3 decimals. */
export const fmtWEL = (n: number) => fmt(n, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const fmtISR = (n: number) => fmt(n, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
export const fmt1 = (n: number) => fmt(n, { maximumFractionDigits: 1 });
export const fmt2 = (n: number) => fmt(n, { maximumFractionDigits: 2 });

/** p-values: scientific notation below 0.001 (§4.2); rendered LTR via <bdi>/CSS. */
export function fmtP(p: number): string {
  if (p < 0.001) return `p = ${p.toExponential(1)}`;
  return `p = ${fmt(p, { minimumFractionDigits: 2, maximumFractionDigits: 3 })}`;
}

/** Significance stars (§12): *** / ** / * / n.s. */
export function sigStars(p: number): string {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "n.s.";
}

export function fmtBytes(bytes: number): string {
  const mb = bytes / 1048576;
  return `${fmt(mb, { maximumFractionDigits: 1 })} MB`;
}

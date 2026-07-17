import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, useSearchParams } from "react-router-dom";

import { setLocale, type Locale } from "../i18n";
import { useTheme } from "../theme/useTheme";

const PRIMARY = [
  { to: "/", key: "overview", end: true },
  { to: "/replays", key: "replays" },
  { to: "/results", key: "results" },
  { to: "/ablation", key: "ablation" },
  { to: "/robustness", key: "robustness" },
  { to: "/generalization", key: "generalization" },
] as const;

const MORE = [
  { to: "/benchmark", key: "benchmark" },
  { to: "/method", key: "method" },
  { to: "/reproducibility", key: "reproducibility" },
  { to: "/about", key: "about" },
] as const;

function FlameGlyph() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 2c1 3.5-1.5 5-1.5 8a4.5 4.5 0 0 0 9 .4C21.5 15.5 18 21 12 21s-8.5-4.5-7.5-9C5.2 8.6 8 7.5 8.5 4.5 10 5.5 10.7 7 10.6 8.6 11.5 6.8 12.4 4.5 12 2Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Sticky top bar (§5.2): wordmark, primary nav + More, language switch, theme toggle. */
export function TopBar() {
  const { t, i18n } = useTranslation();
  const { theme, toggle } = useTheme();
  const [moreOpen, setMoreOpen] = useState(false);
  const [params, setParams] = useSearchParams();
  const moreRef = useRef<HTMLDivElement>(null);

  // The "More" menu closes on Escape and on any click outside it (§13: no traps).
  useEffect(() => {
    if (!moreOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMoreOpen(false);
    };
    const onPointer = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) setMoreOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onPointer);
    };
  }, [moreOpen]);

  const setParam = (key: string, value: string) => {
    const p = new URLSearchParams(params);
    p.set(key, value);
    setParams(p, { replace: true });
  };

  const switchLang = () => {
    const next: Locale = i18n.language === "ar" ? "en" : "ar";
    setLocale(next);
    setParam("lang", next);
  };

  const switchTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    toggle();
    setParam("theme", next);
  };

  const withParams = (to: string) => {
    const q = params.toString();
    return q ? `${to}?${q}` : to;
  };

  return (
    <header className="topbar">
      <NavLink to={withParams("/")} className="wordmark">
        <FlameGlyph />
        <span>{t("app.title")}</span>
      </NavLink>
      <nav className="nav-primary" aria-label={t("nav.more")}>
        {PRIMARY.map((item) => (
          <NavLink
            key={item.key}
            to={withParams(item.to)}
            end={"end" in item ? item.end : false}
            className="nav-link"
          >
            {t(`nav.${item.key}`)}
          </NavLink>
        ))}
        <div style={{ position: "relative" }} ref={moreRef}>
          <button
            className="nav-link"
            aria-expanded={moreOpen}
            aria-haspopup="menu"
            onClick={() => setMoreOpen((v) => !v)}
          >
            {t("nav.more")} ▾
          </button>
          {moreOpen && (
            <div
              role="menu"
              style={{
                position: "absolute",
                insetBlockStart: "100%",
                insetInlineEnd: 0,
                minWidth: 220,
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-card)",
                boxShadow: "var(--shadow-card)",
                padding: "var(--sp-1)",
                display: "flex",
                flexDirection: "column",
                zIndex: 60,
              }}
            >
              {MORE.map((item) => (
                <NavLink
                  key={item.key}
                  to={withParams(item.to)}
                  className="nav-link"
                  role="menuitem"
                  onClick={() => setMoreOpen(false)}
                >
                  {t(`nav.${item.key}`)}
                </NavLink>
              ))}
            </div>
          )}
        </div>
      </nav>
      <div className="topbar-controls">
        <button className="lang-switch" onClick={switchLang} aria-label={t("lang.label")}>
          {t("lang.switch")}
        </button>
        <button
          className="icon-btn"
          onClick={switchTheme}
          aria-pressed={theme === "dark"}
          title={theme === "dark" ? t("theme.light") : t("theme.dark")}
          aria-label={theme === "dark" ? t("theme.light") : t("theme.dark")}
        >
          {theme === "dark" ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
              <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.6" />
              <path
                d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path
                d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}

import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { Callout } from "../components/Callout";
import { DataTable } from "../components/DataTable";

/** Hyperparameters from paper Appendix A (checkpoint configurations). */
const HYPERPARAMS: [string, string][] = [
  ["Grid size / agents / crop", "32×32 / 3 / 9×9"],
  ["Max episode steps · sim. minutes per step", "150 · 30"],
  ["Commander macro-step interval (T)", "10"],
  ["Commander assignment rule", "threat-weighted asset, v/(1+d)"],
  ["Tactical CNN feature dim / target dim", "64 / 16"],
  ["Communication rounds (R)", "2"],
  ["BC demonstrations / epochs / lr", "30 episodes / 60 / 1e-3"],
  ["PPO fine-tune env steps", "100,000"],
  ["Actor / critic learning rate", "3e-4 / 1e-3"],
  ["PPO clip / GAE λ / epochs / batch", "0.2 / 0.95 / 4 / 256"],
  ["Entropy coef / shaping coef", "0.02 / 0.05"],
  ["Reward ISR weight (α)", "20"],
];

/**
 * Interactive SVG rebuild of the paper's architecture figure — a real, themable,
 * RTL-tolerant SVG (never a screenshot of the TikZ figure, §6.8). The replay-grid map
 * itself never mirrors; this block diagram is direction-neutral.
 */
function ArchitectureDiagram() {
  const { t } = useTranslation();
  const blocks = [
    { id: "commander", title: t("method.archCommander"), desc: t("method.archCommanderDesc", { n: 10 }) },
    { id: "tactical", title: t("method.archTactical"), desc: t("method.archTacticalDesc", { r: 2 }) },
    { id: "env", title: t("method.archEnv"), desc: t("method.archEnvDesc") },
  ];
  return (
    <div className="grid-2" style={{ gridTemplateColumns: "1fr" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 0, maxWidth: 640 }}>
        {blocks.map((b, i) => (
          <div key={b.id}>
            <div
              className="card"
              style={{
                borderInlineStart: `3px solid ${b.id === "tactical" ? "var(--accent)" : "var(--border)"}`,
              }}
            >
              <h3 className="card-title">{b.title}</h3>
              <p style={{ fontSize: 14, color: "var(--text-secondary)", margin: "var(--sp-1) 0 0" }}>
                {b.desc}
              </p>
            </div>
            {i < blocks.length - 1 && (
              <div aria-hidden style={{ textAlign: "center", color: "var(--text-muted)", padding: "2px 0" }}>
                <svg width="16" height="20" viewBox="0 0 16 20" fill="none">
                  <path d="M8 1v14m0 0-5-5m5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Method() {
  const { t } = useTranslation();
  return (
    <div className="page">
      <h1>{t("method.title")}</h1>
      <p className="page-summary">{t("method.summary")}</p>

      <section className="section">
        <h2>{t("method.architecture")}</h2>
        <ArchitectureDiagram />
      </section>

      <section className="section">
        <h2>{t("method.storyboard")}</h2>
        <ol
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))",
            gap: "var(--sp-3)",
            listStyle: "none",
            margin: 0,
            padding: 0,
          }}
        >
          {(["s1", "s2", "s3", "s4"] as const).map((s, i) => (
            <li key={s} className="card" style={{ padding: "var(--sp-4)" }}>
              <span
                aria-hidden
                style={{ color: "var(--accent)", fontWeight: 650, fontSize: 18, display: "block" }}
              >
                {i + 1}
              </span>
              <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>
                {t(`method.storySteps.${s}`)}
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section className="section">
        <h2>{t("method.training")}</h2>
        <div className="grid-2">
          <Callout title="1 · BC">{t("method.trainingBC")}</Callout>
          <Callout title="2 · PPO">{t("method.trainingPPO")}</Callout>
        </div>
      </section>

      <section className="section">
        <h2>{t("method.hyperparams")}</h2>
        <DataTable
          caption={t("method.hyperparams")}
          columns={[{ header: "" }, { header: "", numeric: true }]}
          rows={HYPERPARAMS.map(([k, v]) => [k, <code key="v">{v}</code>])}
        />
        <p className="card-footnote">
          <code>AAAI Template/AnonymousSubmission2027.tex · Appendix A</code>
        </p>
      </section>

      <section className="section">
        <Callout title={t("method.scopeTitle")}>
          {t("method.scope")} <Link to="/ablation">{t("nav.ablation")}</Link>
        </Callout>
      </section>
    </div>
  );
}

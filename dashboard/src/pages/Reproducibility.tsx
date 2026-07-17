import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Callout } from "../components/Callout";
import { DataTable } from "../components/DataTable";
import { LoadError, Skeleton } from "../components/Loading";
import { dataUrl, useData, type Meta, type Reproducibility as ReproData } from "../lib/data";

const DOWNLOADS = [
  "phase3_summary.json",
  "phase3_raw.csv",
  "ablation_summary.json",
  "ablation_results.csv",
  "robustness_summary.json",
  "robustness_results.csv",
  "generalization_summary.json",
  "generalization_raw.csv",
  "transfer_summary.json",
  "transfer_matrix_raw.csv",
  "MANIFEST.sha256",
  "FREEZE.json",
];

function CodeBlock({ text }: { text: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-block">
      <button
        className="copy-btn"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            // Clipboard permission denied — the command remains selectable.
          }
        }}
      >
        {copied ? t("actions.copied") : t("actions.copy")}
      </button>
      <pre>
        <code>{text}</code>
      </pre>
    </div>
  );
}

export default function Reproducibility() {
  const { t } = useTranslation();
  const repro = useData<ReproData>("reproducibility.json");
  const { data: meta } = useData<Meta>("meta.json");

  return (
    <div className="page">
      <h1>{t("repro.title")}</h1>
      <p className="page-summary">{t("repro.summary")}</p>

      <section className="section">
        <h2>{t("repro.protocol")}</h2>
        <ul style={{ margin: 0, paddingInlineStart: "1.2em", maxWidth: "80ch" }}>
          {(["seeds", "eval", "unit", "determinism"] as const).map((k) => (
            <li key={k} style={{ marginBlockEnd: "var(--sp-1)" }}>
              {t(`repro.protocolItems.${k}`)}
            </li>
          ))}
        </ul>
      </section>

      {repro.error ? (
        <LoadError retry={repro.retry} />
      ) : !repro.data ? (
        <Skeleton height={400} />
      ) : (
        <>
          <section className="section">
            <h2>{t("repro.hashes")}</h2>
            <p className="page-summary">
              {t("repro.hashesSubtitle", { n: repro.data.n_checkpoints })}
            </p>
            <DataTable
              caption={t("repro.hashes")}
              columns={[{ header: t("repro.file") }, { header: "SHA-256" }]}
              rows={Object.entries(repro.data.seed42_checkpoint_sha256).map(([name, digest]) => [
                <code key="n">{name}</code>,
                <code key="d">{digest.slice(0, 16)}…</code>,
              ])}
            />
            {meta && (
              <p className="card-footnote">
                {t("footer.fingerprint")}: <code>{meta.fingerprint}</code>
              </p>
            )}
          </section>

          <section className="section">
            <h2>{t("repro.reproduce")}</h2>
            <CodeBlock text={repro.data.reproduction_command} />
          </section>

          <section className="section">
            <h2>{t("repro.verify")}</h2>
            <p className="page-summary">{t("repro.verifySteps")}</p>
            <CodeBlock text={repro.data.verify_command} />
          </section>

          <section className="section">
            <h2>{t("repro.compute")}</h2>
            <p className="page-summary">{repro.data.compute}</p>
          </section>
        </>
      )}

      <section className="section">
        <h2>{t("repro.downloads")}</h2>
        <Callout>{t("repro.downloadsNote")}</Callout>
        <ul
          style={{
            listStyle: "none",
            margin: "var(--sp-4) 0 0",
            padding: 0,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(min(280px, 100%), 1fr))",
            gap: "var(--sp-2)",
          }}
        >
          {DOWNLOADS.map((f) => (
            <li key={f}>
              <a href={dataUrl(`data/downloads/${f}`)} download>
                <code>{f}</code>
              </a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

import type { ReactNode } from "react";

interface Props {
  title?: string;
  children: ReactNode;
}

/** Neutral informational card for honesty notes (§12) — never an alert style. */
export function Callout({ title, children }: Props) {
  return (
    <div className="callout" role="note">
      {title && (
        <p style={{ marginBlockEnd: "var(--sp-1)" }}>
          <strong>{title}</strong>
        </p>
      )}
      {children}
    </div>
  );
}

interface Props {
  label: string;
  value: string;
  context?: string;
}

/** Stat tile (§8.6): label → big tabular numeral → context line. Never colored good/bad. */
export function StatTile({ label, value, context }: Props) {
  return (
    <div className="stat-tile">
      <div className="label">{label}</div>
      <div className="value">
        <bdi dir="ltr">{value}</bdi>
      </div>
      {context && <div className="context">{context}</div>}
    </div>
  );
}

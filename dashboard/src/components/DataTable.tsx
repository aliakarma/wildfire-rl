import type { ReactNode } from "react";

export interface Column {
  header: ReactNode;
  numeric?: boolean;
}

interface Props {
  columns: Column[];
  rows: ReactNode[][];
  caption?: string;
}

/**
 * Data table (§12): sticky header, tabular numerals, numbers end-aligned.
 * The canonical accessible alternative for every chart (§13).
 */
export function DataTable({ columns, rows, caption }: Props) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        {caption && <caption className="visually-hidden">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={i} scope="col" className={c.numeric ? "num" : undefined}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci} className={columns[ci]?.numeric ? "num" : undefined}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";

import {
  applyRetention,
  createBackup,
  downloadAuthenticated,
  fetchBackups,
  fetchDiagnosticBundle,
  fetchEnvDiagnostics,
  fetchRetentionPreview,
  fetchRetentionSettings,
  fetchUpdateReadiness,
  restoreBackup,
  saveRetentionSettings,
  type BackupInfo,
  type DiagnosticBundle,
  type EnvDiagnostic,
  type RetentionPreview,
  type RetentionSettings,
  type UpdateReadiness,
} from "./api";

/**
 * Settings → Diagnostics / operations panel.
 * Decision support only — never places trades.
 */
export function OperationsPanel({
  accountId,
  readOnlyPhone = false,
}: {
  accountId: string;
  readOnlyPhone?: boolean;
}) {
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [envDiag, setEnvDiag] = useState<EnvDiagnostic | null>(null);
  const [bundle, setBundle] = useState<DiagnosticBundle | null>(null);
  const [retention, setRetention] = useState<RetentionSettings | null>(null);
  const [preview, setPreview] = useState<RetentionPreview | null>(null);
  const [backups, setBackups] = useState<BackupInfo[]>([]);
  const [update, setUpdate] = useState<UpdateReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmRestoreId, setConfirmRestoreId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [env, ret, prev, bak, upd] = await Promise.all([
        fetchEnvDiagnostics(),
        fetchRetentionSettings(),
        fetchRetentionPreview(),
        fetchBackups(),
        fetchUpdateReadiness(),
      ]);
      setEnvDiag(env);
      setRetention(ret);
      setPreview(prev);
      setBackups(bak);
      setUpdate(upd);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not load diagnostics.");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const run = async (label: string, action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      await action();
      setStatus(label);
      await load();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Operation failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack-lg operations-panel">
      <p className="muted" role="note">
        Exports and diagnostics are decision-support artifacts only. Position Pilot never places,
        stages, or cancels orders. Credentials, tokens, cookies, prompts, licensed full article
        text, and raw environment values are excluded from diagnostics.
      </p>
      {status ? (
        <p className="status-ok" role="status">
          {status}
        </p>
      ) : null}
      {error ? (
        <p className="status-error" role="alert">
          {error}
        </p>
      ) : null}

      <section className="panel-section" aria-labelledby="exports-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Exports</p>
            <h2 id="exports-heading">Portfolio and history</h2>
          </div>
        </div>
        <div className="inline-actions">
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("Portfolio CSV downloaded.", () =>
                downloadAuthenticated(
                  `/api/v1/exports/portfolio.csv?account_id=${encodeURIComponent(accountId)}`,
                  "portfolio.csv",
                ),
              )
            }
          >
            Download portfolio CSV
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("History CSV downloaded.", () =>
                downloadAuthenticated("/api/v1/exports/history.csv", "portfolio-history.csv"),
              )
            }
          >
            Download history CSV
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("HTML snapshot downloaded.", () =>
                downloadAuthenticated(
                  `/api/v1/exports/snapshot.html?account_id=${encodeURIComponent(accountId)}`,
                  "portfolio-snapshot.html",
                ),
              )
            }
          >
            Printable HTML
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("PDF snapshot downloaded.", () =>
                downloadAuthenticated(
                  `/api/v1/exports/snapshot.pdf?account_id=${encodeURIComponent(accountId)}`,
                  "portfolio-snapshot.pdf",
                ),
              )
            }
          >
            Printable PDF
          </button>
        </div>
      </section>

      <section className="panel-section" aria-labelledby="diag-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Diagnostics</p>
            <h2 id="diag-heading">Redacted bundle and .env checks</h2>
          </div>
        </div>
        {envDiag ? (
          <dl className="cadence-facts">
            <div>
              <dt>.env exists</dt>
              <dd>{envDiag.exists ? "Yes" : "No"}</dd>
            </div>
            <div>
              <dt>Gitignored</dt>
              <dd>{envDiag.gitignored ? "Yes" : "No"}</dd>
            </div>
            <div>
              <dt>Tracked by git</dt>
              <dd>
                {envDiag.tracked_by_git == null
                  ? "Unknown"
                  : envDiag.tracked_by_git
                    ? "Yes — remove immediately"
                    : "No"}
              </dd>
            </div>
            <div>
              <dt>Permissions</dt>
              <dd>
                {envDiag.permission_mode ?? "—"}
                {envDiag.broadly_readable ? " (broad — prefer 0600)" : ""}
              </dd>
            </div>
          </dl>
        ) : null}
        {envDiag?.warnings?.length ? (
          <ul className="plain-list" role="list">
            {envDiag.warnings.map((warning) => (
              <li key={warning} className="status-error">
                {warning}
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No .env warnings. Values are never read or returned.</p>
        )}
        <div className="inline-actions">
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("Diagnostic bundle loaded.", async () => {
                setBundle(await fetchDiagnosticBundle());
              })
            }
          >
            Load redacted JSON bundle
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void run("Diagnostic zip downloaded.", () =>
                downloadAuthenticated(
                  "/api/v1/diagnostics/bundle.zip",
                  "position-pilot-diagnostics.zip",
                ),
              )
            }
          >
            Download diagnostic zip
          </button>
        </div>
        {bundle ? (
          <pre className="diagnostic-json" tabIndex={0} aria-label="Redacted diagnostic bundle">
            {JSON.stringify(
              {
                generated_at: bundle.generated_at,
                app_version: bundle.app_version,
                schema_version: bundle.schema_version,
                provider_status: bundle.provider_status,
                counts: bundle.counts,
                redaction: bundle.redaction,
                disclaimer: bundle.disclaimer,
              },
              null,
              2,
            )}
          </pre>
        ) : null}
      </section>

      <section className="panel-section" aria-labelledby="retention-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Retention</p>
            <h2 id="retention-heading">Local operational data</h2>
          </div>
        </div>
        {retention ? (
          readOnlyPhone ? (
            <div className="stack-form">
              <p className="microcopy">
                Portfolio snapshots: {retention.portfolio_snapshots_days} days · Catalysts:{" "}
                {retention.catalyst_events_days} days · Articles: {retention.article_metadata_days}{" "}
                days · Recommendation history:{" "}
                {retention.recommendation_history_days === 0
                  ? "indefinite"
                  : `${retention.recommendation_history_days} days`}
              </p>
              <p className="muted">Retention settings are read-only on phone widths.</p>
            </div>
          ) : (
            <form
              className="stack-form desktop-only-actions"
              onSubmit={(event) => {
                event.preventDefault();
                void run("Retention settings saved.", async () => {
                  const next = await saveRetentionSettings({
                    portfolio_snapshots_days: retention.portfolio_snapshots_days,
                    catalyst_events_days: retention.catalyst_events_days,
                    article_metadata_days: retention.article_metadata_days,
                    recommendation_history_days: retention.recommendation_history_days,
                    transaction_history: retention.transaction_history,
                  });
                  setRetention(next);
                  setPreview(await fetchRetentionPreview());
                });
              }}
            >
              <label className="compact-field">
                Portfolio snapshot days
                <input
                  type="number"
                  min={30}
                  max={3650}
                  value={retention.portfolio_snapshots_days}
                  onChange={(event) =>
                    setRetention({
                      ...retention,
                      portfolio_snapshots_days: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label className="compact-field">
                Catalyst event days
                <input
                  type="number"
                  min={30}
                  max={3650}
                  value={retention.catalyst_events_days}
                  onChange={(event) =>
                    setRetention({
                      ...retention,
                      catalyst_events_days: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label className="compact-field">
                Article metadata days
                <input
                  type="number"
                  min={7}
                  max={365}
                  value={retention.article_metadata_days}
                  onChange={(event) =>
                    setRetention({
                      ...retention,
                      article_metadata_days: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label className="compact-field">
                Recommendation history days (0 = indefinite)
                <input
                  type="number"
                  min={0}
                  max={3650}
                  value={retention.recommendation_history_days}
                  onChange={(event) =>
                    setRetention({
                      ...retention,
                      recommendation_history_days: Number(event.target.value),
                    })
                  }
                />
              </label>
              <p className="microcopy">
                Audit-critical data (transactions, rolls, trader decisions, audit events, and
                recommendation history) is never purged by ordinary retention. Older portfolio
                snapshots compact into durable daily summaries after one year.
              </p>
              <div className="inline-actions desktop-only-actions">
                <button type="submit" disabled={busy}>
                  Save retention settings
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    void run("Retention preview refreshed.", async () => {
                      setPreview(await fetchRetentionPreview());
                    })
                  }
                >
                  Preview purge
                </button>
                <button
                  type="button"
                  className="danger-action"
                  disabled={busy}
                  onClick={() => {
                    if (
                      !window.confirm(
                        "Apply retention now? This permanently deletes expired operational rows and compacts older portfolio snapshots into daily summaries. Audit-critical data is always preserved.",
                      )
                    ) {
                      return;
                    }
                    void run("Retention applied.", async () => {
                      await applyRetention(true);
                    });
                  }}
                >
                  Apply retention
                </button>
              </div>
            </form>
          )
        ) : null}
        {preview ? (
          <div className="retention-preview" aria-live="polite">
            <p className="microcopy">Would delete (preview):</p>
            <ul className="plain-list">
              {Object.entries(preview.would_delete).map(([key, count]) => (
                <li key={key}>
                  {key}: {count}
                </li>
              ))}
              {!Object.keys(preview.would_delete).length ? <li>Nothing pending.</li> : null}
            </ul>
            <p className="microcopy">
              Preserved: {preview.audit_critical_preserved.join("; ")}
            </p>
          </div>
        ) : null}
      </section>

      <section className="panel-section" aria-labelledby="backup-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Backups</p>
            <h2 id="backup-heading">Create, download, restore</h2>
          </div>
        </div>
        <p className="muted">
          SQLite application state only. Credentials stay in .env and are never packaged. Restore is
          blocked while monitoring is active and always creates a pre-restore backup.
        </p>
        {readOnlyPhone ? (
          <p className="muted">Backup creation is available on desktop widths.</p>
        ) : (
          <div className="inline-actions desktop-only-actions">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void run("Backup created.", async () => {
                  await createBackup();
                })
              }
            >
              Create backup
            </button>
          </div>
        )}
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Local backups table">
          <table className="data-table dense">
            <caption className="sr-only">Local database backups</caption>
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col">Reason</th>
                <th scope="col">Schema</th>
                <th scope="col">Integrity</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {backups.map((item) => (
                <tr key={item.backup_id}>
                  <th scope="row">{item.filename}</th>
                  <td>{item.reason}</td>
                  <td className="tabular">{item.schema_version ?? "—"}</td>
                  <td>{item.integrity_ok ? "ok" : "failed"}</td>
                  <td>
                    <div className="inline-actions">
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void run("Backup downloaded.", () =>
                            downloadAuthenticated(
                              `/api/v1/backups/${encodeURIComponent(item.backup_id)}`,
                              item.filename,
                            ),
                          )
                        }
                      >
                        Download
                      </button>
                      {readOnlyPhone ? null : (
                        <button
                          type="button"
                          className="danger-action desktop-only-actions"
                          disabled={busy}
                          onClick={() => {
                            if (confirmRestoreId !== item.backup_id) {
                              setConfirmRestoreId(item.backup_id);
                              setStatus("Click Restore again to confirm atomic restore.");
                              return;
                            }
                            void run("Restore completed.", async () => {
                              await restoreBackup(item.backup_id, true);
                              setConfirmRestoreId(null);
                            });
                          }}
                        >
                          {confirmRestoreId === item.backup_id ? "Confirm restore" : "Restore"}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              {!backups.length ? (
                <tr>
                  <td colSpan={5} className="muted">
                    No backups yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel-section" aria-labelledby="update-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Updates</p>
            <h2 id="update-heading">Readiness (never auto-installed)</h2>
          </div>
        </div>
        {update ? (
          <>
            <dl className="cadence-facts">
              <div>
                <dt>Current</dt>
                <dd>{update.current_version}</dd>
              </div>
              <div>
                <dt>Latest known</dt>
                <dd>{update.latest_version ?? "Not probed (no package manager calls)"}</dd>
              </div>
              <div>
                <dt>Schema</dt>
                <dd>
                  v{update.schema_version}
                  {update.schema_migrations_pending ? " (pending)" : ""}
                </dd>
              </div>
              <div>
                <dt>Monitoring</dt>
                <dd>{update.monitoring_active ? "Active — stop before update" : "Idle"}</dd>
              </div>
            </dl>
            {update.blocked_reason ? (
              <p className="status-error" role="status">
                {update.blocked_reason}
              </p>
            ) : null}
            <p className="muted">{update.note}</p>
            <ol className="update-steps">
              {update.reversible_instructions.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </>
        ) : null}
      </section>
    </div>
  );
}

'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  FileText,
  Download,
  ArrowLeft,
  ShieldCheck,
  AlertTriangle,
  ExternalLink,
  Layers,
  Database,
  Globe,
  CheckCircle,
  XCircle,
  Copy,
  Check,
  Clock,
  Sparkles,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { StatusBadge } from '@/components/StatusBadge';
import { apiClient } from '@/lib/api';
import { ResearchSession } from '@/types/api';

export default function ReportDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params?.id as string;

  const [session, setSession] = useState<ResearchSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'evidence' | 'citations' | 'conflicts'>('summary');
  const [copiedId, setCopiedId] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);
    apiClient
      .getResearchSession(id)
      .then((data) => setSession(data))
      .catch((err) => {
        setError(err?.message || 'Report not found or inaccessible.');
      })
      .finally(() => setLoading(false));
  }, [id]);

  const copyResearchId = () => {
    if (!id) return;
    navigator.clipboard.writeText(id);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 1500);
  };

  return (
    <div className="flex-1 flex flex-col">
      <Header
        title={session?.title || 'Research Report'}
        subtitle={`Session ID: ${id}`}
        actionButton={
          <div className="flex items-center gap-2">
            <Link
              href="/reports"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-300 text-xs transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Back</span>
            </Link>

            {session && (
              <>
                <a
                  href={apiClient.getExportUrl(session.id, 'markdown')}
                  download={`research-${session.id}.md`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-200 text-xs transition-colors"
                  title="Download Markdown Report"
                >
                  <Download className="h-3.5 w-3.5 text-accent-cyan" />
                  <span>Markdown</span>
                </a>
                <a
                  href={apiClient.getExportUrl(session.id, 'json')}
                  download={`research-${session.id}.json`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-200 text-xs transition-colors"
                  title="Download JSON Artifact"
                >
                  <Download className="h-3.5 w-3.5 text-accent-blue" />
                  <span>JSON</span>
                </a>
              </>
            )}
          </div>
        }
      />

      <main className="flex-1 p-6 max-w-6xl mx-auto w-full space-y-6">
        {loading ? (
          <div className="space-y-4 p-8">
            <div className="h-10 w-2/3 bg-surfaceHover rounded-lg animate-pulse" />
            <div className="h-40 bg-surfaceHover rounded-xl animate-pulse" />
            <div className="h-60 bg-surfaceHover rounded-xl animate-pulse" />
          </div>
        ) : error || !session ? (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-8 text-center space-y-3">
            <XCircle className="h-12 w-12 text-rose-400 mx-auto" />
            <h3 className="text-lg font-bold text-white">Report Not Found</h3>
            <p className="text-sm text-rose-300 max-w-md mx-auto">{error}</p>
            <Link
              href="/reports"
              className="inline-block px-4 py-2 rounded-lg bg-surface border border-border text-white text-xs hover:bg-surfaceHover mt-2"
            >
              &larr; Return to Reports List
            </Link>
          </div>
        ) : (
          <>
            {/* Meta Card */}
            <section className="rounded-xl border border-border bg-surface p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-3">
                    <StatusBadge status={session.status} size="md" />
                    <button
                      onClick={copyResearchId}
                      className="flex items-center gap-1.5 text-xs font-mono text-slate-400 hover:text-white bg-background px-2.5 py-1 rounded-md border border-border transition-colors"
                      title="Copy Research ID"
                    >
                      <span>{session.id}</span>
                      {copiedId ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                    </button>
                  </div>
                  <h3 className="text-xl font-bold text-white pt-1">{session.title || session.query}</h3>
                  <p className="text-xs text-slate-400 font-mono">Original Query: &quot;{session.query}&quot;</p>
                </div>

                <div className="flex sm:flex-col items-end gap-1.5 text-xs text-slate-400 font-mono">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5 text-slate-500" />
                    {new Date(session.created_at).toLocaleString()}
                  </span>
                  <span>Tenant: {session.organization_id}</span>
                </div>
              </div>

              {/* Grounding & Verification Indicators */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-border">
                <div className="flex items-center gap-2.5 p-3 rounded-lg bg-background/50 border border-border">
                  {session.is_grounded ? (
                    <ShieldCheck className="h-5 w-5 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                  )}
                  <div>
                    <p className="text-[11px] text-slate-400 font-mono">Grounding Status</p>
                    <p className="text-xs font-semibold text-white">
                      {session.is_grounded ? 'Strictly Grounded' : 'Ungrounded / Warning'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 p-3 rounded-lg bg-background/50 border border-border">
                  {session.sufficient_evidence ? (
                    <CheckCircle className="h-5 w-5 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                  )}
                  <div>
                    <p className="text-[11px] text-slate-400 font-mono">Evidence Completeness</p>
                    <p className="text-xs font-semibold text-white">
                      {session.sufficient_evidence ? 'Sufficient Evidence' : 'Insufficient Evidence'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 p-3 rounded-lg bg-background/50 border border-border">
                  <Globe className="h-5 w-5 text-accent-cyan" />
                  <div>
                    <p className="text-[11px] text-slate-400 font-mono">Web Investigation</p>
                    <p className="text-xs font-semibold text-white capitalize">
                      {session.metadata?.web_status?.status || (session.web_evidence?.length ? 'Completed' : 'Skipped')}
                    </p>
                  </div>
                </div>
              </div>
            </section>

            {/* Navigation Tabs */}
            <div className="flex border-b border-border gap-6 text-sm">
              <button
                onClick={() => setActiveTab('summary')}
                className={`pb-3 font-medium transition-colors border-b-2 flex items-center gap-2 ${
                  activeTab === 'summary'
                    ? 'border-accent-cyan text-accent-cyan'
                    : 'border-transparent text-slate-400 hover:text-white'
                }`}
              >
                <Sparkles className="h-4 w-4" />
                Synthesis Report
              </button>
              <button
                onClick={() => setActiveTab('evidence')}
                className={`pb-3 font-medium transition-colors border-b-2 flex items-center gap-2 ${
                  activeTab === 'evidence'
                    ? 'border-accent-cyan text-accent-cyan'
                    : 'border-transparent text-slate-400 hover:text-white'
                }`}
              >
                <Database className="h-4 w-4" />
                Retrieved Evidence ({session.internal_evidence?.length || 0} int / {session.web_evidence?.length || 0} web)
              </button>
              <button
                onClick={() => setActiveTab('citations')}
                className={`pb-3 font-medium transition-colors border-b-2 flex items-center gap-2 ${
                  activeTab === 'citations'
                    ? 'border-accent-cyan text-accent-cyan'
                    : 'border-transparent text-slate-400 hover:text-white'
                }`}
              >
                <Layers className="h-4 w-4" />
                Verified Citations ({session.citations?.length || 0})
              </button>
              <button
                onClick={() => setActiveTab('conflicts')}
                className={`pb-3 font-medium transition-colors border-b-2 flex items-center gap-2 ${
                  activeTab === 'conflicts'
                    ? 'border-accent-cyan text-accent-cyan'
                    : 'border-transparent text-slate-400 hover:text-white'
                }`}
              >
                <AlertTriangle className="h-4 w-4" />
                Conflicts ({session.conflicts?.length || 0})
              </button>
            </div>

            {/* Tab 1: Synthesis Summary */}
            {activeTab === 'summary' && (
              <section className="space-y-6">
                <div className="rounded-xl border border-border bg-surface p-6 shadow-xl space-y-4">
                  <h4 className="text-base font-semibold text-white flex items-center gap-2">
                    <FileText className="h-4 w-4 text-accent-cyan" />
                    Executive Summary & Findings
                  </h4>
                  <div className="prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed whitespace-pre-wrap font-sans">
                    {session.summary || 'No synthesis summary generated.'}
                  </div>
                </div>

                {/* Sources Attributed */}
                {session.sources && session.sources.length > 0 && (
                  <div className="rounded-xl border border-border bg-surface p-6 shadow-xl space-y-3">
                    <h5 className="text-xs font-semibold text-slate-400 font-mono uppercase tracking-wider">
                      Sources Consulted ({session.sources.length})
                    </h5>
                    <div className="flex flex-wrap gap-2">
                      {session.sources.map((src, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-background border border-border text-xs font-mono text-slate-300"
                        >
                          {src.startsWith('http') ? (
                            <a
                              href={src}
                              target="_blank"
                              rel="noreferrer"
                              className="text-accent-cyan hover:underline flex items-center gap-1"
                            >
                              <Globe className="h-3 w-3" />
                              {src}
                              <ExternalLink className="h-2.5 w-2.5" />
                            </a>
                          ) : (
                            <span className="flex items-center gap-1">
                              <Database className="h-3 w-3 text-accent-blue" />
                              {src}
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Tab 2: Evidence Explorer */}
            {activeTab === 'evidence' && (
              <section className="space-y-6">
                {/* Internal Evidence */}
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-white flex items-center gap-2 font-mono">
                    <Database className="h-4 w-4 text-accent-blue" />
                    Internal Private Knowledge Evidence ({session.internal_evidence?.length || 0})
                  </h4>
                  {(!session.internal_evidence || session.internal_evidence.length === 0) ? (
                    <p className="text-xs text-slate-500 italic p-4 bg-surface rounded-lg border border-border">
                      No internal document evidence retrieved.
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {session.internal_evidence.map((ev, i) => (
                        <div key={i} className="p-4 rounded-xl border border-border bg-surface space-y-2 text-xs">
                          <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
                            <span className="text-accent-blue font-semibold">{ev.source_uri}</span>
                            <span>score: {ev.score?.toFixed(3) ?? 'N/A'}</span>
                          </div>
                          <p className="text-slate-200 font-mono bg-background/60 p-2.5 rounded-lg border border-border/60">
                            {ev.snippet}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Web Evidence */}
                <div className="space-y-3 pt-4 border-t border-border">
                  <h4 className="text-sm font-semibold text-white flex items-center gap-2 font-mono">
                    <Globe className="h-4 w-4 text-accent-cyan" />
                    Autonomous Web Evidence ({session.web_evidence?.length || 0})
                  </h4>
                  {(!session.web_evidence || session.web_evidence.length === 0) ? (
                    <p className="text-xs text-slate-500 italic p-4 bg-surface rounded-lg border border-border">
                      No autonomous web evidence gathered for this session.
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {session.web_evidence.map((ev, i) => (
                        <div key={i} className="p-4 rounded-xl border border-border bg-surface space-y-2 text-xs">
                          <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
                            <a
                              href={ev.source_uri}
                              target="_blank"
                              rel="noreferrer"
                              className="text-accent-cyan hover:underline flex items-center gap-1 font-semibold"
                            >
                              {ev.source_uri}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                            <span>score: {ev.score?.toFixed(3) ?? 'N/A'}</span>
                          </div>
                          <p className="text-slate-200 font-mono bg-background/60 p-2.5 rounded-lg border border-border/60">
                            {ev.snippet}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* Tab 3: Verified Citations */}
            {activeTab === 'citations' && (
              <section className="space-y-3">
                <h4 className="text-sm font-semibold text-white flex items-center gap-2 font-mono">
                  <Layers className="h-4 w-4 text-accent-cyan" />
                  Strictly Verified Citations ({session.citations?.length || 0})
                </h4>
                {(!session.citations || session.citations.length === 0) ? (
                  <p className="text-xs text-slate-500 italic p-4 bg-surface rounded-lg border border-border">
                    No verified citations recorded.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {session.citations.map((c, i) => (
                      <div key={i} className="p-4 rounded-xl border border-border bg-surface space-y-2 text-xs">
                        <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
                          <span className="text-slate-200 font-medium">[{i + 1}] {c.source_uri}</span>
                          <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px]">
                            Verified
                          </span>
                        </div>
                        {c.chunk_id && (
                          <div className="text-[11px] font-mono text-slate-500 flex gap-2">
                            <span>doc: {c.document_id}</span>
                            <span>•</span>
                            <span>chunk: {c.chunk_id}</span>
                          </div>
                        )}
                        <p className="text-slate-300 bg-background/80 p-3 rounded-lg border border-border/80 italic">
                          &ldquo;{c.snippet}&rdquo;
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* Tab 4: Detected Conflicts */}
            {activeTab === 'conflicts' && (
              <section className="space-y-3">
                <h4 className="text-sm font-semibold text-white flex items-center gap-2 font-mono">
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  Cross-Source Contradictions & Conflicts ({session.conflicts?.length || 0})
                </h4>
                {(!session.conflicts || session.conflicts.length === 0) ? (
                  <div className="p-8 text-center bg-surface rounded-xl border border-border space-y-2">
                    <CheckCircle className="h-8 w-8 text-emerald-400 mx-auto" />
                    <p className="text-sm text-slate-300 font-medium">No Conflicts Detected</p>
                    <p className="text-xs text-slate-500">
                      Internal documents and autonomous web sources are in alignment.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {session.conflicts.map((conflict, i) => (
                      <div
                        key={i}
                        className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 space-y-3 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-amber-300 text-sm">
                            Topic: {conflict.topic}
                          </span>
                          <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 text-[10px] font-mono">
                            Contradiction
                          </span>
                        </div>
                        <div className="space-y-1.5">
                          <p className="text-[11px] font-mono text-slate-400">Conflicting Assertions:</p>
                          <ul className="list-disc pl-5 space-y-1 text-slate-300">
                            {conflict.conflicting_assertions?.map((item, j) => (
                              <li key={j}>{item}</li>
                            ))}
                          </ul>
                        </div>
                        {conflict.resolution_note && (
                          <div className="pt-2 border-t border-amber-500/20 text-slate-300 text-[11px]">
                            <strong className="text-amber-200">Resolution Note:</strong> {conflict.resolution_note}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

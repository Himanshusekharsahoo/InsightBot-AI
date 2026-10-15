'use client';

import React, { useState, useEffect, useRef, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import {
  Compass,
  Play,
  Ban,
  Clock,
  ArrowRight,
  AlertTriangle,
  Globe,
  Database,
  Cpu,
  Layers,
  FileCheck,
  CheckCircle,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { StatusBadge } from '@/components/StatusBadge';
import { apiClient } from '@/lib/api';
import { ResearchProgressEvent, ResearchStatus, ResearchJobStatusResponse } from '@/types/api';

const STAGE_ICONS: Record<string, React.ReactNode> = {
  queued: <Clock className="h-4 w-4 text-blue-400" />,
  retrieving_internal: <Database className="h-4 w-4 text-accent-cyan animate-pulse" />,
  internal_complete: <Database className="h-4 w-4 text-emerald-400" />,
  browser_started: <Globe className="h-4 w-4 text-accent-cyan animate-pulse" />,
  browser_complete: <Globe className="h-4 w-4 text-emerald-400" />,
  browser_unavailable: <Globe className="h-4 w-4 text-amber-400" />,
  synthesizing: <Cpu className="h-4 w-4 text-purple-400 animate-pulse" />,
  validating_citations: <FileCheck className="h-4 w-4 text-accent-cyan animate-pulse" />,
  completed: <CheckCircle className="h-4 w-4 text-emerald-400" />,
  interrupted: <AlertTriangle className="h-4 w-4 text-amber-400" />,
};

function ResearchWorkspaceContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialTask = searchParams.get('task') || '';
  const initialJobId = searchParams.get('job_id') || '';

  const [taskQuery, setTaskQuery] = useState(initialTask);
  const [maxSteps, setMaxSteps] = useState(25);
  const [targetDomains, setTargetDomains] = useState('');

  // Active Job State
  const [activeJobId, setActiveJobId] = useState<string | null>(initialJobId || null);
  const [jobStatus, setJobStatus] = useState<ResearchStatus | null>(null);
  const [progressStage, setProgressStage] = useState<string | null>(null);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressMessage, setProgressMessage] = useState<string | null>(null);
  const [isTerminal, setIsTerminal] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  const cleanupSubscriptionRef = useRef<(() => void) | null>(null);

  // Subscribe to progress events when activeJobId changes
  useEffect(() => {
    if (!activeJobId) return;

    // Check status immediately
    apiClient
      .getResearchStatus(activeJobId)
      .then((statusRes: ResearchJobStatusResponse) => {
        setJobStatus(statusRes.status);
        setProgressStage(statusRes.progress_stage || null);
        setProgressPercent(statusRes.progress_percent);
        setProgressMessage(statusRes.progress_message || null);
        setIsTerminal(statusRes.is_terminal);
        if (statusRes.error) setErrorMsg(statusRes.error);
      })
      .catch((err) => {
        console.warn('Could not fetch initial status', err);
      });

    // Start SSE stream with polling fallback
    const unsubscribe = apiClient.subscribeResearchProgress(
      activeJobId,
      (event: ResearchProgressEvent) => {
        setJobStatus(event.status);
        setProgressStage(event.progress_stage || null);
        setProgressPercent(event.progress_percent || 0);
        setProgressMessage(event.progress_message || null);
        setIsTerminal(event.is_terminal);
        if (event.error) setErrorMsg(event.error);
      },
      (err) => {
        console.warn('Subscription error', err);
      }
    );

    cleanupSubscriptionRef.current = unsubscribe;

    return () => {
      if (cleanupSubscriptionRef.current) {
        cleanupSubscriptionRef.current();
        cleanupSubscriptionRef.current = null;
      }
    };
  }, [activeJobId]);

  const handleStartResearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!taskQuery.trim() || isSubmitting) return;

    setIsSubmitting(true);
    setErrorMsg(null);
    setJobStatus('queued');
    setProgressPercent(0);
    setProgressStage('queued');
    setProgressMessage('Submitting research job to background orchestrator...');
    setIsTerminal(false);

    try {
      const domainsList = targetDomains
        .split(',')
        .map((d) => d.trim())
        .filter(Boolean);

      const res = await apiClient.startResearch({
        task: taskQuery.trim(),
        max_steps: maxSteps,
        target_domains: domainsList.length > 0 ? domainsList : undefined,
        sync: false, // Explicitly asynchronous background job
      });

      const assignedId = res.research_id;
      if (!assignedId) {
        throw new Error('Backend failed to return a research_id.');
      }

      setActiveJobId(assignedId);
      router.replace(`/research?job_id=${encodeURIComponent(assignedId)}`);
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to submit research job');
      setJobStatus('failed');
      setIsTerminal(true);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelJob = async () => {
    if (!activeJobId || isCancelling || isTerminal) return;
    setIsCancelling(true);
    try {
      await apiClient.cancelResearch(activeJobId);
      setJobStatus('cancelled');
      setProgressStage('cancelled');
      setProgressMessage('Research job cancelled by user.');
      setIsTerminal(true);
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to cancel job');
    } finally {
      setIsCancelling(false);
    }
  };

  const handleReset = () => {
    if (cleanupSubscriptionRef.current) {
      cleanupSubscriptionRef.current();
      cleanupSubscriptionRef.current = null;
    }
    setActiveJobId(null);
    setJobStatus(null);
    setProgressStage(null);
    setProgressPercent(0);
    setProgressMessage(null);
    setIsTerminal(false);
    setErrorMsg(null);
    router.replace('/research');
  };

  return (
    <div className="flex-1 flex flex-col">
      <Header
        title="Autonomous Research Workspace"
        subtitle="Submit long-running research jobs with live SSE milestones and verifiable evidence"
      />

      <main className="flex-1 p-6 max-w-5xl mx-auto w-full space-y-6">
        {/* Research Input Form */}
        <section className="rounded-xl border border-border bg-surface p-6 shadow-xl space-y-4">
          <form onSubmit={handleStartResearch} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1.5">
                Research Task or Question <span className="text-accent-cyan">*</span>
              </label>
              <textarea
                rows={3}
                value={taskQuery}
                onChange={(e) => setTaskQuery(e.target.value)}
                disabled={activeJobId !== null && !isTerminal}
                placeholder="What objective should InsightBot investigate? (e.g. Ingested AWS policies vs public compliance guidelines)"
                className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none focus:ring-1 focus:ring-accent-cyan disabled:opacity-50"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Max Investigation Steps: <span className="font-mono text-accent-cyan">{maxSteps}</span>
                </label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  step={5}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(Number(e.target.value))}
                  disabled={activeJobId !== null && !isTerminal}
                  className="w-full accent-accent-cyan"
                />
                <div className="flex justify-between text-[11px] text-slate-500 font-mono mt-1">
                  <span>5 (Fast)</span>
                  <span>25 (Standard)</span>
                  <span>50 (Deep)</span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Target Domain Whitelist (Optional, comma-separated)
                </label>
                <input
                  type="text"
                  value={targetDomains}
                  onChange={(e) => setTargetDomains(e.target.value)}
                  disabled={activeJobId !== null && !isTerminal}
                  placeholder="e.g. postgresql.org, github.com"
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none font-mono"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-border/60">
              <span className="text-xs text-slate-500 font-mono">
                Execution: Managed Background Asynchronous Worker
              </span>

              <div className="flex items-center gap-2">
                {activeJobId && isTerminal && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="px-4 py-2 rounded-lg border border-border text-slate-300 hover:text-white text-xs transition-colors"
                  >
                    Start New Investigation
                  </button>
                )}

                {(!activeJobId || isTerminal) && (
                  <button
                    type="submit"
                    disabled={!taskQuery.trim() || isSubmitting}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent-cyan hover:bg-accent-cyan/90 disabled:opacity-50 text-background font-semibold text-xs transition-all shadow-md shadow-accent-cyan/10"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    <span>{isSubmitting ? 'Submitting...' : 'Start Autonomous Research'}</span>
                  </button>
                )}
              </div>
            </div>
          </form>
        </section>

        {/* Active Job Tracking & Live Progress Card */}
        {activeJobId && (
          <section className="rounded-xl border border-border bg-surface p-6 shadow-xl space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-4 border-b border-border">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 flex items-center justify-center text-accent-cyan">
                  <Compass className="h-5 w-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-white text-base">Active Research Job</h3>
                    <StatusBadge status={jobStatus || 'queued'} />
                  </div>
                  <p className="text-xs font-mono text-slate-400 mt-0.5">ID: {activeJobId}</p>
                </div>
              </div>

              {!isTerminal && (
                <button
                  onClick={handleCancelJob}
                  disabled={isCancelling}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 text-xs font-medium transition-colors"
                >
                  <Ban className="h-3.5 w-3.5" />
                  <span>{isCancelling ? 'Cancelling...' : 'Cancel Research'}</span>
                </button>
              )}
            </div>

            {/* Progress Bar & Stage Info */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 flex items-center gap-2 font-mono">
                  {STAGE_ICONS[progressStage || 'queued'] || <Layers className="h-4 w-4 text-slate-400" />}
                  <span>Stage: <strong className="text-white">{progressStage || 'queued'}</strong></span>
                </span>
                <span className="font-mono font-semibold text-accent-cyan">{progressPercent}%</span>
              </div>

              <div className="h-2.5 w-full bg-background rounded-full overflow-hidden border border-border">
                <div
                  className={`h-full transition-all duration-300 rounded-full ${
                    jobStatus === 'failed'
                      ? 'bg-rose-500'
                      : jobStatus === 'cancelled'
                      ? 'bg-slate-500'
                      : jobStatus === 'interrupted'
                      ? 'bg-amber-500'
                      : jobStatus === 'completed'
                      ? 'bg-emerald-500'
                      : 'bg-gradient-to-r from-accent-cyan to-accent-blue'
                  }`}
                  style={{ width: `${Math.max(progressPercent, 4)}%` }}
                />
              </div>

              {progressMessage && (
                <p className="text-xs text-slate-400 font-mono mt-1 flex items-center gap-1.5">
                  <span className="text-accent-cyan">&gt;</span>
                  {progressMessage}
                </p>
              )}
            </div>

            {/* Terminal Result Banner */}
            {isTerminal && (
              <div className="pt-4 border-t border-border/60">
                {jobStatus === 'completed' && (
                  <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <CheckCircle className="h-6 w-6 text-emerald-400 shrink-0" />
                      <div>
                        <p className="font-semibold text-white text-sm">Research Successfully Completed</p>
                        <p className="text-xs text-emerald-300/80 mt-0.5">
                          Evidence has been gathered, synthesized, and verified against internal & web sources.
                        </p>
                      </div>
                    </div>
                    <Link
                      href={`/reports/${activeJobId}`}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-background font-semibold text-xs shrink-0 transition-colors shadow-md"
                    >
                      <span>View Full Report</span>
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                )}

                {jobStatus === 'completed_unverified' && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="h-6 w-6 text-amber-400 shrink-0" />
                      <div>
                        <p className="font-semibold text-white text-sm">Completed (Unverified Citations)</p>
                        <p className="text-xs text-amber-300/80 mt-0.5">
                          Research finished, but some citations could not be strictly verified. Review the report with caution.
                        </p>
                      </div>
                    </div>
                    <Link
                      href={`/reports/${activeJobId}`}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-background font-semibold text-xs shrink-0 transition-colors shadow-md"
                    >
                      <span>Inspect Evidence</span>
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                )}

                {jobStatus === 'provider_unavailable' && (
                  <div className="rounded-lg border border-purple-500/30 bg-purple-500/10 p-4 space-y-2">
                    <div className="flex items-center gap-2 text-purple-300 text-sm font-semibold">
                      <AlertTriangle className="h-5 w-5 text-purple-400" />
                      Upstream AI Provider Unavailable
                    </div>
                    <p className="text-xs text-purple-300/80">
                      The upstream LLM or browser service encountered rate limits, transient outages, or missing credentials. No fake results were generated.
                    </p>
                  </div>
                )}

                {jobStatus === 'failed' && (
                  <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 space-y-3">
                    <div className="flex items-center gap-2 text-rose-300 text-sm font-semibold">
                      <AlertTriangle className="h-5 w-5 text-rose-400" />
                      Research Execution Failed
                    </div>
                    <p className="text-xs text-rose-300/80 font-mono">
                      {errorMsg || 'An internal error occurred during synthesis.'}
                    </p>
                    <div className="flex items-center gap-3 pt-1">
                      <button
                        type="button"
                        onClick={handleStartResearch}
                        disabled={isSubmitting}
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-semibold text-xs transition-colors shadow-sm"
                      >
                        <Play className="h-3.5 w-3.5 fill-current" />
                        <span>Retry Research Job</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleReset}
                        className="px-3 py-1.5 rounded-lg border border-border text-slate-300 hover:text-white text-xs transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                )}

                {jobStatus === 'interrupted' && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 space-y-3">
                    <div className="flex items-center gap-2 text-amber-300 text-sm font-semibold">
                      <AlertTriangle className="h-5 w-5 text-amber-400" />
                      Research Execution Interrupted
                    </div>
                    <p className="text-xs text-amber-300/80">
                      {errorMsg || 'The research execution was interrupted by an application restart or server shutdown. In-memory execution state was discarded to preserve data integrity.'}
                    </p>
                    <div className="flex items-center gap-3 pt-1">
                      <button
                        type="button"
                        onClick={handleStartResearch}
                        disabled={isSubmitting}
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-600 text-background font-semibold text-xs transition-colors shadow-sm"
                      >
                        <Play className="h-3.5 w-3.5 fill-current" />
                        <span>Resubmit Research Job</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleReset}
                        className="px-3 py-1.5 rounded-lg border border-border text-slate-300 hover:text-white text-xs transition-colors"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                )}

                {jobStatus === 'cancelled' && (
                  <div className="rounded-lg border border-slate-700 bg-slate-800/40 p-4 text-xs text-slate-400 font-mono">
                    Job was cancelled by user request. All background tasks and streaming queues have been terminated cleanly.
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default function ResearchWorkspacePage() {
  return (
    <Suspense
      fallback={
        <div className="flex-1 flex items-center justify-center p-12 text-xs font-mono text-slate-400">
          Loading workspace...
        </div>
      }
    >
      <ResearchWorkspaceContent />
    </Suspense>
  );
}

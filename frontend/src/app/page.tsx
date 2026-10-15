'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Compass,
  FileText,
  MessageSquare,
  Database,
  ArrowRight,
  Sparkles,
  RefreshCw,
  AlertCircle,
  Clock,
  Layers,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { StatusBadge } from '@/components/StatusBadge';
import { apiClient } from '@/lib/api';
import { ResearchSession, Conversation, Document, HealthResponse } from '@/types/api';

export default function DashboardPage() {
  const router = useRouter();
  const [researchSessions, setResearchSessions] = useState<ResearchSession[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [quickQuery, setQuickQuery] = useState('');

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sessionsData, convsData, docsData, healthData] = await Promise.allSettled([
        apiClient.listResearchSessions(10, 0),
        apiClient.listConversations(10, 0),
        apiClient.listDocuments(10, 0),
        apiClient.getHealth(),
      ]);

      if (sessionsData.status === 'fulfilled') setResearchSessions(sessionsData.value);
      if (convsData.status === 'fulfilled') setConversations(convsData.value);
      if (docsData.status === 'fulfilled') setDocuments(docsData.value);
      if (healthData.status === 'fulfilled') setHealth(healthData.value);

      // If all failed, show error
      if (
        sessionsData.status === 'rejected' &&
        convsData.status === 'rejected' &&
        docsData.status === 'rejected'
      ) {
        setError('Failed to connect to InsightBot API server. Ensure backend is running.');
      }
    } catch (err: any) {
      setError(err?.message || 'Error loading dashboard');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const handleQuickLaunch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!quickQuery.trim()) return;
    router.push(`/research?task=${encodeURIComponent(quickQuery.trim())}`);
  };

  return (
    <div className="flex-1 flex flex-col">
      <Header
        title="Autonomous Research Dashboard"
        subtitle="Live state, grounding status, and unified knowledge overview"
        actionButton={
          <button
            onClick={loadDashboardData}
            aria-label="Refresh Dashboard Data"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-300 text-xs transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-accent-cyan' : ''}`} />
            Refresh
          </button>
        }
      />

      <main className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
        {/* Error Alert Banner */}
        {error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 flex items-start gap-3 text-rose-300 text-sm">
            <AlertCircle className="h-5 w-5 text-rose-400 mt-0.5 shrink-0" />
            <div className="flex-1">
              <p className="font-medium">Connection Error</p>
              <p className="text-xs text-rose-400/90 mt-0.5">{error}</p>
            </div>
            <button
              onClick={loadDashboardData}
              className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 rounded text-xs transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Quick Launch Hero Input */}
        <section className="rounded-2xl border border-accent-cyan/20 bg-gradient-to-r from-surface via-surface to-accent-cyan/5 p-6 shadow-xl relative overflow-hidden">
          <div className="relative z-10 max-w-3xl space-y-3">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan text-xs font-mono">
              <Sparkles className="h-3.5 w-3.5" />
              Autonomous Dual-Source Investigation
            </div>
            <h3 className="text-2xl font-bold text-white tracking-tight">
              What would you like to investigate today?
            </h3>
            <p className="text-sm text-slate-400">
              InsightBot synthesizes private documents with autonomous browser evidence and enforces strict verification to eliminate hallucinations.
            </p>

            <form onSubmit={handleQuickLaunch} className="flex gap-2 pt-2">
              <input
                type="text"
                value={quickQuery}
                onChange={(e) => setQuickQuery(e.target.value)}
                placeholder="e.g. Verify PostgreSQL 17 replication requirements and cloud support..."
                className="flex-1 rounded-xl border border-border bg-background/80 px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none focus:ring-1 focus:ring-accent-cyan backdrop-blur-sm"
              />
              <button
                type="submit"
                disabled={!quickQuery.trim()}
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-accent-cyan hover:bg-accent-cyan/90 disabled:opacity-50 disabled:pointer-events-none text-background font-semibold text-sm transition-all shadow-lg shadow-accent-cyan/20"
              >
                <span>Launch</span>
                <ArrowRight className="h-4 w-4" />
              </button>
            </form>
          </div>
        </section>

        {/* Real Metrics Grid */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 flex items-center justify-center text-accent-cyan">
              <Compass className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-medium text-slate-400">Research Sessions</p>
              <p className="text-2xl font-bold text-white mt-0.5">
                {loading ? '...' : researchSessions.length}
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-accent-blue/10 border border-accent-blue/20 flex items-center justify-center text-accent-blue">
              <Database className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-medium text-slate-400">Knowledge Documents</p>
              <p className="text-2xl font-bold text-white mt-0.5">
                {loading ? '...' : documents.length}
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-accent-purple/10 border border-accent-purple/20 flex items-center justify-center text-accent-purple">
              <MessageSquare className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-medium text-slate-400">Conversations</p>
              <p className="text-2xl font-bold text-white mt-0.5">
                {loading ? '...' : conversations.length}
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
            <div className="h-12 w-12 rounded-lg bg-accent-emerald/10 border border-accent-emerald/20 flex items-center justify-center text-accent-emerald">
              <Layers className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-medium text-slate-400">Engine Health</p>
              <p className="text-sm font-bold text-white mt-1 capitalize flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    health?.status === 'ok' ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                {health?.status || 'Unknown'}
              </p>
            </div>
          </div>
        </section>

        {/* Content Split: Recent Research & Recent Chat */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Research Sessions (2 cols) */}
          <section className="lg:col-span-2 rounded-xl border border-border bg-surface overflow-hidden flex flex-col">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-accent-cyan" />
                <h4 className="font-semibold text-white text-sm">Recent Research Sessions</h4>
              </div>
              <Link
                href="/reports"
                className="text-xs text-accent-cyan hover:underline flex items-center gap-1"
              >
                View all <ArrowRight className="h-3 w-3" />
              </Link>
            </div>

            <div className="p-4 flex-1">
              {loading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 rounded-lg bg-surfaceHover animate-pulse" />
                  ))}
                </div>
              ) : researchSessions.length === 0 ? (
                <div className="text-center py-10 space-y-2">
                  <Compass className="h-8 w-8 text-slate-600 mx-auto" />
                  <p className="text-sm text-slate-400">No research sessions recorded yet.</p>
                  <Link
                    href="/research"
                    className="inline-block text-xs text-accent-cyan hover:underline"
                  >
                    Start your first research session &rarr;
                  </Link>
                </div>
              ) : (
                <div className="divide-y divide-border/60">
                  {researchSessions.slice(0, 5).map((session) => (
                    <div
                      key={session.id}
                      className="py-3 first:pt-0 last:pb-0 flex items-center justify-between gap-4"
                    >
                      <div className="min-w-0 flex-1">
                        <Link
                          href={`/reports/${session.id}`}
                          className="font-medium text-sm text-slate-200 hover:text-accent-cyan transition-colors truncate block"
                        >
                          {session.title || session.query}
                        </Link>
                        <div className="flex items-center gap-3 text-xs text-slate-500 mt-1 font-mono">
                          <span>{session.id}</span>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(session.created_at).toLocaleDateString()}
                          </span>
                          <span>•</span>
                          <span>{session.sources?.length || 0} sources</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        <StatusBadge status={session.status} />
                        <Link
                          href={`/reports/${session.id}`}
                          className="p-1.5 rounded-lg border border-border bg-surfaceHover hover:border-slate-600 text-slate-400 hover:text-white transition-colors text-xs"
                          title="Open Report"
                        >
                          <ArrowRight className="h-3.5 w-3.5" />
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          {/* Recent Conversations (1 col) */}
          <section className="rounded-xl border border-border bg-surface overflow-hidden flex flex-col">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-accent-purple" />
                <h4 className="font-semibold text-white text-sm">Grounded Chat Threads</h4>
              </div>
              <Link
                href="/chat"
                className="text-xs text-accent-purple hover:underline flex items-center gap-1"
              >
                Open Chat <ArrowRight className="h-3 w-3" />
              </Link>
            </div>

            <div className="p-4 flex-1">
              {loading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-12 rounded-lg bg-surfaceHover animate-pulse" />
                  ))}
                </div>
              ) : conversations.length === 0 ? (
                <div className="text-center py-10 space-y-2">
                  <MessageSquare className="h-8 w-8 text-slate-600 mx-auto" />
                  <p className="text-sm text-slate-400">No conversations yet.</p>
                  <Link href="/chat" className="inline-block text-xs text-accent-purple hover:underline">
                    Start a grounded conversation &rarr;
                  </Link>
                </div>
              ) : (
                <div className="space-y-2">
                  {conversations.slice(0, 5).map((conv) => (
                    <Link
                      key={conv.id}
                      href={`/chat?id=${conv.id}`}
                      className="block p-2.5 rounded-lg border border-border bg-background/50 hover:bg-surfaceHover hover:border-slate-600 transition-all"
                    >
                      <p className="font-medium text-xs text-slate-200 truncate">{conv.title}</p>
                      <p className="text-[11px] text-slate-500 font-mono mt-1">
                        {new Date(conv.updated_at).toLocaleDateString()}
                      </p>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

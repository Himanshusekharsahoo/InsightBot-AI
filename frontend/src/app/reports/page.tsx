'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  FileText,
  Search,
  Filter,
  ArrowRight,
  Download,
  Trash2,
  Clock,
  Layers,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { StatusBadge } from '@/components/StatusBadge';
import { apiClient } from '@/lib/api';
import { ResearchSession } from '@/types/api';

export default function ReportsListPage() {
  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [page, setPage] = useState(0);
  const limit = 20;

  const loadSessions = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.listResearchSessions(limit, page * limit);
      setSessions(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load research reports');
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Are you sure you want to delete research session '${id}'?`)) return;

    try {
      await apiClient.deleteResearchSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
    } catch (err: any) {
      alert(`Delete failed: ${err?.message}`);
    }
  };

  const filteredSessions = sessions.filter((s) => {
    const matchesSearch =
      s.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
      s.query.toLowerCase().includes(searchFilter.toLowerCase()) ||
      s.id.toLowerCase().includes(searchFilter.toLowerCase());
    const matchesStatus = statusFilter === 'all' || s.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="flex-1 flex flex-col">
      <Header
        title="Research Reports Archive"
        subtitle="Saved autonomous multi-source investigations, evidence trails, and exportable artifacts"
        actionButton={
          <Link
            href="/research"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-cyan text-background font-semibold text-xs hover:bg-accent-cyan/90 transition-colors shadow-sm"
          >
            New Research
          </Link>
        }
      />

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* Controls: Search & Status Filter */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Filter by title, query, or ID..."
              className="w-full rounded-xl border border-border bg-surface pl-9 pr-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none"
            />
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Filter className="h-3.5 w-3.5" />
              <span>Status:</span>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs text-white focus:border-accent-cyan focus:outline-none"
              >
                <option value="all">All Statuses</option>
                <option value="completed">Completed</option>
                <option value="completed_unverified">Completed (Unverified)</option>
                <option value="provider_unavailable">Provider Unavailable</option>
                <option value="failed">Failed</option>
                <option value="cancelled">Cancelled</option>
              </select>
            </div>

            <button
              onClick={loadSessions}
              aria-label="Refresh Reports"
              className="p-2 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-400 hover:text-white transition-colors"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-accent-cyan' : ''}`} />
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 flex items-center gap-3 text-rose-300 text-sm">
            <AlertCircle className="h-5 w-5 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Reports Table / Cards */}
        <div className="rounded-xl border border-border bg-surface overflow-hidden shadow-xl">
          {loading ? (
            <div className="p-8 space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-14 bg-surfaceHover rounded-lg animate-pulse" />
              ))}
            </div>
          ) : filteredSessions.length === 0 ? (
            <div className="py-16 text-center space-y-3">
              <FileText className="h-10 w-10 text-slate-600 mx-auto" />
              <p className="text-sm font-medium text-slate-300">No research reports found</p>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                {searchFilter || statusFilter !== 'all'
                  ? 'Try adjusting your search query or status filter.'
                  : 'Start a research task in the workspace to generate persistent reports.'}
              </p>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {filteredSessions.map((session) => (
                <Link
                  key={session.id}
                  href={`/reports/${session.id}`}
                  className="block p-4 hover:bg-surfaceHover/50 transition-colors group"
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center gap-2.5">
                        <h4 className="font-semibold text-sm text-white group-hover:text-accent-cyan transition-colors truncate">
                          {session.title || session.query}
                        </h4>
                        <StatusBadge status={session.status} />
                      </div>
                      <p className="text-xs text-slate-400 truncate font-mono">
                        Query: &quot;{session.query}&quot;
                      </p>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-mono text-slate-400 shrink-0">
                      <div className="flex items-center gap-1.5">
                        <Clock className="h-3.5 w-3.5 text-slate-500" />
                        <span>{new Date(session.created_at).toLocaleString()}</span>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <Layers className="h-3.5 w-3.5 text-slate-500" />
                        <span>{session.citations?.length || 0} citations</span>
                      </div>

                      {/* Export Shortcuts */}
                      <a
                        href={apiClient.getExportUrl(session.id, 'markdown')}
                        download
                        onClick={(e) => e.stopPropagation()}
                        title="Export Markdown"
                        className="p-1.5 rounded-lg border border-border hover:border-slate-500 text-slate-400 hover:text-white"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </a>

                      <button
                        onClick={(e) => handleDelete(session.id, e)}
                        title="Delete Session"
                        className="p-1.5 rounded-lg border border-border hover:border-rose-500/50 text-slate-400 hover:text-rose-400 transition-colors"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>

                      <ArrowRight className="h-4 w-4 text-slate-500 group-hover:text-accent-cyan transition-colors" />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
          <span>Page {page + 1}</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0 || loading}
              className="px-3 py-1.5 rounded-lg border border-border bg-surface disabled:opacity-40 hover:bg-surfaceHover"
            >
              Previous
            </button>
            <button
              onClick={() => setPage(page + 1)}
              disabled={sessions.length < limit || loading}
              className="px-3 py-1.5 rounded-lg border border-border bg-surface disabled:opacity-40 hover:bg-surfaceHover"
            >
              Next
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

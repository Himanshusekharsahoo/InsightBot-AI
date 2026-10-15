import React from 'react';
import { ResearchStatus } from '@/types/api';
import { CheckCircle, AlertTriangle, XCircle, Clock, Loader2, Ban, CloudOff } from 'lucide-react';

interface StatusBadgeProps {
  status: ResearchStatus | string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'sm' }) => {
  const pad = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm font-medium';

  switch (status) {
    case 'completed':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 ${pad}`}
        >
          <CheckCircle className="h-3.5 w-3.5 text-emerald-400" />
          Completed
        </span>
      );
    case 'completed_unverified':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-300 ${pad}`}
        >
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          Completed (Unverified)
        </span>
      );
    case 'in_progress':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 ${pad}`}
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
          In Progress
        </span>
      );
    case 'queued':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-300 ${pad}`}
        >
          <Clock className="h-3.5 w-3.5 text-blue-400" />
          Queued
        </span>
      );
    case 'provider_unavailable':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-300 ${pad}`}
        >
          <CloudOff className="h-3.5 w-3.5 text-purple-400" />
          Provider Unavailable
        </span>
      );
    case 'failed':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-400 ${pad}`}
        >
          <XCircle className="h-3.5 w-3.5 text-rose-400" />
          Failed
        </span>
      );
    case 'cancelled':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-slate-600/40 bg-slate-700/20 text-slate-400 ${pad}`}
        >
          <Ban className="h-3.5 w-3.5 text-slate-400" />
          Cancelled
        </span>
      );
    case 'interrupted':
      return (
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-300 ${pad}`}
        >
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          Interrupted
        </span>
      );
    default:
      return (
        <span className={`inline-flex items-center rounded-full border border-slate-700 bg-slate-800 text-slate-300 ${pad}`}>
          {status}
        </span>
      );
  }
};

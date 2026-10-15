'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Compass,
  FileText,
  MessageSquare,
  Database,
  Shield,
  Layers,
  Menu,
  X,
  Activity,
} from 'lucide-react';
import { apiClient } from '@/lib/api';
import { HealthResponse } from '@/types/api';

const NAV_ITEMS = [
  { label: 'Dashboard', href: '/', icon: LayoutDashboard },
  { label: 'Research', href: '/research', icon: Compass },
  { label: 'Reports', href: '/reports', icon: FileText },
  { label: 'AI Chat', href: '/chat', icon: MessageSquare },
  { label: 'Knowledge Base', href: '/knowledge', icon: Database },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    apiClient
      .getHealth()
      .then((data) => {
        if (mounted) setHealth(data);
      })
      .catch(() => {
        if (mounted)
          setHealth({
            status: 'offline',
            service: 'InsightBot AI',
            version: 'unknown',
            database: 'unavailable',
            storage_mode: 'unknown',
            pgvector_enabled: false,
          });
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <>
      {/* Mobile Top Bar */}
      <div className="md:hidden flex items-center justify-between bg-surface border-b border-border px-4 py-3 sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <Layers className="h-6 w-6 text-accent-cyan" />
          <span className="font-bold text-white tracking-tight">InsightBot AI</span>
        </div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle Navigation Menu"
          className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-surfaceHover transition-colors"
        >
          {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Sidebar Overlay for Mobile */}
      {isOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/60 z-40 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Main Sidebar Container */}
      <aside
        className={`fixed md:sticky top-0 left-0 z-50 h-screen w-64 bg-surface border-r border-border flex flex-col justify-between transition-transform duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div>
          {/* Logo / Brand Header */}
          <div className="p-5 border-b border-border">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-accent-blue/20 border border-accent-cyan/30 flex items-center justify-center group-hover:border-accent-cyan/60 transition-colors">
                <Layers className="h-5 w-5 text-accent-cyan" />
              </div>
              <div>
                <h1 className="font-bold text-base text-white tracking-tight flex items-center gap-1.5">
                  InsightBot <span className="text-accent-cyan">AI</span>
                </h1>
                <p className="text-[11px] font-mono text-slate-400">Autonomous RAG & Web</p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === '/'
                  ? pathname === '/'
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20 shadow-sm shadow-accent-cyan/5'
                      : 'text-slate-400 hover:text-white hover:bg-surfaceHover'
                  }`}
                >
                  <Icon className={`h-4 w-4 ${isActive ? 'text-accent-cyan' : 'text-slate-400'}`} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* System Health / Status Indicator Footer */}
        <div className="p-4 border-t border-border bg-background/50">
          <div className="rounded-lg border border-border p-3 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5 text-slate-400" />
                Backend Status
              </span>
              <span className="flex items-center gap-1.5">
                <span
                  className={`h-2 w-2 rounded-full ${
                    health?.status === 'ok'
                      ? 'bg-emerald-400 animate-pulse'
                      : health?.status === 'degraded'
                      ? 'bg-amber-400'
                      : 'bg-rose-500'
                  }`}
                />
                <span
                  className={`font-mono text-[11px] ${
                    health?.status === 'ok'
                      ? 'text-emerald-400'
                      : health?.status === 'degraded'
                      ? 'text-amber-400'
                      : 'text-rose-400'
                  }`}
                >
                  {health?.status || 'checking...'}
                </span>
              </span>
            </div>
            <div className="flex items-center justify-between text-slate-500 font-mono text-[11px]">
              <span>Storage</span>
              <span className="text-slate-300">
                {health?.database === 'connected' ? health?.storage_mode : 'disconnected'}
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

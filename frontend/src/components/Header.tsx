'use client';

import React, { useEffect, useState } from 'react';
import { Shield, Key, Building2, Sliders, Check } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { TenantContext } from '@/types/api';

interface HeaderProps {
  title: string;
  subtitle?: string;
  actionButton?: React.ReactNode;
}

export const Header: React.FC<HeaderProps> = ({ title, subtitle, actionButton }) => {
  const [tenant, setTenant] = useState<TenantContext | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [orgIdInput, setOrgIdInput] = useState('');
  const [workspaceIdInput, setWorkspaceIdInput] = useState('');
  const [savedFeedback, setSavedFeedback] = useState(false);

  const loadTenant = () => {
    apiClient
      .getAuthMe()
      .then((t) => setTenant(t))
      .catch(() => {
        setTenant({
          organization_id: 'default',
          workspace_id: null,
          api_key_id: null,
          is_admin: false,
        });
      });
  };

  useEffect(() => {
    loadTenant();
    if (typeof window !== 'undefined') {
      setApiKeyInput(sessionStorage.getItem('insightbot_api_key') || '');
      setOrgIdInput(sessionStorage.getItem('insightbot_org_id') || '');
      setWorkspaceIdInput(sessionStorage.getItem('insightbot_workspace_id') || '');
    }
  }, []);

  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    if (typeof window !== 'undefined') {
      if (apiKeyInput.trim()) {
        sessionStorage.setItem('insightbot_api_key', apiKeyInput.trim());
      } else {
        sessionStorage.removeItem('insightbot_api_key');
      }

      if (orgIdInput.trim()) {
        sessionStorage.setItem('insightbot_org_id', orgIdInput.trim());
      } else {
        sessionStorage.removeItem('insightbot_org_id');
      }

      if (workspaceIdInput.trim()) {
        sessionStorage.setItem('insightbot_workspace_id', workspaceIdInput.trim());
      } else {
        sessionStorage.removeItem('insightbot_workspace_id');
      }
    }
    setSavedFeedback(true);
    setTimeout(() => {
      setSavedFeedback(false);
      setShowModal(false);
      loadTenant();
      window.location.reload();
    }, 600);
  };

  return (
    <>
      <header className="border-b border-border bg-surface/50 backdrop-blur-md px-6 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 sticky top-0 z-30">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">{title}</h2>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
        </div>

        <div className="flex items-center gap-3">
          {actionButton}

          {/* Tenant Indicator Button */}
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-surfaceHover text-slate-300 hover:text-white hover:border-slate-600 transition-colors text-xs font-mono"
            title="Configure Tenant or API Key"
          >
            <Building2 className="h-3.5 w-3.5 text-accent-cyan" />
            <span>org: <span className="text-accent-cyan">{tenant?.organization_id || 'default'}</span></span>
            {tenant?.workspace_id && (
              <span className="text-slate-400">/ ws: {tenant.workspace_id}</span>
            )}
            <Sliders className="h-3 w-3 text-slate-400 ml-1" />
          </button>
        </div>
      </header>

      {/* Tenant & Auth Settings Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-xl border border-border bg-surface p-6 shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-border mb-4">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-accent-cyan" />
                <h3 className="font-semibold text-white">Tenant & Authorization</h3>
              </div>
              <button
                onClick={() => setShowModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSaveSettings} className="space-y-4 text-sm">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  API Key (Optional / Bearer Token)
                </label>
                <input
                  type="password"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  placeholder="Paste tenant API key..."
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none font-mono text-xs"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Stored in browser sessionStorage for this session.
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  X-Organization-Id Override (Optional)
                </label>
                <input
                  type="text"
                  value={orgIdInput}
                  onChange={(e) => setOrgIdInput(e.target.value)}
                  placeholder="e.g. org-alpha"
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none font-mono text-xs"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  X-Workspace-Id Context (Optional)
                </label>
                <input
                  type="text"
                  value={workspaceIdInput}
                  onChange={(e) => setWorkspaceIdInput(e.target.value)}
                  placeholder="e.g. ws-security"
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none font-mono text-xs"
                />
              </div>

              <div className="pt-2 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-3 py-1.5 rounded-lg border border-border text-slate-400 hover:text-white text-xs"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-accent-cyan hover:bg-accent-cyan/90 text-background font-medium text-xs transition-colors"
                >
                  {savedFeedback ? (
                    <>
                      <Check className="h-3.5 w-3.5" />
                      Saved!
                    </>
                  ) : (
                    'Save Context'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

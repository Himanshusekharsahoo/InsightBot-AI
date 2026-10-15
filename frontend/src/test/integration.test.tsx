import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DashboardPage from '@/app/page';
import { apiClient } from '@/lib/api';

// Mock router
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ id: 'res-test-123' }),
}));

describe('Frontend Pages Integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders Dashboard with real API metrics and empty state', async () => {
    vi.spyOn(apiClient, 'getHealth').mockResolvedValue({
      status: 'ok',
      service: 'InsightBot AI',
      version: '1.0.0',
      database: 'connected',
      storage_mode: 'native_pgvector',
      pgvector_enabled: true,
    });
    vi.spyOn(apiClient, 'listResearchSessions').mockResolvedValue([]);
    vi.spyOn(apiClient, 'listConversations').mockResolvedValue([]);
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);

    render(<DashboardPage />);

    expect(screen.getByText('Autonomous Research Dashboard')).toBeInTheDocument();
    expect(screen.getByText('What would you like to investigate today?')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('No research sessions recorded yet.')).toBeInTheDocument();
      expect(screen.getByText('No conversations yet.')).toBeInTheDocument();
    });
  });

  it('renders Dashboard with retrieved research sessions and counts', async () => {
    const mockSession = {
      id: 'res-alpha-1',
      organization_id: 'default',
      title: 'PostgreSQL 17 Research',
      query: 'Verify PostgreSQL 17 features',
      status: 'completed' as const,
      progress_percent: 100,
      summary: 'PostgreSQL 17 supports memory improvements.',
      sources: ['architecture.txt'],
      internal_evidence: [],
      web_evidence: [],
      conflicts: [],
      citations: [
        {
          source_uri: 'architecture.txt',
          document_id: 'doc-1',
          chunk_id: 'chk-1',
          snippet: 'PostgreSQL 17 features',
        },
      ],
      is_grounded: true,
      sufficient_evidence: true,
      metadata: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    vi.spyOn(apiClient, 'getHealth').mockResolvedValue({
      status: 'ok',
      service: 'InsightBot AI',
      version: '1.0.0',
      database: 'connected',
      storage_mode: 'native_pgvector',
      pgvector_enabled: true,
    });
    vi.spyOn(apiClient, 'listResearchSessions').mockResolvedValue([mockSession]);
    vi.spyOn(apiClient, 'listConversations').mockResolvedValue([]);
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);

    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText('PostgreSQL 17 Research')).toBeInTheDocument();
      expect(screen.getByText('res-alpha-1')).toBeInTheDocument();
    });
  });

  it('renders connection error alert when backend is unreachable', async () => {
    vi.spyOn(apiClient, 'getHealth').mockRejectedValue(new Error('Network error: Connection refused'));
    vi.spyOn(apiClient, 'listResearchSessions').mockRejectedValue(new Error('Network error'));
    vi.spyOn(apiClient, 'listConversations').mockRejectedValue(new Error('Network error'));
    vi.spyOn(apiClient, 'listDocuments').mockRejectedValue(new Error('Network error'));

    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText('Connection Error')).toBeInTheDocument();
      expect(
        screen.getByText('Failed to connect to InsightBot API server. Ensure backend is running.')
      ).toBeInTheDocument();
    });
  });
});

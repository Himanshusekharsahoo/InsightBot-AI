import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InsightBotClient, ApiError } from '@/lib/api';

describe('InsightBotClient', () => {
  let client: InsightBotClient;
  const mockBaseUrl = 'http://localhost:8000';

  beforeEach(() => {
    client = new InsightBotClient(mockBaseUrl);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('correctly constructs export URLs', () => {
    const mdUrl = client.getExportUrl('res-123', 'markdown');
    expect(mdUrl).toBe('http://localhost:8000/research/res-123/export?format=markdown');

    const jsonUrl = client.getExportUrl('res-456', 'json');
    expect(jsonUrl).toBe('http://localhost:8000/research/res-456/export?format=json');
  });

  it('handles successful API requests and returns JSON', async () => {
    const mockHealth = {
      status: 'ok',
      service: 'InsightBot AI',
      version: '1.0.0',
      database: 'connected',
      storage_mode: 'native_pgvector',
      pgvector_enabled: true,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    } as Response);

    const result = await client.getHealth();
    expect(result).toEqual(mockHealth);
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/health',
      expect.objectContaining({
        headers: expect.objectContaining({
          Accept: 'application/json',
        }),
      })
    );
  });

  it('throws ApiError with status and detailed error message on 4xx/5xx', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: "Research job 'res-999' not found." }),
    } as Response);

    await expect(client.getResearchStatus('res-999')).rejects.toThrow(
      "Research job 'res-999' not found."
    );
  });

  it('starts research job with queued status', async () => {
    const mockResponse = {
      status: 'queued',
      task: 'Verify compliance',
      summary: 'Research job accepted and queued for background processing.',
      research_id: 'res-queued-123',
      progress_stage: 'queued',
      progress_percent: 0,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    } as Response);

    const res = await client.startResearch({ task: 'Verify compliance', sync: false });
    expect(res.research_id).toBe('res-queued-123');
    expect(res.status).toBe('queued');
  });

  it('cancels research job via POST /research/{id}/cancel', async () => {
    const mockCancelResponse = {
      cancelled: true,
      research_id: 'res-123',
      status: 'cancelled',
      message: "Research job 'res-123' successfully cancelled.",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockCancelResponse,
    } as Response);

    const res = await client.cancelResearch('res-123');
    expect(res.cancelled).toBe(true);
    expect(res.status).toBe('cancelled');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/research/res-123/cancel',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('falls back to polling when SSE subscription fails or disconnects', async () => {
    vi.useFakeTimers();

    // Mock EventSource throwing error immediately
    const mockEventSource = vi.fn().mockImplementation(() => {
      throw new Error('EventSource unavailable');
    });
    vi.stubGlobal('EventSource', mockEventSource);

    // Mock getResearchStatus response
    const mockStatusResponse = {
      research_id: 'res-poll-1',
      organization_id: 'default',
      status: 'completed',
      progress_stage: 'completed',
      progress_percent: 100,
      progress_message: 'Research complete',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      is_terminal: true,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockStatusResponse,
    } as Response);

    const onProgress = vi.fn();
    const onError = vi.fn();

    const unsubscribe = client.subscribeResearchProgress('res-poll-1', onProgress, onError);

    // Advance timer by 2000ms to trigger polling interval
    await vi.advanceTimersByTimeAsync(2100);

    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({
        research_id: 'res-poll-1',
        status: 'completed',
        progress_percent: 100,
        is_terminal: true,
      })
    );

    unsubscribe();
    vi.useRealTimers();
  });
});

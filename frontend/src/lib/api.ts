import {
  HealthResponse,
  TenantContext,
  Document,
  DocumentChunk,
  KnowledgeSearchResult,
  KnowledgeAnswer,
  ResearchSession,
  ResearchJobStatusResponse,
  ResearchProgressEvent,
  ResearchCreatePayload,
  Conversation,
  ConversationWithMessages,
  ConversationMessage,
  ConversationCreatePayload,
  ConversationMessagePayload,
} from '@/types/api';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export class InsightBotClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    // If running in browser, default to proxy path `/api/backend` to avoid CORS & hide secrets.
    // If NEXT_PUBLIC_API_URL is specified (e.g. tests or direct), use that.
    this.baseUrl =
      baseUrl ||
      (typeof window !== 'undefined'
        ? '/api/backend'
        : process.env.INSIGHTBOT_API_URL || 'http://localhost:8000');
  }

  private getHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      ...extraHeaders,
    };

    if (typeof window !== 'undefined') {
      const apiKey = sessionStorage.getItem('insightbot_api_key');
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
      const workspaceId = sessionStorage.getItem('insightbot_workspace_id');
      if (workspaceId) {
        headers['X-Workspace-Id'] = workspaceId;
      }
      const orgId = sessionStorage.getItem('insightbot_org_id');
      if (orgId) {
        headers['X-Organization-Id'] = orgId;
      }
    }

    return headers;
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
    const headers = this.getHeaders(
      (options.headers as Record<string, string>) || {}
    );

    const config: RequestInit = {
      ...options,
      headers: {
        'Accept': 'application/json',
        ...headers,
      },
    };

    const res = await fetch(url, config);

    if (!res.ok) {
      let errorMsg = `HTTP ${res.status} ${res.statusText}`;
      let errorData: any = null;
      try {
        errorData = await res.json();
        if (errorData?.detail) {
          errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
        } else if (errorData?.message) {
          errorMsg = errorData.message;
        }
      } catch {
        // Response wasn't JSON
      }
      throw new ApiError(errorMsg, res.status, errorData);
    }

    if (res.status === 204) {
      return {} as T;
    }

    return res.json();
  }

  // System & Tenant
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getAuthMe(): Promise<TenantContext> {
    return this.request<TenantContext>('/auth/me');
  }

  // Research
  async startResearch(payload: ResearchCreatePayload): Promise<any> {
    return this.request('/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async getResearchStatus(researchId: string): Promise<ResearchJobStatusResponse> {
    return this.request<ResearchJobStatusResponse>(`/research/${encodeURIComponent(researchId)}/status`);
  }

  async cancelResearch(researchId: string): Promise<{ cancelled: boolean; status: string; message: string }> {
    return this.request(`/research/${encodeURIComponent(researchId)}/cancel`, {
      method: 'POST',
    });
  }

  async listResearchSessions(
    limit: number = 20,
    offset: number = 0,
    workspaceId?: string
  ): Promise<ResearchSession[]> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (workspaceId) params.append('workspace_id', workspaceId);
    return this.request<ResearchSession[]>(`/research?${params.toString()}`);
  }

  async getResearchSession(researchId: string): Promise<ResearchSession> {
    return this.request<ResearchSession>(`/research/${encodeURIComponent(researchId)}`);
  }

  async deleteResearchSession(researchId: string): Promise<{ deleted: boolean; research_id: string }> {
    return this.request(`/research/${encodeURIComponent(researchId)}`, {
      method: 'DELETE',
    });
  }

  getExportUrl(researchId: string, format: 'markdown' | 'json' = 'markdown'): string {
    return `${this.baseUrl}/research/${encodeURIComponent(researchId)}/export?format=${format}`;
  }

  // Real-time EventSource with Polling Fallback
  subscribeResearchProgress(
    researchId: string,
    onProgress: (event: ResearchProgressEvent) => void,
    onError: (err: any) => void
  ): () => void {
    let isCleanedUp = false;
    let eventSource: EventSource | null = null;
    let pollingInterval: any = null;

    const startPollingFallback = () => {
      if (isCleanedUp || pollingInterval) return;
      pollingInterval = setInterval(async () => {
        try {
          const statusRes = await this.getResearchStatus(researchId);
          onProgress({
            research_id: statusRes.research_id,
            status: statusRes.status,
            progress_stage: statusRes.progress_stage,
            progress_percent: statusRes.progress_percent,
            progress_message: statusRes.progress_message,
            is_terminal: statusRes.is_terminal,
            error: statusRes.error,
          });
          if (statusRes.is_terminal) {
            clearInterval(pollingInterval);
            pollingInterval = null;
          }
        } catch (pollErr) {
          onError(pollErr);
        }
      }, 2000);
    };

    try {
      const sseUrl = `${this.baseUrl}/research/${encodeURIComponent(researchId)}/events`;
      eventSource = new EventSource(sseUrl);

      eventSource.addEventListener('progress', (e) => {
        try {
          const parsed: ResearchProgressEvent = JSON.parse(e.data);
          onProgress(parsed);
          if (parsed.is_terminal && eventSource) {
            eventSource.close();
          }
        } catch (parseErr) {
          console.warn('Failed to parse SSE event data', parseErr);
        }
      });

      eventSource.onerror = (e) => {
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        // Fall back to polling upon SSE failure or disconnect
        startPollingFallback();
      };
    } catch {
      startPollingFallback();
    }

    return () => {
      isCleanedUp = true;
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
    };
  }

  // Conversations & Multi-Turn Chat
  async listConversations(limit: number = 50, offset: number = 0, workspaceId?: string): Promise<Conversation[]> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (workspaceId) params.append('workspace_id', workspaceId);
    return this.request<Conversation[]>(`/conversations?${params.toString()}`);
  }

  async createConversation(payload: ConversationCreatePayload = {}): Promise<Conversation> {
    return this.request<Conversation>('/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async getConversation(conversationId: string): Promise<ConversationWithMessages> {
    return this.request<ConversationWithMessages>(`/conversations/${encodeURIComponent(conversationId)}`);
  }

  async updateConversation(conversationId: string, title: string): Promise<Conversation> {
    return this.request<Conversation>(`/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
  }

  async deleteConversation(conversationId: string): Promise<{ deleted: boolean; conversation_id: string }> {
    return this.request(`/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    });
  }

  async getConversationMessages(conversationId: string, limit: number = 100): Promise<ConversationMessage[]> {
    const params = new URLSearchParams({ limit: String(limit) });
    return this.request<ConversationMessage[]>(
      `/conversations/${encodeURIComponent(conversationId)}/messages?${params.toString()}`
    );
  }

  async sendConversationMessage(
    conversationId: string,
    payload: ConversationMessagePayload
  ): Promise<ConversationMessage> {
    return this.request<ConversationMessage>(
      `/conversations/${encodeURIComponent(conversationId)}/messages`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );
  }

  // Knowledge & Documents
  async listDocuments(limit: number = 50, offset: number = 0, sourceType?: string): Promise<Document[]> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (sourceType) params.append('source_type', sourceType);
    return this.request<Document[]>(`/knowledge/documents?${params.toString()}`);
  }

  async getDocument(docId: string): Promise<Document> {
    return this.request<Document>(`/knowledge/documents/${encodeURIComponent(docId)}`);
  }

  async getDocumentChunks(docId: string): Promise<DocumentChunk[]> {
    return this.request<DocumentChunk[]>(`/knowledge/documents/${encodeURIComponent(docId)}/chunks`);
  }

  async deleteDocument(docId: string): Promise<{ deleted: boolean; id: string }> {
    return this.request(`/knowledge/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });
  }

  async uploadDocument(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);

    const headers = this.getHeaders();
    // Do not set Content-Type header so browser sets multipart boundary automatically
    const url = `${this.baseUrl}/knowledge/documents`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        ...headers,
      },
      body: formData,
    });

    if (!res.ok) {
      let errorMsg = `Upload failed: ${res.statusText}`;
      try {
        const errorData = await res.json();
        if (errorData?.detail) errorMsg = errorData.detail;
      } catch {}
      throw new ApiError(errorMsg, res.status);
    }

    return res.json();
  }

  async searchKnowledge(
    query: string,
    topK: number = 5,
    hybrid: boolean = false,
    documentId?: string
  ): Promise<KnowledgeSearchResult[]> {
    return this.request<KnowledgeSearchResult[]>('/knowledge/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        hybrid,
        document_id: documentId,
      }),
    });
  }

  async askKnowledge(question: string, topK: number = 5, documentId?: string): Promise<KnowledgeAnswer> {
    return this.request<KnowledgeAnswer>('/knowledge/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: topK,
        document_id: documentId,
      }),
    });
  }
}

export const apiClient = new InsightBotClient();

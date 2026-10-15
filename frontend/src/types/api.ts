export type ResearchStatus =
  | 'queued'
  | 'in_progress'
  | 'completed'
  | 'completed_unverified'
  | 'failed'
  | 'provider_unavailable'
  | 'cancelled'
  | 'interrupted';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  database: string;
  storage_mode: string;
  pgvector_enabled: boolean;
}

export interface TenantContext {
  organization_id: string;
  workspace_id?: string | null;
  api_key_id?: string | null;
  is_admin: boolean;
}

export interface Document {
  id: string;
  source_id?: string | null;
  filename: string;
  source_type: string;
  mime_type: string;
  content_hash?: string | null;
  processing_status: string;
  error_message?: string | null;
  metadata: Record<string, any>;
  organization_id?: string | null;
  workspace_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  metadata: Record<string, any>;
  organization_id?: string | null;
  workspace_id?: string | null;
  created_at: string;
}

export interface KnowledgeSearchResult {
  document_id: string;
  chunk_id: string;
  content: string;
  score: number;
  chunk_index: number;
  metadata: Record<string, any>;
}

export interface SourceAttribution {
  source_filename: string;
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  page_number?: number | null;
  section_heading?: string | null;
  snippet: string;
  score: number;
}

export interface KnowledgeAnswer {
  question: string;
  answer: string;
  sources: SourceAttribution[];
  is_grounded: boolean;
  sufficient_context: boolean;
  insufficient_evidence: boolean;
  created_at: string;
}

export interface CitationItem {
  source_type?: string;
  source_uri?: string;
  document_id?: string;
  chunk_id?: string;
  snippet?: string;
  is_verified?: boolean;
}

export interface ConflictItem {
  topic: string;
  conflicting_assertions: string[];
  severity?: string;
  resolution_note?: string;
}

export interface EvidenceItem {
  source_type?: string;
  source_uri: string;
  document_id?: string;
  chunk_id?: string;
  snippet: string;
  score?: number;
  metadata?: Record<string, any>;
}

export interface ResearchSession {
  id: string;
  organization_id: string;
  workspace_id?: string | null;
  title: string;
  query: string;
  status: ResearchStatus;
  progress_stage?: string | null;
  progress_percent: number;
  progress_message?: string | null;
  summary: string;
  sources: string[];
  internal_evidence: EvidenceItem[];
  web_evidence: EvidenceItem[];
  conflicts: ConflictItem[];
  citations: CitationItem[];
  is_grounded: boolean;
  sufficient_evidence: boolean;
  error?: string | null;
  metadata: Record<string, any>;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at: string;
}

export interface ResearchJobStatusResponse {
  research_id: string;
  organization_id: string;
  workspace_id?: string | null;
  status: ResearchStatus;
  progress_stage?: string | null;
  progress_percent: number;
  progress_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at: string;
  error?: string | null;
  is_terminal: boolean;
}

export interface ResearchProgressEvent {
  research_id: string;
  status: ResearchStatus;
  progress_stage?: string | null;
  progress_percent: number;
  progress_message?: string | null;
  is_terminal: boolean;
  error?: string | null;
}

export interface Conversation {
  id: string;
  organization_id: string;
  workspace_id?: string | null;
  title: string;
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources: SourceAttribution[];
  is_grounded: boolean;
  insufficient_evidence: boolean;
  metadata: Record<string, any>;
  created_at: string;
}

export interface ConversationWithMessages extends Conversation {
  messages: ConversationMessage[];
}

export interface ResearchCreatePayload {
  task: string;
  max_steps?: number;
  target_domains?: string[];
  sync?: boolean;
}

export interface ConversationCreatePayload {
  title?: string;
  workspace_id?: string;
  metadata?: Record<string, any>;
}

export interface ConversationMessagePayload {
  content: string;
  top_k?: number;
  similarity_threshold?: number;
  max_history_turns?: number;
}

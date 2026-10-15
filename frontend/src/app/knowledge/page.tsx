'use client';

import React, { useEffect, useState, useRef } from 'react';
import {
  Database,
  Upload,
  Search,
  Trash2,
  Layers,
  FileText,
  Clock,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  X,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { apiClient } from '@/lib/api';
import { Document, DocumentChunk, KnowledgeSearchResult } from '@/types/api';

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Upload State
  interface UploadNotification {
    type: 'uploading' | 'success' | 'validation_error' | 'error';
    message: string;
    details?: string;
    filename?: string;
  }

  const [isUploading, setIsUploading] = useState(false);
  const [uploadNotification, setUploadNotification] = useState<UploadNotification | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Chunk Inspector State
  const [inspectDoc, setInspectDoc] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  // Search Tool State
  const [searchQuery, setSearchQuery] = useState('');
  const [isHybrid, setIsHybrid] = useState(false);
  const [topK, setTopK] = useState(5);
  const [searchResults, setSearchResults] = useState<KnowledgeSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);

  const loadDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.listDocuments(100, 0);
      setDocuments(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size === 0) {
      setUploadNotification({
        type: 'validation_error',
        message: 'Empty File',
        details: `Document '${file.name}' is empty (0 bytes) and cannot be ingested.`,
        filename: file.name,
      });
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    setIsUploading(true);
    setUploadNotification({
      type: 'uploading',
      message: `Ingesting and chunking '${file.name}'...`,
      filename: file.name,
    });

    try {
      const ingested = await apiClient.uploadDocument(file);
      setDocuments((prev) => {
        const exists = prev.some((d) => d.id === ingested.id);
        if (exists) {
          return prev.map((d) => (d.id === ingested.id ? ingested : d));
        }
        return [ingested, ...prev];
      });
      setUploadNotification({
        type: 'success',
        message: `Successfully ingested '${ingested.filename}'`,
        details: `Document ID: ${ingested.id} • Type: ${ingested.source_type.toUpperCase()} • Status: ${ingested.processing_status}`,
        filename: ingested.filename,
      });
    } catch (err: any) {
      const status = err?.status;
      const isValidation = status === 400 || status === 422;
      setUploadNotification({
        type: isValidation ? 'validation_error' : 'error',
        message: isValidation ? 'Document Validation Failed' : 'Document Processing Failed',
        details: err?.message || 'Failed to process document.',
        filename: file.name,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete document '${filename}' and all its vector embeddings?`))
      return;

    try {
      await apiClient.deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      if (inspectDoc?.id === docId) {
        setInspectDoc(null);
        setChunks([]);
      }
    } catch (err: any) {
      alert(`Delete failed: ${err?.message}`);
    }
  };

  const handleInspectChunks = async (doc: Document) => {
    setInspectDoc(doc);
    setLoadingChunks(true);
    try {
      const chunkList = await apiClient.getDocumentChunks(doc.id);
      setChunks(chunkList);
    } catch (err: any) {
      alert(`Could not load chunks: ${err?.message}`);
      setChunks([]);
    } finally {
      setLoadingChunks(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim() || searching) return;

    setSearching(true);
    try {
      const results = await apiClient.searchKnowledge(searchQuery.trim(), topK, isHybrid);
      setSearchResults(results);
    } catch (err: any) {
      alert(`Search failed: ${err?.message}`);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      <Header
        title="Knowledge Base & Ingestion"
        subtitle="Manage ingested private documents, inspect deterministic chunks, and run hybrid semantic search"
        actionButton={
          <div className="flex items-center gap-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".txt,.md,.pdf,.docx,.markdown"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-accent-cyan text-background font-semibold text-xs hover:bg-accent-cyan/90 disabled:opacity-50 transition-colors shadow-sm"
            >
              <Upload className="h-3.5 w-3.5" />
              <span>{isUploading ? 'Ingesting...' : 'Ingest Document'}</span>
            </button>
          </div>
        }
      />

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* Upload Notification Banner */}
        {uploadNotification && (
          <div
            className={`rounded-xl border p-4 flex items-start justify-between gap-3 text-xs transition-all ${
              uploadNotification.type === 'uploading'
                ? 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan'
                : uploadNotification.type === 'success'
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : uploadNotification.type === 'validation_error'
                ? 'border-amber-500/30 bg-amber-500/10 text-amber-400'
                : 'border-rose-500/30 bg-rose-500/10 text-rose-400'
            }`}
          >
            <div className="flex items-start gap-3">
              {uploadNotification.type === 'uploading' && (
                <RefreshCw className="h-4 w-4 animate-spin shrink-0 mt-0.5" />
              )}
              {uploadNotification.type === 'success' && (
                <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
              )}
              {uploadNotification.type === 'validation_error' && (
                <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
              )}
              {uploadNotification.type === 'error' && (
                <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="space-y-1">
                <div className="flex items-center gap-2 font-semibold">
                  <span>{uploadNotification.message}</span>
                  {uploadNotification.type === 'validation_error' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-amber-500/20 text-amber-300 uppercase tracking-wider">
                      Validation Error
                    </span>
                  )}
                  {uploadNotification.type === 'error' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-rose-500/20 text-rose-300 uppercase tracking-wider">
                      Processing Error
                    </span>
                  )}
                  {uploadNotification.type === 'success' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-500/20 text-emerald-300 uppercase tracking-wider">
                      Ready
                    </span>
                  )}
                </div>
                {uploadNotification.details && (
                  <p className="text-slate-300 font-mono text-[11px] leading-relaxed break-all">
                    {uploadNotification.details}
                  </p>
                )}
              </div>
            </div>
            {uploadNotification.type !== 'uploading' && (
              <button
                onClick={() => setUploadNotification(null)}
                className="text-slate-400 hover:text-white p-1 rounded transition-colors shrink-0"
                title="Dismiss"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        )}

        {/* Semantic / Hybrid Search Tool Box */}
        <section className="rounded-xl border border-border bg-surface p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
              <h3 className="font-semibold text-white text-sm">Semantic & Hybrid Search Inspection</h3>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono">
              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={isHybrid}
                  onChange={(e) => setIsHybrid(e.target.checked)}
                  className="rounded border-border bg-background text-accent-cyan focus:ring-accent-cyan"
                />
                <span>Hybrid (Dense Vector + BM25 Lexical)</span>
              </label>
              <div className="flex items-center gap-1.5 text-slate-400">
                <span>Top K:</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="rounded border border-border bg-background px-2 py-0.5 text-xs text-white"
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
              </div>
            </div>
          </div>

          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Query private knowledge chunks... (e.g. database versions, compliance)"
                className="w-full rounded-xl border border-border bg-background pl-9 pr-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none"
              />
            </div>
            <button
              type="submit"
              disabled={!searchQuery.trim() || searching}
              className="px-4 py-2.5 rounded-xl bg-accent-cyan hover:bg-accent-cyan/90 disabled:opacity-50 text-background font-semibold text-xs transition-colors shrink-0"
            >
              {searching ? 'Searching...' : 'Search'}
            </button>
            {searchResults && (
              <button
                type="button"
                onClick={() => setSearchResults(null)}
                className="px-3 py-2.5 rounded-xl border border-border text-slate-400 hover:text-white text-xs"
              >
                Clear
              </button>
            )}
          </form>

          {/* Search Results Display */}
          {searchResults && (
            <div className="pt-3 border-t border-border space-y-3">
              <p className="text-xs font-mono text-slate-400">
                Found {searchResults.length} matching candidate chunks:
              </p>
              {searchResults.length === 0 ? (
                <p className="text-xs text-slate-500 italic p-3 bg-background rounded-lg border border-border">
                  No matching chunks exceeded the minimum similarity threshold.
                </p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {searchResults.map((res, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-lg border border-border bg-background text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <span className="text-accent-cyan font-semibold">doc: {res.document_id}</span>
                        <span className="text-emerald-400 font-bold">score: {res.score.toFixed(3)}</span>
                      </div>
                      <p className="text-slate-300 font-mono text-[11px] bg-surface/60 p-2 rounded border border-border/60">
                        {res.content}
                      </p>
                      <div className="text-[10px] font-mono text-slate-500">chunk: {res.chunk_id}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Documents Table */}
        <section className="rounded-xl border border-border bg-surface overflow-hidden shadow-xl">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-accent-blue" />
              <h3 className="font-semibold text-white text-sm">
                Ingested Documents ({documents.length})
              </h3>
            </div>
            <button
              onClick={loadDocuments}
              aria-label="Refresh Documents List"
              className="p-1.5 rounded-lg border border-border bg-surface hover:bg-surfaceHover text-slate-400 hover:text-white transition-colors"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-accent-cyan' : ''}`} />
            </button>
          </div>

          <div className="p-4">
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-14 bg-surfaceHover rounded-lg animate-pulse" />
                ))}
              </div>
            ) : error ? (
              <div className="p-6 text-center text-rose-300 text-xs">
                <AlertCircle className="h-6 w-6 text-rose-400 mx-auto mb-2" />
                {error}
              </div>
            ) : documents.length === 0 ? (
              <div className="py-14 text-center space-y-2 text-xs text-slate-500">
                <FileText className="h-8 w-8 text-slate-600 mx-auto" />
                <p className="font-medium text-slate-300">No documents ingested yet</p>
                <p>Click &quot;Ingest Document&quot; above to upload text, markdown, or PDF files.</p>
              </div>
            ) : (
              <div className="divide-y divide-border">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="py-3 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                  >
                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-accent-cyan shrink-0" />
                        <span className="font-semibold text-sm text-white truncate">{doc.filename}</span>
                        <span className="px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300 uppercase">
                          {doc.source_type}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-slate-500 font-mono">
                        <span>ID: {doc.id}</span>
                        <span>•</span>
                        <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                        <span>•</span>
                        <span className="text-emerald-400">{doc.processing_status}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => handleInspectChunks(doc)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surfaceHover hover:border-slate-500 text-slate-300 hover:text-white text-xs font-mono transition-colors"
                      >
                        <Layers className="h-3.5 w-3.5 text-accent-cyan" />
                        <span>Inspect Chunks</span>
                      </button>

                      <button
                        onClick={() => handleDelete(doc.id, doc.filename)}
                        title="Delete Document"
                        className="p-1.5 rounded-lg border border-border hover:border-rose-500/50 text-slate-400 hover:text-rose-400 transition-colors"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Chunk Inspection Modal / Drawer */}
        {inspectDoc && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="w-full max-w-3xl rounded-xl border border-border bg-surface flex flex-col max-h-[85vh] shadow-2xl overflow-hidden">
              <div className="p-4 border-b border-border flex items-center justify-between bg-surface">
                <div>
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <Layers className="h-4 w-4 text-accent-cyan" />
                    Chunk Inspector: {inspectDoc.filename}
                  </h3>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">ID: {inspectDoc.id}</p>
                </div>
                <button
                  onClick={() => {
                    setInspectDoc(null);
                    setChunks([]);
                  }}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white text-xs"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {loadingChunks ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="h-20 bg-surfaceHover rounded-lg animate-pulse" />
                    ))}
                  </div>
                ) : chunks.length === 0 ? (
                  <p className="text-center py-8 text-xs text-slate-500 font-mono">
                    No extracted chunks found for this document.
                  </p>
                ) : (
                  chunks.map((chk) => (
                    <div
                      key={chk.id}
                      className="p-3.5 rounded-lg border border-border bg-background space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <span className="font-semibold text-accent-cyan">
                          Chunk Index: #{chk.chunk_index}
                        </span>
                        <span>ID: {chk.id}</span>
                      </div>
                      <p className="text-slate-200 font-mono text-[11px] bg-surface/50 p-3 rounded border border-border/60 whitespace-pre-wrap">
                        {chk.content}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

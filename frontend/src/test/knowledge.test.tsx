import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import KnowledgeBasePage from '@/app/knowledge/page';
import { apiClient, ApiError } from '@/lib/api';

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => '/knowledge',
  useSearchParams: () => new URLSearchParams(),
}));

describe('KnowledgeBasePage Ingestion Reliability and UX', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders knowledge base page with initial empty document list', async () => {
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);

    render(<KnowledgeBasePage />);

    expect(screen.getByText('Knowledge Base & Ingestion')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('No documents ingested yet')).toBeInTheDocument();
    });
  });

  it('shows success banner upon successful document ingestion', async () => {
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);
    const mockDoc = {
      id: 'doc-alpha-123',
      organization_id: 'default',
      filename: 'architecture_spec.md',
      source_type: 'markdown',
      mime_type: 'text/markdown',
      content_hash: 'hash123',
      processing_status: 'completed' as const,
      metadata: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    vi.spyOn(apiClient, 'uploadDocument').mockResolvedValue(mockDoc);

    const { container } = render(<KnowledgeBasePage />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput).toBeTruthy();

    const file = new File(['# Architecture\nValid content.'], 'architecture_spec.md', {
      type: 'text/markdown',
    });
    Object.defineProperty(file, 'size', { value: 35, configurable: true });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText("Successfully ingested 'architecture_spec.md'")).toBeInTheDocument();
      expect(screen.getByText('Ready')).toBeInTheDocument();
      expect(screen.getAllByText(/doc-alpha-123/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows validation banner when uploading an empty file', async () => {
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);
    const uploadSpy = vi.spyOn(apiClient, 'uploadDocument');

    const { container } = render(<KnowledgeBasePage />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const emptyFile = new File([], 'empty.txt', { type: 'text/plain' });
    fireEvent.change(fileInput, { target: { files: [emptyFile] } });

    await waitFor(() => {
      expect(screen.getByText('Empty File')).toBeInTheDocument();
      expect(screen.getByText('Validation Error')).toBeInTheDocument();
      expect(screen.getByText(/empty \(0 bytes\)/)).toBeInTheDocument();
    });

    expect(uploadSpy).not.toHaveBeenCalled();
  });

  it('shows validation error banner when API returns HTTP 400', async () => {
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);
    vi.spyOn(apiClient, 'uploadDocument').mockRejectedValue(
      new ApiError("Document 'corrupt.pdf' contains no extractable text.", 400)
    );

    const { container } = render(<KnowledgeBasePage />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const corruptFile = new File(['%PDF-1.4...invalid'], 'corrupt.pdf', {
      type: 'application/pdf',
    });
    Object.defineProperty(corruptFile, 'size', { value: 20, configurable: true });
    fireEvent.change(fileInput, { target: { files: [corruptFile] } });

    await waitFor(() => {
      expect(screen.getByText('Document Validation Failed')).toBeInTheDocument();
      expect(screen.getByText('Validation Error')).toBeInTheDocument();
      expect(
        screen.getByText("Document 'corrupt.pdf' contains no extractable text.")
      ).toBeInTheDocument();
    });
  });

  it('shows processing error banner when API returns HTTP 500', async () => {
    vi.spyOn(apiClient, 'listDocuments').mockResolvedValue([]);
    vi.spyOn(apiClient, 'uploadDocument').mockRejectedValue(
      new ApiError('Document ingestion failed: RuntimeError', 500)
    );

    const { container } = render(<KnowledgeBasePage />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const validFile = new File(['Sample content'], 'guide.txt', {
      type: 'text/plain',
    });
    Object.defineProperty(validFile, 'size', { value: 14, configurable: true });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    await waitFor(() => {
      expect(screen.getByText('Document Processing Failed')).toBeInTheDocument();
      expect(screen.getByText('Processing Error')).toBeInTheDocument();
      expect(
        screen.getByText('Document ingestion failed: RuntimeError')
      ).toBeInTheDocument();
    });

    // Dismiss banner
    const dismissBtn = screen.getByTitle('Dismiss');
    fireEvent.click(dismissBtn);

    await waitFor(() => {
      expect(screen.queryByText('Document Processing Failed')).not.toBeInTheDocument();
    });
  });
});

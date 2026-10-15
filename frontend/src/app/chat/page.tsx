'use client';

import React, { useEffect, useState, useRef, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import {
  MessageSquare,
  Plus,
  Send,
  Trash2,
  Edit2,
  Check,
  X,
  ShieldCheck,
  AlertTriangle,
  Database,
  Layers,
  ChevronDown,
  ChevronRight,
  Bot,
  User,
  Sparkles,
} from 'lucide-react';
import { Header } from '@/components/Header';
import { apiClient } from '@/lib/api';
import { Conversation, ConversationMessage, SourceAttribution } from '@/types/api';

function ChatContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlConvId = searchParams.get('id');

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConvId, setSelectedConvId] = useState<string | null>(urlConvId || null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [inputContent, setInputContent] = useState('');
  const [loadingConvs, setLoadingConvs] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [titleEditValue, setTitleEditValue] = useState('');
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversations = React.useCallback(async (autoSelectFirst = true) => {
    setLoadingConvs(true);
    try {
      const convs = await apiClient.listConversations(50, 0);
      setConversations(convs);
      if (autoSelectFirst && convs.length > 0 && !selectedConvId) {
        setSelectedConvId(convs[0].id);
      }
    } catch (err) {
      console.warn('Failed to load conversations', err);
    } finally {
      setLoadingConvs(false);
    }
  }, [selectedConvId]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // When selected conversation changes, load messages
  useEffect(() => {
    if (!selectedConvId) {
      setMessages([]);
      return;
    }

    setLoadingMessages(true);
    apiClient
      .getConversationMessages(selectedConvId)
      .then((msgs) => {
        setMessages(msgs);
        setTimeout(scrollToBottom, 50);
      })
      .catch((err) => {
        console.warn('Failed to load messages', err);
      })
      .finally(() => setLoadingMessages(false));
  }, [selectedConvId]);

  const handleCreateConversation = async () => {
    try {
      const newConv = await apiClient.createConversation({ title: 'New Conversation' });
      setConversations([newConv, ...conversations]);
      setSelectedConvId(newConv.id);
      router.replace(`/chat?id=${newConv.id}`);
    } catch (err: any) {
      alert(`Could not create conversation: ${err?.message}`);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputContent.trim() || sendingMessage) return;

    let targetConvId = selectedConvId;
    if (!targetConvId) {
      // Auto-create conversation if none selected
      try {
        const newConv = await apiClient.createConversation({
          title: inputContent.trim().slice(0, 40),
        });
        setConversations([newConv, ...conversations]);
        targetConvId = newConv.id;
        setSelectedConvId(newConv.id);
      } catch (err: any) {
        alert(`Failed to create conversation: ${err?.message}`);
        return;
      }
    }

    const userText = inputContent.trim();
    setInputContent('');

    // Optimistically add user message to list
    const optimisticUserMsg: ConversationMessage = {
      id: `temp-${Date.now()}`,
      conversation_id: targetConvId,
      role: 'user',
      content: userText,
      sources: [],
      is_grounded: true,
      insufficient_evidence: false,
      metadata: {},
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, optimisticUserMsg]);
    setSendingMessage(true);
    setTimeout(scrollToBottom, 50);

    try {
      const assistantResponse = await apiClient.sendConversationMessage(targetConvId, {
        content: userText,
        top_k: 5,
      });
      setMessages((prev) => [...prev, assistantResponse]);
      setTimeout(scrollToBottom, 50);
    } catch (err: any) {
      // Show error turn
      const errorMsg: ConversationMessage = {
        id: `err-${Date.now()}`,
        conversation_id: targetConvId,
        role: 'assistant',
        content: `Error: ${err?.message || 'Failed to generate grounded response.'}`,
        sources: [],
        is_grounded: false,
        insufficient_evidence: true,
        metadata: { error: true },
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setSendingMessage(false);
    }
  };

  const handleStartRename = (conv: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingTitleId(conv.id);
    setTitleEditValue(conv.title);
  };

  const handleSaveRename = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!titleEditValue.trim()) return;
    try {
      await apiClient.updateConversation(convId, titleEditValue.trim());
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, title: titleEditValue.trim() } : c))
      );
      setEditingTitleId(null);
    } catch (err: any) {
      alert(`Rename failed: ${err?.message}`);
    }
  };

  const handleDeleteConversation = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this conversation?')) return;
    try {
      await apiClient.deleteConversation(convId);
      const remaining = conversations.filter((c) => c.id !== convId);
      setConversations(remaining);
      if (selectedConvId === convId) {
        setSelectedConvId(remaining[0]?.id || null);
      }
    } catch (err: any) {
      alert(`Delete failed: ${err?.message}`);
    }
  };

  const toggleSource = (msgId: string) => {
    setExpandedSources((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <Header
        title="Grounded Multi-Turn Chat"
        subtitle="Conversational assistant with strict private document citations and multi-turn context"
      />

      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar: Conversations List */}
        <aside className="w-80 border-r border-border bg-surface flex flex-col shrink-0">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <span className="font-semibold text-xs text-slate-300 font-mono uppercase tracking-wider">
              Conversations
            </span>
            <button
              onClick={handleCreateConversation}
              className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/20 text-xs font-medium transition-colors"
            >
              <Plus className="h-3.5 w-3.5" />
              New
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {loadingConvs ? (
              <div className="space-y-2 p-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-10 bg-surfaceHover rounded-lg animate-pulse" />
                ))}
              </div>
            ) : conversations.length === 0 ? (
              <div className="text-center py-10 px-4 text-xs text-slate-500">
                <MessageSquare className="h-8 w-8 text-slate-600 mx-auto mb-2" />
                No conversation threads yet. Start a new chat above.
              </div>
            ) : (
              conversations.map((conv) => {
                const isSelected = conv.id === selectedConvId;
                const isEditing = editingTitleId === conv.id;

                return (
                  <div
                    key={conv.id}
                    onClick={() => {
                      setSelectedConvId(conv.id);
                      router.replace(`/chat?id=${conv.id}`);
                    }}
                    className={`group flex items-center justify-between p-2.5 rounded-lg text-xs cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-accent-cyan/10 text-white border border-accent-cyan/20 font-medium'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-surfaceHover'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <MessageSquare
                        className={`h-3.5 w-3.5 shrink-0 ${
                          isSelected ? 'text-accent-cyan' : 'text-slate-500'
                        }`}
                      />
                      {isEditing ? (
                        <input
                          type="text"
                          value={titleEditValue}
                          onClick={(e) => e.stopPropagation()}
                          onChange={(e) => setTitleEditValue(e.target.value)}
                          className="bg-background border border-border rounded px-1.5 py-0.5 text-xs text-white focus:outline-none focus:border-accent-cyan w-full"
                          autoFocus
                        />
                      ) : (
                        <span className="truncate">{conv.title}</span>
                      )}
                    </div>

                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0 ml-2">
                      {isEditing ? (
                        <>
                          <button
                            onClick={(e) => handleSaveRename(conv.id, e)}
                            className="p-1 hover:text-emerald-400"
                            title="Save"
                          >
                            <Check className="h-3 w-3" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingTitleId(null);
                            }}
                            className="p-1 hover:text-slate-400"
                            title="Cancel"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={(e) => handleStartRename(conv, e)}
                            className="p-1 hover:text-white"
                            title="Rename"
                          >
                            <Edit2 className="h-3 w-3" />
                          </button>
                          <button
                            onClick={(e) => handleDeleteConversation(conv.id, e)}
                            className="p-1 hover:text-rose-400"
                            title="Delete"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* Right Main Area: Chat Messages & Input */}
        <section className="flex-1 flex flex-col bg-background min-w-0">
          {/* Messages Stream */}
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            {loadingMessages ? (
              <div className="space-y-4 max-w-3xl mx-auto">
                <div className="h-16 bg-surface rounded-xl animate-pulse w-3/4 ml-auto" />
                <div className="h-32 bg-surface rounded-xl animate-pulse w-4/5" />
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3">
                <div className="h-14 w-14 rounded-2xl bg-accent-purple/10 border border-accent-purple/20 flex items-center justify-center text-accent-purple">
                  <Sparkles className="h-7 w-7" />
                </div>
                <h4 className="text-lg font-bold text-white">Grounded Q&A Conversation</h4>
                <p className="text-xs text-slate-400 max-w-md">
                  Ask questions about your ingested private documents. Every answer is strictly attributed with verifiable document citations.
                </p>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto space-y-6">
                {messages.map((msg) => {
                  const isUser = msg.role === 'user';
                  const showSources = expandedSources[msg.id];

                  return (
                    <div
                      key={msg.id}
                      className={`flex gap-3.5 ${isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      {!isUser && (
                        <div className="h-8 w-8 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center text-accent-cyan shrink-0 mt-0.5">
                          <Bot className="h-4 w-4" />
                        </div>
                      )}

                      <div className={`space-y-2 max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}>
                        <div
                          className={`rounded-2xl p-4 text-sm leading-relaxed ${
                            isUser
                              ? 'bg-accent-cyan text-background font-medium rounded-br-sm shadow-md shadow-accent-cyan/10'
                              : 'bg-surface border border-border text-slate-200 rounded-bl-sm shadow-xl'
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{msg.content}</p>

                          {/* Insufficient Evidence Warning Banner */}
                          {!isUser && msg.insufficient_evidence && (
                            <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-amber-300">
                              <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                              <span>Notice: Insufficient private evidence to answer with high confidence.</span>
                            </div>
                          )}
                        </div>

                        {/* Citations & Attribution Dropdown */}
                        {!isUser && msg.sources && msg.sources.length > 0 && (
                          <div className="space-y-1.5">
                            <button
                              onClick={() => toggleSource(msg.id)}
                              className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 hover:text-accent-cyan transition-colors"
                            >
                              {showSources ? (
                                <ChevronDown className="h-3 w-3" />
                              ) : (
                                <ChevronRight className="h-3 w-3" />
                              )}
                              <Layers className="h-3 w-3 text-accent-cyan" />
                              <span>{msg.sources.length} Verified Sources Attributed</span>
                            </button>

                            {showSources && (
                              <div className="space-y-2 pt-1">
                                {msg.sources.map((src: SourceAttribution, idx: number) => (
                                  <div
                                    key={idx}
                                    className="p-3 rounded-lg border border-border bg-surface text-xs space-y-1.5"
                                  >
                                    <div className="flex items-center justify-between text-slate-400 font-mono text-[10px]">
                                      <span className="text-accent-cyan font-semibold flex items-center gap-1">
                                        <Database className="h-2.5 w-2.5" />
                                        {src.source_filename}
                                      </span>
                                      <span>score: {src.score?.toFixed(3)}</span>
                                    </div>
                                    <div className="text-[10px] font-mono text-slate-500">
                                      doc: {src.document_id} • chunk: {src.chunk_id}
                                    </div>
                                    <p className="text-slate-300 font-mono text-[11px] bg-background/50 p-2 rounded border border-border/60">
                                      &ldquo;{src.snippet}&rdquo;
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {isUser && (
                        <div className="h-8 w-8 rounded-lg bg-surfaceHover border border-border flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                          <User className="h-4 w-4" />
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Assistant Typing Indicator */}
                {sendingMessage && (
                  <div className="flex gap-3.5 justify-start">
                    <div className="h-8 w-8 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center text-accent-cyan shrink-0 mt-0.5">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="rounded-2xl rounded-bl-sm p-4 bg-surface border border-border text-slate-400 text-xs flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-accent-cyan animate-ping" />
                      <span>Retrieving evidence and synthesizing grounded response...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Message Input Composer */}
          <div className="p-4 border-t border-border bg-surface/50 backdrop-blur-sm">
            <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto flex gap-2">
              <input
                type="text"
                value={inputContent}
                onChange={(e) => setInputContent(e.target.value)}
                disabled={sendingMessage}
                placeholder="Ask a question grounded in your private knowledge base..."
                className="flex-1 rounded-xl border border-border bg-background px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-accent-cyan focus:outline-none focus:ring-1 focus:ring-accent-cyan disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!inputContent.trim() || sendingMessage}
                className="flex items-center justify-center px-4 py-3 rounded-xl bg-accent-cyan hover:bg-accent-cyan/90 disabled:opacity-50 text-background font-semibold transition-all shadow-md shadow-accent-cyan/10"
                title="Send Message"
              >
                <Send className="h-4 w-4" />
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <div className="flex-1 flex items-center justify-center p-12 text-xs font-mono text-slate-400">
          Loading conversation...
        </div>
      }
    >
      <ChatContent />
    </Suspense>
  );
}

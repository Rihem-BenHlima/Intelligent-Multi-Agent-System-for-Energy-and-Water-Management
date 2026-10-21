import React, { useEffect, useRef, useState } from 'react';
import {
  Bot,
  X,
  Maximize2,
  Send,
  History,
  Sparkles,
  Plus,
  ChevronDown,
  Trash2,
} from 'lucide-react';
import { ChatSession } from '../types';
import { deriveTitleFromMessage } from '../utils/chatTitle';
import { API_BASE } from '../config';

interface ChatbotWidgetProps {
  sessions: ChatSession[];
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>;
  activeSessionId: string;
  setActiveSessionId: (id: string) => void;
  onOpenFullPage: () => void;
  onLeave: () => void;
}

export const ChatbotWidget: React.FC<ChatbotWidgetProps> = ({
  sessions,
  setSessions,
  activeSessionId,
  setActiveSessionId,
  onOpenFullPage,
  onLeave,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showHistoryDropdown, setShowHistoryDropdown] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    if (isOpen) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession.messages, isTyping, isOpen]);

  const handleSendMessage = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim()) return;

    const userText = inputMessage.trim();
    const timestamp = new Date().toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });

    const userMsg = {
      id: `msg-${Date.now()}`,
      sender: 'user' as const,
      text: userText,
      timestamp,
    };

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSession.id) {
          const isFirstRealMessage = s.messages.length === 1;
          return {
            ...s,
            ...(isFirstRealMessage ? { title: deriveTitleFromMessage(userText) } : {}),
            lastMessage: userText,
            messages: [...s.messages, userMsg],
          };
        }
        return s;
      })
    );

    setInputMessage('');
    setIsTyping(true);

    fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: activeSession.id, message: userText }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const aiReply: string = data?.response || 'No response received.';
        const aiMsg = {
          id: `msg-${Date.now() + 1}`,
          sender: 'ai' as const,
          text: aiReply,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        };

        setSessions((prev) =>
          prev.map((s) => {
            if (s.id === activeSession.id) {
              return {
                ...s,
                lastMessage: aiReply,
                messages: [...s.messages, aiMsg],
              };
            }
            return s;
          })
        );
      })
      .catch(() => {
        const aiReply = "Sorry, I couldn't reach the assistant just now. Please try again.";
        const aiMsg = {
          id: `msg-${Date.now() + 1}`,
          sender: 'ai' as const,
          text: aiReply,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        };

        setSessions((prev) =>
          prev.map((s) => {
            if (s.id === activeSession.id) {
              return {
                ...s,
                lastMessage: aiReply,
                messages: [...s.messages, aiMsg],
              };
            }
            return s;
          })
        );
      })
      .finally(() => {
        setIsTyping(false);
      });
  };

  const buildGreetingSession = (title: string): ChatSession => {
    const newId = `session-${Date.now()}`;
    const greeting = 'Hello! I’m your AI assistant — ask me about consumption, anomalies, or forecasts using your real data.';
    return {
      id: newId,
      title,
      date: 'Just now',
      lastMessage: greeting,
      messages: [
        {
          id: `msg-${Date.now()}`,
          sender: 'ai',
          text: greeting,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        },
      ],
    };
  };

  const handleCreateNewSession = () => {
    const newSession = buildGreetingSession('New Energy Query');
    setSessions((prev) => [newSession, ...prev.filter((s) => s.messages.length > 1)]);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const remaining = sessions.filter((s) => s.id !== id);
    if (remaining.length === 0) {
      const fresh = buildGreetingSession('New Energy Query');
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
      return;
    }
    setSessions(remaining);
    if (id === activeSessionId) {
      setActiveSessionId(remaining[0].id);
    }
  };

  return (
    <>
      {/* Small Circle Floating Action Button - hidden when AI Assistant popover is open */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-40 w-12 h-12 rounded-full bg-[#1f3bb3] hover:bg-[#183096] text-white flex items-center justify-center shadow-xl transition-all hover:scale-105 active:scale-95 focus:outline-none cursor-pointer group"
          title="AI Assistant Chat"
          id="floating-chatbot-fab"
        >
          <Bot className="w-6 h-6 transition-transform group-hover:rotate-12" />
          <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-emerald-500 border-2 border-white rounded-full animate-pulse" />
        </button>
      )}

      {/* Floating Chatbot Popover Box */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-[360px] sm:w-[400px] h-[520px] bg-white rounded-2xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden animate-fade-in">
          {/* Header */}
          <div className="bg-[#1f3bb3] text-white p-3.5 flex items-center justify-between shrink-0 relative">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                <Bot className="w-5 h-5 text-indigo-100" />
              </div>
              <div>
                <h4 className="text-xs font-bold leading-tight">AI Assistant</h4>
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              {/* History Emoji Button with Dropdown */}
              <div className="relative">
                <button
                  onClick={() => setShowHistoryDropdown(!showHistoryDropdown)}
                  className="p-1.5 rounded-md hover:bg-white/10 text-indigo-200 hover:text-white transition-colors cursor-pointer"
                  title="Chat History"
                >
                  <History className="w-4 h-4" />
                </button>

                {showHistoryDropdown && (
                  <div className="absolute right-0 top-full mt-2 w-56 bg-white text-slate-800 rounded-xl shadow-xl border border-slate-200 py-2 z-50 animate-fade-in">
                    <div className="px-3 py-1 text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between border-b border-slate-100 pb-1.5 mb-1">
                      <span>Conversations</span>
                      <button
                        onClick={() => {
                          handleCreateNewSession();
                          setShowHistoryDropdown(false);
                        }}
                        className="text-indigo-600 hover:text-indigo-800 text-[11px] font-semibold cursor-pointer"
                      >
                        + New
                      </button>
                    </div>
                    <div className="max-h-40 overflow-y-auto">
                      {/* Untouched, greeting-only sessions are not history. */}
                      {sessions.filter((s) => s.messages.length > 1).length === 0 && (
                        <div className="px-3 py-1.5 text-[11px] text-slate-400 italic">No past conversations yet</div>
                      )}
                      {sessions.filter((s) => s.messages.length > 1).map((s) => (
                        <div
                          key={s.id}
                          className={`w-full flex items-center justify-between hover:bg-slate-50 group ${
                            s.id === activeSession.id ? 'bg-indigo-50' : ''
                          }`}
                        >
                          <button
                            onClick={() => {
                              setActiveSessionId(s.id);
                              setShowHistoryDropdown(false);
                            }}
                            className={`flex-1 min-w-0 text-left px-3 py-1.5 text-xs truncate flex items-center justify-between cursor-pointer ${
                              s.id === activeSession.id
                                ? 'text-indigo-700 font-bold'
                                : 'text-slate-700'
                            }`}
                          >
                            <span className="truncate">{s.title}</span>
                            <span className="text-[9px] text-slate-400 ml-1 shrink-0">
                              {s.date}
                            </span>
                          </button>
                          <button
                            onClick={(e) => handleDeleteSession(s.id, e)}
                            className="p-1.5 mr-1 text-slate-300 hover:text-red-500 rounded shrink-0 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                            title="Delete conversation"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <button
                onClick={() => {
                  setIsOpen(false);
                  onOpenFullPage();
                }}
                className="p-1.5 rounded-md hover:bg-white/10 text-indigo-200 hover:text-white transition-colors cursor-pointer"
                title="Expand to Full Dedicated Chat Page"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setIsOpen(false);
                  onLeave();
                }}
                className="p-1.5 rounded-md hover:bg-white/10 text-indigo-200 hover:text-white transition-colors cursor-pointer"
                title="Close chat"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Message History Feed — min-h-0 required alongside flex-1 for
              overflow-y-auto to actually engage (same fix as ChatbotPage);
              without it this div grows to fit all messages instead of
              scrolling within the fixed h-[520px] container. */}
          <div className="flex-1 min-h-0 p-3.5 overflow-y-auto space-y-3 bg-slate-50/30">
            {activeSession.messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${
                  msg.sender === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-xs leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-[#1f3bb3] text-white rounded-br-xs'
                      : 'bg-white text-slate-800 border border-slate-200/80 shadow-2xs rounded-bl-xs'
                  }`}
                >
                  {msg.text}
                </div>
                <span className="text-[9px] text-slate-400 mt-1 px-1">
                  {msg.timestamp}
                </span>
              </div>
            ))}

            {isTyping && (
              <div className="flex items-center gap-1.5 text-slate-400 text-xs p-2">
                <Sparkles className="w-3.5 h-3.5 text-indigo-500 animate-spin" />
                <span className="text-[11px] italic">AI is analyzing meter telemetry...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Footer */}
          <form
            onSubmit={handleSendMessage}
            className="p-3 bg-white border-t border-slate-200 flex items-center gap-2 shrink-0"
          >
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about consumption, tariffs, or trends..."
              className="flex-1 bg-slate-100/80 text-xs text-slate-800 placeholder-slate-400 py-2 pl-3 pr-3 rounded-xl border border-slate-200 focus:outline-none focus:border-indigo-500 transition-colors"
            />
            <button
              type="submit"
              disabled={!inputMessage.trim() || isTyping}
              className="p-2 bg-[#1f3bb3] hover:bg-[#183096] disabled:opacity-40 text-white rounded-xl transition-colors cursor-pointer shrink-0"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>
      )}
    </>
  );
};

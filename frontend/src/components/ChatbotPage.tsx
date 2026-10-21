import React, { useEffect, useRef, useState } from 'react';
import {
  Bot,
  Send,
  Plus,
  Sparkles,
  History,
  Trash2,
} from 'lucide-react';
import { ChatSession } from '../types';
import { deriveTitleFromMessage } from '../utils/chatTitle';
import { API_BASE } from '../config';

interface ChatbotPageProps {
  sessions: ChatSession[];
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>;
  activeSessionId: string;
  setActiveSessionId: (id: string) => void;
  onLeave: () => void;
}

export const ChatbotPage: React.FC<ChatbotPageProps> = ({
  sessions,
  setSessions,
  activeSessionId,
  setActiveSessionId,
  onLeave,
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showHistoryDropdown, setShowHistoryDropdown] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => { onLeave(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession.messages, isTyping]);

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

  const handleCreateNewChat = () => {
    const newSession = buildGreetingSession('New Conversation');
    setSessions((prev) => [newSession, ...prev.filter((s) => s.messages.length > 1)]);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const remaining = sessions.filter((s) => s.id !== id);
    if (remaining.length === 0) {
      const fresh = buildGreetingSession('New Conversation');
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
    <div className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden flex flex-col h-[750px] max-w-full">
      {/* Main Chat Canvas Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-white">
        {/* Top Chat Banner */}
        <div className="p-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-slate-50/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-[#1f3bb3] text-white">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-800">
                  {activeSession.title}
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {activeSession.date} • AI Energy Assistant
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCreateNewChat}
              className="px-3 py-1.5 text-xs font-semibold text-[#1f3bb3] bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
              title="Start new conversation"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New Chat</span>
            </button>

            {/* History Dropdown Button */}
            <div className="relative">
              <button
                onClick={() => setShowHistoryDropdown(!showHistoryDropdown)}
                className="px-2.5 py-1.5 text-xs font-semibold text-slate-600 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
                title="Chat History"
              >
                <History className="w-3.5 h-3.5 text-indigo-600" />
                <span>History</span>
              </button>

              {showHistoryDropdown && (
                <div className="absolute right-0 top-full mt-2 w-64 bg-white text-slate-800 rounded-xl shadow-xl border border-slate-200 py-2 z-50 animate-fade-in">
                  <div className="px-3 py-1 text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between border-b border-slate-100 pb-1.5 mb-1">
                    <span>Conversations</span>
                    <button
                      onClick={() => {
                        handleCreateNewChat();
                        setShowHistoryDropdown(false);
                      }}
                      className="text-[#1f3bb3] hover:text-indigo-800 text-[11px] font-semibold cursor-pointer"
                    >
                      + New
                    </button>
                  </div>
                  <div className="max-h-56 overflow-y-auto">
                    {/* Only conversations the user actually wrote in — a
                        freshly-opened, untouched session is not history. */}
                    {sessions.filter((s) => s.messages.length > 1).length === 0 && (
                      <div className="px-3 py-2 text-[11px] text-slate-400 italic">No past conversations yet</div>
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
                          className={`flex-1 min-w-0 text-left px-3 py-2 text-xs truncate flex items-center justify-between cursor-pointer ${
                            s.id === activeSession.id
                              ? 'text-indigo-700 font-bold'
                              : 'text-slate-700'
                          }`}
                        >
                          <span className="truncate">{s.title}</span>
                          <span className="text-[9px] text-slate-400 ml-2 shrink-0">
                            {s.date}
                          </span>
                        </button>
                        <button
                          onClick={(e) => handleDeleteSession(s.id, e)}
                          className="p-1.5 mr-1 text-slate-300 hover:text-red-500 rounded shrink-0 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                          title="Delete conversation"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Message Feed Canvas — min-h-0 is required alongside flex-1 for
            overflow-y-auto to actually engage here; without it, this div
            (and its flex-1 parent above) grow to fit ALL messages instead
            of respecting the outer h-[750px] container, so nothing ever
            scrolls and content just gets clipped by that container's
            overflow-hidden once the conversation gets long. */}
        <div className="flex-1 min-h-0 p-4 lg:p-6 overflow-y-auto space-y-4 bg-slate-50/20">
          {activeSession.messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.sender === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-semibold text-slate-400">
                  {msg.sender === 'user' ? 'You' : 'AI Assistant'}
                </span>
                <span className="text-[10px] text-slate-300">•</span>
                <span className="text-[10px] text-slate-400">
                  {msg.timestamp}
                </span>
              </div>

              <div
                className={`group relative max-w-[85%] sm:max-w-[75%] rounded-2xl p-4 text-xs leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-[#1f3bb3] text-white rounded-tr-xs shadow-xs'
                    : 'bg-white text-slate-800 border border-slate-200 shadow-2xs rounded-tl-xs'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.text}</p>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex items-center gap-2 text-slate-500 text-xs p-3 bg-white rounded-xl border border-slate-200 max-w-xs animate-pulse">
              <Sparkles className="w-4 h-4 text-indigo-600 animate-spin" />
              <span>Analyzing query...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input Footer */}
        <form
          onSubmit={handleSendMessage}
          className="p-4 bg-white border-t border-slate-200 flex items-center gap-3"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask AI about consumption blocks, cost reduction, or anomalies..."
              className="w-full text-xs text-slate-800 placeholder-slate-400 py-3 pl-4 pr-12 rounded-xl bg-slate-50 border border-slate-200 focus:outline-none focus:border-indigo-500 transition-colors"
            />
            <button
              type="submit"
              disabled={!inputMessage.trim() || isTyping}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-[#1f3bb3] hover:bg-[#183096] disabled:opacity-40 text-white rounded-lg transition-colors cursor-pointer"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

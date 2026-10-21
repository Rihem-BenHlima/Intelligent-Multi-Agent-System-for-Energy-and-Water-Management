import { ChatSession } from '../types';

// Initial conversation greeting state.
export const INITIAL_CHAT_SESSIONS: ChatSession[] = [
  {
    id: 'session-1',
    title: 'New Conversation',
    date: 'Today',
    lastMessage: 'Hello! I’m your AI assistant — ask me about consumption, anomalies, or forecasts using your real data.',
    messages: [
      {
        id: 'msg-1',
        sender: 'ai',
        text: 'Hello! I’m your AI assistant — ask me about consumption, anomalies, or forecasts using your real data.',
        timestamp: 'Just now',
      },
    ],
  },
];

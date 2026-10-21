export interface MetricItem {
  id: string;
  title: string;
  value: string;
  change: string;
  isPositive: boolean;
  hideArrow?: boolean;
  subtext?: string;
}

export interface ChartDataPoint {
  day: string;
  thisWeek: number;
  lastWeek: number;
}

export interface StackedBarDataPoint {
  month: string;
  thisWeek: number;
  lastWeek: number;
}

export interface NavSubItem {
  id: string;
  label: string;
  badge?: string;
  href?: string;
}

export interface NavGroupItem {
  id: string;
  label: string;
  icon: string;
  subItems?: NavSubItem[];
  href?: string;
}

export interface NavSection {
  title?: string;
  items: NavGroupItem[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
  category?: 'peak' | 'saving' | 'general' | 'tariff';
}

export interface ChatSession {
  id: string;
  title: string;
  date: string;
  lastMessage: string;
  messages: ChatMessage[];
}

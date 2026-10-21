import { NavSection } from '../types';

// Navigation structure for sidebar.
export const NAVIGATION_SECTIONS: NavSection[] = [
  {
    items: [
      { id: 'dashboard', label: 'Dashboard', icon: 'LayoutDashboard' },
      { id: 'demand-forecast', label: 'Demand Forecast', icon: 'TrendingUp' },
      { id: 'anomaly-detection', label: 'Anomaly Detection', icon: 'AlertTriangle' },
      { id: 'reports', label: 'Reports', icon: 'FileText' },
      { id: 'chatbot', label: 'AI Assistant', icon: 'Bot' },
    ],
  },
  {
    items: [
      {
        id: 'account-settings',
        label: 'Settings',
        icon: 'Settings',
      },
    ],
  },
];

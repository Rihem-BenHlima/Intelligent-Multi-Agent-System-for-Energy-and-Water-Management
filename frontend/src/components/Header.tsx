import React, { useState } from 'react';
import {
  Bell,
  Menu,
} from 'lucide-react';
import { useNotifications } from '../hooks/useNotifications';
import { Avatar } from './Avatar';

interface HeaderProps {
  onMobileMenuToggle: () => void;
  activeNavItem?: string;
  utilityType?: 'energy' | 'water';
  userName?: string;
  userEmail?: string;
  userPhoto?: string | null;
  onOpenSettings?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  onMobileMenuToggle,
  activeNavItem = 'dashboard',
  utilityType = 'energy',
  userName = 'User',
  userEmail = 'user@example.com',
  userPhoto = null,
  onOpenSettings,
}) => {
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const notifications = useNotifications();

  const isEnergy = utilityType === 'energy';

  let title = 'Dashboard';
  let subtitle = 'Your performance summary this week';

  if (activeNavItem === 'demand-forecast') {
    title = isEnergy ? 'Energy Demand Forecast' : 'Water Demand Forecast';
    subtitle = `Predictive AI modeling comparing ${isEnergy ? 'power' : 'water'} demand forecast`;
  } else if (activeNavItem === 'anomaly-detection') {
    title = 'Anomaly Detection';
    subtitle = 'Automatic detection of unusual consumption patterns';
  } else if (activeNavItem === 'reports') {
    title = 'Reports';
    subtitle = 'Generate custom AI energy reports and audit history';
  } else if (activeNavItem === 'chatbot') {
    title = 'AI Assistant';
    subtitle = 'Interactive assistant for energy management & analysis';
  } else if (activeNavItem === 'account-settings') {
    title = 'Settings';
    subtitle = 'Manage your account and preferences';
  }

  return (
    <header className="bg-white border-b border-slate-200 px-4 lg:px-8 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 sticky top-0 z-30 shadow-xs">
      {/* Left Greeting */}
      <div className="flex items-center gap-3">
        <button
          onClick={onMobileMenuToggle}
          className="p-1.5 text-slate-600 hover:text-slate-900 lg:hidden focus:outline-none"
          id="mobile-menu-hamburger-btn"
        >
          <Menu className="w-6 h-6" />
        </button>

        <div>
          <h1 className="text-xl md:text-2xl font-bold text-slate-800 tracking-tight">
            {title}
          </h1>
          <p className="text-xs text-slate-500 font-normal mt-0.5">
            {subtitle}
          </p>
        </div>
      </div>

      {/* Right Tools & Profile */}
      <div className="flex items-center gap-2 md:gap-3 self-end md:self-auto">
        {/* Notifications Bell */}
        <div className="relative">
          <button
            onClick={() => setNotificationsOpen(!notificationsOpen)}
            className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-full transition-colors relative"
            title="Notifications"
            id="header-notifications-bell-btn"
          >
            <Bell className="w-4 h-4" />
            {notifications.length > 0 && (
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full ring-2 ring-white" />
            )}
          </button>
          {notificationsOpen && (
            <div className="absolute right-0 top-10 w-72 bg-white border border-slate-200 shadow-xl rounded-lg p-3 z-50">
              <h4 className="text-xs font-bold text-slate-800 border-b pb-2 mb-2">Notifications</h4>
              <div className="space-y-2 text-xs text-slate-600 max-h-80 overflow-y-auto">
                {notifications.length === 0 ? (
                  <p className="p-1.5 text-slate-400 text-center py-4">No notifications right now</p>
                ) : (
                  notifications.map((n) => (
                    <p key={n.id} className="p-1.5 hover:bg-slate-50 rounded leading-relaxed">
                      {n.icon} {n.text}
                    </p>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* User Profile Avatar */}
        <div className="relative ml-1">
          <button
            onClick={() => setProfileOpen(!profileOpen)}
            className="flex items-center focus:outline-none rounded-full"
            id="header-user-avatar-btn"
            title={userName}
          >
            <Avatar name={userName} photo={userPhoto} size={32} />
          </button>
          {profileOpen && (
            <div className="absolute right-0 top-10 w-52 bg-white border border-slate-200 shadow-xl rounded-lg py-1 z-50 text-xs text-slate-700">
              <div className="px-4 py-2.5 border-b border-slate-100 flex items-center gap-2.5">
                <Avatar name={userName} photo={userPhoto} size={32} />
                <div className="min-w-0">
                  <p className="font-semibold text-slate-800 truncate">{userName}</p>
                  <p className="text-[11px] text-slate-500 truncate">{userEmail}</p>
                </div>
              </div>
              <button
                onClick={() => { setProfileOpen(false); onOpenSettings?.(); }}
                className="w-full text-left px-4 py-2 hover:bg-slate-50"
              >
                Edit profile
              </button>
              <button className="w-full text-left px-4 py-2 hover:bg-slate-50 text-red-600 border-t border-slate-100">Log Out</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

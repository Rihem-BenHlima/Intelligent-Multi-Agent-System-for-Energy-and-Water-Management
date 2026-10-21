import React, { useState } from 'react';
import {
  LayoutDashboard,
  Grid,
  AlignRight,
  Layers,
  Box,
  FileText,
  Code,
  BarChart2,
  Table,
  AlertCircle,
  Bell,
  Smile,
  MapPin,
  User,
  AlertTriangle,
  Folder,
  ShoppingCart,
  ChevronDown,
  Menu,
  X,
  Star,
  Settings,
  TrendingUp,
  Bot,
} from 'lucide-react';
import { NAVIGATION_SECTIONS } from '../data/mockData';

interface SidebarProps {
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  activeItem: string;
  setActiveItem: (id: string) => void;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
  onOpenSettings?: () => void;
  sidebarColor?: string;
}

const ICON_MAP: Record<string, React.ElementType> = {
  LayoutDashboard,
  Bot,
  Grid,
  AlignRight,
  Layers,
  Box,
  FileText,
  Code,
  BarChart2,
  Table,
  AlertCircle,
  Bell,
  Smile,
  MapPin,
  User,
  AlertTriangle,
  Folder,
  ShoppingCart,
  Settings,
  TrendingUp,
};

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  setCollapsed,
  activeItem,
  setActiveItem,
  mobileOpen,
  setMobileOpen,
  onOpenSettings,
  sidebarColor = '#181b23',
}) => {
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    dashboard: true,
  });

  const toggleGroup = (id: string) => {
    setOpenGroups((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const renderSection = (
    section: (typeof NAVIGATION_SECTIONS)[0],
    idx: number
  ) => (
    <div key={idx} className="space-y-1">
      {section.title && !collapsed && (
        <div className="px-3 text-[10px] font-bold text-[#626d7f] tracking-wider uppercase mb-2">
          {section.title}
        </div>
      )}

      {section.items.map((item) => {
        const IconComponent = ICON_MAP[item.icon] || Grid;
        const isActive = activeItem === item.id;
        const hasSub = item.subItems && item.subItems.length > 0;
        const isGroupOpen = !!openGroups[item.id];

        return (
          <div key={item.id} className="space-y-1">
            <button
              onClick={() => {
                if (hasSub && !collapsed) {
                  toggleGroup(item.id);
                } else {
                  setActiveItem(item.id);
                  if ((item.id === 'settings' || item.id === 'account-settings') && onOpenSettings) {
                    onOpenSettings();
                  }
                }
              }}
              title={collapsed ? item.label : undefined}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-md text-sm font-medium transition-all ${
                isActive
                  ? 'bg-[#1b2545] text-white border-l-4 border-[#1f3bb3]'
                  : 'text-white hover:bg-[#202534] hover:text-white'
              }`}
              id={`nav-item-${item.id}`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <IconComponent
                  className={`w-4 h-4 shrink-0 ${
                    isActive ? 'text-[#4d70ff]' : 'text-slate-300'
                  }`}
                />
                {!collapsed && (
                  <span className="truncate text-[13px] text-white font-medium">{item.label}</span>
                )}
              </div>

              {!collapsed && hasSub && (
                <ChevronDown
                  className={`w-3.5 h-3.5 shrink-0 transition-transform text-[#626d7f] ${
                    isGroupOpen ? 'rotate-180' : ''
                  }`}
                />
              )}
            </button>

            {/* Submenu Items */}
            {!collapsed && hasSub && isGroupOpen && (
              <div className="pl-9 pr-2 py-1 space-y-1 border-l border-[#262c3e] ml-5">
                {item.subItems!.map((sub) => (
                  <button
                    key={sub.id}
                    onClick={() => setActiveItem(sub.id)}
                    className={`w-full text-left py-1.5 px-2 rounded text-xs transition-colors ${
                      activeItem === sub.id
                        ? 'text-white font-semibold'
                        : 'text-[#7d8a9e] hover:text-white'
                    }`}
                    id={`subnav-item-${sub.id}`}
                  >
                    {sub.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar Container */}
      <aside
        style={{ backgroundColor: sidebarColor }}
        className={`fixed top-0 left-0 bottom-0 z-50 text-[#919dae] flex flex-col transition-all duration-300 select-none shadow-xl border-r border-[#222736] ${
          collapsed ? 'w-20' : 'w-64'
        } ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Top Header / Logo Bar */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-[#222736] shrink-0">
          <div className="flex items-center gap-3 overflow-hidden">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-[#222736] transition-colors focus:outline-none"
              title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
              id="sidebar-toggle-btn"
            >
              <Menu className="w-5 h-5" />
            </button>

            {!collapsed && (
              <div className="flex items-center gap-2 whitespace-nowrap">
                <span className="font-extrabold text-lg tracking-tight">
                  <span className="text-[#1f3bb3] font-extrabold">SFM</span>
                  <span className="text-white font-extrabold">Connect</span>
                </span>
              </div>
            )}
          </div>

          <button
            onClick={() => setMobileOpen(false)}
            className="lg:hidden p-1 text-slate-400 hover:text-white"
            id="mobile-close-sidebar-btn"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation Items List */}
        <div className="flex-1 flex flex-col justify-between overflow-y-auto py-4 px-3 scrollbar-thin scrollbar-thumb-slate-700">
          {/* Top Section */}
          <div className="space-y-6">
            {NAVIGATION_SECTIONS.slice(0, 1).map((section, idx) =>
              renderSection(section, idx)
            )}
          </div>

          {/* Bottom Section (User Pages & Settings) */}
          <div className="space-y-6 mt-auto pt-4 border-t border-[#222736]/80">
            {NAVIGATION_SECTIONS.slice(1).map((section, idx) =>
              renderSection(section, idx)
            )}
          </div>
        </div>
      </aside>
    </>
  );
};

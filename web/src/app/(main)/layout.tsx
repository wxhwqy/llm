"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Users,
  MessageCircle,
  BookOpen,
  User,
  Sparkles,
  Menu,
  Sun,
  Moon,
  LogIn,
  LogOut,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useState, useEffect } from "react";
import { useCurrentUser, useLogout } from "@/hooks/use-queries";
import { useAuthStore } from "@/stores/auth-store";

const navItems = [
  { href: "/characters", label: "角色卡", icon: Users },
  { href: "/chat", label: "聊天", icon: MessageCircle },
  { href: "/worldbooks", label: "世界书", icon: BookOpen },
  { href: "/profile", label: "我的", icon: User },
];

function NavLink({
  href,
  label,
  icon: Icon,
  active,
  onClick,
}: {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  active: boolean;
  onClick?: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        active
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
      )}
    >
      <Icon className="h-4 w-4" />
      {label}
    </Link>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-8 w-8" />;

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-8 w-8"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      {theme === "dark" ? (
        <Sun className="h-4 w-4" />
      ) : (
        <Moon className="h-4 w-4" />
      )}
    </Button>
  );
}

function UserMenu() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const logout = useLogout();

  if (!user) {
    return (
      <Link href="/login">
        <Button variant="ghost" size="sm" className="gap-1.5 text-xs">
          <LogIn className="h-3.5 w-3.5" />
          登录
        </Button>
      </Link>
    );
  }

  const initial = user.username.charAt(0).toUpperCase();
  const roleLabel = user.role === "admin" ? "管理员" : "用户";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-1.5 outline-none">
          <span className="text-xs text-muted-foreground hidden sm:inline">
            {roleLabel}
          </span>
          <div className="h-8 w-8 rounded-full bg-violet-600 flex items-center justify-center text-xs font-medium text-white">
            {initial}
          </div>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <div className="px-2 py-1.5">
          <p className="text-sm font-medium">{user.username}</p>
          <p className="text-xs text-muted-foreground">{user.email}</p>
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => router.push("/profile")}>
          <User className="h-4 w-4 mr-2" />
          个人资料
        </DropdownMenuItem>
        {user.role === "admin" && (
          <DropdownMenuItem onClick={() => router.push("/admin/users")}>
            <Shield className="h-4 w-4 mr-2" />
            用户管理
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => {
            logout.mutate(undefined, {
              onSuccess: () => router.push("/characters"),
            });
          }}
          className="text-destructive focus:text-destructive"
        >
          <LogOut className="h-4 w-4 mr-2" />
          退出登录
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Fetch current user on layout mount (silent — no redirect if unauthenticated)
  useCurrentUser();

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      {/* Desktop Header */}
      <header className="shrink-0 z-50 border-b bg-background/80 backdrop-blur-sm">
        <div className="flex h-14 items-center gap-4 px-4 lg:px-6">
          {/* Mobile menu */}
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-4">
              <SheetTitle className="sr-only">导航菜单</SheetTitle>
              <div className="flex items-center gap-2 mb-6 mt-2">
                <Sparkles className="h-5 w-5 text-violet-500" />
                <span className="text-lg font-bold">AI Chat</span>
              </div>
              <nav className="flex flex-col gap-1">
                {navItems.map((item) => (
                  <NavLink
                    key={item.href}
                    {...item}
                    active={pathname.startsWith(item.href)}
                    onClick={() => setMobileOpen(false)}
                  />
                ))}
              </nav>
            </SheetContent>
          </Sheet>

          {/* Logo */}
          <Link href="/characters" className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-violet-500" />
            <span className="text-lg font-bold hidden sm:inline">AI Chat</span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1 ml-6">
            {navItems.map((item) => (
              <NavLink
                key={item.href}
                {...item}
                active={pathname.startsWith(item.href)}
              />
            ))}
          </nav>

          {/* Right side */}
          <div className="ml-auto flex items-center gap-1.5">
            <ThemeToggle />
            <UserMenu />
          </div>
        </div>
      </header>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur-sm">
        <div className="flex items-center justify-around h-14">
          {navItems.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex flex-col items-center gap-0.5 px-3 py-1 text-xs transition-colors",
                  active ? "text-violet-500" : "text-muted-foreground"
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-hidden pb-14 md:pb-0">{children}</main>
    </div>
  );
}

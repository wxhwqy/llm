"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth-store";
import { useCurrentUser } from "@/hooks/use-queries";

/**
 * Returns a function that checks auth before executing a callback.
 * If user is not logged in, redirects to /login.
 */
export function useRequireAuth() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);

  return <T extends unknown[]>(fn: (...args: T) => void) => {
    return (...args: T) => {
      if (!user) {
        router.push("/login");
        return;
      }
      fn(...args);
    };
  };
}

/**
 * Hook for pages that require authentication.
 * Redirects to /login if not authenticated after loading completes.
 * Returns { user, isLoading }.
 */
export function useAuthGuard() {
  const router = useRouter();
  const { data, isLoading } = useCurrentUser();
  const user = data?.data ?? null;

  useEffect(() => {
    if (!isLoading && !user) {
      router.replace("/login");
    }
  }, [isLoading, user, router]);

  return { user, isLoading };
}

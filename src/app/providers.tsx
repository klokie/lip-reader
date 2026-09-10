"use client"

import posthog from "posthog-js"
import { PostHogProvider as PHProvider } from "posthog-js/react"
import { useEffect, type ReactNode } from "react"

export const PostHogProvider = ({ children }: { children: ReactNode }) => {
  useEffect(() => {
    const key = process.env.NEXT_PUBLIC_POSTHOG_KEY
    if (!key) return

    posthog.init(key, {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://eu.i.posthog.com",
      // harmless on the direct host, required once POSTHOG_HOST is a proxy
      ui_host: "https://eu.posthog.com",
      capture_pageview: true,
      capture_pageleave: true,
      autocapture: true,
    })
  }, [])

  return <PHProvider client={posthog}>{children}</PHProvider>
}

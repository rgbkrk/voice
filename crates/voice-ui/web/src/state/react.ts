import { useMemo, useSyncExternalStore } from "react";
import type { Observable, Subscription } from "rxjs";
import { voiceStore } from "./voice-store";
import type { VoiceUiState } from "@/types";

export function useVoiceSelector<T>(selector: (state: VoiceUiState) => T): T {
  const selected$ = useMemo(() => voiceStore.select(selector), [selector]);

  return useSyncExternalStore(
    (onStoreChange) => subscribe(selected$, onStoreChange),
    () => selector(voiceStore.snapshot()),
    () => selector(voiceStore.snapshot()),
  );
}

function subscribe<T>(observable: Observable<T>, onStoreChange: () => void) {
  let subscription: Subscription | undefined;
  subscription = observable.subscribe(() => {
    onStoreChange();
  });

  return () => subscription?.unsubscribe();
}

import requests
from config import config

# ---------------------------------------------------------------------------
# Token estimation constant (rough: 1 token ≈ 4 characters)
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4
MAX_HISTORY_TOKENS = 3000    # Summarize when history exceeds this limit
RECENT_MESSAGES_TO_KEEP = 4  # Always preserve the last N messages verbatim


class ConversationMemory:
    """
    Stores conversation history for the backend session.

    DL06 upgrade: Instead of hard-cutting history at N messages,
    we estimate token usage and summarize old messages with Groq
    when the context gets too long. This prevents the AI from
    "forgetting" critical early context (e.g. "I'm trapped with my 4-year-old").
    """

    def __init__(self):
        self.history = []

    # ── Public API ────────────────────────────────────────────────────────

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._maybe_summarize()

    def add_ai_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._maybe_summarize()

    def get_history(self) -> list:
        return self.history

    def clear(self):
        self.history = []

    # ── Internal helpers ──────────────────────────────────────────────────

    def _estimate_tokens(self) -> int:
        """Rough token count based on total character length."""
        total_chars = sum(len(m["content"]) for m in self.history)
        return total_chars // CHARS_PER_TOKEN

    def _maybe_summarize(self):
        """
        If history is too long, summarize the older portion with Groq
        and replace it with a single summary message to preserve context
        without overflowing the LLM's context window.
        """
        if self._estimate_tokens() <= MAX_HISTORY_TOKENS:
            return  # Nothing to do

        if len(self.history) <= RECENT_MESSAGES_TO_KEEP:
            return  # Too few messages to summarize

        # Split history: old messages to summarize + recent to keep verbatim
        old_messages = self.history[:-RECENT_MESSAGES_TO_KEEP]
        recent_messages = self.history[-RECENT_MESSAGES_TO_KEEP:]

        print(f"[Memory] Context limit reached (~{self._estimate_tokens()} tokens). "
              f"Summarizing {len(old_messages)} old messages...")

        summary_text = self._summarize_with_groq(old_messages)

        # Replace old messages with a compact summary
        summary_entry = {
            "role": "assistant",
            "content": f"[Earlier conversation summary: {summary_text}]"
        }
        self.history = [summary_entry] + recent_messages
        print(f"[Memory] Summarized. History now has {len(self.history)} entries.")

    def _summarize_with_groq(self, messages: list) -> str:
        """
        Uses Groq to produce a concise summary of old conversation turns,
        so critical context (like user's situation) is never lost.
        """
        if not config.GROQ_API_KEY:
            # Fallback: join last few messages manually if no API key
            return " | ".join(
                f"{m['role']}: {m['content'][:80]}" for m in messages[-6:]
            )

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a conversation summarizer. "
                                "Summarize the following conversation turns into 2-3 concise sentences. "
                                "Preserve ALL critical facts: the user's location, their situation, "
                                "any people with them (especially children), and any danger they mentioned. "
                                "Be factual and brief."
                            )
                        },
                        {"role": "user", "content": transcript}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200
                },
                timeout=10
            )
            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[Memory] Summarization failed: {e}. Using raw truncation.")
            return " | ".join(
                f"{m['role']}: {m['content'][:80]}" for m in messages[-4:]
            )


# Global memory instance shared across the entire backend session
global_memory = ConversationMemory()

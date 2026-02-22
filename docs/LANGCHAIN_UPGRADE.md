# Suggestions: Upgrading to LangChain

Your pipeline uses **injected `ILLMProvider`** for vision and decision; providers (Ollama, OpenAI, HuggingFace) implement `chat` / `chat_vision`. These suggestions keep that design and add LangChain where it helps.

---

## 1. Where LangChain Fits

- **Behind the existing interface**  
  Implement `ILLMProvider` with a LangChain-backed adapter. The pipeline, `VisionService`, and `DecisionService` stay unchanged; only the provider implementation uses LangChain.

- **Optional deeper integration**  
  Later you can refactor services to use LCEL chains (prompt + model + output parser) for observability and structured output, while still depending on abstractions (e.g. a “chain” interface) rather than concrete LangChain types in the pipeline.

---

## 2. What You Gain

| Benefit | How |
|--------|-----|
| **Unified model interface** | `ChatOllama`, `ChatOpenAI`, etc. behind one `ILLMProvider` |
| **Retries / robustness** | LangChain retry logic and error handling |
| **Structured output** | `with_structured_output()` for decision JSON (and optionally vision) → fewer parse failures |
| **Observability** | Optional LangSmith tracing for debugging and cost |
| **Prompt management** | Use `ChatPromptTemplate` / `PromptTemplate` if you want prompts in code instead of files |
| **Future tool use** | If you add tools later, LangChain’s tool-calling is ready |

---

## 3. Recommended Approach: Adapter First

**Step 1 – Add optional dependency**

In `pyproject.toml`:

```toml
[project.optional-dependencies]
langchain = [
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",   # ChatOllama, etc.
    # or: "langchain-ollama>=0.2.0", "langchain-openai>=0.2.0"
]
```

Install: `uv pip install -e ".[langchain]"` (or equivalent).

**Step 2 – LangChain-backed provider implementing `ILLMProvider`**

- Add something like `providers/langchain_provider.py`.
- It holds a LangChain chat model (e.g. `ChatOllama`, `ChatOpenAI` from `langchain_community` or `langchain_ollama`).
- `chat(messages, **kwargs)` → build LangChain messages from `messages`, call `model.invoke(...)`, return `content` string.
- `chat_vision(messages, **kwargs)` → same; most LangChain chat models support multimodal messages (image URL or base64) like your current `image_url` usage.
- `generate(prompt, **kwargs)` → single user message, invoke, return `LLMResponse(text=..., model=...)`.

Map your existing kwargs (`model`, `max_tokens`, `temperature`, `top_p`) to the LangChain model’s constructor or `invoke` bindings. Keep the same contract so `VisionService` and `DecisionService` need no changes.

**Step 3 – Wire it in the factory**

In `providers/factory.py`, add a branch (e.g. `provider == "langchain_ollama"` or `"ollama_lc"`) that builds the LangChain chat model and wraps it in your new `LangChainProvider` (or `LangChainOllamaProvider`). `main.py` continues to call `create_provider(...)`; only config (or env) switches to the LangChain-backed provider.

**Step 4 – Optional: structured output for decision**

- In `DecisionService`, instead of “call provider + regex/JSON parse”, you could:
  - Build a small LCEL chain: system prompt + user prompt → chat model → `with_structured_output(YourDecisionSchema)`.
  - The chain runs inside a LangChain-backed provider, or the provider exposes a method that runs this chain and returns the same decision dict your pipeline expects.
- This reduces `_parse_decision_json` and markdown/code-block issues. You can do the same later for vision (e.g. structured bill extraction) if you want.

---

## 4. Design Choices to Keep

- **Pipeline and services stay provider-agnostic**  
  They depend only on `ILLMProvider` (and your other interfaces). No `import langchain` in `pipeline/` or `services/` unless you introduce an abstraction that hides “chain” behind an interface.

- **Config-driven provider selection**  
  Keep using `config.llm.provider` (and optionally a flag like `use_langchain: true`) so you can switch between current HTTP providers and LangChain-backed ones without code changes.

- **Vision message format**  
  Your `image_to_data_url` + `{"type": "image_url", "image_url": {"url": data_url}}` is compatible with LangChain’s multimodal message format; the adapter just needs to pass that content through to the LangChain model’s `invoke`.

---

## 5. Optional: Deeper LCEL Integration Later

If you later want chains inside the app (not only inside the provider):

- **Vision**  
  Chain: “system + user prompt with image” → chat model → optional structured output (e.g. Pydantic for bill fields). The chain lives behind an interface (e.g. `IVisionExtractor`) so the pipeline still doesn’t depend on LangChain.

- **Decision**  
  Chain: “system + user (bill + policy)” → chat model → `with_structured_output(DecisionSchema)`. Same idea: implement `IDecisionService` with this chain so the pipeline stays unchanged.

- **Observability**  
  Enable LangSmith (env vars) and pass `run_id` / `trace_id` into chain invocations so runs show up in the same trace as your pipeline.

---

## 6. Minimal Code Sketch (Adapter)

```python
# providers/langchain_provider.py (conceptual)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama  # or ChatOpenAI

class LangChainOllamaProvider(BaseLLMProvider):
    def __init__(self, base_url=None, model="llama3.2", timeout_sec=120, ...):
        self._model = ChatOllama(base_url=base_url, model=model, ...)

    def chat(self, messages, **kwargs):
        lc_messages = self._to_lc_messages(messages)
        out = self._model.invoke(lc_messages, max_tokens=kwargs.get("max_tokens", 4096), ...)
        return out.content

    def chat_vision(self, messages, **kwargs):
        return self.chat(messages, **kwargs)  # same; content can include image_url
```

Implement `_to_lc_messages` so that your existing `[{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}]` becomes LangChain’s `HumanMessage(content=[...])`. Then register this in the factory and use it from config.

---

## 7. Summary

| Action | Purpose |
|--------|--------|
| Add optional `langchain` extra | Avoid hard dependency; use only when upgrading |
| Implement `ILLMProvider` via LangChain (adapter) | Use LangChain models without changing pipeline or services |
| Map `chat` / `chat_vision` to `model.invoke` | Keep current message shape (including vision) |
| Optional: `with_structured_output` for decision (and later vision) | Fewer JSON parse errors and cleaner code |
| Keep provider behind config/factory | Switch between current and LangChain providers via config |

This gives you a clear path to LangChain without breaking your existing architecture.

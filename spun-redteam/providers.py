#!/usr/bin/env python3
"""Backend adapters for local AI servers.

The eval harness only needs one operation from a server: given a chat message
list, return the assistant's reply text. Different local servers expose that
behind different HTTP shapes, so each is wrapped in a Provider with a common
interface:

    p = get_provider("openai", "http://gpu-box:8000/v1", api_key=None)
    reply = p.chat("mymodel", [{"role": "user", "content": "hi"}],
                   temperature=0.0, max_tokens=250, timeout=120)

Supported `--provider` values
  openai   OpenAI-compatible /v1/chat/completions  (serve_mistral.py, vLLM,
           LM Studio, llama.cpp --api, LocalAI, text-generation-webui openai ext)
  ollama   Ollama native /api/chat                 (http://host:11434)
  frontier any hosted OpenAI-compatible chat API; no default endpoint, pass
           --endpoint or set FRONTIER_ENDPOINT (key from FRONTIER_API_KEY)

Stdlib only (urllib) so it runs on locked-down networks with no extra installs.
Servers that require a key get `Authorization: Bearer <key>`; pass it with
--api-key or a per-provider environment variable
(OPENAI_API_KEY / OLLAMA_API_KEY / FRONTIER_API_KEY).
"""

import json
import os
import ssl
import urllib.error
import urllib.request


def _ssl_context(insecure=False):
    """CA-verifying context backed by certifi when present (many venvs ship no
    system trust store on Windows). insecure=True disables verification for
    TLS-intercepting corporate proxies only."""
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _post_json(url, payload, timeout, headers=None, ctx=None):
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    kw = {"context": ctx} if url.lower().startswith("https") else {}
    with urllib.request.urlopen(req, timeout=timeout, **kw) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url, timeout, headers=None, ctx=None):
    req = urllib.request.Request(url, headers=headers or {})
    kw = {"context": ctx} if url.lower().startswith("https") else {}
    with urllib.request.urlopen(req, timeout=timeout, **kw) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Provider:
    """Common interface. Subclasses implement _build/_extract and endpoints."""

    name = "base"

    def __init__(self, endpoint, api_key=None, timeout=120, insecure=False):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.ctx = _ssl_context(insecure)

    def _auth_headers(self):
        return {"Authorization": "Bearer %s" % self.api_key} if self.api_key else {}

    def build_payload(self, model, messages, temperature, max_tokens):
        raise NotImplementedError

    def _chat_url(self):
        raise NotImplementedError

    def _extract(self, body):
        raise NotImplementedError

    def chat(self, model, messages, temperature=0.0, max_tokens=250, timeout=None):
        body = _post_json(
            self._chat_url(),
            self.build_payload(model, messages, temperature, max_tokens),
            timeout or self.timeout,
            self._auth_headers(),
            self.ctx,
        )
        return self._extract(body)

    def health(self):
        """True if the server responds. Never raises."""
        try:
            self.list_models()
            return True
        except Exception:
            return False

    def list_models(self):
        raise NotImplementedError


class OpenAIProvider(Provider):
    """/v1/chat/completions — the de-facto standard exposed by serve_mistral.py,
    vLLM, LM Studio, llama.cpp server, LocalAI, TGI's OpenAI route, etc.
    `endpoint` should already include the /v1 suffix."""

    name = "openai"

    def build_payload(self, model, messages, temperature, max_tokens):
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _chat_url(self):
        return self.endpoint + "/chat/completions"

    def _extract(self, body):
        msg = body["choices"][0]["message"]
        # reasoning models may leave content empty and put the
        # text in reasoning_content; fall back so it isn't mislabelled EMPTY
        content = msg.get("content") or ""
        if not content.strip():
            content = msg.get("reasoning_content") or content
        return content

    def list_models(self):
        body = _get_json(self.endpoint + "/models", self.timeout,
                         self._auth_headers(), self.ctx)
        return [m.get("id", "") for m in body.get("data", [])]


class OllamaProvider(Provider):
    """Ollama native API. `endpoint` is the base host, e.g.
    http://gpu-box:11434 (the /api path is added here)."""

    name = "ollama"

    def build_payload(self, model, messages, temperature, max_tokens):
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,   # thinking-capable models (Qwen3) else return empty content
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def _chat_url(self):
        return self.endpoint + "/api/chat"

    def _extract(self, body):
        # non-streaming chat returns {"message": {"role","content","thinking"?}, ...}
        msg = body.get("message", {})
        content = (msg.get("content") or "").strip()
        if not content:                      # fall back to thinking if content empty
            content = msg.get("thinking") or ""
        return content

    def list_models(self):
        body = _get_json(self.endpoint + "/api/tags", self.timeout,
                         self._auth_headers(), self.ctx)
        return [m.get("name", "") for m in body.get("models", [])]


_REGISTRY = {
    "openai": OpenAIProvider,
    "vllm": OpenAIProvider,       # aliases for the same wire format
    "lmstudio": OpenAIProvider,
    "llamacpp": OpenAIProvider,
    "localai": OpenAIProvider,
    "tgi": OpenAIProvider,
    "frontier": OpenAIProvider,   # hosted OpenAI-compatible API (endpoint required)
    "ollama": OllamaProvider,
}

# sensible default base URLs when --endpoint is omitted
DEFAULT_ENDPOINTS = {
    "openai": "http://localhost:8085/v1",
    "ollama": "http://localhost:11434",
}

# which env var holds the key, per provider name
_KEY_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "vllm": ["OPENAI_API_KEY"],
    "lmstudio": ["OPENAI_API_KEY"],
    "llamacpp": ["OPENAI_API_KEY"],
    "localai": ["OPENAI_API_KEY"],
    "tgi": ["OPENAI_API_KEY"],
    "frontier": ["FRONTIER_API_KEY"],
    "ollama": ["OLLAMA_API_KEY"],
}


def get_provider(name, endpoint=None, api_key=None, timeout=120, insecure=False):
    name = (name or "openai").lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError("unknown provider %r (choices: %s)"
                         % (name, ", ".join(sorted(_REGISTRY))))
    if not endpoint and name == "frontier":
        endpoint = os.environ.get("FRONTIER_ENDPOINT")
        if not endpoint:
            raise ValueError("provider 'frontier' has no default endpoint: pass "
                             "--endpoint or set FRONTIER_ENDPOINT")
    if not endpoint:
        endpoint = DEFAULT_ENDPOINTS.get(name, DEFAULT_ENDPOINTS["openai"])
    if api_key is None:
        for var in _KEY_ENV.get(name, []):
            if os.environ.get(var):
                api_key = os.environ[var]
                break
    return cls(endpoint, api_key=api_key, timeout=timeout, insecure=insecure)


if __name__ == "__main__":
    # quick connectivity probe:  python providers.py <provider> [endpoint]
    import sys
    prov = sys.argv[1] if len(sys.argv) > 1 else "openai"
    ep = sys.argv[2] if len(sys.argv) > 2 else None
    p = get_provider(prov, ep)
    print("provider:", p.name, "endpoint:", p.endpoint)
    if p.health():
        print("reachable. models:", ", ".join(p.list_models()) or "(none listed)")
    else:
        print("NOT reachable at", p.endpoint)

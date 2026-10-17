"""
llm_client.py — ตัวกลางเรียก LLM ที่ใช้ร่วมกันทั้งระบบ baseline (ข้อ 1) และระบบปรับปรุง (ข้อ 2)

จุดประสงค์: ให้ "LLM ที่ใช้สร้างบริบท (context) ตอน index และ LLM ที่ใช้ generate คำตอบสุดท้าย"
เป็นตัวเดียวกันในทั้งสองระบบ ตามเงื่อนไขของ assignment (ห้ามเปลี่ยน LLM ที่ generate คำตอบสุดท้าย)

รองรับ 2 provider เลือกด้วย environment variable LLM_PROVIDER:

  1) LLM_PROVIDER=claude-cli   (ค่า default)
     เรียก Claude ผ่าน Claude Code CLI แบบ non-interactive (`claude -p`) ไม่ต้องใช้ API key
     - LLM_MODEL        ชื่อโมเดล เช่น sonnet / opus / haiku (default: sonnet)
     - CLAUDE_CLI_PATH  path ของ binary `claude` (ถ้าไม่ตั้งจะหาใน PATH และใน Claude Desktop app)
     หมายเหตุ: CLI ไม่รับ max_tokens / temperature จึงถูกละไว้

  2) LLM_PROVIDER=openai
     endpoint แบบ OpenAI-compatible — ค่า default คือ OpenRouter + qwen/qwen-2.5-72b-instruct
     ตรงกับโค้ดต้นฉบับของวิชา (ใช้ OPENROUTER_API_KEY) และใช้กับ Ollama ได้
     (LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:7b  LLM_API_KEY=ollama)
"""
import glob
import os
import shutil
import subprocess
import threading
import time

PROVIDER = os.environ.get("LLM_PROVIDER", "claude-cli").lower()

_DEFAULT_MODEL = {
    "claude-cli": "sonnet",
    "openai": "qwen/qwen-2.5-72b-instruct",
}
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODEL.get(PROVIDER, "sonnet"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

_openai_client = None
_openai_lock = threading.Lock()


def describe():
    """คืนข้อความบอกว่า LLM ที่ใช้คือตัวไหน (ไว้พิมพ์ใน log)"""
    if PROVIDER == "claude-cli":
        return f"claude-cli (model={LLM_MODEL}, bin={_find_claude_cli()})"
    return f"openai-compatible (model={LLM_MODEL}, base_url={LLM_BASE_URL})"


def _find_claude_cli():
    p = os.environ.get("CLAUDE_CLI_PATH")
    if p and os.path.exists(p):
        return p
    p = shutil.which("claude")
    if p:
        return p
    # Claude Desktop app (macOS) จะมี CLI ฝังอยู่
    candidates = sorted(glob.glob(os.path.expanduser(
        "~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        "ไม่พบ Claude CLI — ติดตั้งจาก https://claude.ai/install.sh แล้วรัน `claude` เพื่อ /login "
        "หรือตั้ง CLAUDE_CLI_PATH / เปลี่ยนไปใช้ LLM_PROVIDER=openai")


def _call_claude_cli(prompt, system, retries=3):
    cli = _find_claude_cli()
    cmd = [cli, "-p", "--output-format", "text", "--model", LLM_MODEL,
           "--tools", "", "--max-turns", "1", "--no-session-persistence",
           "--system-prompt", system]
    # ตัด env ของ Claude Code session แม่ออก เพื่อให้เรียกซ้อนกันได้
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    last_err = ""
    for attempt in range(retries):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=600)
        out = proc.stdout.strip()
        if proc.returncode == 0 and out and not out.startswith("Not logged in"):
            return out
        last_err = (proc.stderr or proc.stdout or "").strip()[:500]
        if "Not logged in" in (proc.stdout + proc.stderr):
            raise RuntimeError("Claude CLI ยังไม่ได้ login — เปิด terminal แล้วรัน `claude` จากนั้นพิมพ์ /login")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Claude CLI error: {last_err}")


def _call_openai(prompt, system, max_tokens, temperature):
    global _openai_client
    from openai import OpenAI
    if not LLM_API_KEY:
        raise ValueError("OPENROUTER_API_KEY (หรือ LLM_API_KEY) environment variable not set")
    with _openai_lock:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    completion = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def call_llm(prompt, max_tokens=500, temperature=0.2, system="You are a helpful assistant."):
    """เรียก LLM ด้วย prompt เดียว คืนข้อความคำตอบ (signature เดียวกับ call_qwen_api ของต้นฉบับ)"""
    try:
        if PROVIDER == "claude-cli":
            return _call_claude_cli(prompt, system)
        return _call_openai(prompt, system, max_tokens, temperature)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"LLM error: {str(e)}"


def check_ready():
    """ทดสอบว่า LLM เรียกได้จริง (ใช้ก่อนเริ่ม index เพื่อไม่ให้เสียเวลา)"""
    out = call_llm("Reply with exactly: OK", max_tokens=5, temperature=0)
    if out.startswith("LLM error"):
        raise RuntimeError(out)
    return out


if __name__ == "__main__":
    print(describe())
    print("test ->", check_ready())

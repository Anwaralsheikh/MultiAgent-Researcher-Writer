import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from groq import Groq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
from researcher import run_researcher, ResearcherState
from copywriter import run_copywriter, CopywriterState

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_MESSAGES = 10  # Keep only last 10 messages to prevent context growth
MAX_TOKENS = 1024  # Reduced for supervisor
MAX_RETRIES = 3    # Maximum retries for rate limit errors
BASE_DELAY = 2     # Base delay in seconds for exponential backoff

# ─── Client ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ─── State ────────────────────────────────────────────────────────────────────
@dataclass
class SupervisorState:
    task:              str
    messages:          List[dict]    = field(default_factory=list)
    research_report:   Optional[str] = None
    linkedin_post:     Optional[str] = None
    finished:          bool          = False

# ─── Tools Definition ─────────────────────────────────────────────────────────
SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "handoff_to_researcher",
            "description": "Assign the research task to the Researcher agent. Call this FIRST.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The specific research task for the researcher."
                    }
                },
                "required": ["task_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handoff_to_copywriter",
            "description": "Assign the writing task to the Copywriter agent. Call ONLY after researcher finishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The specific writing task for the copywriter."
                    }
                },
                "required": ["task_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Mark the task as complete. Call ONLY after copywriter finishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished."
                    }
                },
                "required": ["summary"]
            }
        }
    }
]

# ─── Tool Execution ───────────────────────────────────────────────────────────
def handoff_to_researcher(task_description: str, state: SupervisorState) -> str:
    print(f"\n Supervisor → Researcher")
    print(f"  Task: {task_description}")
    researcher_state = run_researcher(task=task_description)
    if researcher_state.research_report:
        state.research_report = researcher_state.research_report
        return f"Researcher finished. Report generated ({len(state.research_report)} chars)."
    return "Researcher finished but no report was generated."


def handoff_to_copywriter(task_description: str, state: SupervisorState) -> str:
    if not state.research_report:
        return "Cannot run copywriter — no research report. Run researcher first."
    print(f"\n Supervisor → Copywriter")
    print(f"  Task: {task_description}")
    copywriter_state = run_copywriter(
        task=task_description,
        research_report=state.research_report
    )
    if copywriter_state.linkedin_post:
        state.linkedin_post = copywriter_state.linkedin_post
        return f"Copywriter finished. Post generated ({len(state.linkedin_post)} chars)."
    return "Copywriter finished but no post was generated."


def finish(summary: str, state: SupervisorState) -> str:
    print(f"\n Finishing: {summary}")
    state.finished = True
    return f"Task completed: {summary}"


def execute_tool(tool_name: str, tool_args: dict, state: SupervisorState) -> str:
    if tool_name == "handoff_to_researcher":
        return handoff_to_researcher(tool_args["task_description"], state)
    elif tool_name == "handoff_to_copywriter":
        return handoff_to_copywriter(tool_args["task_description"], state)
    elif tool_name == "finish":
        return finish(tool_args["summary"], state)
    return f"Unknown tool: {tool_name}"

# ─── System Prompt ────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a team of AI agents to create LinkedIn content.

Your team:
- Researcher: searches the web and generates research reports
- Copywriter: writes LinkedIn posts based on research

Your workflow (ALWAYS follow this order):
1. handoff_to_researcher → get the research done first
2. handoff_to_copywriter → write the LinkedIn post using the research
3. finish → mark the task as complete

Rules:
- NEVER skip the researcher step
- NEVER call copywriter before researcher is done
- Be specific in task descriptions"""

# ─── Helper Functions ────────────────────────────────────────────────────────
def truncate_messages(messages: List[dict], max_messages: int = MAX_MESSAGES) -> List[dict]:
    """Keep only the last max_messages to prevent context from growing too large."""
    if len(messages) <= max_messages:
        return messages
    
    # Always keep the system prompt
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    
    # Get the last max_messages - (1 if system exists)
    keep = max_messages - (1 if system_msg else 0)
    recent = messages[-keep:] if keep > 0 else []
    
    if system_msg:
        return [system_msg] + recent
    return recent


def call_llm_with_retry(messages: List[dict], tools: List[dict], max_tokens: int = MAX_TOKENS) -> dict:
    """Call Groq API with retry logic for rate limit errors."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
            )
            return response
        except RateLimitError as e:
            last_error = e
            delay = BASE_DELAY * (2 ** attempt)
            print(f"Rate limit error (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay}s...")
            time.sleep(delay)
        except APIError as e:
            last_error = e
            delay = BASE_DELAY * (2 ** attempt)
            print(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except APIConnectionError as e:
            last_error = e
            delay = BASE_DELAY * (2 ** attempt)
            print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    
    raise last_error if last_error else Exception("Unknown error occurred")


# ─── Supervisor Loop ──────────────────────────────────────────────────────────
def run_supervisor(task: str, max_steps: int = 10) -> SupervisorState:
    state = SupervisorState(task=task)
    state.messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Create a LinkedIn post about: {task}"}
    ]

    print(f"\n Supervisor starting: {task}")
    print("=" * 60)

    step = 0
    while not state.finished and step < max_steps:
        step += 1
        print(f"\n[Supervisor Step {step}]")

        try:
            
            context = f"""Current status:
- Research done: {state.research_report is not None}
- LinkedIn post done: {state.linkedin_post is not None}"""

            messages_with_context = state.messages + [
                {"role": "user", "content": context}
            ]

            # ── Call the LLM with retry logic ─────────────────────────────────
            response = call_llm_with_retry(
                messages=messages_with_context,
                tools=SUPERVISOR_TOOLS,
                max_tokens=MAX_TOKENS
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                print("Supervisor done")
                state.finished = True
                break

            state.messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"Supervisor → {tool_name}")
                result = execute_tool(tool_name, tool_args, state)
                state.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "name":         tool_name,
                    "content":      result
                })

            # Truncate messages to prevent context growth
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

            if state.finished:
                break
                
        except Exception as e:
            print(f"Error in supervisor loop: {e}")
            state.messages.append({
                "role": "user",
                "content": f"Error occurred: {e}. Please continue if possible."
            })
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

    return state

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    final_state = run_supervisor(
        task="write linkein post for The the differance between langchain and langgraph"
    )

    print("\n" + "=" * 60)
    print("📱 FINAL LINKEDIN POST:")
    print("=" * 60)
    print(final_state.linkedin_post or "No post generated")

    print("\n" + "=" * 60)
    print("RESEARCH REPORT (preview):")
    print("=" * 60)
    if final_state.research_report:
        print(final_state.research_report[:500] + "...")

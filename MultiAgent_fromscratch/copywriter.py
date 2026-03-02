import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from groq import Groq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_MESSAGES = 10  # Keep only last 10 messages to prevent context growth
MAX_TOKENS = 2048  # Reduced from 4096 to stay within limits
MAX_RETRIES = 3    # Maximum retries for rate limit errors
BASE_DELAY = 2     # Base delay in seconds for exponential backoff

# ─── Client ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ─── State ────────────────────────────────────────────────────────────────────
@dataclass
class CopywriterState:
    task:             str
    research_report:  str                
    messages:         List[dict] = field(default_factory=list)
    linkedin_post:    Optional[str] = None
    finished:         bool = False

# ─── Tools Definition (what the LLM sees) ─────────────────────────────────────
COPYWRITER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "review_research_report",
            "description": (
                "Review the research report to understand the topic before writing. "
                "Always call this FIRST before writing anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_linkedin_post",
            "description": (
                "Save the final LinkedIn post when you are happy with it. "
                "Call this ONLY when the post is complete and polished."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the post (used as filename)."
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The full LinkedIn post content. "
                            "Use emojis, hooks, and line breaks for readability."
                        )
                    }
                },
                "required": ["title", "content"]
            }
        }
    }
]

# ─── Tool Execution ───────────────────────────────────────────────────────────
def review_research_report(state: CopywriterState) -> str:
    """Return the research report to the LLM."""
    print("Reviewing research report...")
    if state.research_report:
        return state.research_report
    return "No research report available."


def save_linkedin_post(title: str, content: str, state: CopywriterState) -> str:
    """Save the LinkedIn post to state and to a file."""
    print(f"Saving LinkedIn post: {title}")

    state.linkedin_post = content
    state.finished      = True

    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/{title.replace(' ', '_')[:50]}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")

    return f"LinkedIn post saved to {filename}"


def execute_tool(tool_name: str, tool_args: dict, state: CopywriterState) -> str:
    """Route tool calls to the correct function."""
    if tool_name == "review_research_report":
        return review_research_report(state)

    elif tool_name == "save_linkedin_post":
        return save_linkedin_post(
            title=tool_args["title"],
            content=tool_args["content"],
            state=state
        )

    else:
        return f"Unknown tool: {tool_name}"

# ─── System Prompt ────────────────────────────────────────────────────────────
COPYWRITER_SYSTEM_PROMPT = """You are an expert LinkedIn copywriter. Your job is to write engaging, professional LinkedIn posts based on research.

Your writing process:
1. Call review_research_report FIRST to understand the topic
2. Write a compelling LinkedIn post based on the research
3. Call save_linkedin_post when done

LinkedIn post structure:
- 🪝 Hook (first line must grab attention)
- Story or insight (2-3 short paragraphs)
- Key takeaways (3-5 bullet points)
- Call to action (end with a question or invitation)

Rules:
- Use emojis strategically (not excessively)
- Keep paragraphs short (2-3 lines max)
- Write in a professional but conversational tone
- Always base the post on the research report
- Post length: 150-300 words"""

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


# ─── Copywriter Agent — Core Loop ─────────────────────────────────────────────
def run_copywriter(task: str, research_report: str, max_steps: int = 10) -> CopywriterState:
    """
    The main copywriter agent loop.

    Flow:
    1. Review research report
    2. Write LinkedIn post
    3. Save and finish
    """
    state = CopywriterState(
        task=task,
        research_report=research_report
    )

    # Initialize conversation
    state.messages = [
        {
            "role":    "system",
            "content": COPYWRITER_SYSTEM_PROMPT
        },
        {
            "role":    "user",
            "content": f"Write a LinkedIn post about: {task}"
        }
    ]

    print(f"\n  Copywriter starting: {task}")
    print("=" * 50)

    step = 0
    while not state.finished and step < max_steps:
        step += 1
        print(f"\n  [Step {step}]")

        try:
            # ── Call the LLM with retry logic ─────────────────────────────────
            response = call_llm_with_retry(
                messages=state.messages,
                tools=COPYWRITER_TOOLS,
                max_tokens=MAX_TOKENS
            )

            msg = response.choices[0].message

            # ── No tool call → LLM is done ────────────────────────────────────────
            if not msg.tool_calls:
                print("Copywriter finished (no more tool calls)")
                if not state.finished and msg.content:
                    state.linkedin_post = msg.content
                    state.finished      = True
                break

            # ── Add assistant message to history ──────────────────────────────────
            state.messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # ── Execute each tool call ─────────────────────────────────────────────
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"Tool: {tool_name}")

                tool_result = execute_tool(tool_name, tool_args, state)

                state.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "name":         tool_name,
                    "content":      tool_result
                })

            # Truncate messages to prevent context growth
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

            if state.finished:
                print("LinkedIn post saved!")
                break
                
        except Exception as e:
            print(f"Error in copywriter loop: {e}")
            state.messages.append({
                "role": "user",
                "content": f"Error occurred: {e}. Please continue if possible or save your post."
            })
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

    if step >= max_steps:
        print(f"Max steps ({max_steps}) reached")

    return state


# ─── Run (standalone test) ────────────────────────────────────────────────────
# if __name__ == "__main__":

#     dummy_report = """
#     ## Overview

# The research on the impact of AI Agents on the future of work in 2025 suggests that while AI has the potential to automate many jobs, it also has the potential to create new ones. However, the pace and magnitude of change are still unclear, and it is essential to understand the impact on specific occupations and workers.

# ## Key Facts & Statistics

# * 30% of current U.S. jobs could be automated by 2030, with 60% having tasks significantly modified by AI.
# * 200,000 to 300,000 U.S. jobs may have been displaced or foregone due to AI in 2025.

# ## Main Concepts

# * AI Agents: AI systems designed to work alongside humans to enhance productivity and decision-making.   
# * Job displacement: The loss of jobs due to automation or AI.
# * New job creation: The creation of new jobs due to the need for workers to manage and maintain AI systems.

# ## Real-world Examples & Applications

# * AI-powered chatbots and virtual assistants that create new customer service jobs.
# * AI-driven diagnosis and treatment in the healthcare industry.

# ## Sources

# * PwC. (2025). AI agents: reimagining the future of work, your workforce and workers.
# * McKinsey. (2025). Superagency in the workplace: empowering people to unlock AI's full potential at work.
# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agents.html
# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agents.html
# https://www.mckinsey.com/mgi/our-research/generative-ai-and-the-future-of-work-in-america
# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.
# * SHRM. (2025). Automation, Generative AI, and Job Displacement Risk in U.S. Employment.

# https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agents.html
# https://www.mckinsey.com/mgi/our-research/generative-ai-and-the-future-of-work-in-america
# https://www.shrm.org/topics-tools/research/automation-generative-ai-job-displacement-risk-hr-employment  
#     """

#     state = run_copywriter(
#         task="The future of AI Agents in the workplace",
#         research_report=dummy_report
#     )

#     print("\n" + "=" * 60)
#     print("FINAL LINKEDIN POST:")
#     print("=" * 60)

#     if state.linkedin_post:
#         print(state.linkedin_post)
#     else:
#         print("No post generated")

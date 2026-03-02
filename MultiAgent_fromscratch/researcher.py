import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from groq import Groq, APIError, APIConnectionError, RateLimitError
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_MESSAGES = 10  # Keep only last 10 messages to prevent context growth
MAX_TOKENS = 2048  # Reduced from 4096 to stay within limits
MAX_RETRIES = 3    # Maximum retries for rate limit errors
BASE_DELAY = 2     # Base delay in seconds for exponential backoff

# ─── Clients ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
tavily      = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ─── State ────────────────────────────────────────────────────────────────────
@dataclass
class ResearcherState:
    task:             str
    messages:         List[dict]  = field(default_factory=list)
    research_report:  Optional[str] = None
    finished:         bool          = False

# ─── Tools Definition (what the LLM sees) ─────────────────────────────────────
RESEARCHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web and get a list of results including title, url, "
                "and a short summary of each webpage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 3).",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_content_from_webpage",
            "description": (
                "Extract the full content from one or more webpages. "
                "Use this after search_web when you need deeper information from a specific page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to extract content from."
                    }
                },
                "required": ["urls"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_research_report",
            "description": (
                "Save the final research report when you have gathered enough information. "
                "Call this tool ONLY when you are done researching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic of the research."
                    },
                    "report": {
                        "type": "string",
                        "description": (
                            "The full research report in markdown format. "
                            "Include all key findings, facts, and sources."
                        )
                    }
                },
                "required": ["topic", "report"]
            }
        }
    }
]

# ─── Tool Execution (what actually runs) ──────────────────────────────────────
def search_web(query: str, num_results: int = 3) -> dict:
    """Search the web using Tavily."""
    print(f"Searching: {query}")
    response = tavily.search(query=query, max_results=min(num_results, 3))

    results = []
    for r in response["results"]:
        results.append({
            "title":           r.get("title", ""),
            "url":             r.get("url", ""),
            "content_preview": r.get("content", "")[:500]  
        })

    return {"query": query, "results": results}


def extract_content_from_webpage(urls: List[str]) -> list:
    """Extract full content from webpages using Tavily Extract."""
    print(f"Extracting content from {len(urls)} page(s)...")
    try:
        from langchain_tavily import TavilyExtract
        extractor = TavilyExtract()
        results   = extractor.invoke(input={"urls": urls})["results"]
        return results
    except Exception as e:
        
        return [{"url": url, "content": f"Could not extract: {e}"} for url in urls]


def save_research_report(topic: str, report: str, state: ResearcherState) -> str:
    """Save the research report to state and to a file."""
    print(f"Saving research report: {topic}")

    state.research_report = report
    state.finished        = True

    os.makedirs("research_outputs", exist_ok=True)
    filename = f"research_outputs/{topic.replace(' ', '_')[:50]}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{report}")

    return f"Research report saved successfully to {filename}"


def execute_tool(tool_name: str, tool_args: dict, state: ResearcherState) -> str:
    """Route tool calls to the correct function."""
    if tool_name == "search_web":
        result = search_web(
            query=tool_args["query"],
            num_results=tool_args.get("num_results", 3)
        )
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "extract_content_from_webpage":
        result = extract_content_from_webpage(tool_args["urls"])
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "save_research_report":
        result = save_research_report(
            topic=tool_args["topic"],
            report=tool_args["report"],
            state=state
        )
        return result

    else:
        return f"Unknown tool: {tool_name}"

# ─── System Prompt ────────────────────────────────────────────────────────────
RESEARCHER_SYSTEM_PROMPT = """You are an expert research agent. Your job is to gather comprehensive information on a given topic.

Your research process:
1. Start with 2-3 broad web searches to understand the topic
2. Use extract_content_from_webpage on the most relevant URLs for deeper info
3. Do 1-2 more targeted searches if needed to fill gaps
4. When you have enough information, call save_research_report with a comprehensive report

Report format (markdown):
- ## Overview
- ## Key Facts & Statistics  
- ## Main Concepts
- ## Real-world Examples & Applications
- ## Sources

Rules:
- Always search at least 2 times before saving the report
- The report must be detailed and well-structured
- Include the URLs of your sources in the report
- Only call save_research_report when you are truly done"""

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
                model="llama-3.3-70b-versatile",  # Changed to better rate limit model
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
            )
            return response
        except RateLimitError as e:
            last_error = e
            delay = BASE_DELAY * (2 ** attempt)  # Exponential backoff
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
    
    # If all retries failed, raise the last error
    raise last_error if last_error else Exception("Unknown error occurred")


# ─── Researcher Agent — Core Loop ─────────────────────────────────────────────
def run_researcher(task: str, max_steps: int = 20) -> ResearcherState:
    """
    The main researcher agent loop.
    
    Flow:
    1. Send task to LLM with tools
    2. If LLM calls a tool → execute it → add result → repeat
    3. If LLM doesn't call a tool → done
    """
    state = ResearcherState(task=task)

    # Initialize conversation
    state.messages = [
        {
            "role":    "system",
            "content": RESEARCHER_SYSTEM_PROMPT
        },
        {
            "role":    "user",
            "content": f"Research this topic thoroughly: {task}"
        }
    ]

    print(f"\n🔬 Researcher starting: {task}")
    print("=" * 50)

    step = 0
    while not state.finished and step < max_steps:
        step += 1
        print(f"\n  [Step {step}]")

        try:
            # ── Call the LLM with retry logic ─────────────────────────────────
            response = call_llm_with_retry(
                messages=state.messages,
                tools=RESEARCHER_TOOLS,
                max_tokens=MAX_TOKENS
            )

            usage = response.usage
            print(f"Tokens — prompt: {usage.prompt_tokens} | completion: {usage.completion_tokens} | total: {usage.total_tokens}")

            msg = response.choices[0].message

            # ── No tool call → LLM is done ────────────────────────────────────────
            if not msg.tool_calls:
                print("Researcher finished (no more tool calls)")
                # إذا ما حفظ report لسا، خليه يحفظ
                if not state.finished and msg.content:
                    state.research_report = msg.content
                    state.finished        = True
                break

            # ── Add assistant message to history ─────────────────────────────────
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

            # ── Execute each tool call ────────────────────────────────────────────
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"Tool: {tool_name}")

                tool_result = execute_tool(tool_name, tool_args, state)

                # Add tool result to conversation history
                state.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "name":         tool_name,
                    "content":      tool_result
                })

            # Truncate messages to prevent context growth
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

             
            if state.finished:
                print("Research report saved!")
                break
                
        except Exception as e:
            print(f"Error in researcher loop: {e}")
            state.messages.append({
                "role": "user",
                "content": f"Error occurred: {e}. Please continue if possible or save your report."
            })
            # Try to continue by truncating and retrying
            state.messages = truncate_messages(state.messages, MAX_MESSAGES)

    if step >= max_steps:
        print(f"  ⚠️  Max steps ({max_steps}) reached")

    return state

# # ─── Run ──────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     # اختبر الـ researcher بنفسه
#     state = run_researcher(
#         task="The impact of AI Agents on the future of work in 2025"
#     )

#     print("\n" + "=" * 60)
#     print("FINAL RESEARCH REPORT:")
#     print("=" * 60)

#     if state.research_report:
#         print(state.research_report)
#     else:
#         print(" No report generated")

#     print(f"\n Finished: {state.finished}")
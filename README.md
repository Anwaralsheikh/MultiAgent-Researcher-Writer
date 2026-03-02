# MultiAgent Researcher & Writer

**Multi-Agent System** built from scratch (no frameworks) that researches any topic and generates a professional LinkedIn post.

---

## Architecture

```
User
  ↓
Supervisor Agent        ← orchestrates the workflow
  ├── Researcher Agent  ← searches the web & writes research report  
  └── Copywriter Agent  ← writes LinkedIn post based on research
```

### Agent Responsibilities

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Orchestrates workflow, decides which agent runs next | `handoff_to_researcher`, `handoff_to_copywriter`, `finish` |
| **Researcher** | Searches the web, extracts content, generates report | `search_web`, `extract_content_from_webpage`, `save_research_report` |
| **Copywriter** | Reads research report, writes LinkedIn post | `review_research_report`, `save_linkedin_post` |

---

## Project Structure

```
MultiAgent_fromscratch/
├── researcher.py    # Researcher Agent
├── copywriter.py    # Copywriter Agent
├── supervisor.py    # Supervisor Agent (entry point)
├── main.py          # Run the full system
research_outputs/    # Generated research reports (markdown)
outputs/             # Generated LinkedIn posts (markdown)
.env                 # API keys (never commit this)
.env.example         # API keys template
```

---

## Setup

### 1.Install UV to Manage Python Projects

[UV](https://docs.astral.sh/uv/) is a python project manager that replaces pip, poetry, pyenv, and more. It's 10-100x faster than pip and I recommend using it for all of your python projects.

For MacOS it's easiest to install with homebrew:

```bash
brew install uv
```

For MacOS and Linux you can also run the following curl command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
 ```

For Windows using irm:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

There are several other ways to install uv. For more information, see the uv [documentation](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Clone the repo
```bash
git clone https://github.com/your-username/MultiAgent-Researcher-Writer.git
cd MultiAgent-Researcher-Writer
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Add API keys
```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the system
```bash
uv run MultiAgent_fromscratch/supervisor.py
```

---

## API Keys

| Key  | Where to get |
|---|---|
| `GROQ_API_KEY` |  [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |

---

##  Models Used

| Agent | Model | Reason |
|---|---|---|
| Supervisor | `llama-3.3-70b-versatile` | Decision making |
| Researcher | `llama-3.3-70b-versatile` | Large context for research |
| Copywriter | `llama-3.3-70b-versatile` | Quality writing |

---

## How It Works

```
1. User gives a topic to Supervisor
        ↓
2. Supervisor → handoff_to_researcher
        ↓
3. Researcher runs ReAct loop:
   - search_web() × 2-3
   - extract_content_from_webpage()
   - save_research_report()
        ↓
4. Supervisor → handoff_to_copywriter
        ↓
5. Copywriter runs:
   - review_research_report()
   - save_linkedin_post()
        ↓
6. Supervisor → finish() 
```

---

## Key Concepts

**Why from scratch (no LangGraph/CrewAI)?**
Building without frameworks gives full control and deep understanding of how multi-agent systems actually work under the hood.

**Why separate agents?**
Each agent has one job — separation of concerns makes the system easier to debug, test, and extend.

**Why Supervisor pattern?**
The Supervisor decides dynamically which agent runs next based on the current state, making the system flexible and extensible.

---

## Output Example

**Research Report** → saved to `research_outputs/topic_name.md`

**LinkedIn Post** → saved to `outputs/topic_name.md`

```
AI Agents are changing the way we work.

Here's what you need to know about the future of work in 2025...
[Full post in outputs/ folder]
```

---

## Roadmap

- [ ] Add memory (long-term storage)
- [ ] Add more content types (Twitter, Blog post)
- [ ] Add human-in-the-loop approval
- [ ] Add LangSmith monitoring
- [ ] Build REST API with FastAPI
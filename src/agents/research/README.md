# DR-in-KG 2.0: Deep Research System

> A systematic deep research system based on **Dynamic Topic Queue** architecture, enabling multi-agent collaboration across three phases: **Planning → Researching → Reporting**.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Three-Phase Pipeline](#three-phase-pipeline)
- [Core Data Structures](#core-data-structures)
- [Agent Responsibilities](#agent-responsibilities)
- [Tool Integration](#tool-integration)
- [Citation System](#citation-system)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Architecture Overview

```
User Input Topic
    ↓
╔══════════════════════════════════════════════════════════════╗
║  Phase 1: Planning                                           ║
║  ├─ RephraseAgent: Topic optimization (with user feedback)   ║
║  ├─ DecomposeAgent: Subtopic decomposition (RAG-enhanced)    ║
║  └─ Initialize DynamicTopicQueue                             ║
╚══════════════════════════════════════════════════════════════╝
    ↓
╔══════════════════════════════════════════════════════════════╗
║  Phase 2: Researching (Dynamic Loop)                         ║
║  ├─ ManagerAgent: Queue scheduling & task distribution       ║
║  ├─ ResearchAgent: Sufficiency check & query planning        ║
║  ├─ Tool Execution: RAG / Web / Paper / Code                 ║
║  └─ NoteAgent: Information compression & ToolTrace creation  ║
║                                                              ║
║  Execution Modes: Series (sequential) | Parallel (concurrent)║
╚══════════════════════════════════════════════════════════════╝
    ↓
╔══════════════════════════════════════════════════════════════╗
║  Phase 3: Reporting                                          ║
║  ├─ Deduplication: Remove redundant topics                   ║
║  ├─ Outline Generation: Three-level heading structure        ║
║  └─ Report Writing: Markdown with inline citations           ║
╚══════════════════════════════════════════════════════════════╝
    ↓
Final Research Report (Markdown)
```

### Directory Structure

```
src/agents/research/
├── main.py                  # CLI entry point
├── research_pipeline.py     # Three-phase pipeline orchestration
├── data_structures.py       # TopicBlock, ToolTrace, DynamicTopicQueue
├── agents/
│   ├── base_agent.py        # Base agent with LLM integration
│   ├── rephrase_agent.py    # Topic optimization
│   ├── decompose_agent.py   # Subtopic decomposition
│   ├── manager_agent.py     # Queue management
│   ├── research_agent.py    # Research decisions
│   ├── note_agent.py        # Information compression
│   └── reporting_agent.py   # Report generation
├── prompts/
│   ├── en/                  # English prompts
│   └── cn/                  # Chinese prompts
└── utils/
    ├── citation_manager.py  # Citation ID management
    ├── json_utils.py        # JSON parsing utilities
    └── token_tracker.py     # Token usage tracking
```

---

## Quick Start

### CLI Usage

```bash
# Quick mode (fast research, ~5-10 min)
python src/agents/research/main.py --topic "Deep Learning Basics" --preset quick

# Medium mode (balanced depth)
python src/agents/research/main.py --topic "Transformer Architecture" --preset medium

# Deep mode (thorough research)
python src/agents/research/main.py --topic "Graph Neural Networks" --preset deep

# Auto mode (agent decides depth)
python src/agents/research/main.py --topic "Reinforcement Learning" --preset auto
```

### Python API

```python
import asyncio
from src.agents.research import ResearchPipeline
from src.core.core import get_llm_config, load_config_with_main

async def main():
    # Load configuration (from main.yaml, agent params from agents.yaml)
    config = load_config_with_main("main.yaml")
    llm_config = get_llm_config()

    # Create pipeline with optional progress callback
    pipeline = ResearchPipeline(
        config=config,
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        kb_name="ai_textbook",  # Override knowledge base
        progress_callback=lambda event: print(f"Progress: {event}")
    )

    # Run research
    result = await pipeline.run(topic="Attention Mechanisms")
    print(f"Report: {result['final_report_path']}")

asyncio.run(main())
```

---

## Three-Phase Pipeline

### Phase 1: Planning

**Goal**: Transform user input into a structured research plan with subtopics.

#### 1.1 RephraseAgent (Topic Optimization)

- **Input**: User's original topic
- **Process**:
  1. Analyze and optimize the topic for research
  2. Support multi-turn user interaction for refinement
  3. LLM judges user satisfaction to decide continuation
- **Output**: Optimized research topic with clear focus and scope

```json
{
  "topic": "Optimized, specific, and researchable topic description (200-400 words)"
}
```

#### 1.2 DecomposeAgent (Subtopic Decomposition)

Supports two modes configured via `planning.decompose.mode`:

**Manual Mode** (`mode: "manual"`):
1. Generate N sub-queries based on `initial_subtopics` setting
2. Execute RAG retrieval for each query to gather background knowledge
3. Generate exactly N subtopics based on RAG context

**Auto Mode** (`mode: "auto"`):
1. Execute single RAG retrieval using the main topic
2. LLM autonomously identifies and generates subtopics (up to `auto_max_subtopics`)
3. Prioritizes most relevant and important aspects

- **Output**: List of subtopics with titles and overviews

```json
{
  "main_topic": "Topic",
  "sub_topics": [
    {"title": "Subtopic 1", "overview": "2-3 sentence description"},
    {"title": "Subtopic 2", "overview": "..."}
  ],
  "total_subtopics": 5,
  "mode": "manual|auto"
}
```

#### 1.3 Queue Initialization

Each subtopic is wrapped into a `TopicBlock` with status `PENDING` and added to `DynamicTopicQueue`.

---

### Phase 2: Researching

**Goal**: Deeply research each topic block through iterative tool calls and knowledge accumulation.

#### Execution Modes

| Mode | Description | Configuration |
|:---:|:---|:---:|
| **Series** | Sequential processing of topics | `execution_mode: "series"` |
| **Parallel** | Concurrent processing with semaphore | `execution_mode: "parallel"` |

Parallel mode uses `max_parallel_topics` to limit concurrency and provides thread-safe operations.

#### 2.1 ManagerAgent (Queue Scheduling)

- **Responsibilities**:
  - Get next `PENDING` task and mark as `RESEARCHING`
  - Mark tasks as `COMPLETED` or `FAILED`
  - Add dynamically discovered topics to queue
  - Provide queue statistics

- **State Transitions**:
```
PENDING → RESEARCHING → COMPLETED
              ↓
           FAILED (on error)
```

#### 2.2 ResearchAgent (Research Decisions)

For each TopicBlock, executes an iterative research loop:

**Per-Iteration Steps**:

1. **Check Sufficiency** (`check_sufficiency`):
   - Evaluates if current knowledge is sufficient
   - Considers iteration mode (fixed vs. flexible)
   - Returns `is_sufficient` and reasoning

2. **Generate Query Plan** (`generate_query_plan`):
   - Selects appropriate tool based on research phase
   - Generates specific query for knowledge gaps
   - May discover new related topics (dynamic splitting)

```json
{
  "query": "Specific query statement",
  "tool_type": "rag_hybrid|web_search|paper_search|...",
  "rationale": "Selection reasoning",
  "new_sub_topic": "(Optional) Discovered topic",
  "new_topic_score": 0.0-1.0,
  "should_add_new_topic": true|false
}
```

3. **Tool Execution**: Pipeline calls the selected tool
4. **Note Recording**: NoteAgent processes results

**Iteration Modes**:
- `fixed`: Must complete all iterations (conservative about stopping)
- `flexible`: Agent can stop early when knowledge is sufficient

#### 2.3 NoteAgent (Information Compression)

- **Input**: Raw tool output, query, topic context
- **Process**: Extracts key information, generates concise summary
- **Output**: `ToolTrace` with:
  - `citation_id`: Unique reference ID (e.g., `CIT-3-01`)
  - `summary`: Compressed knowledge (200-500 words)
  - `raw_answer`: Complete original response

#### 2.4 Dynamic Topic Discovery

During research, ResearchAgent may discover related topics:
- If `should_add_new_topic = true` and `new_topic_score ≥ min_score`
- ManagerAgent adds new TopicBlock to queue tail
- Ensures no duplicate topics

---

### Phase 3: Reporting

**Goal**: Generate structured Markdown report with citations.

#### 3.1 Deduplication (`deduplicate_blocks`)

- LLM identifies semantically duplicate topic blocks
- Merges or removes redundant content
- Returns cleaned block list

#### 3.2 Outline Generation (`generate_outline`)

Creates three-level heading structure:
- Level 1 (`#`): Report title
- Level 2 (`##`): Main sections (Introduction, Core Sections, Conclusion)
- Level 3 (`###`): Subsections within each section

```json
{
  "title": "# Research Topic",
  "introduction": "## Introduction",
  "sections": [
    {
      "title": "## Section Title",
      "instruction": "Section guidance",
      "block_id": "block_1",
      "subsections": [
        {"title": "### Subsection", "instruction": "..."}
      ]
    }
  ],
  "conclusion": "## Conclusion"
}
```

#### 3.3 Report Writing (`write_report`)

- **Primary**: LLM generates complete Markdown report
- **Fallback**: Local template assembly if LLM fails

**Citation Format**:
- Inline: `[[CIT-3-01](#ref-cit-3-01)]`
- References section with anchor IDs: `<a id="ref-cit-3-01"></a>`

---

## Core Data Structures

### TopicBlock

Minimum scheduling unit in the queue:

```python
@dataclass
class TopicBlock:
    block_id: str           # e.g., "block_1"
    sub_topic: str          # Topic title
    overview: str           # Topic description
    status: TopicStatus     # PENDING|RESEARCHING|COMPLETED|FAILED
    tool_traces: list[ToolTrace]  # Tool call history
    iteration_count: int    # Current iteration
    created_at: str
    updated_at: str
    metadata: dict
```

### ToolTrace

Single tool call record:

```python
@dataclass
class ToolTrace:
    tool_id: str            # Unique ID (timestamp-based)
    citation_id: str        # Citation reference (e.g., CIT-3-01)
    tool_type: str          # Tool name
    query: str              # Query issued
    raw_answer: str         # Complete tool response
    summary: str            # Compressed summary
    timestamp: str
```

### DynamicTopicQueue

Core memory and scheduling center:

```python
class DynamicTopicQueue:
    research_id: str
    blocks: list[TopicBlock]
    max_length: int | None

    # Methods
    add_block(sub_topic, overview) -> TopicBlock
    get_pending_block() -> TopicBlock | None
    mark_researching(block_id) -> bool
    mark_completed(block_id) -> bool
    has_topic(sub_topic) -> bool
    get_statistics() -> dict
```

---

## Agent Responsibilities

| Agent | Phase | Input | Output |
|:---:|:---:|:---|:---|
| **RephraseAgent** | Planning | User topic | Optimized topic JSON |
| **DecomposeAgent** | Planning | Topic + RAG context | Subtopics list |
| **ManagerAgent** | Researching | Queue | Task assignments |
| **ResearchAgent** | Researching | TopicBlock + context | Query plans |
| **NoteAgent** | Researching | Tool output | ToolTrace |
| **ReportingAgent** | Reporting | Queue + topic | Markdown report |

---

## Tool Integration

Available tools (configurable via `researching.enable_*`):

| Tool | Type | Query Format | Use Case |
|:---:|:---:|:---|:---|
| `rag_hybrid` | RAG | Natural language | Comprehensive knowledge retrieval |
| `rag_naive` | RAG | Natural language | Basic vector search |
| `query_item` | Entity | Item ID (e.g., "Theorem 3.1") | Specific entity lookup |
| `paper_search` | External | English keywords | Academic research |
| `web_search` | External | Natural language | Real-time information |
| `run_code` | Code | Python code | Calculations, visualization |

**Tool Selection Strategy**:
- Early iterations: Focus on RAG tools
- Middle iterations: Add Paper/Web search for depth
- Late iterations: Fill gaps with all available tools

---

## Citation System

### CitationManager

Manages global citation IDs with two formats:

- **Planning stage**: `PLAN-01`, `PLAN-02`, ...
- **Research stage**: `CIT-{block}-{seq}` (e.g., `CIT-3-01`)

### Citation Flow

1. ResearchAgent calls tool
2. CitationManager generates unique `citation_id`
3. NoteAgent creates ToolTrace with citation_id
4. ReportingAgent inserts inline citations in report
5. References section lists all citations with anchor links

### In-Report Format

```markdown
This is a key finding [[CIT-3-01](#ref-cit-3-01)].

## References

### Subtopic 1
<a id="ref-cit-3-01"></a> [CIT-3-01] rag_hybrid: Query about concept X
<a id="ref-cit-3-02"></a> [CIT-3-02] web_search: Latest developments in X
```

---

## Configuration

### Key Settings

**Agent Parameters** (`config/agents.yaml`):
```yaml
research:
  temperature: 0.5
  max_tokens: 12000
```

**Module Settings** (`config/main.yaml` - research section):
```yaml
research:
  # Planning Phase
  planning:
    rephrase:
      enabled: false              # Enable topic rephrasing
      max_iterations: 3
    decompose:
      mode: "auto"                # "manual" or "auto"
      initial_subtopics: 5        # For manual mode
      auto_max_subtopics: 8       # For auto mode

  # Researching Phase
  researching:
    execution_mode: "parallel"    # "series" or "parallel"
    max_parallel_topics: 5
    max_iterations: 5
    new_topic_min_score: 0.85     # Threshold for dynamic topics

    # Tool switches
    enable_rag_hybrid: true
    enable_rag_naive: true
    enable_paper_search: true
    enable_web_search: true
    enable_run_code: true

  # Queue
  queue:
    max_length: 5                 # Maximum topics
```

### Preset Modes

| Preset | Subtopics | Iterations | Mode | Use Case |
|:---:|:---:|:---:|:---:|:---|
| `quick` | 1 | 1 | Fixed | Fast overview |
| `medium` | 5 | 4 | Fixed | Balanced depth |
| `deep` | 8 | 7 | Fixed | Thorough research |
| `auto` | ≤8 | ≤6 | Flexible | Agent decides |

---

## Output Files

```
data/user/research/
├── reports/
│   ├── research_YYYYMMDD_HHMMSS.md      # Final Markdown report
│   └── research_*_metadata.json          # Statistics and metadata
└── cache/
    └── research_YYYYMMDD_HHMMSS/
        ├── queue.json                    # DynamicTopicQueue state
        ├── citations.json                # Citation registry
        ├── step1_planning.json           # Planning results
        ├── planning_progress.json        # Planning events
        ├── researching_progress.json     # Research events
        ├── reporting_progress.json       # Reporting events
        ├── outline.json                  # Report outline
        └── token_cost_summary.json       # Token usage
```

---

## Design Principles

1. **Dynamic Topic Queue**: Replaces static tree structure with flexible queue-based scheduling
2. **Strict JSON**: All agent prompts enforce JSON output; parsing uses `json_utils` validation
3. **Environment-based Model**: LLM model loaded only from environment variables
4. **Context Management**: NoteAgent compression prevents token explosion
5. **Extensible Tools**: Easy to add new tools in ResearchAgent's query plan
6. **Clickable Citations**: Inline references linked to anchor points in References section

---

## FAQ

**Q: ModuleNotFoundError: research_pipeline**

Run from project root:
```bash
python src/agents/research/main.py --topic "..." --preset quick
```

**Q: How to use a different knowledge base?**

```bash
# Via CLI (uses config default)
# Edit config/main.yaml: research.rag.kb_name

# Via Python API
pipeline = ResearchPipeline(..., kb_name="my_kb")
```

**Q: How to enable/disable specific tools?**

Edit `config/main.yaml`:
```yaml
research:
  researching:
    enable_web_search: false
    enable_paper_search: false
```

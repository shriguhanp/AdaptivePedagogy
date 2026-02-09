# Analysis Loop - Deep Research Phase

## Overview

Analysis Loop is the first phase of the dual-loop architecture, responsible for **deeply understanding user questions**. Through iterative research, it dynamically collects and organizes knowledge until the question is sufficiently understood.

## Architecture

```
┌─────────────────────────────────────────┐
│         Analysis Loop (Research Phase)   │
│                                          │
│  ┌──────────┐    ┌──────────┐            │
│  │Investigate│ -> │   Note   │ ────┐      │
│  └──────────┘    └──────────┘     │      │
│        ▲            │             │      │
│        └────────────┴─────────────┘      │
│                    │                     │
│            Stop Condition Met            │
│                    │                     │
│                    ▼                     │
│              InvestigateMemory           │
└─────────────────────────────────────────┘
```

## Core Agents

### 1. InvestigateAgent

**Role**: Research planner and tool executor. Based on knowledge chain summaries and user questions, generates multiple high-value queries and calls tools to retrieve raw data.

**Key Features**:
1. Identifies knowledge gaps from cite_id summaries; generates multiple `[TOOL]tool_type | query` commands in one pass if needed.
2. Immediately calls tools (`rag_naive`, `rag_hybrid`, `web_search`, `query_item`) for each query to get `raw_result`.
3. When `[TOOL] none` appears, knowledge is sufficient and Analysis Loop automatically stops.

**Output Structure**:
```python
{
    'reasoning': 'List of knowledge gaps identified this round',
    'should_stop': False,
    'knowledge_item_ids': ['c01', 'c02'],  # List of new cite_ids this round
    'actions': [
        {'tool_type': 'rag_hybrid', 'query': 'Section 2.5...', 'cite_id': 'c01'},
        {'tool_type': 'rag_naive', 'query': 'Lambda range in algorithm', 'cite_id': 'c02'}
    ]
}
```

### 2. NoteAgent

**Role**: Note-taker. For each knowledge item in `knowledge_item_ids`, sequentially calls LLM to convert `raw_result` into summaries and extract citations.

**Key Features**:
1. Automatically iterates through multiple knowledge items in one round, independently constructing Prompt → LLM → parsing summary.
2. Summaries must be concise and extract citable sources from `raw_result`.
3. After processing, updates `InvestigateMemory` and `CitationMemory` for use by Solve Loop.

**Output Structure**:
```python
{
    'success': True,
    'processed_items': 2,
    'details': [
        {'cite_id': 'c01', 'citations_count': 2},
        {'cite_id': 'c02', 'citations_count': 1}
    ],
    'failed': []
}
```

## Memory System

### InvestigateMemory

Persistent JSON structure that chains all knowledge collected during the analysis phase. Each tool call generates a unique `cite_id` and is recorded in `knowledge_chain`.

**Structure**:
```python
{
  "user_question": "What is linear convolution?",
  "knowledge_chain": [
    {
      "cite_id": "c01",
      "tool_type": "rag_hybrid",
      "query": "Section 2.5 Linear convolution definition and properties",
      "raw_result": "...RAG source text...",
      "summary": "Linear convolution derives from convolution integral; core properties include commutativity and associativity.",
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "reflections": {
    "remaining_questions": [],
    "updated_at": "..."
  },
  "metadata": {
    "total_iterations": 2,
    "total_knowledge_items": 1,
    "coverage_rate": 1.0,
    "avg_confidence": 0.9
  }
}
```

**Storage Location**: `{output_dir}/investigate_memory.json`

## Configuration

Configure Analysis Loop in `config.yaml`:

```yaml
system:
  max_analysis_iterations: 5  # Maximum iterations

agents:
  investigate_agent:
    enabled: true
    model: "gpt-4o"
    temperature: 0.4

  note_agent:
    enabled: true
    model: "gpt-4o"
    temperature: 0.3
```

## Usage Example

```python
from solve_agents.memory import InvestigateMemory
from solve_agents.analysis_loop import InvestigateAgent, NoteAgent

memory = InvestigateMemory.load_or_create(
    output_dir="./output",
    user_question="What is linear convolution?"
)

investigate_agent = InvestigateAgent(config, api_key, base_url)
note_agent = NoteAgent(config, api_key, base_url)

for i in range(max_iterations):
    investigate_result = await investigate_agent.process(
        question=question,
        memory=memory,
        citation_memory=citation_memory,
        kb_name="ai_textbook",
        output_dir="./output"
    )

    cite_ids = investigate_result.get('knowledge_item_ids', [])
    if cite_ids:
        await note_agent.process(
            question=question,
            memory=memory,
            new_knowledge_ids=cite_ids,
            citation_memory=citation_memory,
            output_dir="./output"
        )

    if investigate_result.get('should_stop'):
        break

memory.save()
```

## Workflow Example

**Question**: "What is linear convolution?"

**Round 1**:
1. **Investigate**: Outputs two queries
   - `[TOOL]rag_hybrid | Linear convolution definition`
   - `[TOOL]rag_naive | Geometric interpretation of convolution`
   Gets cite_ids `c01/c02`.
2. **Note**: Processes `c01/c02` sequentially, generates summaries and extracts 3 citations.

**Round 2**:
1. **Investigate**: Outputs
   - `[TOOL]rag_hybrid | Convolution calculation examples`
   - `[TOOL]web_search | Latest application scenarios`
   Gets cite_ids `c03/c04`.
2. **Note**: Processes `c03/c04`, supplements calculation procedures and practical examples.

**Round 3**:
1. **Investigate**: `[TOOL]none` - reasoning: "Definition, derivation, and examples all covered".
2. Analysis Loop completes.

**Result**: Collects 4 knowledge items; NoteAgent automatically updates summaries and citations; ready for Solve Loop consumption.

## Key Features

### 1. Dynamic Research
- No preset query count
- Dynamically determines based on actual needs
- Automatically judges when to stop

### 2. Tool Selection
- Prioritizes knowledge base (RAG)
- Uses web search when necessary
- Intelligently selects most appropriate tools

### 3. Citation Management
- Automatically extracts key citations
- Provides evidence for Solve Loop
- Supports subsequent citation formatting

### 4. Persistent Memory
- JSON format storage
- Supports checkpoint resumption
- Facilitates debugging and review

## Integration with Solve Loop

After Analysis Loop completes, `InvestigateMemory` is passed to Solve Loop:

```python
# Analysis Loop completes
investigate_memory.save()

# Solve Loop consumes
solve_memory = SolveMemory.load_or_create(...)
plan_result = await plan_agent.process(
    question=question,
    investigate_memory=investigate_memory,  # Pass analysis results
    solve_memory=solve_memory
)
```

Solve Loop can access:
- `investigate_memory.knowledge_chain`: All collected knowledge
- `investigate_memory.get_all_citations()`: All citations
- `investigate_memory.get_knowledge_summary()`: Knowledge summary

## Debugging and Monitoring

### View Memory Files

```bash
cat output/investigate_memory.json | jq .
```

### Monitor Iteration Progress

Logs display detailed information for each round:

```
--- Analysis Loop Round 1 ---
  [Investigate] Reasoning: Still missing calculation examples
    • rag_hybrid: Section 2.4 convolution definition -> cite_id=c01
    • rag_naive: Geometric interpretation of convolution -> cite_id=c02
  [Note] Processed 2 knowledge items
```

### Performance Statistics

```python
metadata = investigate_memory.metadata
print(f"Total iterations: {metadata['total_iterations']} rounds")
print(f"Total knowledge items: {metadata['total_knowledge_items']}")
print(f"Coverage rate: {metadata['coverage_rate']:.2%}")
print(f"Average confidence: {metadata['avg_confidence']:.2f}")
```

## Common Questions

**Q: How to adjust maximum iteration count?**

Set in `config.yaml`:
```yaml
system:
  max_analysis_iterations: 5  # Default 5 rounds
```

**Q: How to force web search usage?**

Modify the Investigate Agent prompt or pass parameters when calling.

**Q: How to view extracted citations?**

```python
citations = investigate_memory.get_all_citations()
for citation in citations:
    print(citation)
```

**Q: What if memory file becomes too large?**

Periodically clean up `raw_content`, keeping only `summary` and `citations`.

---

**Related Documentation**:
- [Solve Loop README](../solve_loop/README.md)
- [Main README](../README.md)

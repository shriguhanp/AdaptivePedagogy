# Tools Module

This module provides a collection of various tools used in the DeepTutor system, including code execution, knowledge retrieval, web search, paper search, and more.

## 📁 Directory Structure

```
tools/
├── __init__.py              # Module initialization, unified export of all tools
├── code_executor.py         # Code execution tool ⭐
├── query_item_tool.py       # Numbered item query tool ⭐
├── rag_tool.py              # RAG retrieval tool ⭐
├── web_search.py            # Web search tool
├── paper_search_tool.py     # Paper search tool
├── tex_downloader.py        # LaTeX source download tool
├── tex_chunker.py           # LaTeX text chunking tool
└── README.md                # This document
```

## 🚀 Quick Start

### Import Tools

```python
from src.tools import (
    run_code,              # Code execution
    run_code_sync,         # Synchronous code execution
    query_numbered_item,   # Query numbered items
    rag_search,            # RAG retrieval
    web_search,            # Web search
    PaperSearchTool,       # Paper search (optional)
    TexDownloader,         # LaTeX download (optional)
    TexChunker,            # LaTeX chunking (optional)
)
```

## 📋 Tool Details

### 1. Code Execution Tool (`code_executor.py`)

Execute Python code in an isolated workspace, preserving the original input/output structure.

**Key Features:**
- Asynchronous and synchronous code execution
- Isolated workspace environment
- Secure filesystem access control
- Support for persistent workspaces
- Automatic cleanup of temporary files

**Usage Example:**

```python
from src.tools import run_code, run_code_sync

# Asynchronous execution
result = await run_code(
    code="print('Hello, World!')",
    workspace="my_workspace"
)

# Synchronous execution
result = run_code_sync(
    code="x = 1 + 1\nprint(x)",
    workspace="my_workspace"
)

# Result format
# {
#     "status": "success" | "error",
#     "output": str,      # Standard output
#     "error": str,       # Error message (if any)
#     "exit_code": int,   # Exit code
# }
```

**Configuration:**
- Configure via `tools.run_code` in `solve_config.yaml` or `question_config.yaml`
- Environment variables:
  - `RUN_CODE_WORKSPACE`: Workspace directory
  - `RUN_CODE_ALLOWED_ROOTS`: List of allowed root directories

---

### 2. Numbered Item Query Tool (`query_item_tool.py`)

Query numbered items in the knowledge base, such as definitions, theorems, formulas, figures, etc.

**Key Features:**
- Support for multiple numbered item types: definitions, theorems, formulas, figures, examples, remarks, etc.
- Support for fuzzy and exact matching
- Returns multiple matching results
- Backward compatible with single result format

**Usage Example:**

```python
from src.tools import query_numbered_item

# Query definition
result = query_numbered_item(
    identifier="Definition 1.1",
    kb_name="ai_textbook"
)

# Query formula
result = query_numbered_item(
    identifier="(1.2.1)",
    kb_name="ai_textbook"
)

# Query figure
result = query_numbered_item(
    identifier="Figure 2.5",
    kb_name="ai_textbook",
    max_results=3
)

# Result format
# {
#     "identifier": str,      # Original query identifier
#     "type": str,            # Type: formula/definition/theorem/lemma/figure/example/remark
#     "status": str,          # success/failed
#     "count": int,           # Number of matched items
#     "items": [              # List of all matched items
#         {
#             "identifier": str,
#             "type": str,
#             "content": str
#         }
#     ],
#     "content": str,         # Backward compatible: single item content or merged content
#     "error": str            # Error message (only when failed)
# }
```

**Supported Identifier Formats:**
- Definition/Theorem: `"Definition 1.1"`, `"Theorem 2.3"`
- Formula: `"(1.2.1)"`, `"(2.3.5)"`
- Figure: `"Figure 1.1"`, `"Figure 2.5"`
- Example/Remark: `"Example 1.1"`, `"Remark 2.1"`

---

### 3. RAG Retrieval Tool (`rag_tool.py`)

Knowledge base query tool based on RAG (Retrieval-Augmented Generation).

**Key Features:**
- Support for multiple query modes: `local`, `global`, `hybrid`, `naive`
- Based on LightRAG and RAG-Anything frameworks
- Automatic knowledge base connection management
- Support for custom LLM and Embedding configurations

**Usage Example:**

```python
from src.tools import rag_search

# Asynchronous query
result = await rag_search(
    query="What is machine learning?",
    kb_name="ai_textbook",
    mode="hybrid"  # local/global/hybrid/naive
)

# Result format
# {
#     "query": str,
#     "answer": str,           # RAG-generated answer
#     "context": str,           # Retrieved context
#     "prompt": str,            # Full prompt (if only_need_prompt=True)
#     "status": str,            # success/error
#     "error": str              # Error message (if any)
# }
```

**Query Mode Description:**
- `local`: Local retrieval, focusing on specific topics
- `global`: Global retrieval, obtaining overall knowledge
- `hybrid`: Hybrid mode, combining local and global (recommended)
- `naive`: Simple retrieval mode

**Configuration:**
- Configure LLM and Embedding via `main.yaml`
- Knowledge base path: default `./knowledge_bases`

---

### 4. Web Search Tool (`web_search.py`)

Perform web search using Perplexity API.

**Key Features:**
- Real-time web search based on Perplexity API
- Automatic saving of search results
- Support for verbose output mode

**Usage Example:**

```python
from src.tools import web_search

# Perform search
result = web_search(
    query="latest deep learning research progress",
    output_dir="./search_results",  # Optional: save results
    verbose=True
)

# Result format
# {
#     "query": str,
#     "answer": str,           # Search result summary
#     "result_file": str       # Saved file path (if output_dir provided)
# }
```

**Dependencies:**
- Requires `perplexity` package
- Requires `PERPLEXITY_API_KEY` environment variable

**Installation:**
```bash
pip install perplexity
```

---

### 5. Paper Search Tool (`paper_search_tool.py`)

ArXiv paper search tool.

**Key Features:**
- Search ArXiv papers
- Parse paper metadata
- Format paper information
- Support for sorting by relevance or date
- Support for year limits

**Usage Example:**

```python
from src.tools import PaperSearchTool

tool = PaperSearchTool()

# Search papers
papers = await tool.search_papers(
    query="transformer attention mechanism",
    max_results=5,
    years_limit=3,        # Last 3 years
    sort_by="relevance"   # relevance or date
)

# Result format
# [
#     {
#         "title": str,
#         "authors": list[str],
#         "year": int,
#         "abstract": str,
#         "url": str,
#         "arxiv_id": str,
#         "published": str,
#         "categories": list[str]
#     },
#     ...
# ]

# Format paper information
formatted = tool.format_paper_info(papers[0])
```

**Dependencies:**
- Requires `arxiv` package

**Installation:**
```bash
pip install arxiv
```

---

### 6. LaTeX Source Download Tool (`tex_downloader.py`)

Download LaTeX source code of papers from ArXiv.

**Key Features:**
- Download LaTeX source archive from ArXiv
- Automatically extract and locate main tex file
- Read tex file content
- Support for tar.gz and zip formats

**Usage Example:**

```python
from src.tools import TexDownloader

downloader = TexDownloader(workspace_dir="./workspace")

# Download LaTeX source
result = await downloader.download_tex(
    arxiv_id="2301.00001"
)

# Result format (TexDownloadResult)
# result.success: bool
# result.tex_path: str | None      # Main tex file path
# result.tex_content: str | None    # Tex file content
# result.error: str | None          # Error message (if any)

if result.success:
    print(f"Tex file: {result.tex_path}")
    print(f"Content length: {len(result.tex_content)}")
```

**Helper Function:**
```python
from src.tools import read_tex_file

# Directly read tex file
content = read_tex_file(tex_path)
```

---

### 7. LaTeX Text Chunking Tool (`tex_chunker.py`)

Intelligently chunk LaTeX content for processing long documents.

**Key Features:**
- Chunk by section or token count
- Token estimation (based on GPT tokenizer)
- Maintain context coherence (overlap between chunks)
- Support for custom chunking strategies

**Usage Example:**

```python
from src.tools import TexChunker

chunker = TexChunker(model="gpt-4o")

# Estimate token count
token_count = chunker.estimate_tokens(tex_content)

# Chunk by section
chunks = chunker.chunk_by_section(
    tex_content,
    max_tokens_per_chunk=4000,
    overlap_tokens=200
)

# Chunk by token count
chunks = chunker.chunk_by_tokens(
    tex_content,
    max_tokens=4000,
    overlap_tokens=200
)

# Result format
# [
#     {
#         "content": str,        # Chunk content
#         "tokens": int,         # Token count
#         "start_line": int,     # Start line number
#         "end_line": int        # End line number
#     },
#     ...
# ]
```

**Dependencies:**
- Requires `tiktoken` package

**Installation:**
```bash
pip install tiktoken
```

---

## 🔧 Configuration

### Environment Variables

The tools module uses the following environment variables:

```bash
# Code execution
RUN_CODE_WORKSPACE=/path/to/workspace
RUN_CODE_ALLOWED_ROOTS=/path1,/path2

# LLM configuration (for RAG)
LLM_MODEL=gpt-4o
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1

# Web search
PERPLEXITY_API_KEY=your-perplexity-key

# Knowledge base path
KB_BASE_DIR=./knowledge_bases
```

### Configuration Files

Tool configuration is mainly in the following configuration files:

- `config/main.yaml`: Main configuration file (LLM, Embedding, etc.)
- `config/solve_config.yaml`: Solver configuration (includes `tools.run_code`)
- `config/question_config.yaml`: Question configuration (includes `tools.run_code`)

---

## 📦 Dependency Management

### Required Dependencies

Dependencies required by all tools:
- `lightrag`: RAG framework
- `raganything`: RAG-Anything framework

### Optional Dependencies

Some tools require additional dependencies:

```bash
# Web search
pip install perplexity

# Paper search and LaTeX processing
pip install arxiv tiktoken requests
```

**Note:** If some optional dependencies are not installed, the related tools will fail to import, but this will not affect the use of other tools. `__init__.py` gracefully handles import failures.

---

## 🔍 Usage Scenarios

### Scenario 1: Code Execution and Verification

```python
from src.tools import run_code_sync

# Execute mathematical calculations
result = run_code_sync(
    code="""
import numpy as np
x = np.array([1, 2, 3])
print(x.mean())
""",
    workspace="math_workspace"
)
```

### Scenario 2: Knowledge Base Query

```python
from src.tools import query_numbered_item, rag_search

# Query specific definition
definition = query_numbered_item("Definition 3.1", kb_name="ai_textbook")

# RAG retrieval of related concepts
context = await rag_search(
    query="Explain backpropagation algorithm",
    kb_name="ai_textbook",
    mode="hybrid"
)
```

### Scenario 3: Paper Research

```python
from src.tools import PaperSearchTool, TexDownloader, TexChunker

# Search related papers
tool = PaperSearchTool()
papers = await tool.search_papers("transformer architecture", max_results=3)

# Download and process LaTeX source
downloader = TexDownloader("./workspace")
result = await downloader.download_tex(papers[0]["arxiv_id"])

if result.success:
    chunker = TexChunker()
    chunks = chunker.chunk_by_section(result.tex_content, max_tokens_per_chunk=4000)
```

### Scenario 4: Web Information Retrieval

```python
from src.tools import web_search

# Search for latest information
result = web_search(
    query="latest deep learning breakthroughs in 2024",
    verbose=True
)
```

---

## 🛠️ Development Guide

### Adding New Tools

1. Create a new tool file in the `tools/` directory
2. Implement the tool function or class
3. Import and export in `__init__.py`
4. Update this document

### Tool Interface Specification

It is recommended that tool functions follow this interface specification:

```python
def tool_function(
    required_param: str,
    optional_param: str | None = None,
    **kwargs
) -> dict:
    """
    Tool function description

    Args:
        required_param: Required parameter description
        optional_param: Optional parameter description
        **kwargs: Other parameters

    Returns:
        dict: Result dictionary containing status, data, error fields
    """
    pass
```

### Error Handling

All tools should:
- Return a unified dictionary format result
- Include a `status` field (`"success"` or `"error"`)
- Provide an `error` field on failure
- Use logging to record error information

---

## 📝 Notes

1. **Code Execution Security**: `code_executor` runs in an isolated environment, but security configuration should still be noted
2. **API Keys**: Web search and RAG require corresponding API keys, please keep them secure
3. **Knowledge Base Path**: Ensure the knowledge base path is configured correctly
4. **Async Functions**: Some tools are asynchronous and need to be called with `await` or in an async environment
5. **Dependency Installation**: Some tools require additional dependencies, please install according to usage scenarios

---

## 🔗 Related Documentation

- [Knowledge Base Module Documentation](../knowledge/README.md)
- [Core Module Documentation](../core/README.md)
- [Agents Module Documentation](../agents/README.md)

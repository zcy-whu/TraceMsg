# **TraceMsg**

## **Overview**

Commit messages provide records of code changes and help developers understand and maintain software. To reduce the manual effort involved in writing them, commit message generation (CMG) approaches automatically generate commit messages that describe code changes in natural language. Existing CMG approaches generally treat CMG as a standalone task and optimize generated messages primarily for similarity to developer-written references. 
However, this objective may not properly reflect the practical usefulness of generated commit messages. This limitation matters in current AI-assisted software engineering workflows, where commit messages are used as shared information to serve different software engineering activities. We propose \toolname, a collaborative multi-agent framework that rethinks CMG as a dynamic process guided by issue--commit traceability, a representative software engineering activity. It coordinates message generation, traceability reasoning, and feedback-guided refinement, allowing generated messages to serve traceability while using reasoning feedback to guide issue-conditioned refinement.

## **Dataset**

TraceMsg uses the curated Apache issue-commit traceability benchmark under the Realistic Distribution Setting (RDS). Our data can be accessed here: 



Expected repository paths:

- **Commit data**: `data/<project>/all_commits_info.pkl`
  - Contains commit-level metadata, generated change summaries, generated root-cause/rationale fields, and validity flags.
- **Gold links**: `data/<project>/gold_links.pkl`
  - Maps issue IDs to ground-truth fixing commit IDs.
- **Evaluation CSV**: `data/constructed_evaluation_dataset/zhang_dataset/test_<project>_link.csv`
  - Contains RDS issue-candidate rows with issue text, candidate commit IDs, and labels.

## **Installation**

### **Prerequisites**

- Python 3.9.25
- PyTorch 2.4.1
- Conda is recommended
- A Qwen-compatible API key for online LLM stages

### **Setup**

If installing only the TraceMsg dependencies:

```bash
pip install -r TraceMsg/requirements.txt
```

Set the Qwen API key before running reranking, feedback, or final stages:

```bash
export QWEN_API_KEY="your_qwen_api_key"
```

The code uses the local embedding model if available:

```bash
code/sentence-transformers/all-MiniLM-L6-v2
```

Otherwise, Sentence-Transformers will load `sentence-transformers/all-MiniLM-L6-v2`.

## **Entry File**

Main entry:

```bash
TraceMsg/run_tracemsg.py
```

Knowledge-base construction entry:

```bash
TraceMsg/build_knowledge_base.py
```

Example full run on Isis:

```bash
cd TraceMsg
python run_tracemsg.py --project isis --mode all
```

Run only dense retrieval without an API key:

```bash
cd TraceMsg
python run_tracemsg.py --project isis --mode all --stages retrieve
```

Build the Ambari maintenance-pattern knowledge base from historical links:

```bash
cd TraceMsg
python build_knowledge_base.py \
  --project ambari \
  --historical_csv ../data/ground_truth_dataset/zhang_dataset/ambari_link.csv
```

For a quick API smoke test, limit the number of extracted historical pairs:

```bash
cd TraceMsg
python build_knowledge_base.py --project ambari --limit_pairs 3
```

The builder streams the large CSV with `chunksize` and uses only rows where `label == 1`. Any issue ID that appears in `data/ambari/gold_links.pkl` is skipped so current evaluation issues are not used to construct patterns. Commit-side abstractions are built from the generated structured `WHAT`/`WHY` fields, not developer-written messages.

Run a quick smoke test on five issues:

```bash
cd TraceMsg
python run_tracemsg.py --project isis --mode all --limit_issues 5
```

Run only later stages after retrieval cache exists:

```bash
cd TraceMsg
python run_tracemsg.py --project isis --mode all --stages rerank feedback final
```

## **Output**

Outputs are written to:

```bash
data/<project>/tracemsg_results/
```

Main files:

- `stage0_results_<mode>.csv`: dense retrieval results.
- `rerank_results_<mode>.csv`: first-round LLM reranking results.
- `final_top10_rerank_results_<mode>.csv`: final reranking after feedback-guided `WHY` refinement.
- `summary_<mode>.json`: metrics summary for completed stages.

Intermediate caches are written to:

```bash
data/<project>/tracemsg_cache/
```

Cache folders include retrieval results, initial rerank results, mode-isolated feedback judgments, refined `WHY` statements, and final rankings. These caches support resuming interrupted runs without mixing WHAT/WHY ablations.

## **Code Structure**

```text
TraceMsg/
  README.md
  requirements.txt
  run_tracemsg.py
  tracemsg/
    config.py              # central parameters and dataset paths
    data.py                # commit, issue, and gold-link loading
    generation_agent.py    # WHAT/WHY generation and WHY refinement
    traceability_agent.py  # dense retrieval and sliding-window reranking
    feedback_agent.py      # pattern retrieval and pair-level feedback
    knowledge_builder.py   # offline pattern extraction from historical links
    pipeline.py            # end-to-end workflow
    evaluation.py          # Hit@k, Recall@k, MRR
    llm.py                 # Qwen-compatible LLM client and JSON parsing
```


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECTS = ("ambari", "calcite", "groovy", "ignite", "isis", "netbeans")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEMSG_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TraceMsgConfig:
    # Absolute defaults make the entry points independent of the caller's cwd.
    repo_root: Path = REPO_ROOT
    data_root: Path = TRACEMSG_ROOT / "data"
    ground_truth_root: Path = REPO_ROOT / "data" / "ground_truth_dataset" / "zhang_dataset"
    evaluation_root: Path = (
        REPO_ROOT / "data" / "constructed_evaluation_dataset" / "zhang_dataset"
    )
    output_root: Path = TRACEMSG_ROOT / "data"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    retrieval_ratio: float = 0.02
    retrieval_mode: str = "all"

    llm_model: str = "qwen3.6-plus"
    llm_temperature: float = 0.1
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key_env: str = "QWEN_API_KEY"

    judge_model: str = "qwen3.6-plus"
    ranker_model: str = "qwen3.6-plus"
    max_workers: int = 5

    rerank_window_size: int = 5
    rerank_step_size: int = 3
    initial_top_k: int = 10
    feedback_pattern_top_k: int = 3
    feedback_rounds: int = 1

    index_dir_name: str = "tracemsg_indexes"
    cache_dir_name: str = "tracemsg_cache"
    result_dir_name: str = "tracemsg_results"

    require_valid_diff: bool = True
    exclude_multi_issue_commits: bool = True

    @property
    def local_embedding_model_path(self) -> Path:
        return self.repo_root / "code" / "sentence-transformers" / "all-MiniLM-L6-v2"

    def project_dir(self, project: str) -> Path:
        return self.data_root / project

    def commits_path(self, project: str) -> Path:
        return self.project_dir(project) / "all_commits_info.pkl"

    def gold_links_path(self, project: str) -> Path:
        return self.project_dir(project) / "gold_links.pkl"

    def evaluation_csv_path(self, project: str) -> Path:
        return self.evaluation_root / f"test_{project}_link.csv"

    def historical_link_csv_path(self, project: str) -> Path:
        return self.ground_truth_root / f"{project}_link.csv"

    def output_dir(self, project: str) -> Path:
        return self.output_root / project

    def cache_dir(self, project: str) -> Path:
        return self.output_dir(project) / self.cache_dir_name

    def result_dir(self, project: str) -> Path:
        return self.output_dir(project) / self.result_dir_name

    def knowledge_base_path(self, project: str) -> Path:
        local_path = self.output_dir(project) / "knowledge_base" / "knowledge_base.json"
        legacy_path = (
            self.repo_root
            / "data"
            / "zcy"
            / project
            / "knowledge_base"
            / "knowledge_base.json"
        )
        if not local_path.exists() and legacy_path.exists():
            return legacy_path
        return local_path

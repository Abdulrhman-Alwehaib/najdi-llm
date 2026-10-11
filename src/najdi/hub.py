from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from najdi.env import hf_token


@lru_cache(maxsize=1)
def api() -> HfApi:
    return HfApi(token=hf_token())


def resolve(name: str) -> str:
    return name if "/" in name else f"{api().whoami()['name']}/{name}"


def create_private(name: str, repo_type: str = "dataset") -> str:
    repo_id = api().create_repo(resolve(name), repo_type=repo_type, private=True, exist_ok=True).repo_id
    info = api().dataset_info(repo_id) if repo_type == "dataset" else api().model_info(repo_id)
    if not info.private:
        raise RuntimeError(f"{repo_id} exists and is public")
    return repo_id


def _payload(value: Path | str | bytes):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return value


def push(name: str, files: dict[str, Path | str | bytes], message: str, repo_type: str = "dataset") -> str:
    repo_id = create_private(name, repo_type)
    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=_payload(value)) for path, value in files.items()
    ]
    api().create_commit(repo_id, repo_type=repo_type, commit_message=message, operations=operations)
    return repo_id


def upload_folder(repo_id: str, folder: Path, path_in_repo: str, message: str, allow: list[str] | None = None) -> None:
    kwargs = dict(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=message,
        ignore_patterns=[".env", "*.env", "_*"],
    )
    if path_in_repo:
        kwargs["delete_patterns"] = "*"
    if allow:
        kwargs["allow_patterns"] = allow
    api().upload_folder(**kwargs)


def download(name: str, filename: str, repo_type: str = "dataset", force: bool = False) -> Path:
    path = hf_hub_download(resolve(name), filename, repo_type=repo_type, token=hf_token(), force_download=force)
    return Path(path)

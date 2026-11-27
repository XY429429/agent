from __future__ import annotations
import os
from dataclasses import dataclass,replace
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent

def env_bool(name:str,default:bool=False)->bool:
    raw=os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in{"1","true","yes","on"}

@dataclass(frozen=True)
class Settings:
    mode:str="llm"
    db_path:Path=PROJECT_ROOT / "demo.sqlite"
    max_revisions:int=2
    require_approval:bool=False
    log_level:str="INFO"
    openai_mode:str="deepseek-flash"

    checkpointer:str="sqlite"
    postgres_uri:str=""
    api_key:str=""
    require_auth:bool=False
    rate_limit_per_min:int=120


"""读取环境配置"""
def load_settings()->Settings:
    api_key=os.getenv("APIKEY","")
    settings=Settings(
        mode=os.getenv("APP_MODE","llm").lower(),
        db_path=Path(os.getenv("APP_DB_PATH",str(PROJECT_ROOT / "demo.sqlite"))),
        max_revisions=int(os.getenv("MAX_REVISIONS",'2')),
        require_approval=env_bool("REQUIRE_APPROVAL",False),
        log_level=os.getenv("LOG_LEVEL","INFO").upper(),
        openai_mode=os.getenv("MODEL","deepseek-flash"),
        checkpointer=os.getenv("CHECKPOINTER","SQLITE").lower(),
        postgres_uri=os.getenv("POSTGRES_URI", ""),
        api_key=api_key,
        require_auth=env_bool("REQUIRE_AUTH", bool(api_key)),
        rate_limit_per_min=int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
    )
    if settings.mode not in {"mock", "llm","single"}:
        raise ValueError(f"APP_MODE must be mock or llm, got {settings.mode!r}")
    if settings.max_revisions < 0:
        raise ValueError("MAX_REVISIONS must be >= 0")
    if settings.checkpointer not in {"sqlite", "postgres"}:
        raise ValueError("CHECKPOINTER must be sqlite or postgres")
    if settings.checkpointer == "postgres" and not settings.postgres_uri:
        raise ValueError("POSTGRES_URI is required when CHECKPOINTER=postgres")
    if settings.require_auth and not settings.api_key:
        raise ValueError("API_KEY is required when REQUIRE_AUTH=1")
    return settings

def with_overrides(settings:Settings,**kwargs)->Settings:
    return replace(settings,**kwargs)
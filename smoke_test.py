from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


PROMPT = "What is the capital of Karnataka? Answer in one short sentence."


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    env_var: str
    model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ProviderConfig("anthropic", "ANTHROPIC_API_KEY", "claude-3-5-haiku-latest"),
    "google": ProviderConfig("google", "GOOGLE_API_KEY", "gemini-2.0-flash"),
    "groq": ProviderConfig("groq", "GROQ_API_KEY", "llama-3.1-8b-instant"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate each supported chat provider and make a simple API call."
    )
    parser.add_argument(
        "--provider",
        choices=["all", *PROVIDERS.keys()],
        default="all",
        help="Provider to test. Default: all configured providers.",
    )
    return parser.parse_args()


def build_model(config: ProviderConfig):
    if config.name == "openai":
        return ChatOpenAI(model=config.model)
    if config.name == "anthropic":
        return ChatAnthropic(model=config.model)
    if config.name == "google":
        return ChatGoogleGenerativeAI(model=config.model)
    if config.name == "groq":
        return ChatGroq(model=config.model)
    raise ValueError(f"Unsupported provider: {config.name}")


def run_provider(config: ProviderConfig) -> bool:
    api_key = os.getenv(config.env_var)
    print(f"\n=== {config.name} ===")

    if not api_key:
        print(f"SKIP: missing {config.env_var}")
        return False

    try:
        model = build_model(config)
        response = model.invoke(PROMPT)
    except Exception as exc:  # pragma: no cover - this is a smoke test path
        print(f"FAIL: {exc}")
        return False

    print(f"Model: {config.model}")
    print(f"Prompt: {PROMPT}")
    print(f"Response: {response.content}")
    return True


def main() -> int:
    load_dotenv()
    args = parse_args()

    provider_names = PROVIDERS.keys() if args.provider == "all" else [args.provider]
    attempted = 0
    passed = 0

    for provider_name in provider_names:
        attempted += 1
        if run_provider(PROVIDERS[provider_name]):
            passed += 1

    print(f"\nSummary: {passed}/{attempted} provider checks passed")
    return 0 if passed == attempted else 1


if __name__ == "__main__":
    sys.exit(main())
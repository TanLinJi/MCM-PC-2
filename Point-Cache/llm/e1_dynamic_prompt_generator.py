"""Generic LLM prompt generation utilities for E1 text prototype enhancement.

This module generates class-level point-cloud-aware descriptions.

Important:
- It only uses dataset candidate class names.
- It does not use test labels.
- It does not use test point cloud content.
- It saves generated prompts so that experiments remain reproducible.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_llm_api_key(api_key_file="llm/secrets/llm_api_key.txt"):
    """Read LLM API key from a local file or environment variable.

    The local key file should contain one line like:
        sk-xxxxxxxxxxxxxxxx
    """
    key_path = Path(api_key_file)

    if key_path.exists():
        api_key = key_path.read_text(encoding="utf-8").strip()
        if api_key:
            return api_key

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if api_key:
        return api_key

    raise RuntimeError(
        "LLM API key not found. Put it in "
        "'Point-Cache/llm/secrets/llm_api_key.txt' or set LLM_API_KEY."
    )


def safe_name(name):
    """Convert a string to a safe file-name component."""
    name = str(name).replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def strip_json_code_fence(text):
    """Remove markdown JSON fences if the LLM returns them."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def parse_prompt_list(content):
    """Parse LLM output into a list of prompt strings."""
    content = strip_json_code_fence(content)

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    prompts = []
    for line in content.splitlines():
        line = line.strip()
        line = re.sub(r"^[0-9]+[.)]\s*", "", line)
        line = line.strip("-• \t")
        if line:
            prompts.append(line)

    return prompts


def build_llm_request(classname, prompt_count, model, temperature):
    """Build an OpenAI-compatible chat completion request for one class."""
    system_prompt = (
        "You generate concise English descriptions for 3D point cloud object recognition. "
        "Focus on geometric structure, object parts, shape, symmetry, and spatial layout. "
        "Do not describe colors, textures, photos, paintings, or camera styles. "
        "Return only a JSON array of strings."
    )

    user_prompt = (
        f"Generate {prompt_count} different point-cloud-aware descriptions for the class "
        f"'{classname}'. Each description should be one complete sentence and should help "
        "a vision-language model recognize this object from a 3D point cloud. "
        "Return only a JSON array of strings."
    )

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1600,
    }


def call_openai_compatible_api(api_key, api_base_url, payload):
    """Call an OpenAI-compatible chat completion API using the standard library."""
    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        api_base_url,
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP error {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc

    response_json = json.loads(response_text)
    return response_json["choices"][0]["message"]["content"]


def get_prompt_cache_path(args, dataset_name):
    """Return the JSON path used to save or load generated prompts."""
    cache_dir = Path(getattr(args, "prompt_cache_dir", "results/E1_text_prototype_enhancement/prompts"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    provider = safe_name(getattr(args, "llm_provider", "deepseek"))
    model_name = safe_name(getattr(args, "llm_model", "deepseek-v4-pro"))
    prompt_count = int(getattr(args, "dynamic_prompt_count", 25))
    dataset_name = safe_name(dataset_name or getattr(args, "dataset", "unknown_dataset"))

    file_name = f"{dataset_name}_{provider}_{model_name}_{prompt_count}_prompts.json"
    return cache_dir / file_name


def generate_llm_prompts(classnames, args, dataset_name=None):
    """Generate or load LLM prompts for dataset candidate class names."""
    classnames = [str(name).replace("_", " ") for name in classnames]

    cache_path = get_prompt_cache_path(args, dataset_name)
    force_regenerate = bool(getattr(args, "force_regenerate_prompts", False))

    if cache_path.exists() and not force_regenerate:
        with open(cache_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        prompts = saved.get("prompts", saved)
        print(f"[E1] Loaded LLM prompts from {cache_path}")
        return prompts

    api_key_file = getattr(args, "llm_api_key_file", "llm/secrets/llm_api_key.txt")
    api_key = read_llm_api_key(api_key_file)

    provider = getattr(args, "llm_provider", "deepseek")
    api_base_url = getattr(args, "llm_api_base_url", "https://api.deepseek.com/chat/completions")
    model = getattr(args, "llm_model", "deepseek-v4-pro")
    prompt_count = int(getattr(args, "dynamic_prompt_count", 25))
    temperature = float(getattr(args, "llm_temperature", 0.7))

    prompts = {}

    for idx, classname in enumerate(classnames, start=1):
        print(f"[E1] Generating LLM prompts [{idx}/{len(classnames)}]: {classname}")

        payload = build_llm_request(
            classname=classname,
            prompt_count=prompt_count,
            model=model,
            temperature=temperature,
        )

        content = call_openai_compatible_api(
            api_key=api_key,
            api_base_url=api_base_url,
            payload=payload,
        )

        class_prompts = parse_prompt_list(content)

        if len(class_prompts) == 0:
            raise RuntimeError(f"LLM returned no valid prompts for class: {classname}")

        prompts[classname] = class_prompts[:prompt_count]

    saved = {
        "prompt_source": getattr(args, "prompt_source", "llm_dynamic_init"),
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_base_url": api_base_url,
        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name or getattr(args, "dataset", "unknown_dataset"),
        "dynamic_prompt_count": prompt_count,
        "temperature": temperature,
        "class_names": classnames,
        "prompts": prompts,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)

    print(f"[E1] Saved LLM prompts to {cache_path}")
    return prompts

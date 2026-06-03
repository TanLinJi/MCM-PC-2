"""E1 dynamic LLM prompt generation utilities.

This module generates class-level descriptions for E1 text prototype enhancement.

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
    """Read LLM API key from a local file or environment variable."""
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
    text = str(text).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def clean_prompt_text(text):
    """Clean one prompt string returned by the LLM.

    This removes common artifacts such as:
    - escaped quotes: \"...\"
    - surrounding quotes: "..."
    - trailing commas from JSON-like line outputs
    """
    text = str(text).strip()
    text = text.strip(",")

    # Convert escaped quotes that may appear after line-based fallback parsing.
    text = text.replace('\\"', '"').strip()

    # Remove one layer of surrounding quotes.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    text = text.strip()
    text = text.strip(",")

    return text


def parse_prompt_list(content):
    """Parse LLM output into a list of prompt strings.

    Supported outputs:
    - JSON array: ["...", "..."]
    - JSON object with prompts/descriptions/sentences/items
    - plain numbered lines
    """
    content = strip_json_code_fence(content)

    if not content:
        return []

    try:
        data = json.loads(content)

        if isinstance(data, list):
            return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]

        if isinstance(data, dict):
            for key in ["prompts", "descriptions", "sentences", "items", "data"]:
                value = data.get(key)
                if isinstance(value, list):
                    return [clean_prompt_text(item) for item in value if clean_prompt_text(item)]
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON array from mixed text.
    left = content.find("[")
    right = content.rfind("]")
    if left != -1 and right != -1 and right > left:
        try:
            data = json.loads(content[left:right + 1])
            if isinstance(data, list):
                return [clean_prompt_text(item) for item in data if clean_prompt_text(item)]
        except json.JSONDecodeError:
            pass

    prompts = []
    for line in content.splitlines():
        line = line.strip()
        line = re.sub(r"^[0-9]+[.)]\s*", "", line)
        line = line.strip("-• \t")
        if line and line not in {"[", "]", "{", "}"}:
            cleaned = clean_prompt_text(line)
        if cleaned:
            prompts.append(cleaned)

    return prompts


def build_llm_request(classname, prompt_count, model, temperature, prompt_mode="multiview_2d3d"):
    """Build an OpenAI-compatible chat completion request for one class.

    prompt_mode:
    - pointcloud_geometry: mainly 3D point cloud geometry.
    - multiview_2d3d: a prompt set containing both 2D visual semantics and 3D point-cloud geometry.
    """
    if prompt_mode == "pointcloud_geometry":
        system_prompt = (
            "You generate concise English descriptions for 3D point cloud object recognition. "
            "Focus on geometric structure, object parts, shape, symmetry, and spatial layout. "
            "Do not describe colors, textures, photos, paintings, or camera styles. "
            "Return only a non-empty JSON array of strings."
        )

        user_prompt = (
            f"Generate exactly {prompt_count} different point-cloud-aware descriptions for the class "
            f"'{classname}'. Each description should be one complete sentence and should help "
            "a vision-language model recognize this object from a 3D point cloud. "
            "Return only a non-empty JSON array of strings. Do not return an empty array."
        )

    elif prompt_mode == "multiview_2d3d":
        system_prompt = (
            "You generate concise English class descriptions for vision-language recognition of 3D point clouds. "
            "The whole description set should contain both 2D visual semantics and 3D point-cloud geometry, "
            "but each individual sentence does not need to contain both. "
            "Return only a JSON array of strings. Do not include explanations, numbering, markdown, or extra text."
        )

        if int(prompt_count) == 10:
            user_prompt = (
                f"Generate exactly 10 descriptions for the class '{classname}'. "
                "The 10 descriptions should be organized conceptually as follows, but return only one flat JSON array of 10 strings: "
                "Descriptions 1-4: 2D visual semantic descriptions, focusing on common visual appearance, recognizable parts, "
                "object identity, and image-level cues useful for CLIP-like text encoders. "
                "Descriptions 5-8: 3D point-cloud geometric descriptions, focusing on shape, structure, parts, symmetry, "
                "spatial layout, and point distribution. "
                "Descriptions 9-10: bridge descriptions that connect visual appearance with 3D geometric structure. "
                "Each description must be a complete English sentence. "
                "Avoid very short fragments. Do not output vague fragments such as 'a hand-carved stone'. "
                "Return only a JSON array of 10 strings."
            )
        else:
            user_prompt = (
                f"Generate exactly {prompt_count} descriptions for the class '{classname}'. "
                "Some descriptions should focus on 2D visual semantics and common appearance. "
                "Some descriptions should focus on 3D point-cloud geometry, shape, parts, symmetry, and spatial layout. "
                "A small number may connect both views. "
                "Each description must be a complete English sentence. "
                "Avoid very short fragments. Return only a JSON array of strings."
            )

    else:
        raise ValueError(f"Unknown LLM prompt mode: {prompt_mode}")

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1200,
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
    prompt_mode = safe_name(getattr(args, "llm_prompt_mode", "multiview_2d3d"))
    prompt_count = int(getattr(args, "dynamic_prompt_count", 25))
    dataset_name = safe_name(dataset_name or getattr(args, "dataset", "unknown_dataset"))

    file_name = f"{dataset_name}_{provider}_{model_name}_{prompt_mode}_{prompt_count}_prompts.json"
    return cache_dir / file_name


def save_prompt_cache(cache_path, args, dataset_name, classnames, prompts, failed_classes=None):
    """Save prompt cache after each generated class."""
    saved = {
        "prompt_source": getattr(args, "prompt_source", "llm_dynamic_init"),
        "llm_provider": getattr(args, "llm_provider", "deepseek"),
        "llm_model": getattr(args, "llm_model", "deepseek-v4-pro"),
        "llm_api_base_url": getattr(args, "llm_api_base_url", "https://api.deepseek.com/chat/completions"),
        "llm_prompt_mode": getattr(args, "llm_prompt_mode", "multiview_2d3d"),
        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name or getattr(args, "dataset", "unknown_dataset"),
        "dynamic_prompt_count": int(getattr(args, "dynamic_prompt_count", 25)),
        "temperature": float(getattr(args, "llm_temperature", 0.7)),
        "class_names": classnames,
        "completed_class_names": sorted(prompts.keys()),
        "failed_classes": failed_classes or [],
        "prompts": prompts,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)


def load_existing_prompts(cache_path):
    """Load existing complete or partial prompt cache."""
    if not cache_path.exists():
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    prompts = saved.get("prompts", saved)
    if not isinstance(prompts, dict):
        return {}

    return prompts


def is_valid_prompt_text(text):
    """Filter out incomplete or obviously low-quality prompt fragments."""
    text = str(text).strip()
    words = text.split()

    if len(words) < 8:
        return False

    if not text.endswith((".", "!", "?")):
        return False

    bad_fragments = {
        "a hand-carved stone",
        "hand-carved stone",
    }
    if text.lower().strip(" .,!?:;") in bad_fragments:
        return False

    return True


def generate_one_class_prompts(classname, args, api_key, api_base_url, model, prompt_count, temperature, prompt_mode):
    """Generate prompts for one class with retry."""
    max_retries = int(getattr(args, "llm_max_retries", 3))

    for attempt in range(1, max_retries + 1):
        payload = build_llm_request(
            classname=classname,
            prompt_count=prompt_count,
            model=model,
            temperature=temperature,
            prompt_mode=prompt_mode,
        )

        content = call_openai_compatible_api(
            api_key=api_key,
            api_base_url=api_base_url,
            payload=payload,
        )

        prompts = [p for p in parse_prompt_list(content) if is_valid_prompt_text(p)]

        if len(prompts) > 0:
            return prompts[:prompt_count]

        print(f"[E1] Empty or invalid LLM output for {classname}. Retry {attempt}/{max_retries}.")

        time.sleep(1.0)

    raise RuntimeError(f"LLM returned no valid prompts for class: {classname}")


def generate_llm_prompts(classnames, args, dataset_name=None):
    """Generate or load LLM prompts for dataset candidate class names."""
    classnames = [str(name).replace("_", " ") for name in classnames]

    cache_path = get_prompt_cache_path(args, dataset_name)
    force_regenerate = bool(getattr(args, "force_regenerate_prompts", False))
    prompt_count = int(getattr(args, "dynamic_prompt_count", 25))

    existing_prompts = {} if force_regenerate else load_existing_prompts(cache_path)

    missing_classes = [
        classname for classname in classnames
        if len(existing_prompts.get(classname, [])) < prompt_count
    ]

    if len(missing_classes) == 0:
        print(f"[E1] Loaded complete LLM prompts from {cache_path}")
        return existing_prompts

    if existing_prompts:
        print(f"[E1] Loaded partial LLM prompts from {cache_path}")
        print(f"[E1] Remaining classes to generate: {len(missing_classes)}")

    api_key_file = getattr(args, "llm_api_key_file", "llm/secrets/llm_api_key.txt")
    api_key = read_llm_api_key(api_key_file)

    api_base_url = getattr(args, "llm_api_base_url", "https://api.deepseek.com/chat/completions")
    model = getattr(args, "llm_model", "deepseek-v4-pro")
    temperature = float(getattr(args, "llm_temperature", 0.7))
    prompt_mode = getattr(args, "llm_prompt_mode", "multiview_2d3d")

    failed_classes = []

    for idx, classname in enumerate(missing_classes, start=1):
        print(f"[E1] Generating LLM prompts [{idx}/{len(missing_classes)}]: {classname}")

        try:
            class_prompts = generate_one_class_prompts(
                classname=classname,
                args=args,
                api_key=api_key,
                api_base_url=api_base_url,
                model=model,
                prompt_count=prompt_count,
                temperature=temperature,
                prompt_mode=prompt_mode,
            )
        except Exception:
            failed_classes.append(classname)
            save_prompt_cache(cache_path, args, dataset_name, classnames, existing_prompts, failed_classes)
            raise

        existing_prompts[classname] = class_prompts
        save_prompt_cache(cache_path, args, dataset_name, classnames, existing_prompts, failed_classes)
        print(f"[E1] Saved partial LLM prompts to {cache_path}")

    print(f"[E1] Saved complete LLM prompts to {cache_path}")
    return existing_prompts

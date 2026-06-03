import json

# 64个手工设计的文本模板
text_prompts = [
    "a point cloud model of {}.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
    "a photo of a {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of a {}.",
    "itap of my {}.",
    "itap of the {}.",
    "a photo of a {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of a {}.",
    "a good photo of the {}.",
    "a bad photo of a {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of a {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of a {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of a {}",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}."
]

text_prompts_pc2_view = [
    "a point cloud depth map of a(n) {}.",
]

with open('llm/mn40_gpt35_prompts.json') as fin:
    mn40_gpt35_prompts = json.load(fin)

with open('llm/mn40_gpt4_prompts.json') as fin:
    mn40_gpt4_prompts = json.load(fin)    

with open('llm/mn40_pointllm_prompts.json') as fin:
    mn40_pointllm_prompts = json.load(fin)   


with open('llm/sonn_gpt35_prompts.json') as fin:
    sonn_gpt35_prompts = json.load(fin)

with open('llm/sonn_gpt4_prompts.json') as fin:
    sonn_gpt4_prompts = json.load(fin)    

with open('llm/sonn_pointllm_prompts.json') as fin:
    sonn_pointllm_prompts = json.load(fin)  


with open('llm/omniobject3d-gpt3.5-turbo.json') as fin:
    omni3d_gpt35_prompts = json.load(fin)
    
omni3d_gpt4_prompts = []
omni3d_pointllm_prompts = []

with open('llm/objaverse_lvis-gpt3.5-turbo.json') as fin:
    o_lvis_gpt35_prompts = json.load(fin)
    
o_lvis_gpt4_prompts = []
o_lvis_pointllm_prompts = []

# E1 smoke-test diagnostic prompt subset.
#
# manual_3d_prompts removes clearly 2D image-style prompts from the original
# Point-Cache manual templates and keeps templates that are more related to
# 3D objects, geometry, shapes, structures, models, and point-cloud semantics.
#
# This prompt source is retained only as a diagnostic smoke-test method.
# It is not the main E1 direction because its accuracy is much lower than
# manual_full in the current ULIP × ModelNet-C severity=2 zero-shot test.
_IMAGE_STYLE_PROMPT_KEYWORDS = (
    "photo",
    "image",
    "picture",
    "painting",
    "sketch",
    "cartoon",
    "drawing",
    "rendering",
    "screenshot",
    "view",
    "camera",
)

_3D_STYLE_PROMPT_KEYWORDS = (
    "3d",
    "point cloud",
    "object",
    "shape",
    "geometry",
    "geometric",
    "structure",
    "model",
    "mesh",
    "surface",
    "solid",
    "scene",
)

manual_3d_prompts = [
    prompt for prompt in text_prompts
    if any(keyword in prompt.lower() for keyword in _3D_STYLE_PROMPT_KEYWORDS)
    and not any(keyword in prompt.lower() for keyword in _IMAGE_STYLE_PROMPT_KEYWORDS)
]

if len(manual_3d_prompts) == 0:
    manual_3d_prompts = text_prompts


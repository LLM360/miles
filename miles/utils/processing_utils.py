import base64
import inspect
import io
import logging
import os

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, chat_template_path: str = None, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if chat_template_path:
        assert os.path.isfile(chat_template_path), (
            f"chat_template_path not found: {chat_template_path}. "
            f"Ensure the path is accessible on this node (e.g. inside the miles repo or on a shared filesystem)."
        )
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Loaded custom chat template from %s", chat_template_path)
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids as lists), "pt" for modality-specific outputs
    result["text_kwargs"] = {**result.get("text_kwargs", {}), "return_tensors": None}
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def processor_requires_medias(processor) -> bool:
    try:
        params = inspect.signature(processor.__call__).parameters
        return "medias" in params and "text" in params
    except (TypeError, ValueError):
        return hasattr(processor, "media_processor")


def call_processor(processor, text, multimodal_inputs: dict | None = None):
    multimodal_inputs = multimodal_inputs or {}

    # for kimi-vl & kimi-2.5
    if processor_requires_medias(processor):
        medias = []
        if images := multimodal_inputs.get("images"):
            medias.extend({"type": "image", "image": image} for image in images)
        if videos := multimodal_inputs.get("videos"):
            medias.extend({"type": "video", "video": video} for video in videos)
        return processor(text=text, medias=medias)

    kwargs = build_processor_kwargs(multimodal_inputs)
    return processor(text=text, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # temporary solution, will write image utils for miles later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

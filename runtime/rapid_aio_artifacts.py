"""Pinned Rapid AIO weights and local-only configuration/tokenizer assets.

Hashes were read from HF LFS metadata or calculated from the pinned small files.
The supporting Qwen files contain no duplicate encoder or VAE weights.
"""
from pathlib import Path

RAPID_REPO = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
RAPID_REVISION = "691024f438640508f8aa86414863fc15edfb8a84"
QWEN_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
QWEN_CODE_REVISION = "6b5e1f5cec987d404be5ac6657db3b9aacb56a89"

_SUPPORT_FILES = (
    ("README.md", "9724c194bef2a6d821090f0cd65774962e8f77e3acbfb2a7cbbdd58c92049902", 7179),
    ("processor/added_tokens.json", "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b", 605),
    ("processor/chat_template.jinja", "a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb", 1017),
    ("processor/merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", 1671853),
    ("processor/preprocessor_config.json", "e98b81513c5d03209ac71194b3cd5303b90956097ffdb9e2aa64d6d1a13ae74e", 826),
    ("processor/special_tokens_map.json", "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd", 613),
    ("processor/tokenizer.json", "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa", 11421896),
    ("processor/tokenizer_config.json", "1b20115c83f30580a8c0fc330f47fad9222dccd4f739fe2d48e62ee93d28ac19", 4727),
    ("processor/video_preprocessor_config.json", "6763a66890170c801051d721ac1679339c2f1a4f4e31076bc58661756670c32f", 910),
    ("processor/vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833),
    ("scheduler/scheduler_config.json", "7ee767e37bae4af31d4eb935e125bb20a2237eeecafd22af6610093865c6f587", 485),
    ("text_encoder/config.json", "4a49f14a1594fd988ab301e491ac522cd9b0e8fd1c9232b806c54568f348c1fb", 3146),
    ("tokenizer/added_tokens.json", "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b", 605),
    ("tokenizer/chat_template.jinja", "44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5", 2427),
    ("tokenizer/merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", 1671853),
    ("tokenizer/special_tokens_map.json", "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd", 613),
    ("tokenizer/tokenizer_config.json", "0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614", 4686),
    ("tokenizer/vocab.json", "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d", 3383407),
    ("transformer/config.json", "43c02875ed3e0d4c1b3a1007b34505d0bd2c1607fc90b2107c5e39c53d187113", 362),
    ("vae/config.json", "c448160dba5ce79c965cb075ee02e18d1c42eb6424f787e5869790d577b56a65", 730),
)


def rapid_aio_file_entries(models_dir: Path) -> dict[str, dict]:
    """Return the setup downloader's ordinary verified-file entries."""
    root = models_dir / "qwen-rapid-aio"
    source_page = f"https://huggingface.co/{RAPID_REPO}"
    files = {
        "avatar_rapid_aio_v19": {
            "label": "Qwen Image Edit Rapid AIO v19",
            "url": f"{source_page}/resolve/{RAPID_REVISION}/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors",
            "source_page": f"{source_page}/blob/{RAPID_REVISION}/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors",
            "sha256": "ba71575515709c9912560d1176b2386eaa49294fedc6ce57b9734aa57e91e5ac",
            "size": 28_431_843_583,
            "destination": root / "Qwen-Rapid-AIO-NSFW-v19.safetensors",
        },
        "avatar_rapid_aio_publisher_readme": {
            "label": "Rapid AIO publisher notes",
            "url": f"{source_page}/resolve/{RAPID_REVISION}/README.md",
            "source_page": source_page,
            "sha256": "076dd4d4c35e60890088f3b31b59c5b5e80c7e796bdc8af349206eda885c7c46",
            "size": 7845,
            "destination": root / "README.md",
        },
        "avatar_rapid_aio_qwen_license": {
            "label": "Qwen Image Apache 2.0 license",
            "url": f"https://raw.githubusercontent.com/QwenLM/Qwen-Image/{QWEN_CODE_REVISION}/LICENSE",
            "source_page": "https://github.com/QwenLM/Qwen-Image",
            "sha256": "b907c8234b565e5ecefff8232dbb6f1d603fbf9a47a3176db734dcae27175d08",
            "size": 11544,
            "destination": root / "LICENSE-Qwen-Image",
        },
    }
    for filename, sha256, size in _SUPPORT_FILES:
        key = filename.replace("/", "_").replace(".", "_")
        files[f"avatar_rapid_aio_config_{key}"] = {
            "label": f"Rapid AIO {filename}",
            "url": f"https://huggingface.co/{QWEN_REPO}/resolve/{QWEN_REVISION}/{filename}",
            "source_page": source_page,
            "sha256": sha256,
            "size": size,
            "destination": root / "config" / filename,
        }
    return files

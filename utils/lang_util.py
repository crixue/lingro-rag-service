from typing import Literal


def lang_def_name (is_chinese_text: bool) -> Literal["cn", "en"]:
    return "cn" if is_chinese_text else "en"

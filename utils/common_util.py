import hashlib
import datetime
import os
from functools import lru_cache
from typing import Optional

import pytz
from datetime import datetime

from lingua import LanguageDetectorBuilder, Language
from tiktoken import get_encoding

from configs.model_config import logger


def get_md5(some_str: str) -> str:
    m = hashlib.md5()
    m.update(some_str)
    return m.hexdigest()


def random_string(rand_len: int) -> str:
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=rand_len))


def remove_none_value_from_dict(some_dict: dict) -> dict:
    return {k: v for k, v in some_dict.items() if v is not None}

def get_beijing_current_time() -> datetime:
    return datetime.now(pytz.timezone('Asia/Shanghai'))

@lru_cache(maxsize=4)
def _get_cached_encoding(encoding_name: str):
    """缓存编码器实例以提高性能"""
    return get_encoding(encoding_name)

def count_tokens(text: str,
                 encoding_name: str = "cl100k_base",
                 max_length: Optional[int] = None
                 ) -> int:
    """
    Returns the number of tokens in a text string.
    """
    if not text:
        return 0

    try:
        # 如果文本过长，先截断以提高性能
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters for token counting")

        encoding = _get_cached_encoding(encoding_name)
        # 使用 disallowed_special=() 避免特殊字符导致的编码错误
        num_tokens = len(encoding.encode(text, disallowed_special=()))
        return num_tokens
    except Exception as e:
        logger.error(f"Failed to count tokens: {e}, input type: {type(text)}")
        # 降级方案：粗略估算（平均每个 token 约 4 个字符）
        return len(text) // 4

def contains_word(target_seq, detect_words):
    return (' ' + detect_words + ' ') in (' ' + target_seq + ' ')

def convert_file_encoding(input_path, output_path, from_codepage, to_codepage):
    command = f'iconv -f {from_codepage} -t {to_codepage} {input_path} > {output_path}'
    try:
        # 使用Popen替换子进程
        import subprocess
        process = subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        if process.returncode != 0:
            if process.stderr is not None:
                logger.error(f"Iconv command execution returned an error with code {process.returncode}: {process.stderr.decode('utf-8')}")
            else:
                logger.error(f"Iconv command execution returned an error with code {process.returncode}")
            raise Exception(f"Iconv command execution returned an error with code {process.returncode}")
        logger.info(f"File {input_path} has been converted from {from_codepage} to {to_codepage} and saved to {output_path}")
    except FileNotFoundError as e:
        logger.exception(f"{e}, iconv not found. Please make sure it's installed and accessible.")
        raise e
    except Exception as e:
        logger.exception(f"Error in convert_file_encoding: {e}")
        raise e

def detect_encoding_and_convert_file(file_path: str, encoding_name: str = "utf-8") -> str:
    """
    Detects the encoding of a file and converts it to the specified encoding.
    """
    def require_to_detected_files():
        return [".txt", ".doc", ".docx",".rtf","xls", ".xlsx", ".xlsm", ".csv", ".log", ".srt"]

    os_path, file_extension = os.path.splitext(file_path)
    if file_extension not in require_to_detected_files():
        logger.info(f"File {file_path} does not require encoding detection.")
        return file_path

    try:
        with open(file_path, "rb") as f:
            # Read the most 2048 bytes of the file
            raw_data = f.read(4096)
    except Exception as e:
        logger.error(f"Failed to open file: {file_path}, error: {e}")
        raise e
    # 使用chardet猜测字符集
    import chardet
    detected_encoding = chardet.detect(raw_data).get('encoding', 'utf-8')
    if detected_encoding is None:
        detected_encoding = "utf-8"
    if detected_encoding != encoding_name:
        logger.info(f"Detected file: {file_path} encoding: {detected_encoding}")
        # 如果不是UTF-8，尝试转换
        try:
            if detected_encoding.lower() == "gb2312":
                detected_encoding = "gb18030"
            utf8_file_path = file_path.split(".")[0] + "_utf8." + file_path.split(".")[1]
            convert_file_encoding(file_path, utf8_file_path, detected_encoding, encoding_name)
            return utf8_file_path
        except Exception as e:
            logger.warning(f"Failed to convert file: {file_path} from {detected_encoding} to {encoding_name}: {e}")
            # If conversion fails, return the original file path
            return file_path

    return file_path

def get_file_name_from_path(full_path: str) -> str:
    return os.path.basename(full_path)


detector = LanguageDetectorBuilder.from_languages(
    Language.CHINESE, Language.ENGLISH
).build()
def detect_if_is_chinese_text(text: str) -> bool:
    """
    检测文本中是否包含中文字符
    """
    if not text or len(text.strip()) == 0:
        return False

    detected_language = detector.detect_language_of(text)
    if not detected_language:
        return False

    if Language.CHINESE == detected_language:
        return True

    return False

def format_template_by_jinja2(template_str: str, **kwargs) -> str:
    """
    使用Jinja2模板引擎格式化字符串
    """
    from jinja2 import Template, StrictUndefined, exceptions

    try:
        template = Template(template_str, undefined=StrictUndefined)
        rendered_str = template.render(**kwargs)
        return rendered_str
    except exceptions.UndefinedError as e:
        logger.error(f"Jinja2 template rendering error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during Jinja2 template rendering: {e}")
        raise e

if __name__ == '__main__':
    # Case 1: 简单变量替换
    try:
        tpl1 = "Hello {{ name }}!"
        res1 = format_template_by_jinja2(tpl1, name="World")
        print("Case 1:", res1)
    except Exception as e:
        print("Case 1 Error:", type(e), e)

    # Case 2: 条件判断
    try:
        tpl2 = "Value is {% if value > 10 %}big{% else %}small{% endif %}"
        res2 = format_template_by_jinja2(tpl2, value=15)
        print("Case 2:", res2)
    except Exception as e:
        print("Case 2 Error:", type(e), e)

    # Case 3: 列表迭代
    try:
        tpl3 = "List: {% for item in items %}{{ item }} {% endfor %}"
        res3 = format_template_by_jinja2(tpl3, items=['a', 'b', 'c'])
        print("Case 3:", res3)
    except Exception as e:
        print("Case 3 Error:", type(e), e)

    # Case 4: 未定义变量（应触发 StrictUndefined -> 异常）
    try:
        tpl4 = "Missing: {{ missing_var }}"
        res4 = format_template_by_jinja2(tpl4)  # 故意不传 missing_var
        print("Case 4:", res4)
    except Exception as e:
        print("Case 4 Expected Error:", type(e), e)

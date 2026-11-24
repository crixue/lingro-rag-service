from utils.common_util import count_tokens


def test_count_tokens():
    text = "这是一个测试文本，用于计算token数量。"
    num_tokens = count_tokens(text)
    print(f"\"{text}\"中的token数量为{num_tokens}个。")


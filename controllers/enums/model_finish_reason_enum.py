from enum import Enum


class ModelFinishReasonEnum(Enum):
    stop = 'stop',
    length = 'length',
    tool_calls = 'tool_calls',
    sensitive = 'sensitive',
    network_error = 'network_error',


if __name__ == '__main__':
    for item in ModelFinishReasonEnum.__members__:
        print(item)
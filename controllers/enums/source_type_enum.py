from enum import Enum


class SourceTypeEnum(Enum):
    UPLOAD_FILES = 'upload_files'
    PLAIN_TEXT = 'plain_text'
    SPEECH_RECOGNIZE_TRANSCRIPT = 'speech_recognize_transcript'
    WEB_SEARCH = 'web_search'
    URLS = 'urls'

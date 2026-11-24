import re

from configs.model_config import logger

PARSE_INFO_KEY_PHRASES = "phrases"
PARSE_INFO_KEY_LIMITATION = "limitation"
PARSE_INFO_KEY_SYNTAX = "syntax"


def parse_query(query: str, other_info: dict[str, str] = None) -> dict[str, list[str]]:
    """
    Parse the query string.
    Args:
        query: The query string
        other_info: Other information to include in the parsed info. key is the pattern to search for, value is the name of the key in the parsed info.
        return: Dictionary of query strings, contains:
          "phrases": List of phrases in the query string;
          "limitation": List of limitations in the query string;
          "syntax": List of syntax in the query string.
    """
    logger.debug(f"[parse_query]-Raw Query: {query}")
    phrases = []
    limitations = []
    syntax = []

    if "|" in query:
        phrases.extend(query.split("|"))
    if "<SEP>" in query:
        phrases.extend(re.findall(r'<SEP>(.*?)</SEP>', query))
    # Remove the <SEP> tag, avoid the unexpected tag in the phrases
    for p in phrases:
        p.replace("<SEP>", "")
        p.replace("</SEP>", "")
        p.strip()

    limitations.extend(re.findall(r'<LIMIT>(.*?)</LIMIT>', query))
    limitations.extend(re.findall(r'<LIMIT>(.*?)<LIMIT>', query))
    for l in limitations:
        l.replace("<LIMIT>", "")
        l.replace("</LIMIT>", "")

    syntax.extend(re.findall(r'<SYNTAX>(.*?)</SYNTAX>', query))
    syntax.extend(re.findall(r'<SYNTAX>(.*?)<SYNTAX>', query))
    for s in syntax:
        s.replace("<SYNTAX>", "")
        s.replace("</SYNTAX>", "")

    parsed_info = {
        PARSE_INFO_KEY_PHRASES: phrases,
        PARSE_INFO_KEY_LIMITATION: limitations,
        PARSE_INFO_KEY_SYNTAX: syntax
    }

    if other_info:
        for key, value in other_info.items():
            parsed_info[value] = re.findall(key, query)

    return parsed_info
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field

from configs.model_config import logger, SERVER_TIMEZONE


class GetCurrentDateTimeInput(BaseModel):
    """Input schema for getting current date and time."""
    timezone: str = Field(
        default=SERVER_TIMEZONE,
        description="The timezone to get the current date and time for. Default is the server's timezone.",
    )


@tool(
    name_or_callable="get_current_date_time_tool",
    description=(
        "Get the current date and time in the specified timezone. "
        "Returns the current date, time, day of week, and timezone information. "
        "By default, it uses the server's configured timezone. "
        "Use this tool when you need to know what time it is now or what date it is today."
    ),
    args_schema=GetCurrentDateTimeInput,
    response_format="content_and_artifact",
)
def get_current_date_time_tool(
        runtime: Annotated[ToolRuntime, InjectedToolArg],
        timezone: Annotated[
            str,
            Field(
                default=SERVER_TIMEZONE,
                description="The timezone to get the current date and time for.",
            ),
        ] = SERVER_TIMEZONE,
) -> tuple[str, dict[str, Any]]:
    """
    获取指定时区的当前日期和时间。

    此工具返回当前的日期、时间、星期几和时区信息。
    默认使用服务器配置的时区。

    Args:
        timezone: 时区名称（例如：Asia/Shanghai, UTC, America/New_York）
        runtime: 运行时配置（自动注入）

    Returns:
        元组 (content_string, artifact_dict)，其中：
        - content_string: 格式化的日期时间信息
        - artifact_dict: 包含详细时间信息的字典

    Examples:
        >>> result = get_current_date_time_tool.invoke({})
        >>> content, artifact = result
        >>> print(content)
        Current Date and Time:
        Date: 2024-01-15
        Time: 14:30:45
        ...
    """
    user_id = runtime.context.user_id if runtime.context else "unknown"

    logger.info(
        f"[get_current_date_time_tool] Getting current date and time for timezone='{timezone}', user_id='{user_id}'"
    )

    try:
        # 获取指定时区的当前时间
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        # 格式化时间信息
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_names_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        # 构建 artifact
        artifact = {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": weekday_names[now.weekday()],
            "weekday_cn": weekday_names_cn[now.weekday()],
            "timezone": timezone,
            "utc_offset": now.strftime("%z"),
        }

        # 构建描述性返回信息
        content = (
            f"Current Date and Time:\n"
            f"Date: {artifact['date']}\n"
            f"Time: {artifact['time']}\n"
            f"Day of Week: {artifact['weekday']} ({artifact['weekday_cn']})\n"
            f"Timezone: {timezone} (UTC{artifact['utc_offset']})\n"
            f"ISO Format: {artifact['datetime']}"
        )

        logger.info(
            f"[get_current_date_time_tool] Successfully retrieved current time: {artifact['datetime']}"
        )

        return content, artifact

    except Exception as e:
        error_msg = f"Failed to get current date and time: {str(e)}"
        logger.exception(f"[get_current_date_time_tool] {error_msg}")
        # 返回错误信息而不是抛出异常，让大模型能够理解发生了什么
        return error_msg, {"error": str(e), "timezone": timezone}


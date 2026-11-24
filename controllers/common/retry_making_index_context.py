import functools
import glob
import json
import os
import pickle
import traceback
import time
from typing import Any

from configs.model_config import MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR, CREATE_INDEX_RETRYABLE_CONTEXT_FILE_DIR_NAME, \
    logger
from exceptions.custom_exception import RetryableStepFailedException

def after_precompute():
    pass

def retryable_step(steps_enum, step):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.retryable_context.can_run_this_step(step):
                logger.warning(f"[retryable_step]-Step {step} pass.")
                return None
            retry_times = 2
            while retry_times > 0:
                try:
                    self.retryable_context.set_other_params(**{
                        "current_step": step,
                        "is_running_failed": False,
                        "request_params": self.retryable_request_params
                    })
                    if (self.retryable_context_info and isinstance(self.retryable_context_info, dict) and
                            self.retryable_context.is_triggered_by_retry_task):
                        kwargs.update(self.retryable_context_info)
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    logger.exception(f"[retryable_step]-Error occurred when set other params: {e}. Try again...")
                    retry_times -= 1
                    if retry_times <= 0:
                        self.retryable_context.is_running_failed = True
                        self.retryable_context.save_failed_context_info(context_info=self.retryable_context_info,
                                                                        error_msg=traceback.format_exc())
                        raise RetryableStepFailedException(
                            message=f"[retryable_step]-Error occurred when set other params: {e}",
                            failed_status=-10 * (len(steps_enum.__members__) - steps_enum[step].value) - 900
                        )
                    time.sleep(1 * (3 - retry_times))
            # if execute the last step and not failed, then delete the context info
            if (steps_enum[step].value == self.retryable_context.total_steps  - 1 and
                    not self.retryable_context.is_running_failed):
                self.retryable_context.delete_context_info()

        return wrapper

    return decorator

class CreateIndexRetryableContext:
    def __init__(self, steps_enum, current_step: str = None, is_running_failed: bool = False,
                 is_triggered_by_retry_task: bool = False, *, id: str = None, user_folder_id: str = None,
                 **kwargs):
        self.steps_enum = steps_enum
        base_retryable_saved_dir = MAKE_KB_RETRYABLE_CONTEXT_FILE_DIR
        self.id = id
        self.current_step = current_step or steps_enum.INITIATE.name
        self.is_running_failed = is_running_failed
        self.error_msg = kwargs.get("error_msg", None)
        self.request_params = kwargs.get("request_params", {})
        knowledge_index_folder_id = user_folder_id if user_folder_id else f"{self.id}"
        self.retryable_context_dir = os.path.join(base_retryable_saved_dir,
                                                  knowledge_index_folder_id,
                                                  CREATE_INDEX_RETRYABLE_CONTEXT_FILE_DIR_NAME)
        self.retryable_context_info_filepath = os.path.join(self.retryable_context_dir,
                                                            f"{self.id}_retryable_context_info.p")
        if not os.path.exists(self.retryable_context_dir):
            os.makedirs(self.retryable_context_dir)
        self.total_steps = kwargs.get("total_steps", len(steps_enum.__members__))
        self.is_triggered_by_retry_task = is_triggered_by_retry_task

    def set_other_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def can_run_this_step(self, step: str):
        if self.is_running_failed:
            if self.steps_enum[step].value != self.steps_enum[self.current_step].value:
                return False
            return True
        else:
            if self.steps_enum[step].value >= self.steps_enum[self.current_step].value:
                return True
            return False

    def save_failed_context_info(self, context_info: Any, error_msg: str):
        if context_info:
            with open(self.retryable_context_info_filepath, "wb") as f:
                pickle.dump(context_info, f)

        self.is_running_failed = True
        self.error_msg = error_msg

        retryable_context_file = os.path.join(self.retryable_context_dir, f"{self.id}_retryable_context.json")
        with open(retryable_context_file, "w") as f:
            context_dict = self.__dict__.copy()
            context_dict.pop("steps_enum", None)
            f.write(json.dumps(context_dict, indent=2, ensure_ascii=False))
            logger.info(
                f"[save_failed_context_info]-Save failed context info for {self.id} to {retryable_context_file} successfully.")

    def delete_context_info(self):
        if os.path.exists(self.retryable_context_dir):
            files = glob.glob(os.path.join(self.retryable_context_dir, f"{self.id}*"))
            for f in files:
                os.remove(f)
                logger.info(f"[delete_context_info]-Delete file: {f} successfully.")
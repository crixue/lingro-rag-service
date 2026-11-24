
class NotHaveEnoughMoneyException(Exception):
    def __init__(self):
        message = "Not have enough money, can not use this service."
        super().__init__(message)
        self.message = message


class RetryableStepFailedException(Exception):
    def __init__(self, message: str = None, failed_status: int = -1000):
        super().__init__(message)
        self.message = message
        self.failed_status = failed_status

    @property
    def status(self):
        return self.failed_status


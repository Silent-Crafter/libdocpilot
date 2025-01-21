from typing import Literal, Optional, Union

class NotALogger:
    
    def __init__(self, module: str):
        self.enable = True
        self.to_file = False
        self.log_file = None
        self.module = module

    def log(self, message: str, log_type: Union[Literal['info', 'error'], str] = "info"):
        if not self.enable:
            return

        handlers = {
            'info': self.info,
            'error': self.error,
        }

        handlers.get(log_type, self._print_custom(log_type))(message, self.module)

    def info(self, message: str, to_file: Optional[bool] = None, *args, **kwargs):
        log_to_file = to_file if to_file is not None else self.to_file
        if log_to_file:
            self._log_file(message, "info")

        if self.enable: self._print_info(message, self.module)

    def error(self, message: str, to_file: Optional[bool] = None, *args, **kwargs):
        log_to_file = to_file if to_file is not None else self.to_file
        if log_to_file:
            self._log_file(message, "info")

        if self.enable: self._print_error(message, self.module)

    def _log_file(self, message: str, log_type: Union[Literal['info', 'error'], str] = "info"):
        raise NotImplementedError

    @staticmethod
    def _print_error(message: str, from_: str):
        print(f"\033[31m[ERR] {from_}: {message}\033[0m")

    @staticmethod
    def _print_info(message: str, from_: str):
        print(f"\033[34m[INFO] {from_}: {message}\033[0m")

    @staticmethod
    def _print_custom(log_type: str):
        def _print_custom(message: str, from_: str):
            print(f"\033[31m[{log_type}] {from_}: {message}\033[0m")

        return _print_custom

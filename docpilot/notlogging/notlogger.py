import textwrap
from typing import Literal, Optional, Union, List

class NotALogger:

    modules: dict[str, 'NotALogger'] = {}
    
    def __init__(self, module: str):
        self._enable = True
        self.to_file = False
        self.log_file = None
        self.module = module

        self.modules[self.module] = self

    def log(self, message: str, log_type: Union[Literal['info', 'error'], str] = "info"):
        if not self._enable:
            return

        handlers = {
            'info': self.info,
            'error': self.error,
        }

        handlers.get(log_type, self._print_custom(log_type))(message, self.module)

    @property
    def enabled(self):
        return self._enable

    @enabled.setter
    def enabled(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("'is_enabled' must be a boolean")

        self._enable = value

    def info(self, message: Union[str, List[str]], to_file: Optional[bool] = None, wrap: bool = True, *args, **kwargs):
        if not self._enable:
            return

        log_to_file = to_file if to_file is not None else self.to_file

        if wrap and isinstance(message, str):
            message = textwrap.wrap(message, 100, replace_whitespace=False, drop_whitespace=False)
            message = [message[0]] + list(map(lambda x: '\t'+x, message[1:]))

        if log_to_file:
            self._log_file(message, "info")

        if isinstance(message, str):
            self._print_info(message, self.module)
            return

        if isinstance(message, list):
            for msg in message:
                self._print_info(msg, self.module)

    def error(self, message: str, to_file: Optional[bool] = None, *args, **kwargs):
        log_to_file = to_file if to_file is not None else self.to_file
        if log_to_file:
            self._log_file(message, "info")

        if self._enable: self._print_error(message, self.module)

    def _log_file(self, message: Union[str, List[str]], log_type: Union[Literal['info', 'error'], str] = "info"):
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

    def __str__(self):
        return str(self.enabled)

    def __repr__(self):
        return str(self.enabled)

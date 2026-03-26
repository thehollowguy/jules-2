import ctypes
from ctypes import c_char_p, c_uint32


class Jules:
    def __init__(self, lib_path: str = "libjules.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.jules_version.restype = c_uint32
        self.lib.jules_run_file_ffi.argtypes = [c_char_p]
        self.lib.jules_run_file_ffi.restype = ctypes.c_uint
        self.lib.jules_check_code_ffi.argtypes = [c_char_p]
        self.lib.jules_check_code_ffi.restype = ctypes.c_uint
        self.lib.jules_error_string.argtypes = [ctypes.c_uint]
        self.lib.jules_error_string.restype = c_char_p

    def version(self) -> int:
        return int(self.lib.jules_version())

    def run_file(self, path: str) -> int:
        return int(self.lib.jules_run_file_ffi(path.encode("utf-8")))

    def check_code(self, source: str) -> int:
        return int(self.lib.jules_check_code_ffi(source.encode("utf-8")))

    def error_string(self, code: int) -> str:
        msg = self.lib.jules_error_string(code)
        return msg.decode("utf-8") if msg else "Unknown"

"""Компьютер, на котором идёт обучение: процессор, ядра, память."""
import functools
import platform

import psutil


@functools.cache
def info():
    """Имя процессора, ядра и потоки, память в ГБ."""
    return {
        "cpu": _cpu_name(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(),
        "ram": round(psutil.virtual_memory().total / 2**30),
    }


def free():
    """Свободная память в ГБ — обучению нужно около 6."""
    return round(psutil.virtual_memory().available / 2**30, 1)


def _cpu_name():
    """Имя из реестра Windows («AMD Ryzen 9 7940HS»); platform.processor() даёт только «AMD64 Family 25…»."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except (ImportError, OSError):
        name = platform.processor()
    return name.split(" w/ ")[0].strip()

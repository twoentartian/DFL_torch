import logging
import sys

def set_logging(log_file_path: str, base_logger):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(name)s] --- %(message)s (%(filename)s:%(lineno)s)")

    file = logging.FileHandler(log_file_path)
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    base_logger.setLevel(logging.DEBUG)
    base_logger.addHandler(file)
    base_logger.addHandler(console)
    base_logger.addHandler(ExitOnExceptionHandler())
    base_logger.info("logging setup complete")

    del file, console, formatter

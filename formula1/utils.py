from functools import wraps
import logging
import datetime as dt

def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        str_length = 18
        func_name = func.__name__[:str_length]
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        logger.info(f"[{func_name: <{str_length}}] shape={result.shape},  time={time_taken}")
        return result
    return wrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger()


def init_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger()

    return logger

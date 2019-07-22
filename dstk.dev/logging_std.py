from collections import defaultdict, Counter
from contextlib import contextmanager
import logging


class MassMessageFilter(logging.Filter):
    def __init__(self, max_num_message=3):
        self.message_value = defaultdict(list)
        self.max_num_message = max_num_message

    def filter(self, record):
        args = record.args
        msg = record.msg
        self.message_value[msg].append(args)

        num_msg = len(self.message_value[msg])
        if num_msg < self.max_num_message:
            return True
        elif num_msg == self.max_num_message:
            record.msg = record.msg + " (omitting further messages since {}x exceeded)".format(self.max_num_message)
            return True
        else:
            return False

    def values(self, msg):
        return self.message_value[msg]

    def get_report(self, msg):
        if self.message_value[msg]:
            values = self.values(msg)
            counter = Counter(values)
            most_common_elem, most_common_count = counter.most_common(1)[0]
            return "Summary: '{}' came {} times with {} distinct elements and most commonly {}x {}".format(msg,
                                                                                                           len(values),
                                                                                                           len(set(
                                                                                                               values)),
                                                                                                           most_common_count,
                                                                                                           ",".join(
                                                                                                               repr(s)
                                                                                                               for s in
                                                                                                               most_common_elem))
        else:
            return ""


@contextmanager
def error_trapping(msg, exc_info=True):
    try:
        yield None
    except Exception:
        logging.error(msg, exc_info=exc_info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")

    with error_trapping("ABC"):
        print("A")
        raise Exception("EXC")
    filter = MassMessageFilter()
    logger.addFilter(filter)
    for i in range(10):
        logger.info("Message %s", i)
    logger.info("Message %s", 8)
    logger.info(filter.get_report("Message %s"))
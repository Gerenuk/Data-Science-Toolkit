
import logging
import time


def format_args(args, kwargs):
    return "({}{})".format(
        ", ".join(map(str, args)),
        ", " + ", ".join("{}={}".format(k, v) for k, v in kwargs.items())
        if kwargs
        else "",
    )


class Experiment:
    def __init__(self, filename):
        self.existing_runs = []

        try:
            run_db_file = open(filename, "rb")
            try:
                while 1:
                    self.existing_runs.append(pickle.load(run_db_file))
            except EOFError:
                pass
            logging.info(
                "Loaded run database {} with {} entries".format(
                    filename, len(self.existing_runs)
                )
            )
        except FileNotFoundError:
            pass

        self.run_db_file_writing = open(filename, "ab")
        self.run_count = 0
        self.last_run_skipped = False

    def __call__(self, func):
        def exp_runner(*args, **kwargs):
            for exist_args, exist_kwargs, exist_result, *_ in self.existing_runs:
                if (exist_args, exist_kwargs) == (args, kwargs):
                    logging.info(
                        "Skipping run {} and returning existing result".format(
                            format_args(args, kwargs)
                        )
                    )
                    self.last_run_skipped = True
                    return exist_result

            self.last_run_skipped = False

            self.run_count += 1
            logging.info(
                "Running run #{} {}".format(self.run_count, format_args(args, kwargs))
            )

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            self.existing_runs.append((args, kwargs, result, duration))
            pickle.dump((args, kwargs, result, duration), self.run_db_file_writing)

            logging.info(
                "Run #{} {} finished after {}sec. Storing in database at position {}.".format(
                    self.run_count,
                    format_args(args, kwargs),
                    round(duration),
                    len(self.existing_runs) - 1,
                )
            )
            return result

        return exp_runner

    def close(self):
        self.run_db_file_writing.close()

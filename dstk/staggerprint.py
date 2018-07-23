from __future__ import print_function
import time
import sys

class StaggerPrint(object):
    def __init__(self, iorate = 1000, max_line_len=None):
        """ Protect IPython Notebooks from large outputs that would
        freeze your browser.
        iorate: # of prints per second allowed
        max_line_len: Truncate lines in case those are big too
        """
        self.print = print
        self.iorate = iorate
        self.max_line_len = max_line_len
        self.print_times = []
        self.char_counts = []
        self.hiatus = False
        self.last_called = 0
        return
    
    def purge_old_records(self, now):
        # Remove any print times and char counts older than 1 second:
        """ Remove any print times and character counts older than one second."""
        num_print_times = len(self.print_times)
        for i in xrange(num_print_times):
            pt = self.print_times[num_print_times - i - 1]
            if now - pt > 1:
                self.print_times.pop(num_print_times - i - 1)
                self.char_counts.pop(num_print_times - i - 1)
                
    def __call__(self, *args, **kwargs):
        """ Print input. Check if print rate has been exceeded and truncate output """
        
        now = time.time()
        if self.hiatus:
            if now - self.last_called > 1:
                self.hiatus = False
            else:
                return
        
        # Prepare output
        output = ""
        output = ' '.join(map(str, args))
        output_length = len(output)
        time_since_last_print = now - self.last_called
        self.print_times.append(now)
        self.char_counts.append(output_length)
        
        # Remove any print times and char counts older than 1 second
        self.purge_old_records(now)
        
        # Limit to "iorate" prints per second.
        if len(self.print_times) > self.iorate:
            self.hiatus = True
            self.print_times = []
            self.char_counts = []
            self.last_called = now
            self.print("... stdout truncated for one second...")
            sys.stdout.flush()
            return
        self.last_called = now
        
        # Let's push that string goodness to the screen.
        if self.max_line_len:
            self.print(output[:max_line_len])
        else:
            self.print(output)
        sys.stdout.flush()

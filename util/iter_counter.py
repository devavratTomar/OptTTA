import os
import numpy as np
import torch
import time


class IterationCounter():
    def __init__(self, opt):
        self.opt = opt
        self.iter_record_path = os.path.join(
            self.opt.checkpoints_dir, 'iter.txt')
        self.steps_so_far = 0
        self.batch_size = opt.batch_size
        self.time_measurements = {}

        automatically_find_resume_iter = opt.continue_train
        if automatically_find_resume_iter:
            try:
                self.steps_so_far = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
                print('Resuming from iteration %d' % (self.steps_so_far))
            except Exception:
                print('Could not load iteration record at %s. '
                      'Starting from beginning.' % self.iter_record_path)
        else:
            self.steps_so_far = 0

    def record_one_iteration(self):
        if self.needs_saving():
            np.savetxt(self.iter_record_path,
                       [self.steps_so_far], delimiter=',', fmt='%d')
            print("Saved current iter count at %s" % self.iter_record_path)
        self.steps_so_far += self.batch_size

    def needs_saving(self):
        return (self.steps_so_far % self.opt.save_freq) < self.batch_size

    def needs_evaluation(self):
        return (self.steps_so_far >= self.opt.evaluation_freq) and \
            ((self.steps_so_far % self.opt.evaluation_freq) < self.batch_size)

    def needs_printing(self):
        return (self.steps_so_far % self.opt.print_freq) < self.batch_size

    def needs_displaying(self):
        return (self.steps_so_far % self.opt.display_freq) < self.batch_size

    def completed_training(self):
        return (self.steps_so_far >= self.opt.total_nimgs)
    
    def is_last_iter(self):
        return self.steps_so_far + self.batch_size >= self.opt.total_nimgs

    class TimeMeasurement:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, type, value, traceback):
            torch.cuda.synchronize()
            start_time = self.start_time
            elapsed_time = (time.time() - start_time) / self.parent.batch_size

            if self.name not in self.parent.time_measurements:
                self.parent.time_measurements[self.name] = elapsed_time
            else:
                prev_time = self.parent.time_measurements[self.name]
                updated_time = prev_time * 0.98 + elapsed_time * 0.02
                self.parent.time_measurements[self.name] = updated_time

    def time_measurement(self, name):
        return IterationCounter.TimeMeasurement(name, self)

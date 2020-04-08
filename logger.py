from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

class Logger:
    def __init__(self):
        self.writer = SummaryWriter(flush_secs=0.5)

    def log(self, message):
        print(f'INFO: {datetime.now()}: {message}')

    def __call__(self, **kwargs):
        message = ""
        for key in kwargs:
            message += f"{key} : {kwargs[key]} "
            self.writer.add_scalar(key, kwargs[key], kwargs["Log iter"])

        self.log(message)
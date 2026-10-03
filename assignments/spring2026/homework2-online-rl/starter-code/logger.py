import csv
import datetime
from collections import defaultdict

import numpy as np
import torch
import torchvision
from termcolor import colored
import wandb

COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('episode_success', 'SR', 'float'),
                      ('total_time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        """Initialize an online average accumulator."""
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        """Add a new observation, optionally weighted by a count."""
        self._sum += value
        self._count += n

    def value(self):
        """Return the current mean, guarding against division by zero."""
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        """Track a named collection of meters and write them to CSV/console."""
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        """Update one named meter inside the group."""
        self._meters[key].update(value, n)

    def _prime_meters(self):
        """Normalize meter names and collect their averaged scalar values."""
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            elif key.startswith('actor'):
                key = key[len('actor') + 1:]
            elif key.startswith('critic'):
                key = key[len('critic') + 1:]
            elif key.startswith('pretrain'):
                key = key[len('pretrain') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        """Drop stale CSV rows when resuming from an earlier episode index."""
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        """Append one row to the CSV file, rebuilding the writer if needed."""
        def _init_csv_writer():
            should_write_header = True
            if self._csv_file_name.exists():
                # Check if existing CSV has compatible fieldnames
                with self._csv_file_name.open('r') as f:
                    reader = csv.DictReader(f)
                    existing_fields = set(reader.fieldnames or [])
                new_fields = set(data.keys())
                
                if new_fields == existing_fields:
                    # Compatible schema: remove stale entries and append
                    self._remove_old_entries(data)
                    should_write_header = False
                else:
                    # Schema mismatch: start fresh
                    self._csv_file_name.unlink()

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                            fieldnames=sorted(data.keys()),
                                            restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        if self._csv_writer is None:
            _init_csv_writer()

        try:
            self._csv_writer.writerow(data)
            self._csv_file.flush()
        except OSError:
            # Networked filesystems can return stale-handle errors on long runs.
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            _init_csv_writer()
            self._csv_writer.writerow(data)
            self._csv_file.flush()

    def _format(self, key, value, ty):
        """Render one metric according to the configured display type."""
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise ValueError(f'invalid format type: {ty}')

    def _dump_to_console(self, data, prefix):
        """Print one formatted metric row to stdout."""
        if prefix == 'train':
            color = 'yellow'
        else:
            color = 'green'
        prefix = colored(prefix, color)
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        """Flush accumulated metrics to disk and console, then clear the group."""
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_wandb=False, wandb_project=None, wandb_entity=None, wandb_group=None, cfg=None):
        """Create per-stream metric groups and optionally initialize Weights & Biases."""
        self._log_dir = log_dir
        self._pretrain_mg = MetersGroup(log_dir / 'pretrain.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._actor_mg = MetersGroup(log_dir / 'actor.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._critic_mg = MetersGroup(log_dir / 'critic.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT)

        if use_wandb:
            wandb.init(project=wandb_project, entity=wandb_entity, group=wandb_group)
        self._use_wandb = use_wandb

    def log(self, key, value, step):
        """Record one scalar metric in the appropriate meter group and backend."""
        assert key.startswith('train') or key.startswith('actor') or key.startswith('critic') or key.startswith('eval') or key.startswith('pretrain')
        if type(value) == torch.Tensor:
            value = value.item()
        if self._use_wandb:
            wandb.log({key: value}, step=step)
        if key.startswith('train'):
            mg = self._train_mg
        elif key.startswith('actor'):
            mg = self._actor_mg
        elif key.startswith('critic'):
            mg = self._critic_mg
        elif key.startswith('eval'):
            mg = self._eval_mg
        else:
            mg = self._pretrain_mg

        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        """Log every metric in a dictionary under the provided metric namespace."""
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        """Flush one or more meter groups depending on the requested stream type."""
        if ty == 'train':
            self._train_mg.dump(step, 'train')
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty == 'critic':
            self._critic_mg.dump(step, 'critic')
        if ty == 'actor':
            self._actor_mg.dump(step, 'actor')
        if ty == 'pretrain':
            self._pretrain_mg.dump(step, 'pretrain')

    def log_and_dump_ctx(self, step, ty):
        """Return a context manager that logs keys and flushes on exit."""
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        """Capture the logger, step, and stream for a scoped logging block."""
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        """Enter the scoped logging context."""
        return self

    def __call__(self, key, value):
        """Log one key/value pair using the context's metric prefix."""
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        """Flush the scoped metric stream when leaving the context."""
        self._logger.dump(self._step, self._ty)

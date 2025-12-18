## RLHF Run Diagnostic Report

Generated: 2025-12-18T00:48:25.764370Z

**Trainer:** DPO | **Status:** Crashed (exception)

### Run Summary
- Steps: 15
- Final Reward Mean: 0.000

- Final DPO Loss: 0.6931
- Mean Win Rate: 61.7%
- Final Win Rate: 75.0%





### Key Insights


1. [HIGH] DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.
   *Ref: Rafailov et al. (2023) 'DPO', Section 4.2 - Loss at ln(2) indicates no preference signal*

2. [MEDIUM] Win rate shows high volatility (std=0.53), indicating inconsistent preference learning.








### Postmortem
**Exit Reason:** exception
- Last Step: 15
- Timestamp: 2025-12-17T19:26:04.880887+00:00


Traceback (truncated):
```
Traceback (most recent call last):
  File "/Users/adityachallapally/Library/CloudStorage/OneDrive-Microsoft/Coding/post-training-toolkit/demo/scripts/minimal_dpo_demo.py", line 201, in <module>
    main()
  File "/Users/adityachallapally/Library/CloudStorage/OneDrive-Microsoft/Coding/post-training-toolkit/demo/scripts/minimal_dpo_demo.py", line 168, in main
    trainer.train()
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 2325, in train
    return inner_training_loop(
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 2756, in _inner_training_loop
    self._maybe_log_save_evaluate(
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 3221, in _maybe_log_save_evaluate
    metrics = self._evaluate(trial, ignore_keys_for_eval)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 3170, in _evaluate
    metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 4513, in evaluate
    self.log(output.metrics)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/trl/trainer/dpo_trainer.py", line 1995, in log
    return super().log(logs, start_time)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer.py", line 3793, in log
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer_callback.py", line 549, in on_log
    return self.call_event("on_log", args, state, control, logs=logs)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer_callback.py", line 556, in call_event
    result = getattr(callback, event)(
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/transformers/trainer_callback.py", line 693, in on_log
    self.training_bar.write(str(shallow_logs))
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/std.py", line 720, in write
    with cls.external_write_mode(file=file, nolock=nolock):
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/contextlib.py", line 117, in __enter__
    return next(self.gen)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/std.py", line 745, in external_write_mode
    inst.clear(nolock=True)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/std.py", line 1319, in clear
    self.sp('')
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/std.py", line 459, in print_status
    fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/std.py", line 452, in fp_write
    fp.write(str(s))
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/tqdm/utils.py", line 196, in inner
    return func(*args, **kwargs)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/Users/adityachallapally/Library/Python/3.9/lib/python/site-packages/wandb/sdk/lib/console_capture.py", line 171, in write_with_callbacks
    n = orig_write(s)
BrokenPipeError: [Errno 32] Broken pipe

```



### Recommended Actions


- DPO loss at random chance: increase learning rate 2-5x, check data quality, or reduce beta.

- Win rate unstable: increase batch size for more stable gradient estimates.


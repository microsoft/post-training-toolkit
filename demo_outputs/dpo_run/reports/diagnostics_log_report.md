## RLHF Run Diagnostic Report

Generated: 2025-12-14T06:01:56.280909Z

**Trainer:** DPO | **Status:** Crashed (exception)

### Run Summary
- Steps: 24

- Final DPO Loss: 0.6931
- Mean Win Rate: 65.6%





### Key Insights


1. [HIGH] DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.
   *Ref: Rafailov et al. (2023) 'DPO', Section 4.2 - Loss at ln(2) indicates no preference signal*

2. [MEDIUM] Win rate shows high volatility (std=0.53), indicating inconsistent preference learning.

3. [LOW] DPO loss has plateaued; consider adjusting learning rate or beta.








### Postmortem
**Exit Reason:** exception
- Last Step: 20
- Timestamp: 2025-12-14T06:00:37.909739+00:00


Traceback (truncated):
```
Traceback (most recent call last):
  File "/home/avasanthc/post-training-system/examples/minimal_dpo_demo.py", line 201, in <module>
    main()
  File "/home/avasanthc/post-training-system/examples/minimal_dpo_demo.py", line 168, in main
    trainer.train()
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 2240, in train
    return inner_training_loop(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 2622, in _inner_training_loop
    self._maybe_log_save_evaluate(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 3095, in _maybe_log_save_evaluate
    metrics = self._evaluate(trial, ignore_keys_for_eval)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 3044, in _evaluate
    metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 4173, in evaluate
    output = eval_loop(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py", line 1509, in evaluation_loop
    initial_output = super().evaluation_loop(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer.py", line 4401, in evaluation_loop
    self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer_callback.py", line 552, in on_prediction_step
    return self.call_event("on_prediction_step", args, state, control)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer_callback.py", line 556, in call_event
    result = getattr(callback, event)(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/transformers/trainer_callback.py", line 659, in on_prediction_step
    self.prediction_bar = tqdm(
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/asyncio.py", line 24, in __init__
    super().__init__(iterable, *args, **kwargs)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/std.py", line 1098, in __init__
    self.refresh(lock_args=self.lock_args)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/std.py", line 1347, in refresh
    self.display()
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/std.py", line 1495, in display
    self.sp(self.__str__() if msg is None else msg)
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/std.py", line 459, in print_status
    fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/std.py", line 452, in fp_write
    fp.write(str(s))
  File "/home/avasanthc/.local/lib/python3.10/site-packages/tqdm/utils.py", line 196, in inner
    return func(*args, **kwargs)
BrokenPipeError: [Errno 32] Broken pipe

```



### Recommended Actions


- DPO loss at random chance: increase learning rate 2-5x, check data quality, or reduce beta.

- DPO loss plateaued: try learning rate warmup/decay or adjust beta parameter.

- Win rate unstable: increase batch size for more stable gradient estimates.


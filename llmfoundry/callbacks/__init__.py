# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import (EarlyStopper, EvalOutputLogging, Generate,
                                LRMonitor, MemoryMonitor, MemorySnapshot,
                                OOMObserver, OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.core import Callback, State
from composer.loggers import Logger
import torch
from transformers import AutoTokenizer
from composer.utils import dist
import numpy as np
from llmfoundry.callbacks.async_eval_callback import AsyncEval
from llmfoundry.callbacks.curriculum_learning_callback import CurriculumLearning
from llmfoundry.callbacks.eval_gauntlet_callback import EvalGauntlet
from llmfoundry.callbacks.fdiff_callback import FDiffMetrics
from llmfoundry.callbacks.hf_checkpointer import HuggingFaceCheckpointer
from llmfoundry.callbacks.log_mbmoe_tok_per_expert_callback import \
    MegaBlocksMoE_TokPerExpert
from llmfoundry.callbacks.monolithic_ckpt_callback import \
    MonolithicCheckpointSaver
from llmfoundry.callbacks.resumption_callbacks import (GlobalLRScaling,
                                                       LayerFreezing)
from llmfoundry.callbacks.scheduled_gc_callback import ScheduledGarbageCollector
from llmfoundry.registry import callbacks, callbacks_with_config

class LossSpikeCallback(Callback):
    def __init__(self, tokenizer_path: str, threshold: float = 2.0, window_size: int = 10):
        self.threshold = threshold
        self.window_size = window_size
        self.loss_history = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.rank = dist.get_global_rank()
        self.world_size = dist.get_world_size()

    def after_backward(self, state: State, logger: Logger) -> None:
        current_loss = state.loss.item()
        self.loss_history.append(current_loss)
        
        print(f"Rank {self.rank} - Current LOSS: {current_loss}")
        
        if len(self.loss_history) >= self.window_size:
            recent_losses = self.loss_history[-self.window_size:]
            mean_loss = np.mean(recent_losses[:-1])
            
            print(f"Rank {self.rank} - MEAN: {mean_loss}")
            print(f"Rank {self.rank} - threshold: {self.threshold}")
            
            if (current_loss - mean_loss) > (self.threshold * mean_loss):
                self.save_batch_data(state, mean_loss)
                
            self.loss_history = self.loss_history[-self.window_size:]

    def save_batch_data(self, state: State, mean_loss: float):
        filename = f"loss_spike_data_step_{state.timestamp.batch.value}_rank_{self.rank}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Step: {state.timestamp.batch.value}\n")
            f.write(f"Rank: {self.rank}\n")
            f.write(f"Current Loss: {state.loss.item()}\n")
            f.write(f"Mean Loss: {mean_loss}\n")
            f.write(f"Loss Difference: {state.loss.item() - mean_loss}\n")
            f.write(f"Loss Ratio: {state.loss.item() / mean_loss}\n\n")
            
            if 'input_ids' in state.batch and isinstance(state.outputs, dict) and 'logits' in state.outputs:
                inputs = state.batch['input_ids']
                outputs = state.outputs['logits']
                
                for i, (input_seq, output_seq) in enumerate(zip(inputs, outputs)):
                    decoded_input = self.tokenizer.decode(input_seq)
                    predicted_tokens = torch.argmax(output_seq, dim=-1)
                    decoded_output = self.tokenizer.decode(predicted_tokens)
                    
                    f.write(f"Sample {i}:\n")
                    f.write(f"Input:\n{decoded_input}\n")
                    f.write("\n")
                    f.write(f"Output:\n{decoded_output}\n")                                   
                    f.write("\n")

        print(f"Rank {self.rank} - Loss spike detected at step {state.timestamp.batch.value}. Batch data saved to {filename}")

    def fit_start(self, state: State, logger: Logger) -> None:
        print(f"Rank {self.rank} - Starting training with {self.world_size} GPUs")
        


callbacks.register('lr_monitor', func=LRMonitor)
callbacks.register('memory_monitor', func=MemoryMonitor)
callbacks.register('memory_snapshot', func=MemorySnapshot)
callbacks.register('speed_monitor', func=SpeedMonitor)
callbacks.register('runtime_estimator', func=RuntimeEstimator)
callbacks.register('optimizer_monitor', func=OptimizerMonitor)
callbacks.register('generate_callback', func=Generate)
callbacks.register('early_stopper', func=EarlyStopper)
callbacks.register('fdiff_metrics', func=FDiffMetrics)
callbacks.register('hf_checkpointer', func=HuggingFaceCheckpointer)
callbacks.register('global_lr_scaling', func=GlobalLRScaling)
callbacks.register('layer_freezing', func=LayerFreezing)
callbacks.register('mono_checkpoint_saver', func=MonolithicCheckpointSaver)
callbacks.register('scheduled_gc', func=ScheduledGarbageCollector)
callbacks.register('oom_observer', func=OOMObserver)
callbacks.register('eval_output_logging', func=EvalOutputLogging)
callbacks.register('mbmoe_tok_per_expert', func=MegaBlocksMoE_TokPerExpert)
callbacks.register("loss_spike", func=LossSpikeCallback)

callbacks_with_config.register('async_eval', func=AsyncEval)
callbacks_with_config.register('curriculum_learning', func=CurriculumLearning)

__all__ = [
    'FDiffMetrics',
    'MonolithicCheckpointSaver',
    'GlobalLRScaling',
    'LayerFreezing',
    'ScheduledGarbageCollector',
    'EvalGauntlet',
    'HuggingFaceCheckpointer',
    'MegaBlocksMoE_TokPerExpert',
    'AsyncEval',
    'CurriculumLearning',
]

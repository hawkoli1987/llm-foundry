from composer.core import Callback, State
from composer.loggers import Logger
import torch
from transformers import AutoTokenizer
from composer.utils import dist
import numpy as np


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
        


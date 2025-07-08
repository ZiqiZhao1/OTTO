import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class StateTrainer(Trainer):


    def train_step(self):
        states, next_states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        next_states, action_target, reward_target = torch.clone(next_states), torch.clone(actions), torch.clone(rewards)

        next_state_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        state_dim = next_state_preds.shape[2]
        next_state_preds = next_state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        next_state_target = next_states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(None, next_state_preds, None,
                            None, next_state_target, None)  # we only need the 2nd one and the 5th one

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/next_state_error'] = torch.mean((next_state_preds-next_state_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

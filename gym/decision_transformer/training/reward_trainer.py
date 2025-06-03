import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class RewardTrainer(Trainer):

    def train_step(self):
        states, next_states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        next_states, action_target, reward_target = torch.clone(next_states), torch.clone(actions), torch.clone(rewards)

        reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        reward_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]
        reward_target = reward_target.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(None, reward_preds, None,
                            None, reward_target, None)  # we only need the 2nd one and the 5th one

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/reward_error'] = torch.mean((reward_preds-reward_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

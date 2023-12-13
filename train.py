import gymnasium as gym
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from utils import make_env, matrix_norm

from utils import parse_args
from agent import Agent_Memory


class Train:
    def __init__(self):
        self.args = parse_args()
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        if self.args.track:
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % (
                    "\n".join(
                        [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                    )
                ),
            )
        else:
            self.writer = None

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )

        # env setup
        # envs = gym.vector.SyncVectorEnv(
        #      [make_env(self.args.env_id, self.args.seed + i, i) for i in range(self.args.num_envs)]
        # )
        # env = DummyVecEnv([lambda:env])
        self.envs = DummyVecEnv(
            [
                make_env(self.args.env_id, self.args.seed + i, i)
                for i in range(self.args.num_envs)
            ]
        )
        self.envs = VecNormalize(self.envs, norm_obs=True, norm_reward=True)
        assert isinstance(
            self.envs.action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        self.agent = Agent_Memory(self.envs, self.args.hidden_size).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.args.num_steps, self.args.num_envs)
            + self.envs.observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.args.num_steps, self.args.num_envs) + self.envs.action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )

    def train(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        next_lstm_state = (
            torch.zeros(
                self.agent.lstm.num_layers,
                self.args.num_envs,
                self.agent.lstm.hidden_size,
            ).to(self.device),
            torch.zeros(
                self.agent.lstm.num_layers,
                self.args.num_envs,
                self.agent.lstm.hidden_size,
            ).to(self.device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        num_updates = self.args.total_timesteps // self.args.batch_size

        for update in range(1, num_updates + 1):
            initial_lstm_state = (
                next_lstm_state[0].clone(),
                next_lstm_state[1].clone(),
            )
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            if self.args.coverage:
                coverage_hist = np.zeros((self.args.num_envs, self.envs.action_space.n))
            else:
                coverage_hist = None

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    if coverage_hist is None:
                        (
                            action,
                            logprob,
                            _,
                            value,
                            next_lstm_state,
                        ) = self.agent.get_action_and_value(
                            next_obs, next_lstm_state, next_done
                        )
                    else:
                        coverage_hist_norm = matrix_norm(coverage_hist)
                        coverage_hist_norm = (
                            torch.FloatTensor(coverage_hist_norm)
                            .to(self.device)
                            .detach()
                        )
                        (
                            action,
                            logprob,
                            _,
                            value,
                            next_lstm_state,
                        ) = self.agent.get_action_and_value(
                            next_obs,
                            next_lstm_state,
                            next_done,
                            coverage_hist=coverage_hist_norm,
                        )
                        coverage_hist[range(self.args.num_envs), action.cpu()] += 1
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.)
                next_obs, reward, done, info = self.envs.step(
                    list(map(int, list(action.cpu().numpy())))
                )
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        print(
                            f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}"
                        )
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "charts/episodic_return",
                                item["episode"]["r"],
                                global_step,
                            )
                            self.writer.add_scalar(
                                "charts/episodic_length",
                                item["episode"]["l"],
                                global_step,
                            )
                        break

            # bootstrap value if not done
            with torch.no_grad():
                # next_value = self.agent.get_value(next_obs).reshape(1, -1)
                next_value = self.agent.get_value(
                    next_obs,
                    next_lstm_state,
                    next_done,
                ).reshape(1, -1)
                if self.args.gae:
                    advantages = torch.zeros_like(self.rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = (
                            self.rewards[t]
                            + self.args.gamma * nextvalues * nextnonterminal
                            - self.values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.args.gamma
                            * self.args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(self.rewards).to(self.device)
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            self.rewards[t]
                            + self.args.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            b_dones = self.dones.reshape(-1)

            # Optimizing the policy and value network
            # b_inds = np.arange(self.args.batch_size)
            assert self.args.num_envs % self.args.num_minibatches == 0
            envsperbatch = self.args.num_envs // self.args.num_minibatches
            envinds = np.arange(self.args.num_envs)
            flatinds = np.arange(self.args.batch_size).reshape(
                self.args.num_steps, self.args.num_envs
            )
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, self.args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[
                        :, mbenvinds
                    ].ravel()  # be really careful about the index

                    (
                        new_a_prob,
                        newlogprob,
                        entropy,
                        newvalue,
                        _,
                    ) = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        (
                            initial_lstm_state[0][:, mbenvinds],
                            initial_lstm_state[1][:, mbenvinds],
                        ),
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()

                    if coverage_hist is None:
                        loss = (
                            pg_loss
                            - self.args.ent_coef * entropy_loss
                            + v_loss * self.args.vf_coef
                        )  # + coverage_loss * 0.025
                    else:
                        coverage_loss = torch.mean(
                            torch.min(
                                torch.mean(coverage_hist_norm, 0),
                                torch.mean(new_a_prob, 0),
                            )
                        )
                        loss = (
                            pg_loss
                            - self.args.ent_coef * entropy_loss
                            + v_loss * self.args.vf_coef
                            + coverage_loss * 0.025
                        )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.args.max_grad_norm
                    )
                    self.optimizer.step()

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if self.writer is not None:
                self.writer.add_scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar(
                    "losses/policy_loss", pg_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/entropy", entropy_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/approx_kl", approx_kl.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/clipfrac", np.mean(clipfracs), global_step
                )
                self.writer.add_scalar(
                    "losses/explained_variance", explained_var, global_step
                )
                self.writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
            print("SPS:", int(global_step / (time.time() - start_time)))

        torch.save(self.agent, f"models/ppo_optimized_{self.args.env_id}.pt")
        self.envs.close()
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    model = Train()
    model.train()

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from SAC.model import (Policy, SoftQNetwork, ReplayBuffer)
from SAC.util import *
import argparse, time
from torch.utils.tensorboard import SummaryWriter


class SAC(object):
    def __init__(self, dataset, num_states, num_actions, env):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_requests = dataset.num_requests
        self.num_servers = dataset.num_locals + dataset.num_edges
        self.num_partitions = dataset.num_partitions

        self.num_states = num_states
        self.num_actions = num_actions
        print("num_states:", num_states, "num_actions", num_actions)

        parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
        # Common arguments
        parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), help='the name of this experiment')
        parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0", help='the id of the gym environment')
        parser.add_argument('--seed', type=int, default=2, help='seed of the experiment')
        parser.add_argument('--episode-length', type=int, default=0, help='the maximum length of each episode')
        parser.add_argument('--total-timesteps', type=int, default=100000, help='total timesteps of the experiments')
        parser.add_argument('--torch-deterministic', type=lambda x: bool(str2bool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
        parser.add_argument('--cuda', type=lambda x: bool(str2bool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by default')
        parser.add_argument('--autotune', type=lambda x: bool(str2bool(x)), default=True, nargs='?', const=True, help='automatic tuning of the entropy coefficient.')

        # Algorithm specific arguments
        parser.add_argument('--buffer-size', type=int, default=10000+1, help='the replay memory buffer size')
        parser.add_argument('--learning-starts', type=int, default=5000, help="timestep to start learning")
        parser.add_argument('--gamma', type=float, default=0.9, help='the discount factor gamma')
        parser.add_argument('--target-network-frequency', type=int, default=1, help="the timesteps it takes to update the target network") # Denis Yarats' implementation delays this by 2. 
        parser.add_argument('--max-grad-norm', type=float, default=10.0, help='the maximum norm for the gradient clipping')
        parser.add_argument('--batch-size', type=int, default=256, help="the batch size of sample from the reply memory") # Worked better in my experiments, still have to do ablation on this. Please remind me
        parser.add_argument('--tau', type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
        parser.add_argument('--alpha', type=float, default=0.2, help="Entropy regularization coefficient.")

        # Additional hyper parameters for tweaks
        ## Separating the learning rate of the policy and value commonly seen: (Original implementation, Denis Yarats)
        parser.add_argument('--policy-lr', type=float, default=1e-4, help='the learning rate of the policy network optimizer')
        parser.add_argument('--q-lr', type=float, default=1e-4, help='the learning rate of the Q network network optimizer')
        parser.add_argument('--policy-frequency', type=int, default=2, help='delays the update of the actor, as per the TD3 paper.')
        # NN Parameterization
        parser.add_argument('--weights-init', default='xavier', const='xavier', nargs='?', choices=['xavier', "orthogonal", 'uniform'], help='weight initialization scheme for the neural networks.')
        parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'], help='weight initialization scheme for the neural networks.')
        parser.add_argument('--ent-c', default=-0.25, type=float, help='target entropy of continuous component.')
        parser.add_argument('--ent-d', default=0.25, type=float, help='target entropy of discrete component.')

        args = parser.parse_args()
        if not args.seed:
            args.seed = int(time.time())

        experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(f"runs/{experiment_name}")
        writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        self.out_c = self.out_d = self.num_servers
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.policy_network = Policy(num_states, self.out_c, self.out_d, env).to(self.device)
        self.q1_network = SoftQNetwork(num_states, self.out_c, self.out_d).to(self.device)
        self.q2_network = SoftQNetwork(num_states, self.out_c, self.out_d).to(self.device)
        self.q1_network_target = SoftQNetwork(num_states, self.out_c, self.out_d).to(self.device)
        self.q2_network_target = SoftQNetwork(num_states, self.out_c, self.out_d).to(self.device)
        self.q1_network_target.load_state_dict(self.q1_network.state_dict())
        self.q2_network_target.load_state_dict(self.q2_network.state_dict())
        values_optimizer = optim.Adam(list(self.q1_network.parameters()) + list(self.q2_network.parameters()), lr=args.q_lr)
        policy_optimizer = optim.Adam(list(self.policy_network.parameters()), lr=args.policy_lr)
        loss_fn = nn.MSELoss()

        # Automatic entropy tuning
        if args.autotune:
            # target_entropy = -float(out_c)
            target_entropy = args.ent_c
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha = log_alpha.exp().detach().cpu().item()
            a_optimizer = optim.Adam([log_alpha], lr=1e-4)

            # target_entropy_d = -0.98 * np.log(1/out_d)
            target_entropy_d = args.ent_d
            log_alpha_d = torch.zeros(1, requires_grad=True, device=self.device)
            alpha_d = log_alpha_d.exp().detach().cpu().item()
            a_d_optimizer = optim.Adam([log_alpha_d], lr=1e-4)
        else:
            alpha = args.alpha
            alpha_d = args.alpha

        # TRY NOT TO MODIFY: start the game
        global_episode = 0
        obs, done = env.reset(), False
        episode_reward, episode_length = 0., 0
        average_reward = 0.

        for global_step in range(1, args.total_timesteps + 1):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                action_c, action_d = self.policy_network.sample([obs])
                action = to_gym_action(action_c, action_d)
            else:
                action_c, action_d, _, _, _ = self.policy_network.get_action([obs], self.device)
                action = to_gym_action(action_c, action_d)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _ = env.next_step(action)
            self.replay_buffer.put((obs, gym_to_buffer(action), reward, next_obs, done))
            episode_reward += reward
            episode_length += 1
            obs = np.array(next_obs)

            # ALGO LOGIC: training.
            if len(self.replay_buffer.buffer) > args.batch_size:  # starts update as soon as there is enough data.
                s_obs, s_actions, s_rewards, s_next_obses, s_dones = self.replay_buffer.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.policy_network.get_action(s_next_obses, self.device)
                    qf1_next_target = self.q1_network_target.forward(s_next_obses, next_state_actions_c, self.device)
                    qf2_next_target = self.q2_network_target.forward(s_next_obses, next_state_actions_c, self.device)

                    min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_prob_d * next_state_log_pi_c - alpha_d * next_state_log_pi_d)
                    next_q_value = torch.Tensor(s_rewards).to(self.device) + (1 - torch.Tensor(s_dones).to(self.device)) * args.gamma * (min_qf_next_target.sum(1)).view(-1)

                s_actions_c, s_actions_d = to_torch_action(s_actions, self.device)
                qf1_a_values = self.q1_network.forward(s_obs, s_actions_c, self.device).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
                qf2_a_values = self.q2_network.forward(s_obs, s_actions_c, self.device).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
                qf1_loss = loss_fn(qf1_a_values, next_q_value)
                qf2_loss = loss_fn(qf2_a_values, next_q_value)
                qf_loss = (qf1_loss + qf2_loss) / 2

                values_optimizer.zero_grad()
                qf_loss.backward()
                values_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.policy_network.get_action(s_obs, self.device)
                        qf1_pi = self.q1_network.forward(s_obs, actions_c, self.device)
                        qf2_pi = self.q2_network.forward(s_obs, actions_c, self.device)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)

                        policy_loss_d = (prob_d * (alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                        policy_loss_c = (prob_d * (alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                        policy_loss = policy_loss_d + policy_loss_c

                        policy_optimizer.zero_grad()
                        policy_loss.backward()
                        policy_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                a_c, a_d, lpi_c, lpi_d, p_d = self.policy_network.get_action(s_obs, self.device)
                            alpha_loss = (-log_alpha * p_d * (p_d * lpi_c + target_entropy)).sum(1).mean()
                            alpha_d_loss = (-log_alpha_d * p_d * (lpi_d + target_entropy_d)).sum(1).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().detach().cpu().item()

                            a_d_optimizer.zero_grad()
                            alpha_d_loss.backward()
                            a_d_optimizer.step()
                            alpha_d = log_alpha_d.exp().detach().cpu().item()

                # update the target network
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(self.q1_network.parameters(), self.q1_network_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(self.q2_network.parameters(), self.q2_network_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if len(self.replay_buffer.buffer) > args.batch_size and global_step % 100 == 0:
                writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # NOTE: additional changes from cleanrl
                writer.add_scalar("losses/alpha_d", alpha_d, global_step)
                writer.add_histogram("actions/discrete", action[0]+1, global_step)
                for i in range(self.num_servers):
                    writer.add_histogram("actions/continuous_{}".format(i), action[1][i], global_step)
                writer.add_scalar("debug/ent_bonus", (- alpha * next_state_prob_d * next_state_log_pi_c - alpha_d * next_state_log_pi_d).sum(1).mean().item(), global_step)
                writer.add_scalar("debug/next_q", next_q_value.mean().item(), global_step)
                writer.add_scalar("debug/policy_loss_c", policy_loss_c.item(), global_step)
                writer.add_scalar("debug/policy_loss_d", policy_loss_d.item(), global_step)
                writer.add_scalar("debug/policy_ent_d", -(prob_d*log_pi_d).sum(1).mean().item(), global_step)
                writer.add_scalar("debug/mean_q", min_qf_pi.mean().item(), global_step)
                writer.add_scalar("debug/mean_r", s_rewards.mean(), global_step)
                writer.add_histogram("debug_q/q_0", min_qf_pi[:, 0].mean().item(), global_step)
                writer.add_histogram("debug_q/q_1", min_qf_pi[:, 1].mean().item(), global_step)
                writer.add_histogram("debug_q/q_2", min_qf_pi[:, 2].mean().item(), global_step)
                writer.add_histogram("debug_pi/pi_0", prob_d[:, 0].mean().item(), global_step)
                writer.add_histogram("debug_pi/pi_1", prob_d[:, 1].mean().item(), global_step)
                writer.add_histogram("debug_pi/pi_2", prob_d[:, 2].mean().item(), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                    writer.add_scalar("losses/alpha_d_loss", alpha_d_loss.item(), global_step)

            if done:
                global_episode += 1  # Outside the loop already means the epsiode is done
                writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                writer.add_scalar("charts/episode_length", episode_length, global_step)
                # Terminal verbosity
                average_reward += 10 - reward * 10
                if global_episode % 10 == 0:
                    print(f"Episode: {global_episode} Step: {global_step}, Ep. Reward: {average_reward / 10}")
                    average_reward = 0

                # Reseting what need to be
                obs, done = env.reset(), False
                episode_reward, episode_length = 0., 0

        writer.close()
        env.close()

        # save the policy
        torch.save(self.policy_network.state_dict(), 'platform.pth')

    def load_weights(self, output, best=False):
        if output is None:
            return
        if best:
            self.actor.load_state_dict(torch.load('{}/actor_best.pkl'.format(output)))
            self.critic.load_state_dict(torch.load('{}/critic_best.pkl'.format(output)))
            self.critic2.load_state_dict(torch.load('{}/critic2_best.pkl'.format(output)))
        else:
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))
            self.critic2.load_state_dict(torch.load('{}/critic2.pkl'.format(output)))

    def save_model(self, output, best=False):
        if best:
            torch.save(self.actor.state_dict(), '{}/actor_best.pkl'.format(output))
            torch.save(self.critic.state_dict(), '{}/critic_best.pkl'.format(output))
            torch.save(self.critic2.state_dict(), '{}/critic2_best.pkl'.format(output))
        else:
            torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
            torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
            torch.save(self.critic2.state_dict(), '{}/critic2.pkl'.format(output))
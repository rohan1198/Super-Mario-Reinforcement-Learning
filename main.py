import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import time
import random

from gym.wrappers import Monitor
from distutils.util import strtobool

from lib.agent import Agent
from lib.experience import ReplayBuffer
from lib.loss import calc_loss
from lib.network import DuelingDQN
from lib.wrappers import make_env




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Super Mario Bros Rainbow DQN")

    parser.add_argument("--exp-name", type = str, default = "MarioRainbowDQN", help = "Name of this experiment")
    parser.add_argument("--gym-id", type = str, default = "SuperMarioBros-1-1-v0", help = "Gym ID")
    parser.add_argument('--learning-rate', type = float, default = 25e-5, help = 'Learning rate of the optimizer')
    parser.add_argument('--seed', type = int, default = 2, help = 'Seed of the experiment')
    parser.add_argument('--total-timesteps', type = int, default = 2000000, help = 'Total timesteps for the experiment')
    parser.add_argument('--cuda', action = "store_true", default = False, help = 'Enable CUDA acceleration')
    parser.add_argument('--prod-mode', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'Run the script in production mode')
    parser.add_argument('--capture-video', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'Capture the videos during training')
    parser.add_argument('--wandb-project-name', type = str, default = "SuperMarioBros", help = "The wandb's project name")

    parser.add_argument("--learning-starts", type = int, default = 1e4, help = "Populate the Buffer")
    parser.add_argument("--target-net-sync", type = int, default = 1e4, help = "Time steps until the target network synchronizes with the main network")
    parser.add_argument('--epsilon-frames', type = int, default = 10**5, help = "Frames for episilon decay")
    parser.add_argument('--epsilon-start', type = float, default = 1.0, help = "Starting value for epsilon")
    parser.add_argument('--epsilon-final', type = float, default = 0.05, help = "Final value for epsilon")
    parser.add_argument('--buffer-size', type = int, default = 90000, help = 'Replay memory buffer size')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'Discount factor')
    parser.add_argument('--batch-size', type = int, default = 32, help = "Batch Size for the memory")

    args = parser.parse_args()

    if args.seed:
        args.seed = int(time.time())

    experiment_name = str(args.exp_name)
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    if args.prod_mode:
        import wandb
        wandb.init(project = args.wandb_project_name, sync_tensorboard = True, config = vars(args), name = experiment_name, monitor_gym = True, save_code = True)
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device == torch.device("cuda"):
        print(torch.cuda.get_device_properties(device))
    else:
        print("Training on CPU!")
    print("\n")

    env = make_env(args.gym_id)
    
    print("Observation space shape: ", env.observation_space.shape, "| Number of actions available: ", env.action_space.n, "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    if args.capture_video:
        env = Monitor(env, f'videos/{experiment_name}', video_callable = lambda episode_id: episode_id % 1 == 0, force = True)
    
    buffer = ReplayBuffer(args.buffer_size)
    agent = Agent(env, buffer)

    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)

    print(net, "\n")

    epsilon = args.epsilon_start
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    for global_step in range(1, args.total_timesteps + 1):
        frame_idx += 1
        epsilon = max(args.epsilon_final, args.epsilon_start - frame_idx / args.epsilon_frames)

        reward = agent.play_step(net, epsilon, device = device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            
            print("-" * 105)
            print(f"|| {frame_idx} frames || {len(total_rewards)} Games || {reward:.3f} Reward || {mean_reward:.3f} Mean Reward || {epsilon:.2f} Epislon || {speed:.2f} frames/sec ||")

            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), "mario-best.dat")

                if best_mean_reward is not None:
                    print("=" * 60)
                    print(f"Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f} model saved!  |")
                    print("=" * 60)
                
                best_mean_reward = mean_reward

        if len(buffer) < args.learning_starts:
            continue

        if frame_idx % args.target_net_sync:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        
        loss_t = calc_loss(batch, net, tgt_net, args.gamma, device = device)
        loss_t.backward()
        optimizer.step()

    writer.close()
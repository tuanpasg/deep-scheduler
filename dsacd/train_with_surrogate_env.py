import argparse
import torch

from surrogate_env_adapter import SurrogateEnvAdapter
from DSACD_multibranch import DSACD


def main(args):
    device = torch.device(args.device)

    env = SurrogateEnvAdapter(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        fallback_action=args.fallback_action,
        device=args.device,
    )

    agent = DSACD(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_rbg=args.n_rbg,
        fallback_action=args.fallback_action,
        device=device,
    )

    obs = env.reset()

    for tti in range(args.ttis):
        env.begin_tti()

        for layer_ctx in env.layer_iter():
            with torch.no_grad():
                logits = agent.actor(
                    layer_ctx.obs.unsqueeze(0),
                    layer_ctx.masks_rbg,
                )  # [n_rbg, act_dim]

                actions = torch.argmax(logits, dim=-1)

            env.apply_layer_actions(layer_ctx, actions)

        env.end_tti()
        transitions = env.export_branch_transitions()

        for tr in transitions:
            agent.replay_buffer.add(tr)

        if agent.replay_buffer.size >= args.learning_starts:
            agent.update()

        if tti % args.log_every == 0:
            print(
                f"[TTI {tti}] "
                f"Buffer={agent.replay_buffer.size}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ttis", type=int, default=1000)
    parser.add_argument("--learning_starts", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--obs_dim", type=int, default=128)
    parser.add_argument("--act_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_rbg", type=int, default=8)
    parser.add_argument("--fallback_action", type=int, default=0)

    args = parser.parse_args()
    main(args)

"""Exercise either the ALFWorld text backend or the CUDA inference stack."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def check_textworld(data_dir):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    train_dir = data_dir / "json_2.1.1" / "train"
    game_file = min(train_dir.glob("pick_and_place_simple-*/trial_*/game.tw-pddl"))
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True)
    env_id = textworld.gym.register_game(
        str(game_file), request_infos, max_episode_steps=50,
        wrappers=[AlfredDemangler(shuffle=False)],
    )
    env = textworld.gym.make(env_id)
    try:
        env.seed(0)
        initial_observation, initial_info = env.reset()
        command = "look"
        if command not in initial_info["admissible_commands"]:
            command = sorted(initial_info["admissible_commands"])[0]
        feedback, score, done, info = env.step(command)
        assert initial_observation and feedback
        assert "won" in info
        return {
            "alfworld": importlib.metadata.version("alfworld"),
            "textworld": importlib.metadata.version("textworld"),
            "game_file": str(game_file.relative_to(data_dir)),
            "game_sha256": hashlib.sha256(game_file.read_bytes()).hexdigest(),
            "observation": initial_observation,
            "action": command,
            "feedback": feedback,
            "score": score,
            "done": done,
            "won": info["won"],
        }
    finally:
        env.close()


def check_cuda():
    import bitsandbytes as bnb
    import torch

    with torch.inference_mode():
        ones = torch.ones((256, 256), dtype=torch.float16, device="cuda")
        assert torch.all(ones @ ones == 256).item(), "FP16 matmul failed"

        torch.manual_seed(0)
        layer = bnb.nn.Linear4bit(
            64, 32, bias=False, compute_dtype=torch.float16, quant_type="nf4"
        ).to(dtype=torch.float16).cuda().eval()
        output = layer(ones[:2, :64].contiguous())
        torch.cuda.synchronize()
        assert layer.weight.quant_state.quant_type == "nf4"
        assert output.is_cuda and output.shape == (2, 32)
        assert torch.isfinite(output).all().item() and torch.any(output != 0).item()

    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_capability(0),
        "bitsandbytes": bnb.__version__,
        "fp16_matmul": "passed",
        "nf4_forward": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["textworld", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "rsi-data" / "alfworld")
    args = parser.parse_args()
    result = check_textworld(args.data_dir) if args.check == "textworld" else check_cuda()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

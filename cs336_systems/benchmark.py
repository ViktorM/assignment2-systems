import argparse
import torch
import numpy as np
from timeit import timeit, default_timer
import sys
import math


# Add assignment2-systems/cs336-basics to path
sys.path.append('/home/viktor4090/Projects/Stanford/CS336/assignment2-systems/cs336-basics')
# sys.path.append('/home/viktor4090/Projects/Stanford/CS336/assignment1-basics')

# Import from cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient


torch.set_float32_matmul_precision('high')


def calc_average_and_std(array):
    # avg = sum(array) / len(array)
    # std = math.sqrt(sum((x - avg)**2 for x in array) / len(array))
    avg = np.mean(array)
    std = np.std(array)
    return avg, std


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Dummy data for benchmarking
    dataset = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(100000,)
    ).numpy()

    # Warm-up steps
    for _ in range(args.warmup_steps):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_gradient(model.parameters(), args.grad_clip)
        optimizer.step()

    # Benchmarking steps
    torch.cuda.synchronize()
    start_time = default_timer()

    forward_times = []
    backward_times = []
    clip_times = []
    optimizer_times = []

    for _ in range(args.benchmark_steps):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        start_forward_time = default_timer()
        logits = model(x)
        forward_time = default_timer() - start_forward_time
        forward_times.append(forward_time)

        torch.cuda.synchronize()
        start_backward_time = default_timer()
        loss = cross_entropy(logits, y)
        loss.backward()
        backward_time = default_timer() - start_backward_time        
        backward_times.append(backward_time)

        torch.cuda.synchronize()
        start_clip_time = default_timer()
        clip_gradient(model.parameters(), args.grad_clip)
        clip_time = default_timer() - start_clip_time
        clip_times.append(clip_time)

        torch.cuda.synchronize()
        start_optimizer_time = default_timer()
        optimizer.step()
        optimizer_time = default_timer() - start_optimizer_time
        optimizer_times.append(optimizer_time)

        torch.cuda.synchronize()
        # print(f"Step {_} loss: {loss.item()}")

    avg_forward_time, std_forward_time = calc_average_and_std(forward_times)
    avg_backward_time, std_backward_time = calc_average_and_std(backward_times)
    avg_clip_time, std_clip_time = calc_average_and_std(clip_times)
    avg_optimizer_time, std_optimizer_time = calc_average_and_std(optimizer_times)

    elapsed_time = default_timer() - start_time
    avg_time = elapsed_time / args.benchmark_steps

    print(f"Average time per iteration: {avg_time:.5f} seconds")
    print(f"Average forward time: {avg_forward_time:.5f} seconds, std: {std_forward_time:.5f}")
    print(f"Average backward time: {avg_backward_time:.5f} seconds, std: {std_backward_time:.5f}")
    print(f"Average clip time: {avg_clip_time:.5f} seconds, std: {std_clip_time:.5f}")
    print(f"Average optimizer time: {avg_optimizer_time:.5f} seconds, std: {std_optimizer_time:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer LM Training Benchmark"
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--compile", type=bool, default=False)

    parsed_args = parser.parse_args()
    train(parsed_args)

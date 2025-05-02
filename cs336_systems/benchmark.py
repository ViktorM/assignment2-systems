import argparse
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
import pandas as pd
from timeit import default_timer
import sys


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

@nvtx.range("forward_pass")
def forward_pass(model, x):
    return model(x)

@nvtx.range("backward_pass")
def backward_pass(loss):
    loss.backward()

@nvtx.range("clip_gradient")
def clip_gradient(model, grad_clip):
    clip_gradient(model.parameters(), grad_clip)

@nvtx.range("optimizer_step")
def optimizer_step(optimizer):
    optimizer.step()


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
    with nvtx.range("warmup"):
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
        logits = forward_pass(model, x)
        forward_time = default_timer() - start_forward_time
        forward_times.append(forward_time)

        torch.cuda.synchronize()
        start_backward_time = default_timer()
        loss = cross_entropy(logits, y)
        backward_pass(loss)
        backward_time = default_timer() - start_backward_time        
        backward_times.append(backward_time)

        torch.cuda.synchronize()
        start_clip_time = default_timer()
        clip_gradient(model, args.grad_clip)
        clip_time = default_timer() - start_clip_time
        clip_times.append(clip_time)

        torch.cuda.synchronize()
        start_optimizer_time = default_timer()
        optimizer_step(optimizer)
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


def benchmark_model(args, model_sizes, output_markdown_file="benchmark_results.md"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for size_name, params in model_sizes.items():
        print(f"Benchmarking model size: {size_name}")

        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=params["d_model"],
            num_layers=params["num_layers"],
            num_heads=params["num_heads"],
            d_ff=params["d_ff"],
            rope_theta=args.rope_theta
        ).to(device)

        if args.compile:
            model = torch.compile(model)

        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        dataset = torch.randint(0, args.vocab_size, (100000,)).numpy()

        # Warm-up steps
        for _ in range(args.warmup_steps):
            x, y = get_batch(dataset, args.batch_size, args.context_length, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            clip_gradient(model.parameters(), args.grad_clip)
            optimizer.step()

        # Benchmarking
        forward_times, backward_times = [], []

        for _ in range(args.benchmark_steps):
            x, y = get_batch(dataset, args.batch_size, args.context_length, device)
            optimizer.zero_grad()

            torch.cuda.synchronize()
            start_forward = default_timer()
            logits = model(x)
            torch.cuda.synchronize()
            forward_time = default_timer() - start_forward
            forward_times.append(forward_time)

            torch.cuda.synchronize()
            start_backward = default_timer()
            loss = cross_entropy(logits, y)
            loss.backward()
            torch.cuda.synchronize()
            backward_time = default_timer() - start_backward
            backward_times.append(backward_time)

            clip_gradient(model.parameters(), args.grad_clip)
            optimizer.step()

        avg_fwd, std_fwd = calc_average_and_std(forward_times)
        avg_bwd, std_bwd = calc_average_and_std(backward_times)

        results.append({
            "Model Size": size_name,
            "Forward Pass Avg (s)": avg_fwd,
            "Forward Pass Std (s)": std_fwd,
            "Backward Pass Avg (s)": avg_bwd,
            "Backward Pass Std (s)": std_bwd,
        })

    df = pd.DataFrame(results)
    markdown_results = df.to_markdown(index=False)
    print(markdown_results)

    # Save the markdown results to a file
    with open(output_markdown_file, "w") as f:
        f.write("# Benchmarking Results\n\n")
        f.write(markdown_results)

    print(f"Results saved to {output_markdown_file}")


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
    #train(parsed_args)

    model_sizes = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    #    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    #    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    #    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

    output_markdown_file = "benchmark_results_" + "warmup_" + str(parsed_args.warmup_steps)
    if parsed_args.compile:
        output_markdown_file += "_compile"
    output_markdown_file += ".md"

    benchmark_model(parsed_args, model_sizes, output_markdown_file)

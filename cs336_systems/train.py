import argparse
import torch
import time
import sys

# Add assignment2-systems/cs336-basics to path
sys.path.append('/home/viktor4090/Projects/Stanford/CS336/assignment2-systems/cs336-basics')
# sys.path.append('/home/viktor4090/Projects/Stanford/CS336/assignment1-basics')

# Import from cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient


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
    start_time = time.perf_counter()

    for _ in range(args.benchmark_steps):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_gradient(model.parameters(), args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        print(f"Step {_} loss: {loss.item()}")

    elapsed_time = time.perf_counter() - start_time
    avg_time = elapsed_time / args.benchmark_steps

    print(f"Average time per iteration: {avg_time:.4f} seconds")


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()
    train(args)

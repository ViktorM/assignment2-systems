# CS336 Assignment 2: Systems and Parallelism

## Section 1: Profiling and Benchmarking

### 1.1.3 End-to-End Benchmarking

#### (a) Benchmarking Script
Write a script to perform basic end-to-end benchmarking of the forward and backward passes in your model.

*Your implementation goes here.*

#### (b) Forward and Backward Pass Timings
Time the forward and backward passes for the model sizes described in §1.1.2.

*Your response goes here.*

#### (c) Warm-Up Steps Impact
Repeat your analysis without the warm-up steps. How does this affect your results?

*Your response goes here.*

### 1.1.4 Nsight Systems Profiler

#### (a) Forward Pass Total Time
What is the total time spent on your forward pass?

*Your response goes here.*

#### (b) CUDA Kernel with Most GPU Time
What CUDA kernel takes the most cumulative GPU time during the forward pass?

*Your response goes here.*

#### (c) Other Significant CUDA Kernels
What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

*Your response goes here.*

#### (d) AdamW Optimizer Step Profiling
Profile running one complete training step with your implementation of AdamW.

*Your response goes here.*

#### (e) Softmax vs. Matrix Multiplication Runtimes
Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer.

*Your response goes here.*

### 1.1.5 Mixed Precision

#### Mixed Precision Accumulation
Run the provided mixed precision accumulation code and comment on the accuracy.

*Your response goes here.*

#### (a) Toy Model Data Types
What are the data types of various components within the autocast context?

- Model parameters:
- Output of first layer:
- Output of layer norm:
- Model's logits:
- Loss:
- Gradients:

#### (b) Layer Normalization and Precision
What parts of layer normalization are sensitive to mixed precision?

*Your response goes here.*

#### (c) Mixed Precision Benchmarking Timings and Trends
Modify your benchmarking script to optionally run the model using mixed precision with BF16.

*Your response goes here.*

### 1.1.6 Profiling Memory

#### (a) Memory Timeline Images and Response
Run your model through the memory profiler. How do your memory timelines look?

- Forward Pass:
- Full Training Step:

*Your commentary goes here.*

#### (b) Peak Memory Usage Table
What is the peak memory usage of each model size?

| Model Size | Forward Pass (MB) | Full Training Step (MB) |
|------------|-------------------|-------------------------|
| Small      |                   |                         |
| Medium     |                   |                         |
| Large      |                   |                         |
| XL         |                   |                         |
| 2.7B       |                   |                         |

#### (c) Mixed Precision Memory Impact
Does mixed-precision significantly affect memory usage?

*Your response goes here.*

#### (d) Size of Activations Tensor
What is the size of a tensor of activations in the Transformer residual stream?

*Your response and derivation go here.*

#### (e) Largest Allocation Size and Source
What is the size of the largest allocations shown and where do they come from?

*Your response goes here.*

## Section 1.2: Optimizing Attention with FlashAttention-2

### 1.2.1 Benchmarking PyTorch Attention
Benchmark your attention implementation at different scales.

*Table with timings, memory usage calculation, and commentary goes here.*

## Section 1.3: Benchmarking JIT-Compiled Attention

#### (a) Compiled vs. Uncompiled Attention Timings
Compare the compiled and uncompiled versions of your attention implementation.

*Comparison table goes here.*

#### (b) Entire Transformer Model Compilation Timings
How does the performance of the forward pass change when compiling your entire Transformer model?

*Comparison table goes here.*

### 1.3.2 FlashAttention-2 Forward Pass

#### (a) PyTorch Autograd Implementation
Write a pure PyTorch autograd function that implements the FlashAttention-2 forward pass.

*Your implementation details here.*

#### (b) Triton Kernel Implementation
Write a Triton kernel for the forward pass of FlashAttention-2.

*Your Triton implementation details here.*

#### (c) Causal Masking Implementation
Add a causal masking flag to your FlashAttention-2 implementation.

*Your implementation details here.*

## Section 2: Distributed Data Parallel Training

### 2.1 Single-Node Distributed Communication
Benchmark the runtime of the all-reduce operation in single-node multi-process setups.

*Benchmarking results and commentary go here.*

### 2.2 Naïve Distributed Data Parallel Training
Write a script to naively perform distributed data parallel training.

*Implementation details go here.*

### Naïve DDP Benchmarking
Benchmark the overhead of your naïve DDP implementation.

*Benchmarking setup and results here.*

### 2.3 Improving Minimal DDP

#### 2.3.1 Communication Calls Reduction
Modify your minimal DDP implementation to communicate a tensor with flattened gradients.

*Benchmarking results and comparison go here.*

#### 2.3.2 Overlapping Computation with Individual Gradients
Implement a DDP class that overlaps gradient communication with computation.

*Class implementation and benchmarking results here.*

#### (b) Profiler Screenshots
Instrument your benchmarking code and provide profiler screenshots.

- Initial DDP:
- Overlapping DDP:

### 2.3.3 Bucketed Parameter Gradients
Implement a Python class using gradient bucketing for DDP.

*Class implementation goes here.*

#### (a) Bucketed DDP Benchmarking
Benchmark your bucketed DDP implementation with various bucket sizes.

*Results and commentary here.*

#### (b) Equation for Optimal Bucket Size
Write equations modeling DDP overhead and optimal bucket size.

*Your equations and explanations here.*

### 2.4 4D Parallelism

#### (a) Memory Calculation
Calculate the memory needed for the XXL model configuration.

*Calculation and response here.*

#### (b) Memory with Sharding
Write an expression for memory per device when sharded.

*Calculation and response here.*


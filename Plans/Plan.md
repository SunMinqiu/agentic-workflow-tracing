# Characterization Agentic Scientific Workflows

1. Introduction

1.1. First explain why systems need characterization: before designing scheduling, resource provisioning, and data management algorithms, we must know what resources real workflows need and what behavior they exhibit.

1.2. Point out the bias in existing research samples
The community lacks public real agentic workflows. We found GenoMAS, SciLink, ... . We also introduced agents into traditional workflows (SRASearch, 1000 Genomes), forming agentic scientific workflows.

A general workflow system should not be designed and evaluated based on the characteristics of a single workflow. Explain why we choose different workflows. And we also introduced agents into traditional workflows, forming agentic scientific workflows.

1.3. Define the scope of our study
First distinguish prior work: earlier studies characterized "traditional" workflow patterns; agentic workflow (not scientific).
This paper studies the performance characteristics of individual components and of the overall scientific workflow.
Then, different from traditional scientific workflows, there's no specific "job". So we measure only **executable operation**, which is an executable operation initiated by the orchestrator, with a clear start, end, and output.

Agentic scientific workflow
├── Inference operations
│   ├── planning
│   ├── reasoning
│   ├── tool selection
│   ├── validation and review
│   └── retry and refinement
├── Tool and scientific-kernel operations
│   ├── CPU computation
│   ├── GPU computation
│   └── storage and dataflow
└── Orchestration operations
    └── This time is not part of LLM inference, nor is it part of scientific tool execution.

Then for each operation, we measure:
- operation count
- runtime

And for tool execution operations, we measure:
- I/O (read/write size)
- peak memory
- CPU utilization
- ...

For LLM inference operations, we measure:
- KVCache size, hit rate
- Input/Output token size
- TTFT, TPOT
- ...

1.4. Finally explain what these data are used for:

- generating synthetic workflows
- building benchmarks
- driving simulation
- estimating task runtime and data size
- improving scheduling
- improving resource provisioning
- storing them together with provenance

1. Then give data and findings for each workflow.



1. Combining the findings, propose several corresponding solutions.

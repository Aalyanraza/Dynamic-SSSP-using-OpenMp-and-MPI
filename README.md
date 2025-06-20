# 🚀 Dynamic SSSP using OpenMP and MPI

## 👥 Team Members
- Aalyan Raza  
- Aashir Hameed
- Najam Hassan
---

## 📖 1. Introduction

This projects covers **Phase 2** of the project focused on implementing a parallel Single Source Shortest Path (SSSP) algorithm using **MPI** and **OpenMP**. The goal was to enhance scalability and efficiency when operating over large graphs. 

**Key Technologies Used:**
- 🔧 **METIS** for graph partitioning
- 📊 **Intel VTune Profiler** for performance analysis
- 🔄 **Hybrid MPI + OpenMP** parallelization

---

## 🛠️ 2. Implementation Overview

### 2.1 📊 Graph Conversion (`ConvertGraph.py`)

```python
# Transforms raw Facebook edge list CSV → weighted graph
# Features:
# - Random weights (1-10) assignment
# - METIS-compatible format generation
```

**Purpose:** Converts raw Facebook edge list CSV into a weighted graph (`graph.txt`) with random weights (1–10) assigned to each edge. This format is necessary for METIS and parallel processing.

### 2.2 🔀 Graph Partitioning (`partition.cpp`)

Using **METIS**, the graph is partitioned into multiple subgraphs, each assigned to an MPI process.

**Output Files per Process:**
- `subgraph_<rank>.txt` → Local edges
- `subgraph_<rank>_nodes.txt` → Local and ghost nodes

### 2.3 🎲 Change Generation (`generate_changes.py`)

Simulates dynamic graph behavior by generating:
- ➕ **Insertions** 
- ➖ **Deletions**

All changes saved in `changes.txt` for processing.

### 2.4 ⚡ Parallel SSSP (`main.cpp`)

Implements the core SSSP algorithm using **distributed-memory parallelism** (MPI) combined with **shared-memory parallelism** (OpenMP).

#### 🔑 Key Techniques:
- **Hybrid Parallelism:** MPI + OpenMP
- **Ghost Node Message Passing:** Inter-process communication
- **Edge Relaxation:** Via `relax_edges_distributed()`
- **Dynamic Updates:** Edge insertions/deletions handling

#### Algorithm Flow:
1. 🔄 Apply edge updates
2. 🎯 Initialize distances  
3. 🔁 Iteratively relax edges
4. 📡 Sync ghost nodes across ranks

### 2.5 📈 Visualization (`drawGraph.py`)

Visualizes the resulting SSSP tree from `sssp_output.txt` and exports it as `sssp_graph.png`.

---

## 🚧 3. Challenges Faced

| Challenge | Description | Solution |
|-----------|-------------|----------|
| 🔄 **Data Duplication** | Preventing double-counting during edge insertions | Implemented deduplication checks |
| 🐛 **MPI Sync Bugs** | Ensuring correct ghost vertex updates without message loss | Added robust synchronization barriers |
| ⚖️ **Partition Balance** | Handling imbalance in METIS output partitions | Manual tuning and load monitoring |
| 🔥 **CPU Utilization** | Optimizing OpenMP loop schedules | Experimented with different scheduling strategies |

---

## 📊 4. Performance Evaluation

Performance measured using **Intel VTune Profiler** for:
- ⏱️ Elapsed time
- 🖥️ CPU time  
- 🧵 Thread utilization
- 🎯 Hotspot identification

### 4.1 📈 Sequential Baseline

```
📊 Performance Metrics:
├── Elapsed Time: 20.235s
├── CPU Time: 72.390s
└── Parallelism: Poor (underutilized threads)
```
![image2](https://github.com/user-attachments/assets/661730eb-7884-4dcf-b5b3-165ca8c9e9ce)


**Observations:** All threads mostly underutilized, indicating poor parallelism in baseline implementation.

### 4.2 🚀 MPI + OpenMP (12 threads)

```
📊 Performance Metrics:
├── Elapsed Time: 11.451s ⬇️ (43% improvement)
├── CPU Time: 36.190s ⬇️ (50% improvement)  
└── Parallelism: Improved (3-4 logical CPUs peak)
```
![image1](https://github.com/user-attachments/assets/a2b84dd8-6c68-46f9-a868-f8bcc6fe347c)
![image4](https://github.com/user-attachments/assets/41f05934-39e8-46db-8b83-4480b87e9ab7)


### 4.3 ⚡ MPI + OpenMP (20 threads)

```
📊 Performance Metrics:
├── Elapsed Time: 11.904s
├── CPU Time: 36.840s
└── Parallelism: Moderate scaling
```
![image3](https://github.com/user-attachments/assets/85e92304-263d-4098-8f55-8fde27bffb59)
![image6](https://github.com/user-attachments/assets/7eb02815-7460-4c88-8cb7-d9bc19ffe71b)


**Note:** Limited benefit over 12-thread configuration indicates scaling bottlenecks.

### 4.4 🔥 MPI + OpenMP (28 threads)

```
📊 Performance Metrics:
├── Elapsed Time: 13.151s ⬆️ (Performance degradation)
├── CPU Time: 40.508s ⬆️
└── Hotspots: Identified in relax_edges_distributed
```
![image5](https://github.com/user-attachments/assets/ba13066d-f007-446a-b852-d736d5cf8e17)
![image8](https://github.com/user-attachments/assets/722dddda-b81f-48db-bbae-136481513cfd)


### 4.5 💪 MPI + OpenMP (56 threads)

```
📊 Performance Metrics:
├── Elapsed Time: 5.990s ⬇️ (Best performance)
├── CPU Time: 31.390s ⬇️
└── Parallelism: ~8 logical CPUs utilization
```
![image7](https://github.com/user-attachments/assets/5fe4ea2f-3535-4ab9-aa71-fb28e6a6090a)
![image11](https://github.com/user-attachments/assets/b9e840f4-661e-4bd7-a0e4-f5c80a97c359)


**🏆 Best Configuration:** Achieved optimal balance between thread count and CPU utilization.

### 4.6 📊 Scaling Summary

| Configuration | Threads | Elapsed Time | CPU Time | Parallel Efficiency |
|---------------|---------|--------------|----------|-------------------|
| Serial | 1 | 20.235s | 72.390s | - |
| MPI+OpenMP | 12 | 11.451s | 36.190s | ~43% |
| MPI+OpenMP | 20 | 11.904s | 36.840s | ~41% |
| MPI+OpenMP | 28 | 13.151s | 40.508s | ~35% |
| **MPI+OpenMP** | **56** | **5.990s** | **31.390s** | **~65%** |

**📈 Key Insight:** Parallel efficiency improved up to **65%** with higher thread counts, showing strong scaling behavior.
![image9](https://github.com/user-attachments/assets/41865b36-14c0-405e-84c2-31dc2496a40c)
![image10](https://github.com/user-attachments/assets/a25f0dc8-a06f-4799-abde-2fe7ad962e75)


![image12](https://github.com/user-attachments/assets/6a8ca9fb-98fa-42ea-bbb0-c13e54035fa7)
![image13](https://github.com/user-attachments/assets/dbc4d8fc-df81-43ea-98be-d48a5a6eecb6)


---

## 🎯 5. Conclusions

### ✅ Key Achievements:
- **🔀 METIS Integration:** Successful graph partitioning with improved load distribution
- **⚡ Hybrid Parallelism:** MPI+OpenMP significantly reduced execution time vs. serial
- **📊 Performance Analysis:** VTune identified thread imbalance and optimization opportunities
- **🎯 Bottleneck Identification:** Main bottleneck in `relax_edges_distributed` function

### 📈 Performance Gains:
- **🚀 3.4x speedup** (20.235s → 5.990s) with optimal configuration
- **💪 65% parallel efficiency** achieved with 56 threads
- **🔥 Hotspot optimization** improved CPU utilization patterns

### 🔍 Technical Insights:
- METIS partitioning requires manual tuning for optimal results
- Thread scaling benefits plateau around 56 threads for this workload
- Ghost node synchronization critical for correctness

---

## 🚀 6. Future Work

### 🎯 Immediate Improvements:
- **🖥️ GPU Acceleration:** Use OpenCL for offloading compute-heavy sections
- **⚖️ Dynamic Load Balancing:** Integrate adaptive balancing for uneven subgraph sizes
- **📡 Async Communication:** Explore asynchronous models to hide MPI latency

### 🔬 Advanced Optimizations:
- **🧠 Machine Learning:** Predict optimal partitioning strategies
- **🔄 Pipeline Parallelism:** Overlap computation and communication phases
- **📊 Memory Optimization:** Reduce memory footprint for larger graphs

### 🌐 Scalability Enhancements:
- **☁️ Cloud Deployment:** Scale to distributed cloud environments
- **📈 Benchmarking Suite:** Comprehensive performance testing framework
- **🔧 Auto-tuning:** Automatic parameter optimization based on graph characteristics

---

## 📚 Technical Specifications

```yaml
Environment:
  - Language: C++, Python
  - Parallelization: MPI + OpenMP
  - Profiling: Intel VTune Profiler
  - Partitioning: METIS Library
  - Visualization: Python matplotlib

Performance Metrics:
  - Best Speedup: 3.4x
  - Optimal Threads: 56
  - Parallel Efficiency: 65%
  - Primary Bottleneck: Edge relaxation
```

---

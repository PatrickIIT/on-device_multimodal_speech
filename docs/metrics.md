# Evaluation Metrics & Mathematical Formulations

## 1. Speech Recognition Quality

### Word Error Rate (WER)
Word Error Rate measures the minimum token-level edit distance between a ground-truth reference sequence and the model hypothesis:

$$\text{WER} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}$$

Where:
* $S$: Number of word substitutions.
* $D$: Number of word deletions.
* $I$: Number of word insertions.
* $C$: Number of correct words.
* $N = S + D + C$: Total number of words in reference transcript.

### Character Error Rate (CER)
Character Error Rate evaluates phonemic and morphological recognition fidelity:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

Where $S_c, D_c, I_c$ represent character-level substitutions, deletions, and insertions over total reference characters $N_c$.

---

## 2. Latency Metrics

### First Chunk Latency (FCL)
The time elapsed from the arrival of the first acoustic chunk to the emission of the first partial decoding token:

$$\text{FCL} = t_{\text{first\_token}} - t_{\text{first\_chunk\_arrival}}$$

### Chunk Processing Time (CPT)
The execution time required to process a single acoustic chunk $k$:

$$\text{CPT}_k = t_{\text{chunk\_end}, k} - t_{\text{chunk\_start}, k}$$

### User-Perceived Latency (UPL) / End-of-Utterance Latency (EUL)
The latency between the end of user speech and the emission of the final transcript token:

$$\text{EUL} = t_{\text{final\_token\_emitted}} - t_{\text{final\_chunk\_submitted}}$$

### Latency Percentiles
For an ordered sequence of $M$ chunk processing times $\{L_{(1)}, L_{(2)}, \dots, L_{(M)}\}$:
* **P50 / Median**: $L_{(\lceil 0.50 \cdot M \rceil)}$
* **P90**: $L_{(\lceil 0.90 \cdot M \rceil)}$
* **P95**: $L_{(\lceil 0.95 \cdot M \rceil)}$

### Cold-Start vs. Warm-Start Latency
* **Cold-Start Latency**: Execution time of the very first inference pass following process initialization (includes graph compilation, weight paging, memory allocation).
* **Warm-Start Latency**: Average execution time across subsequent inference passes once caches and JIT paths are primed.

---

## 3. Computational & System Efficiency

### Real-Time Factor (RTF)
Measures the relationship between model execution time and input audio duration:

$$\text{RTF} = \frac{T_{\text{inference}}}{T_{\text{audio}}}$$

* **Interpretation**:
  * $\text{RTF} < 1.0$: Processing faster than real time (e.g. $\text{RTF} = 0.25$ means 4x faster than real time).
  * $\text{RTF} = 1.0$: Exact real-time processing boundary.
  * $\text{RTF} > 1.0$: System cannot keep up with real-time stream; buffer overflow will occur.

### Compression Ratio
$$\text{Compression Ratio} = \frac{\text{Size}_{\text{uncompressed}}}{\text{Size}_{\text{compressed}}}$$

### Memory Metrics
* **Resident Set Size (RSS)**: Total physical RAM currently allocated to the process in Megabytes (MB).
* **Peak RSS ($\text{RAM}_{\text{peak}}$)**: Maximum physical RAM recorded throughout initialization, streaming, and garbage collection.

### Energy Consumption
Energy dissipated during continuous streaming sessions:

$$E = \int_{0}^{T} V(t) \cdot I(t) \, dt \quad [\text{Watt-hours / Joules}]$$

Where $V(t)$ is battery terminal voltage and $I(t)$ is instantaneous discharge current sampled via `/sys/class/power_supply/battery/`.

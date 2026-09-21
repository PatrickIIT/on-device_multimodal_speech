# Experimental Results Registry

This directory contains empirical benchmark logs, summary tables, and generated figures.

> **CRITICAL RESEARCH INTEGRITY RULE**: Results in this directory must be generated exclusively from actual physical executions and experiments. **Do not manually fabricate, estimate, or edit experimental measurements.**

---

## Directory Organization

* `tables/`: Summary CSV tables consolidating results across experimental trials.
  * `master_results.csv`: Master repository table capturing all evaluated configurations.
* `raw/`: Unprocessed JSON logs generated directly by benchmark and evaluation scripts.
* `figures/`: Rendered Pareto frontier charts, latency distribution histograms, and WER curves.

---

## Master Table Schema

Columns in `tables/master_results.csv`:
```text
experiment,model,dataset,device,precision,sparsity,chunk_ms,wer,cer,model_size_mb,parameters,flops,latency_mean_ms,latency_p50_ms,latency_p90_ms,latency_p95_ms,ram_peak_mb,cpu_percent,rtf,energy_wh,thermal_state
```

Metric values remain empty until experiments are formally executed.

# 30D same-H100 profile, 200 measured updates

This is a measured initial-update throughput and PyTorch allocator-memory comparison on an NVIDIA H100 NVL. Both methods ran sequentially on the same GPU, with 10 warm-up and 200 measured optimizer updates each. The standalone [benchmark notebook](../../benchmark_30d_same_h100_profile200.ipynb) embeds the same SHA-256-locked 30D HCN notebook and NN4MGF source as [the repository benchmark lock](../../sources.json). This directory preserves the two unmodified JSON reports and their CSV summary.

| Method | Mean measured seconds/update | Peak allocated GiB | Planned full updates |
|---|---:|---:|---:|
| HCN | 0.165113 | 7.805 | 50,000 |
| NN4MGF | 1.601046 | 77.815 | 300,000 |

The measured per-update time ratio is **9.70×**. Multiplying these initial-update rates by the respective configured update counts gives a **58.18× projected** full-schedule ratio. No 50,000/300,000-update full pair was completed in this record, and the projection should not be labeled a measured full-run speedup. The two methods use different base inputs per update (1,280 versus 32,768), as the reports show. Peak allocated bytes are PyTorch allocator measurements, not whole-device consumption. The benchmark contains no tail evaluation; the clipping convention affects accuracy reports, not timed training updates.

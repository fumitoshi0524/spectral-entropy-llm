# Historical cosine route

This directory preserves the earlier static-anchor cosine experiment code and configuration as a separate route. It does not generate the archived entropy results in the root `results/` directory. Run it from this directory if needed; use the root `pipeline.py` for the entropy workflow.

The archive retains historical loop and data-loading behavior, including its dataset fallback. The maintained entropy data loader instead fails if the requested C4 samples cannot be loaded. No legacy weights or results are bundled here.

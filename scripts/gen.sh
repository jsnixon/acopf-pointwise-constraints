poetry run julia --project=. scripts/generate.jl \
    --casefile test/case30_ieee.json \
    --n_samples 10000 \
    --min_load 0.8 \
    --max_load 1.2 

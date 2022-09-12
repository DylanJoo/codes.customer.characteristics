for BS in 4096 1024 512 128;do
    # for BS in 1024 512 128;do
    for K in 25 50;do
        python3 train.py -k $K -bs $BS -iters 10000 > K-${K}.BS-${BS}-10000/results/log &
    done
    python3 train.py -k 100 -bs $BS -iters 10000 > K-${K}.BS-${BS}-10000/results/log
done

a=10
for BS in 4096 1024 512 128;do
# for BS in 128;do
    for K in 25 50;do
        mkdir -p K-${K}.BS-${BS}.A-${a}
        python3 train.py -alpha $a -k $K -bs $BS -iters 10000 > K-${K}.BS-${BS}.A-${a}/log &
    done
    mkdir -p K-100.BS-${BS}.A-${a}
    python3 train.py -alpha $a -k 100 -bs $BS -iters 10000 > K-100.BS-${BS}.A-${a}/log
done

a=50
for BS in 4096 1024 512 128;do
    # for BS in 1024 512 128;do
    for K in 25 50;do
        mkdir -p K-${K}.BS-${BS}.A-${a}
        python3 train.py -alpha $a -k $K -bs $BS -iters 10000 > K-${K}.BS-${BS}.A-${a}/log &
    done
    mkdir -p K-100.BS-${BS}.A-${a}
    python3 train.py -alpha $a -k 100 -bs $BS -iters 10000 > K-100.BS-${BS}.A-${a}/log
done

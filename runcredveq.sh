for nunit in 12
do
    for sc in 1. 0.
    do
        for tr in 1
        do
            for nc in 1
            do
                for mb in 0.
                do
                    for cb in 3.
                    do
                        for b in "GGD" "T"
                        do
                            python runcredvdeq.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
                        done
                    done
                done
            done
        done
    done
done


for nunit in 12
do
    for sc in 1. 0.
    do
        for tr in 1
        do
            for nc in 100
            do
                for mb in 0.
                do
                    for cb in 1.1 
                    do
                        for b in "GaussianMixture" "MultivariateGaussian"
                        do
                            python runcredvdeq.py -cb "$mb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
                        done
                    done
                done
            done
        done
    done
done

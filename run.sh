for nunit in 4 8 16 96
do
	for sc in 1.
	do
		for tr in 1 0
		do
			for nc in 4 10 100 500
			do
				for mb in 10000000.
				do
					for cb in 1.1 
					do
						for b in "GaussianMixture" "MultivariateGaussian" "GMM"
						do
							python run.py -cb "$mb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
						done
					done
				done
			done
		done
	done
done


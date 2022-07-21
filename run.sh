for nunit in 16
do
	for sc in 1.
	do
		for tr in 1 0
		do
			for nc in 1
			do
				for mb in 0. 2. 4.
				do
					for cb in 1. 2. 3. 4.
					do
						for b in "GGD" "T"
						do
							python run.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
						done
					done
				done
			done
		done
	done
done


for nunit in 16
do
	for sc in 1.
	do
		for tr in 1 0
		do
			for nc in 500 100 20 4
			do
				for mb in 0. 2. 4.
				do
					for cb in 1.1 
					do
						for b in "GaussianMixture" "MultivariateGaussian"
						do
							python run.py -cb "$mb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
						done
					done
				done
			done
		done
	done
done



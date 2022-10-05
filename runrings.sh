
for nunit in 16
do
	for sc in 1.
	do
		for tr in 1
		do
			for nc in 1
			do
				for mb in 0. 3. 6.
				do
					for cb in 2.
					do
						for b in "T" "GGD"
						do
							python runrings.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
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
		for tr in 1
		do
			for nc in 100
			do
				for mb in 0. 3. 6.
				do
					for cb in 1.1 
					do
						for b in "GaussianMixture" "MultivariateGaussian"
						do
							python runrings.py -cb "$mb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit" -b "$b"
						done
					done
				done
			done
		done
	done
done




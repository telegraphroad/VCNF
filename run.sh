for nunit in 2 4 8 16 64
do
	for sc in 100.
	do
		for cb in 1.001 1.021 1.081 1.101
		do
			for mb in 1.001 1.021 1.081 1.101
			do
				for nc in 2 4 10 20 50 100 500 1000
				do
					for tr in 1 0
					do
						python run.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit"
					done
				done
			done
		done
	done
done


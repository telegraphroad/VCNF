for nunit in 2 4 8 16 64
do
	for sc in 1. 2. 3.
	do
		for cb in 1.001 5. 10. 50.
		do
			for mb in 1.001 5. 10. 50.
			do
				for nc in 2 5 10 20 50 100 500 1000
				do
					for tr in 0 1
					do
						python run.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit"
					done
				done
			done
		done
	done
done


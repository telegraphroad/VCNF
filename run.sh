for nunit in 2 3 4 5 8 16 32 64 96
do
	for sc in 1. 2. 3. 4.
	do
		for cb in 1.001 5. 10. 20. 50. 100.
		do
			for mb in 1.001 5. 10. 20. 50. 100.
			do
				for nc in 2 3 4 5 6 7 8 9 10 15 20 50 100 200 300 400 500 1000 2000
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


for nunit in 4 8 16 32
do
	for sc in 1.
	do
		for tr in 0 1
		do
			for nc in 100 10 4
			do
				for mb in 10. 50.
				do
					for cb in 1.1 
					do
						python run.py -cb "$mb" -mb "$mb"  -sc "$sc" -nc "$nc" -trainable "$tr" -nu "$nunit"
					done
				done
			done
		done
	done
done


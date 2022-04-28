for cb in 1.0001 10. 50.
do
	for mb in 1.0001 10. 50.
	do
		for sc in 1. 2. 3.
		do
			for nc in 2 1000 3 500 4 300
			do
				python run.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc"
			done
		done
	done
done


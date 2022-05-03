for sc in 1. 2. 3.
do
	for cb in 10. 50.
	do
		for mb in 10. 50.
		do
			for nc in 2 500 3 200 4 
			do
				python run.py -cb "$cb" -mb "$mb"  -sc "$sc" -nc "$nc"
			done
		done
	done
done


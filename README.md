# DISH_python
Implementation of the DISH algorithm in Python.

[Full paper](https://doi.org/10.1016/j.swevo.2018.10.013)

## Usage
Sample use can also be seen at the end of the file __main.py__. 
```python
dim = 10 #dimension size of the optimized problem
NP = round(25 * math.log(dim) * math.sqrt(dim)) #population size (recomended setting)
maxFEs = 5000 #maximum number of objective function evaluations
H = 5 #archive size
minPopSize = 4

sphere = Sphere(dim) #defined test function
de = DISH(dim, maxFEs, sphere, H, NP, minPopSize) #initialize DISH
resp = de.run() #run the optimization
print(resp) #print the results
```
Output ``resp`` then includes optimized values ``features`` and value of objective function ``ofv``. Also, the ``id`` of particle is included.

## File descriptions
* __main.py__
  * The main file contains the main class DISH and one sample test function class Sphere.

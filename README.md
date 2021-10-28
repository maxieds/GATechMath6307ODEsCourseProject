# Math6307 ODEs Course Project (Fall 2021)

This project focuses on the exploration of numerical solutions to ODEs in Python. 
We will introduce a few generic methods for solving general purpose ODEs using standardized 
Python extension libraries (see [these examples]()). 
The exposition for the course project is centered on exploring the properties of numerical solutions 
to a few types of chaotic attractors (see [examples in the subdirectories here]()). 
In doing so, we aim to present a set of reusable ideas and methods that viewers can use to understand 
numerical solutions to ODEs more generally that arise in other applications. 

## Running the examples



## Setting up a sane Python environment on MacOS

### Minimum requirements

```bash
$ brew install python@3.9 ipython
$ brew install sage
$ python3.9 -m pip install numpy scipy sympy matplotlib notebook jupyterlab 
$ python3.9 -m pip install gekko ode-toolbox ode-explorer
```

### Configuring the Python path in the local Bash shell 

```bash 
$ cd UtilityScripts
$ /bin/bash ./AddPythonPathToBashConfig.sh
# ... On MacOS:
$ source ~/.bash_profile 
# ... On Linux:
$ source ~/.bashrc
$ cd ..
```

## Links

### Tutorials on solving ODEs numerically 

* [Ordinary Differential Equations (ODE) with Python](https://elc.github.io/posts/ordinary-differential-equations-with-python/)
* [Towards Data Science: Ordinary Differential Equation (ODE) in Python](https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b)
* [Learn Programming: Solve Differential Equations in Python](http://apmonitor.com/che263/index.php/Main/PythonDynamicSim)
* [Dyanmics and Control: Solve Differential Equations with GEKKO](https://apmonitor.com/pdc/index.php/Main/PythonDifferentialEquations) 
* [Solve differential equations in Python (with ODEINT)](https://www.pharmacoengineering.com/2018/11/29/3290/) 

### List of solid Python addon-packages/libraries to assist with numerical analysis of ODEs

* [GEKKO documentation](https://apmonitor.com/wiki/index.php/Main/GekkoPythonOptimization): Note that this package is substantially error prone and hard to use compared to ``scipy``. Nonetheless, it does seem to have some sophisticated solver capability if you understand the internals of the package well.
* [ODE-toolbox](https://ode-toolbox.readthedocs.io/en/master/): Simplifies and automates many useful procedures in simulating ODE solutions numerically or even analytically. The utility still suffers from a lack of computational motivation in Python when working with symbolic independent (indeterminate) parameters. For example, an [example given in the package docs](https://github.com/nest/ode-toolbox/blob/master/tests/lorenz_attractor.json) shows a model for the [Lorenz attractor](https://mathworld.wolfram.com/LorenzAttractor.html) given by 
```bash
{
  "dynamics": [
    {
      "expression": "x' = sigma * (y - x)",
      "initial_value" : "1"
    },
    {
      "expression": "y' = x * (rho - z) - y",
      "initial_value" : "1"
    },
    {
      "expression": "z' = x * y - beta * z",
      "initial_value" : "1"
    }
  ],
  "parameters" : {
    "sigma" : "10",
    "beta" : "8/3",
    "rho" : "28"
  }
}
```





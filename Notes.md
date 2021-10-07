# AI for Social Good – Resource Allocation Problems in Renewable Energy

## Abstract

In recent years AI has proved to be a very useful tool in approaching a variety “social good” problems, in particular the challenge of allocating a finite number of resources to a large range of locations to maximise the potential impact. These large-scale combinatorial optimisation problems have the potential to make a huge difference if tackled correctly and making the right decisions can be incredibly challenging, especially when considering the uncertainty of certain unknown variables. In this project we will focus on the allocation of a budget on the costs to produce, transport, set-up and connect a range of renewable energy sources around the UK. We will attempt to find a maximal allocation of renewable energy sources - options including solar, wind, and wave energy - across a range of different potential sites with different costs attributed to each. A large factor with these renewable sources is the impact that inconsistent weather patterns will have on their output, as such we aim to implement a model that considers this uncertainty and implement a "solver" for this model to maximise the power output of the overall system on a given budget.



## Preplan Notes


#### Resource to allocate:
* Monetary Budget
#### Costs to consider:
* Production/Purchase
* Setup costs of different energy types
  * Connection to the grid costs?
* Transport costs of components from production to end location
* Repair costs (uncertainty on chance of breakdowns over time)
#### Variable/Allocatable  factors:
* Type of energy source (Usable in UK):
    * Solar
    * Wind
    * Hydroelectric? ("Well established in the UK" [1] - Likely not much area for growth and reallocation)
    * Wave/Marine
* Location of allocations 
    * Research required to map out a dataset of applicable locations and survey data of past weather patterns to build a accurate model of weather over time
    * Balancing the more profitable areas with high variance in weather/energy generation with more consistent areas
    * Limit scope to UK (but design for change in overall location to allow for reuse in different regions?)
#### Statistically Uncertain Factors:
* Weather patterns that will affect power generation dependent on allocated regions
* Long term repair costs and degradation of efficiency
#### Maximisation goal:
* Predicted Power generation
#### Potential minimisation goal:
* CO2 Emissions in production & transportation?
* Payback time/profitability?





## Sources
```
1- https://www.energy-uk.org.uk/energy-industry/renewable-generation.html
```
# CS310 Specification - Draft

## Efficient Allocation of Renewable Energy Sources Under Uncertainty Across the UK

## Problem

One of the most substantial threats to the modern world is the climate crisis. With a need for new more renewable technologies and energy being one of the key paths forward, the problem becomes where best to site these new energy sources. With so many possible locations to choose from and a finite budget the automation of these decisions would be an obvious help.

 AI is a very useful tool for tackling optimisation problems where the number of variables creates a number of combinations too big for any real minimisation or maximisation effort to made. As such it is the perfect tool to apply to finding an efficient way of investing a budget in new renewable energy sources. As discussed, with such a wide range of varying locations across the country  the choices of where to allocate funds quickly becomes a complicated one, especially when considering the statistically uncertain factors such as maintenance and most importantly for renewables, weather. 

When considering a choice location and of energy source there are a range of factors to consider in evaluating the value of the decision. The costs of a choice will be impacted by the setup and connection costs, the initial production cost and the cost of repairs, meanwhile the output of a location will vary depending on the weather of any given day. This exposes the other area of the problem, the need to consider the statistical uncertainty of events such as faults requiring repair and "profitable" weather patterns occurring when evaluating a choice.

## Objectives

By the end of the development stage of this project a list of objectives should be met in order to determine the project a success. All of the below should be included to achieve a solution to the problem defined above, as such we expect the final program to meet the following:

### Core Objectives

1. A "performance" function will be able to evaluate a location for a given energy source type, using uncertain variables such as windspeed, sun light time and sun intensity.
2. A "cost" function will evaluate a cost of choosing a location according to the production cost of the energy source, transportation and connection costs for a given location, and an evaluation of repair costs against the chance of a fault occurring.
3. A "solver" function will implement a yet undetermined optimisation problem algorithm suited to an allocation problem of this type, and will make use of the "cost" and "performance" functions as heuristic values.
4. The program be able to access a  data set of locations, curated to be viable for allocation.
5. Given an input budget value the program will return the user a set of allocations chosen as it deems to be the most performant based on the performance function defined in objective 1.
6. The program will run under a fixed time condition such that the user should not be waiting more than 30 seconds for a result.

These are the core parts of the project which are required to have successfully implemented the goal of this project. However these objectives have some areas for expansion which could potentially be investigated given the core aspects are implemented successfully with additional time to spare.

### Potential areas for extension

1. Extend the "performance" function to consider long term trends in the weather to all for the program to make allocations based on future worth.
2. Evaluate more than one of the most relevant algorithms, benchmarking performance on time taken, accuracy, and consistency.
3. Extend the location set by allowing users to input their own location dataset, allowing the program to be used in different countries or more specific areas in the future.

## Methods

To be completed

## Timetable

To be completed

## Resources 

The main resources needed for this project is the historical weather data required to evaluate the suitability of different locations. Using a mix of sources from visualcrossing.com [1] and the Met Office[2], will help ensure a full range of data for each chosen location as well as acting as a fail safe in the unlikely event of either sources removing access to this information. The project will be built in the base Python 3.9 language, this is a widely supported language with no chance of becoming unavailable within the duration of this project. 

Sources:

- https://www.visualcrossing.com/weather-history# - detailed weather data for locations and data
- https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data

## Risks

The main risks this project will face are data loss and issues with falling behind schedule. In order to prevent and mitigate the problems caused by these risk we will take the following measures:

- Data Backup:
  - By using Git as a version control protocol for the project we can make use of GitHub's private repositories to keep a regularly updated backed up to a central online location.
  - To avoid the unlikely case of losing access to the repository causing any issues we will also be pulling up to date versions of the code base and project documentation to at least 2 different computers (a personal Laptop and Desktop most regularly).
- Time management:
  - As part of this document the timetable will help build an expectation of where the project should be every week.
  - Project management tools such as Kanban boards can be used to break down the tasks ensure regular progress is being made on the development.

## Ethical Considerations

 This project does not make use of anything requiring any considerations of this kind.


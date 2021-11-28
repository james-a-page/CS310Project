Things to talk about:

1. Whats been done so far?:
   1. Data collection
      1. Weather apis
      2. Cost information
      3. National grid data of actual generation
   2. Research and plans
   3. Solidified new plans
2. Changed ideas
   1. Changing schedule
   2. New restrictions to the plan and collected data
3. Next steps and going forward
   1. New gantt?
   2. why focusing on performance function first
   3. 3 main sections of project, performance predictor, graph map solver, if possible   





## **Progress Report**

## **Intro**

In the first 8 weeks of this project, as planned, most of the progress made so far has been research to lay the groundwork to ensure a smooth development stage, with this research done we can now refine our plans for the development. In this document we will highlight what new details we have to consider for the project and how what we have learnt has impacted the initial plans for the project. Up to this point the schedule set out in the specification document (appendix of this document) has been followed without problem, so we will begin by examining the research done during this period. Following that we will use this information to determine what has changed with the plan and discuss why these changes are important to producing the best end product possible. Finally we will outline the next steps to be taken in the project and outline an updated schedule for the upcoming weeks and argue that this new schedule is still achievable within the remaining time.

## **Current Progress**

When setting out on this project it was uncertain what types of renewable energy we would be examining after collecting a variety of sources we found that wind and solar generators had the most information available relevant to the UK, therefore the decision was made to focus into these two for more in depth research.

### Approach Research

- Costs of energy source

Information on the costs of installing solar photo-voltaic cells is extremely accessible as many people have looked to install small scale generators into their homes. However as we would be looking to install large scale solar farms with the scope of this project these price estimates may not scale accurately. To ensure we were making accurate estimations we ensured that any sources we used could be cross referenced with another accurate source. We found that the Renewable Power Generation Costs in 2019 report published by the International Renewable Energy Agency (IRENA) reported costs for everything we would need to know, reporting for Solar PV a cost of £1210 per kW on a commercial scale (https://www.irena.org/publications/2020/Jun/Renewable-Power-Costs-in-2019). This would appear to be a very reliable source as it comes from a reputable agency trusted to keep track of many different aspects of renewable energy. Nevertheless we ensured that these numbers were consistent with a report published in 2019 by the UK government which estimated installations between 10-50kW (the upper end of their scale, so most relevant to a large scale commercial project) cost an average of £1139 per kW, and that this number had fallen to £1088 per kW by 2021.

The IRENA report also reports costs of installing onshore and offshore wind farms, with similar cost breakdowns of cost per kW to the solar reports. The reported cost of an offshore wind farm was reported as £3432 per kW in 2019 which is significantly more than installing solar panels. Onshore wind is not as expensive as offshore to set up but again is more expensive than a solar equivalent capcaity at £1349 per kW in Europe

However when you compare the levied cost of electricity from the generator, which is calculated by considering the operations and maintenance costs, lifetime capacity and the economic lifetime of the project (IRENA) we see that onshore wind is significantly cheaper than the other options at a cost of £0.0503 per kWh, compared to the £0.1403 per kWh and £0.0907 per kWh of solar photo-voltaics and offshore respectively. This number does vary across different countries reported and solar may have a higher cost due it being relatively unused on a large scale in the UK and with increased adoption this may get better but currently this shows wind being the most efficient option to consider cost wise.

- Locations of generators

We found that generally with solar farms the location impact is fairly minimal over a small area like the UK, as the only impacting factor ends up being the levels of sunlight over a day. This was one of the main reasons from this point the project is restricting the scope to Wind farms only as will be discussed further in the Updated Plans sections later in this document.

When looking into suitable locations from wind farms another question comes up regarding the difference between offshore and onshore wind. In 2019 the UK had a slightly higher generation capacity of onshore wind than offshore, however the actual generation amounts were equal with both generating 9.9\% of the UK's total power each. This is due to offshore farms having access to more suitable weather conditions resulting in a higher average load factor (average percentage of generation capacity actually generated) of 40.5\% compared to the 26.5\% of onshore wind. From this it may appear advantageous to focus on offshore wind, and the increasing proportion of capacity being offshore over the last few years would also support this, but considering the levied cost of onshore wind farms being cheaper per kW generated (IRENA source) and that historical weather data is more readily available around onshore farms we will aim to focus on allocating onshore farms initially with this project with offshore being an area for expansion later.

<img src="C:\Users\james\Documents\University\Year 3\CS310\CS310Project\Research\RejectedWindApplications.png" alt="RejectedWindApplications" style="zoom:50%;" />

 Choosing a potential site to build a new wind farm is a difficult task as not just the generation can be considered, a perfect site may not be usable if you are not allowed to build there. Many wind projects get rejected as seen in figure 1 which shows all wind farm applications that have been refused recorded in the Renewable Energy Planning Database (source). As such when deciding on locations we have decided to accept some limitations by only considering existing locations and suggesting the project finds the best of these existing projects to expand upon. This is a limitation that can be easily amended in future work if a list of suitable new sites is created. We will also be making use of locations that have been approved but are under or awaiting construction which expands our potential sites vastly as shown in figure 2.

<img src="C:\Users\james\Documents\University\Year 3\CS310\CS310Project\Research\PotentialSites.png" style="zoom: 50%;" />

- Prediction 

As we found that generation rates for wind power vary depending on a variety of weather conditions we decided that the first idea to consider to predict a new generation amount would be to use regression. This will allow us to train a model against weather data for each day with a target being  the generation a wind farm actually produced on that day. By training this on a large number of different wind farms over a large set of days, we hope to be able to accurately estimate the power generated by a wind farm for any weather pattern. We will then use this to predict a generation on an "average" day for every location to predict the performance of the proposed wind farm.\\

Currently the plan is to evaluate a few different regression models to see which finds the best fit, and in the next few weeks we will begin work on constructing the data set and conducting any feature engineering required to extract latent features from the data provided. This may not be necessary and a simple regression model against weather data may be sufficient but by testing a variety of approaches we aim to find the best result possible.

### Data Collected

With the majority of the time up to this point focusing on the research aspects of the project there have not been many technical developments. The main one technical aspect though is the collection of data and a start to the building of our training and testing data sets. 

From the research stage we now have a better idea of what data we are going to require to achieve the aim we set out at the beginning. There are two main things we now know to consider, a comprehensive weather breakdown for a given location at any point in time over the span of our training time range and an actual generation amount for every onshore wind farm on every day of that same training time range.

- Weather historical data sources

In the process of searching for a good source of this weather data we came across a few potential sources. The first was the Met Office historic station data sets. This gave us access to a selection of 37 stations across the UK, each having the the following information for every month since it's opening: 

•Mean daily maximum temperature•Mean daily minimum temperature•Days of air frost•Total rainfall•Total sunshine duration

Unfortunately this did not give us the resolution we needed nor any information on wind speeds that would seem inherently important to a discussion on wind farm performance.

The option we discovered next was visualcrossing.com, this website provided access to data for any location in the UK based on the closest weather stations it has access to. It gives the resolution and span of dates we need to create a comprehensive training data set. We are able to query and date and time range, at an hourly resolution getting information about precipitation, temperature, humidity, wind speed, wind direction, wind gust speed, air pressure, and cloud cover. This amount of information should make for a plentiful feature set hopefully allowing for a accurate prediction model to be found. The weather API system allows us to automate the data collection process although their free model limits to 1000 records a day so we are planing carefully what dates and areas we require for our data set before making any large requests.

- National grid generator output

Finding values of actual generation produced by wind farms was a bit more of an involved process, from our research we have been unable to find any one specific source that gives us the full information we need. However after a bit of searching the National Grid online data portal (https://data.nationalgrideso.com/) and the Elexon Balancing Mechanism Reporting Service (BMRS) (https://www.bmreports.com/bmrs/) we were able to obtain a list of all wind farm Balancing Mechanism Units Ids (BMU\_IDs) which allows us to query the BMRS dashboard to obtain an actual generation amount at half hour intervals for any date it was operational.

 This allowed us to compile a list of all the BMU\_IDs and in the next few weeks let us start to match a location, a date-time and a power output, which can be used in combination with the weather data gathered to build our data set.

## **Updated Plan and next steps**

Now we have the research completed we have a better idea of exactly what is possible for this project. Over the course of this document we have discussed some of the areas we have narrowed the scope to improve the quality of the product we do produce. For example by restricting our predictor to work initially on strictly onshore wind power we hope to ensure that our prediction accuracy can be better than if we were split across different generator types having to consider a variety of different features. 

We believe this decision is for the best of the project as the concepts demonstrated by this more limited project will allow for expansion in the future to more easily implement other sources and locations.

From here our project will consist of 2 main targets, the predictor system and the efficient allocation algorithm. Originally our specification planned the order such that we would implement and research the allocation algorithm before tackling the performance predictor but through the research section it became clear that this would be extremely counter intuitive. Both these components are key areas requiring a lot of focus, and they should be tackled when required instead of preemptively designing the allocation system with no performance heuristic to use with it. As such, the immediate focus now is on designing tests for the first iteration of development focused on the predictive model and then compiling the data highlighted in section 3.1 into one large training data set.

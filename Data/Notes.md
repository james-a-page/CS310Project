# Data Processing

## Sources

Weather - https://www.visualcrossing.com/weather-history#

Using ElexonDataPortal library https://github.com/OSUKED/ElexonDataPortal
https://data.nationalgrideso.com/generation/daily-wind-availability/r/daily_wind_availability

## Data Stages

Initally starting with only wind options as data is most prevalent separate extening implementation will be required in future

1. Get list of generator units & correcsponding location
2. Get data set of actual generation on different dates & corresponding generator
3. Build training set as y = actual generation on date, X features including weather data for that date

### 1

- National grid wind availabilty predictions, extract the unique BMU_Ids &#9745;
- Build Dictionary of locations with ids as key &#9745;

### 2

- Need to select a set of dates randomly within a range to make up sample for data set
- With dates selected, sample generation at 4 key points through out day
- Get weather for location of generator at the 4 points throughout day
```HTTPRequest
GET https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Coventry/2021-11-11/2021-11-11?unitGroup=uk&key=6FF9G8N2T2NZCSR2C2WPZ42QF&include=hours
```

### 3

- 
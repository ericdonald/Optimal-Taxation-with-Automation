# Replication Package <br> [Optimal Taxation with Automation](https://www.ericdonald.com/research/optimal-taxation-with-automation)

## Data Sources:

Below is the list of all data sources required for replication. The first group are those programmatically retrieved via APIs or direct download, and the second group are those contained in the Raw Data folder. The links below are for reference only; a user does not need to visit these sites to extract the data.
To make use of the API commands, the user will need to make a `.keys` file with the following lines:

```
FRED_API = XX
IPUMS_API = XX
```

where `XX` is the user's API key for the relevant data source.

### API/Web Acessible:

- FRED
  - [CPI](https://fred.stlouisfed.org/series/CPIAUCSL)
- IPUMS
  - [1980 Census](https://usa.ipums.org/usa/)
  - [2016 ACS](https://usa.ipums.org/usa/)
- LLM Exposure from [Elondou et al. (2024)](https://github.com/openai/GPTs-are-GPTs/tree/main)

### Contained in Raw Data:

- [2016 SCF](https://www.federalreserve.gov/econres/scf_2016.htm)
- Occupation Exposure Scores and Crosswalk from [Webb (2020)](http://eepurl.com/gxo4zr)
- Capital-Labor Elasticities of Substitution from [Caunedo et al. (2023)](https://capitalbyoccupation.weebly.com/)

## Software Requirements:

### Python

All of the replication codes run on Python `3.11.14`. Prior to running the codes, install the following packages:

| Package | Version |
|---------|---------|
| cyipopt | 1.6.1 |
| ipumspy | 0.7.0 |
| numba | 0.63.1 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| quantecon | 0.10.1 |
| scipy | 1.17.0 |

## Description of Code:

## List of Tables and Figures:

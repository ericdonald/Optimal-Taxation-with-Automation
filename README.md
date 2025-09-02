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
  - CPI
- IPUMS
  - 1980 Census
  - 2016 ACS

### Contained in Raw Data:

- 2016 SCF
- Occupation Exposure Scores
- Occupation Crosswalk
- Capital-Labor Elasticities of Substitution

## Software Requirements:

### Python

All of the replication codes run on Python `3.11.13`. Prior to running the codes, install the following packages:
| Package | Version |
|---------|---------|
| numpy | 2.2.6 |
| pandas | 2.3.2 |
| scipy | 1.16.1 |
| matplotlib | 3.10.5 |
| numba | 0.61.2 |
| quantecon | 0.10.1 |
| statsmodels | 0.14.5 |
| openpyxl | 3.1.5 |

## Description of Code:

## List of Tables and Figures:

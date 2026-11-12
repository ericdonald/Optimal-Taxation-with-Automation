# Replication Package <br> [Optimal Taxation with Automation](https://www.ericdonald.com/research/optimal-taxation-with-automation)

## Data Sources:

Below is the list of all data sources required for replication. All necessary data files are included in the Raw Data folder, so the replication code can be run immediately with `API=0`. Setting `API=1` will re-download the data from the original sources, which may produce small numerical differences if the underlying databases have been updated since the archived data was collected.

The first group are those that can be programmatically retrieved via APIs or direct download, and the second group are those contained exclusively in the Raw Data folder and do not make use of an API. The links below are for reference only; a user does not need to visit these sites to extract the data.

To make use of the API commands, the user will need to make a `.keys` file with the following lines:

```
FRED_API = XX
IPUMS_API = XX
```

where `XX` is the user's API key for the relevant data source.

### API/Web Accessible:

- FRED
  - [CPI](https://fred.stlouisfed.org/series/CPIAUCSL)
- IPUMS
  - [1980 Census](https://usa.ipums.org/usa/)
  - [2016 ACS](https://usa.ipums.org/usa/)
- AI Exposure from [Felten et al. (2021)](https://github.com/AIOE-Data/AIOE/tree/main) and [Eloundou et al. (2024)](https://github.com/openai/GPTs-are-GPTs/tree/main)

### Contained in Raw Data:

- [2016 SCF](https://www.federalreserve.gov/econres/scf_2016.htm)
- Occupational Exposure Scores and Crosswalk from [Webb (2020)](http://eepurl.com/gxo4zr)

## Software Requirements:

### Python

All of the replication codes run on Python `3.12.15`. Prior to running the codes, install the following packages:

| Package | Version |
|---------|---------|
| cyipopt | 1.7.0 |
| ipumspy | 0.9.0 |
| numba | 0.68.0 |
| numpy | 2.5.4 |
| openpyxl | 3.1.5 |
| pandas | 3.0.6 |
| quantecon | 0.13.1 |
| requests | 2.34.2 |
| scipy | 1.18.1 |
| ipopt | 3.14.20 |

## Description of Code:

### Setup Instructions

Before running the code, create the following folders in the repository root:

```
Raw Data/
Clean Data/
Results/
Results/Tables/
Results/Figures/
```

Download the [raw data](https://github.com/ericdonald/Optimal-Taxation-with-Automation/releases/download/v1.0.0/Raw.Data.zip), unzip, and place the file(s) directly in Raw Data/.

## List of Tables and Figures:

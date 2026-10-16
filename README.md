# Housing Affordability Dynamics: Price-to-Rent Ratios Across U.S. Metro Areas, 2020–August 2026

## Project overview

This project examines how home values changed relative to rents across the 100 largest U.S. metropolitan areas from January 2020 through August 2026.

> **Research question:** How did home values change relative to rents across U.S. metropolitan areas, and which metros experienced the largest differences between home-price growth and rent growth?

Using Zillow Research data, the analysis calculates price-to-rent ratios, compares cumulative home-price and rent growth, examines lagged correlations, estimates a linear regression, and applies K-means clustering to identify groups of metros with similar growth patterns.

### Key findings

- Home values increased by approximately 49%, compared with 43% growth in rents over the study period.
- The median price-to-rent ratio peaked at approximately 18.1 in late 2022 before declining to around 17.2 by August 2026.
- McAllen, Texas, had the largest home-price and rent growth divergence among the top 10 metros identified.
- San Jose, California, had the largest increase in the price-to-rent ratio among the top 10 metros identified.
- Rent growth was positively associated with home-value growth 12 months later, but the regression had limited explanatory power (\(R^2 = 0.165\)).
- K-means clustering identified three groups of metropolitan areas based on cumulative home-price and rent growth.

These findings describe associations and trends, not causal effects.

## Data source

**Zillow Research — Housing Data**  
[https://www.zillow.com/research/data/](https://www.zillow.com/research/data/)

The project uses two datasets:

| File | Description |
|---|---|
| `Metro_zhvi.csv` | Zillow Home Value Index (ZHVI) |
| `Metro_zori.csv` | Zillow Observed Rent Index (ZORI) |

The datasets are filtered to January 2020–August 2026 and focus on the 100 largest metropolitan areas by Zillow's `SizeRank`.

The CSV files should be placed in the `Data/` directory. Include them in the repository only if sharing is permitted. Otherwise, download the datasets from Zillow Research and follow the instructions below.

## Repository structure

```text
Housing-Affordability-Analysis/
├── Data/
│   ├── Metro_zhvi.csv
│   └── Metro_zori.csv
├── report/
│   └── Housing_Affordability_Report.pdf
├── presentation/
│   └── Housing_Affordability_Presentation.pdf
├── output/
├── Mini_Project_Thao_Nguyen.ipynb
├── README.md
└── requirements.txt
```

## Required software and packages

- Python 3.9 or later
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`

## Steps to run the analysis

### 1. Get the repository

Clone or download this GitHub repository and open a terminal in the project directory.

### 2. Obtain the data

Download the ZHVI and ZORI datasets from [Zillow Research](https://www.zillow.com/research/data/) and place them in `Data/` with the filenames specified above.

### 3. Install the packages

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `Mini_Project_Thao_Nguyen.ipynb` and select **Kernel → Restart & Run All**.

Run the notebook from the repository root so that the relative paths to the `Data/` directory resolve correctly.

## Methods in brief

- **Data preparation:** Reshapes and merges monthly home-value and rent data, selects the largest metros, and handles missing observations.
- **Price-to-rent ratio:** Calculates home values divided by annualized rents.
- **Growth analysis:** Compares cumulative home-price growth, rent growth, and their divergence.
- **Time-series analysis:** Examines annual growth rates and changes in median price-to-rent ratios.
- **Regression:** Tests the association between rent growth and home-value growth 12 months later.
- **K-means clustering:** Groups metropolitan areas based on cumulative home-price and rent growth.

## Limitations

- The price-to-rent ratio does not account for household income, mortgage rates, taxes, insurance, or maintenance costs.
- Data coverage varies across metropolitan areas.
- The analysis focuses on the 100 largest metros and may not represent smaller cities or rural areas.
- Regression results show associations, not causation.
- K-means results depend on the selected variables and number of clusters.
- Zillow may revise historical data, so results may differ across dataset versions.

## References

- Zillow Research. *Housing Data*. https://www.zillow.com/research/data/
- Additional external sources, code, images, and ideas used in the notebook, report, and presentation should be cited in the relevant materials.

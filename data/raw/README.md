# Raw Data Sources

Due to their large size, the raw datasets used in this research are not included in this repository. To reproduce the analysis, please download the files directly from the official open data portals of the respective cities.

### 📥 How to Download
1. Follow the links provided below.
2. On the portal page, locate the **Export** button (usually situated in the top-right corner).
3. Select the **CSV** format to download the complete dataset.
4. Place the downloaded `.csv` files into this `data/raw/` directory before running the preprocessing pipeline.

### ⚠️ Important: File Naming
Ensure your downloaded `.csv` filenames match the `input` fields in the `DATASETS` dictionary inside `base.py`. The script supports wildcards (e.g., `Philly_*.csv`) and comma-separated lists for split datasets.

*Note: If you only want to run the correlator, you do not need to download raw files. Preprocessed tables are already included in the repository.*

### 🗄️ Datasets

* **Chicago Crimes**  
  [Download from Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/data_preview)

* **Los Angeles Crimes**  
  [Download Part 1 (2010–2019)](https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z/data_preview)  
  [Download Part 2 (2020–Present)](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/data_preview)

* **New York City (NYC) Crimes**  
  [Download from NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/data_preview)

* **Philadelphia Crimes**  
  [Download from OpenDataPhilly](https://opendataphilly.org/datasets/crime-incidents/)

* **San Francisco Crimes**  
  [Download from DataSF](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/data_preview)

* **NYC 911 Calls for Service**  
  [Download from NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Calls-for-Service-Historic-/d6zx-ckhd/data_preview)

* **NYC EMS (Emergency Medical Services) Dispatch**  
  [Download from NYC Open Data](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj/data_preview)
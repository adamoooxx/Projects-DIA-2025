
# Species Distribution Prediction - Backend

This is the Flask backend for the Species Distribution Prediction application.

## Setup Instructions

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the required data folders:
   - Create a `MODEL` folder with machine learning models:
     - `species_prediction_ESP1_optimized.pkl` (for sardines)
     - `species_prediction_ESP2_optimized.pkl` (for rails)
   - Create a `VF` folder with the Morocco shapefile (`T_cote_Maroc_MAJ_MAI_06_2021.shp`)
   - Create a `GRILLE` folder with the grid shapefile (`grill_adam.shp`)

4. Run the application:
   ```
   python app.py
   ```

The server will start at http://localhost:5000

## API Endpoints

- `POST /`: Upload CSV file and generate prediction map
  - Request: multipart/form-data with:
    - `file`: CSV file with required columns
    - `species`: Either "sardine" or "rails"
  - Response: `{"image_path": "path/to/generated/image"}`

- `GET /download/<species>`: Download generated map
  - Path parameter: species ("sardine" or "rails")
  - Response: Image file download

## CSV Format

The CSV file should have the following columns:
- LAT_DD: Latitude (decimal degrees)
- LONG_DD: Longitude (decimal degrees)
- Salinite: Salinity
- Temp: Temperature
- DO: Dissolved Oxygen
- pH: pH level

Values can use comma or period as decimal separator.

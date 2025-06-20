# Affective States

This repository contains the code for the Bachelor End Project of the Bachelor Data Science at Tilburg University (TiU) and Eindhoven University of Technology (TU/e), titled "Simple versus Ensemble Regression Models for Affective State Prediction" by Zita Godrie.

## Project Structure

- `data/` - Place the G-REx dataset by Bota et al. (2024) here (not included in this repository).
- `notebooks/` - Jupyter notebooks for data exploration, feature engineering (including preprocessing), model training and evaluation.
- `results/` - Contains CSV files with model evaluation scores.
- `requirements.txt` - List of Python dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zitagodrie/affective-states.git
   ```
2. Navigate to the project directory:
   ```bash
   cd affective-states
   ```
3. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Place the G-REx dataset in the `data/` directory (see Data Description below).
- Open and run the Jupyter notebooks in the `notebooks/` directory for data analysis, feature engineering, and model evaluation.
- Results will be saved in the `results/` directory.

## Data Description

For a description of the data used in this project, see: 'A real-world dataset of group emotion experiences based on physiological data' by Bota et al. (2024).

*Note: The dataset is not included in this repository due to licensing restrictions. Please refer to the original publication for access.*

## License

This repository is made available under the MIT license.

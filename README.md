# Energy Consumption Predictor ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web application built with Streamlit that predicts energy consumption based on environmental and operational parameters. This application provides facility managers and energy professionals with insights to optimize energy usage and improve efficiency.

![Energy Consumption Predictor Screenshot](https://via.placeholder.com/800x400?text=Energy+Consumption+Predictor)

## Features

- **Real-time Energy Prediction**: Input environmental conditions and building parameters to get instant energy consumption forecasts
- **Interactive Dashboard**: Visualize energy consumption patterns and correlations
- **Efficiency Analysis**: Receive insights and recommendations to optimize energy usage
- **Parameter Comparison**: Compare current settings against optimal ranges
- **User-friendly Interface**: Easy-to-use sliders and input fields for quick adjustments

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [How to Contribute](#how-to-contribute)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/amsoorya/energy-consumption-predictor.git
   cd energy-consumption-predictor
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model or train your own following the instructions in the `/model` directory.

### Requirements

- streamlit>=1.24.0
- pandas>=1.5.0
- numpy>=1.22.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- scikit-learn>=1.0.0
- joblib>=1.1.0

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501).

3. Use the sidebar to input environmental parameters, building characteristics, and system states.

4. Click the "Predict Energy Consumption" button to get results.

5. Explore the different tabs:
   - **Prediction**: View predicted energy consumption and recommendations
   - **Data Visualization**: Analyze historical energy usage patterns
   - **Model Information**: Learn about the prediction model and feature importance

## Project Structure

```
energy-consumption-predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── model/
│   ├── energy_prediction_model.pkl   # Pre-trained model
│   └── train_model.py                # Script for training the model
├── data/
│   ├── sample_data.csv              # Sample data for visualization
│   └── historical_data.csv          # Historical data for model training
├── docs/
│   └── user_guide.md                # User documentation
└── README.md                 # Project documentation
```

## Model Information

The energy prediction model uses machine learning to forecast energy consumption based on multiple input parameters:

- Environmental conditions (temperature, humidity)
- Building characteristics (square footage, occupancy)
- System states (HVAC usage, lighting usage)
- Time parameters (day of week, time of day, season)
- Renewable energy contribution

Performance metrics:
- R² Score: 0.89
- RMSE: 15.7
- MAE: 11.2

## How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Jaya Soorya - [@amsoorya](https://github.com/amsoorya) - amjayasoorya@gmail.com

Project Link: [https://github.com/amsoorya/energy-consumption-predictor](https://github.com/amsoorya/energy-consumption-predictor)

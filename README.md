# 🚍 HU.BER: Optimizing Public Transportation Routes
### **Hackathon 2024 - Challenge 1**  
**Course**: Introduction to Machine Learning
---

## 🏁 Overview
HU.BER is an innovative project aimed at optimizing public transportation in Israel using machine learning techniques. Given the frustration of irregular bus schedules, packed buses, and long travel times, our goal is to improve the efficiency and experience of public transport. We use real-world bus data to predict passenger boardings and trip durations, and provide insights on improving public transit systems.

## 📊 Dataset
The dataset `train_bus_schedule.csv` contains 226,112 records of bus stop information, with various features describing each stop. These features allow us to predict key outcomes like the number of passengers boarding the bus and trip durations.

---

## 🚀 Tasks

### 1. Predicting Passenger Boardings at Bus Stops
- **Input**: Data for a single bus stop (excluding passengers_up).
- **Output**: A CSV `passengers_up_predictions.csv` with two columns:
  - `trip_id_unique_station`: A unique identifier for the stop.
  - `passengers_up`: The predicted number of passengers boarding.
- **Evaluation**: Predictions are evaluated using **Mean Squared Error (MSE)**.

### 2. Predicting Trip Duration
- **Input**: Data for a complete bus trip, excluding the arrival times for intermediate stops.
- **Output**: A CSV `trip_duration_predictions.csv` with two columns:
  - `trip_id_unique`: Unique identifier for the trip.
  - `trip_duration_in_minutes`: Predicted trip duration.
- **Evaluation**: Predictions are evaluated using **MSE**.
---

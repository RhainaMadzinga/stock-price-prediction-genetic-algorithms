# Stock Price Prediction using Genetic Algorithms

A Java-based machine learning project that uses a **Genetic Algorithm (GA)** to predict whether the price of a financial asset will increase over the following 14 days.

## Overview

This project combines **technical indicator feature engineering** with a **Genetic Algorithm** to evolve simple, interpretable prediction rules. The system was developed as part of a Computational Intelligence mini project.

The pipeline includes:
- loading and preprocessing historical price data
- computing technical indicators
- converting indicators into binary input features
- evolving rule-based classifiers using a Genetic Algorithm
- evaluating performance on unseen test data

## Features

- Implemented fully in **Java**
- Uses 5 technical indicators:
  - SMA
  - EMA
  - TBR
  - VOL
  - MOM
- Converts indicators into 6 binary features
- 15-bit chromosome representation
- Tournament selection
- One-point crossover
- Bit-flip mutation
- 70% training / 30% testing split
- Interpretable evolved rules

## Technical Indicators Used

The following indicators were computed from the price series:

- **SMA(10)** – short-term simple moving average
- **SMA(30)** – medium-term simple moving average
- **EMA(10)** – exponential moving average
- **TBR(20)** – trade breakout relative to previous 20-day maximum
- **VOL(20)** – rolling volatility
- **MOM(5)** – short-term momentum

These were transformed into the following binary inputs:

- SMA10 > SMA30
- Price > SMA10
- EMA10 > SMA30
- TBR20 > 0
- VOL20 > 0.02
- MOM5 > 0

## Genetic Algorithm Design

Each individual uses a **15-bit chromosome**:

- 12 bits encode whether each of the 6 binary features is used and whether it is used positively or negated
- 3 bits encode a voting threshold

The GA uses:
- **Population size:** 50
- **Generations:** 100
- **Crossover rate:** 0.8
- **Mutation rate:** 0.01
- **Tournament size:** 3

## Results

Across repeated runs, the model achieved:

- **Mean training accuracy:** ~58.05%
- **Mean test accuracy:** ~54.33%

This showed stable behaviour and performance above random guessing, while also evolving simple and interpretable rules.

## Files

- `rm828.java` – main Java implementation
- `PriceData.csv` – input dataset
- `RM828.pdf` – project report

## How to Run

Compile:

```bash
javac src/rm828.java

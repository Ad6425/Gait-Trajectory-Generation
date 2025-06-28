# Gait-Trajectory-Generation
This project analyzes and predicts human gait kinematics from Kinect treadmill data using an LSTM model.

## Gait Analysis and Prediction using LSTM
This project demonstrates how to process gait data from Kinect treadmill records, train an LSTM model to predict gait angles, and visualize the results. The process includes data loading, cleaning, filtering, interpolation, angle calculation, model training, and plotting actual vs. predicted angles and phase plots.

## Project Structure
The project is designed to be run sequentially in a Google Colab notebook. The main steps are:

- Setup and Data Extraction: Mount Google Drive, unzip the dataset, and define helper functions and configuration parameters.
- Load Participant Data: Load demographic information for participants.
- Process Gait Data: Iterate through individual data files, apply filtering and interpolation to standardize gait cycles, calculate joint angles, and concatenate data into a single DataFrame.
- Prepare Data for LSTM: Prepare the processed data for LSTM training by creating sequences and scaling features.
- Train LSTM Model: Define and train an LSTM model to predict gait angles.
- Evaluate Model (Single Cycle): Test the trained model on a single gait cycle and calculate RMSE.
- Plot Actual vs. Predicted: Visualize the actual and predicted gait angles for the selected test cycle.
- Plot Phase Heatmaps: Generate phase space heatmaps for ankle angles (angle vs. angular velocity) to visualize gait patterns.

## Running the Project on Google Colab
To run this project, you will need a Google account and access to Google Colab.

- Open the Notebook: Open the project notebook in Google Colab.
- Mount Google Drive: Run the first code cell to mount your Google Drive. This is necessary if your dataset is stored there.
- Upload or Link Dataset: Ensure your dataset (104_Kinect_treadmill_records (1).zip and 1_Participants information.xlsx) is accessible from your Colab environment. The notebook expects the zip file to be located at /content/drive/MyDrive/104_Kinect_treadmill_records (1). Modify the path in the second code cell if your file is located elsewhere or if you upload it directly to the Colab session storage.
- Run All Cells: Execute all the code cells in the notebook sequentially. You can do this by going to Runtime > Run all in the Colab menu.
- Dataset Link: [[Link]](https://ieee-dataport.org/open-access/kinematic-gait-data-using-microsoft-kinect-v2-sensor-during-gait-sequences-over)![image](https://github.com/user-attachments/assets/b8e8d9d2-8754-46f8-b69e-0b36de33378e)

Note: The processing step (cell 3) can take some time as it iterates through all participant files. The model training (cell 5) also requires computational resources and time.

## Dependencies
All necessary libraries are imported within the notebook cells. Google Colab provides most common data science libraries pre-installed (like pandas, numpy, scikit-learn, tensorflow, matplotlib).

## Visualizations
The notebook generates the following visualizations:

- Predicted Knee Angle vs. Hip Angle: A phase plot showing the relationship between predicted knee and hip angles over time for a selected individual's cycles.
- Predicted Angle Change per Step vs. Angle (Phase Plots): Phase space plots for various predicted joint angles, showing the angle's rate of change against the angle itself.
- Left Ankle Phase Heatmap: A heatmap visualizing the density of (angle, angular velocity) points for the left ankle across processed cycles of a selected individual.
- Right Ankle Phase Heatmap: A heatmap visualizing the density of (angle, angular velocity) points for the right ankle across processed cycles of a selected individual.
- Actual vs. Predicted Angles: Time series plots comparing the actual and predicted angles for each output joint angle for a single selected gait cycle.
These plots help in understanding the predicted gait patterns and evaluating the model's performance visually.

## Model Architecture
The project uses a simple LSTM model for gait angle prediction:

- Input Layer: Takes a sequence of dynamic features (joint angles, walking speed, stride length).
- LSTM Layers: Stacked LSTM layers (including a Bidirectional LSTM) process the sequence data.
- TimeDistributed Dense Layer: Applies a Dense layer independently to each time step of the LSTM output to predict the output features (joint angles) for that step.
- The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss.

## Results
The notebook outputs the RMSE (Root Mean Squared Error) for each predicted angle on a test set, providing a quantitative measure of the model's performance. Visualizations further illustrate how well the model predicts the angle trajectories and captures the phase space characteristics of gait.

# Explanation of Program Parts
Here is a breakdown of the different parts of the program:

## Data Acquisition
This part of the program is responsible for loading the raw data into the Colab environment and making it accessible for processing.

- Mount Google Drive: The code first mounts your Google Drive to access files stored there. This is done using the google.colab.drive library.
- Unzip Dataset: The compressed dataset file (expected to be a zip file containing Excel files) is then unzipped into a designated directory within the Colab environment. This is handled using Python's zipfile and os libraries, along with a shell command (!unzip).
- Load Participant Information: An Excel file containing demographic information about the participants (like age, sex, height, weight) is loaded into a pandas DataFrame. This data will be used later to enrich the gait data.

## Data Processing
This is a crucial and extensive part of the program where the raw gait data from individual files is cleaned, transformed, and standardized for model training.

- Iterate Through Files: The code loops through each individual participant's data file (expected to be Excel files starting with "K3").
- Load and Select Columns: Each file is loaded into a pandas DataFrame, and only a predefined set of relevant columns (mostly joint coordinates and 'Steps') are initially extracted.
- Match Participant Data: The participant ID from the file name is used to look up and add their personal information (specifically 'Age' and the original 'File name') to the DataFrame.
- Apply Butterworth Filter: A low-pass Butterworth filter is applied to smooth the time-series data for most coordinate columns. This helps remove noise.
- Identify Gait Cycles: The 'Steps' column is used to identify individual gait cycles. A cycle is typically defined by two consecutive step increments.
- Filter Cycles by Length: Cycles whose lengths (number of data points) fall outside a certain range (defined by the mean and standard deviation of all cycle lengths) are discarded. This removes abnormally short or long cycles.
- Calculate Gait Metrics: For each valid cycle, metrics like cycle duration, stride length, and walking speed are calculated.
- Interpolate Cycles: Each remaining gait cycle is then interpolated to a fixed target length (e.g., 50 data points). This standardization is necessary for training the LSTM model, which requires fixed-length input sequences. PchipInterpolator is used for this.
- Calculate Joint Angles: Using the interpolated 3D joint coordinates, the angles at the knees, ankles, and hips are calculated for each time step within the interpolated cycle.
- Concatenate Processed Data: The processed data from all valid cycles across all participants is combined into a single large pandas DataFrame (common_df).

## Model Training
This section defines and trains the LSTM model using the processed and prepared data.

- Prepare Data for LSTM: The common_df is transformed into sequences of data points suitable for an LSTM. This involves:
  - Extracting dynamic features (interpolated joint angles, walking speed, stride length).
  - Creating input sequences (X_dynamic) and corresponding target sequences (y). Each input sequence is a window of sequence_length time steps of dynamic features, and the target sequence is the next sequence_length time steps of output features (joint angles).
  - Splitting the data into training, validation, and test sets.
  - Scaling the dynamic features using StandardScaler to normalize their ranges.
- Define Model Architecture: An LSTM model is defined using TensorFlow/Keras. The architecture includes input layers, LSTM layers (including a Bidirectional LSTM for potentially capturing dependencies in both forward and backward directions), and a TimeDistributed Dense layer to produce predictions for each time step in the output sequence.
- Compile Model: The model is compiled with an optimizer (Adam) and a loss function (Mean Squared Error - MSE), which is appropriate for regression tasks like angle prediction.
- Train Model: The compiled model is trained using the training data (X_train, y_train), with validation data (X_val, y_val) used to monitor performance during training and prevent overfitting.

## Evaluation
After training, the model's performance is evaluated.

- Predict on Test Set: The trained model is used to make predictions on the unseen test data (X_test).
- Calculate RMSE: The Root Mean Squared Error (RMSE) is calculated for each predicted joint angle by comparing the model's predictions (predictions) to the actual values (y_test). RMSE provides a measure of the typical difference between predicted and actual values, in the original units (degrees, after conversion).

## Visualization
This part focuses on visualizing the results to gain insights into the model's performance and the characteristics of the gait data.

- Plot Actual vs. Predicted Angles: For a selected individual's gait cycle, time series plots are generated to visually compare the actual trajectory of each joint angle against the trajectory predicted by the model.
- Plot Predicted Knee vs. Hip Angle: A phase plot is created showing the relationship between the predicted knee and hip angles over the entire concatenated segment of processed cycles for a selected individual.
- Plot Predicted Phase Plots: Phase space plots (angle vs. angular velocity) are generated for each predicted joint angle across the concatenated segment of processed cycles. This visualizes the dynamic behavior of the predicted angles.
- Plot Ankle Phase Heatmaps: Heatmaps are generated for the left and right ankle angles, showing the density of (angle, angular velocity) points across the processed cycles of a selected individual. These heatmaps provide a statistical view of the ankle's phase space trajectory and can reveal common patterns. Data points are repeated to make the density more pronounced in the heatmap visualization. A frequency threshold is applied to hide bins with very few data points.

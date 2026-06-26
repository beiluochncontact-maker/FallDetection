import config
from preprocessing import data
from pathlib import Path
import pandas as pd

# Label reader
def label_reader():
 
    label_root = config.LABEL_DATA_DIR
    label_dict = {}

    for subject_id in range(6, 39):

        subject = f"S{subject_id:02d}"
        excel_path = label_root / f"SA{subject_id:02d}_label.xlsx"
        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            continue

        for _, row in df.iterrows():

            trial = row["Trial ID"]
            onset = row["Fall_onset_frame"]
            impact = row["Fall_impact_frame"]
            label_dict[(subject, trial)] = {

                "onset": 0 if pd.isna(onset) else int(onset),
                "impact": 0 if pd.isna(impact) else int(impact)
            }

    return label_dict


# Sensor reader
def sensor_reader():
  
    sensor_root = config.SENSOR_DATA_DIR
    sensor_dict = {}

    for subject_id in range(6, 39):

        subject = f"S{subject_id:02d}"
        folder = sensor_root / f"SA{subject_id:02d}"

        for csv_path in sorted(folder.glob("*.csv")):

            filename = csv_path.stem
            # S06T02R03

            trial = filename[len(subject):]
            # T02R03

            sensor_dict[(subject, trial)] = {
                "csv_path": csv_path
            }

    return sensor_dict


if __name__ == "__main__":

    sensor = sensor_reader()
    labels = label_reader()

    print(next(iter(sensor.items())))
    print(next(iter(labels.items())))

    print("success")
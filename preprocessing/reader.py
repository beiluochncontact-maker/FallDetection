import config
import re
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
            df["Task Code (Task ID)"] = df["Task Code (Task ID)"].ffill()
            df["Description"] = df["Description"].ffill()
        except FileNotFoundError:
            continue

        for _, row in df.iterrows():

            task_code = str(row["Task Code (Task ID)"])
            match = re.search(r"\((\d+)\)", task_code)
            task_id = int(match.group(1))
            trial_id = row["Trial ID"]
            trial = f"T{task_id:02d}R{trial_id:02d}"

            onset = row["Fall_onset_frame"]
            impact = row["Fall_impact_frame"]
            description = str(row["Description"]).strip()
            fall_type = config.DESCRIPTION_MAP.get(description, -1)
            label_dict[(subject, trial)] = {
                "onset": onset,
                "impact": impact,
                "fall_type": fall_type
            }

    return label_dict


# Sensor reader
def sensor_reader():
  
    sensor_root = config.SENSOR_DATA_DIR
    sensor_dict = {}

    for subject_id in range(6, 39):

        subject = f"S{subject_id:02d}"
        folder = sensor_root / f"SA{subject_id:02d}"

        if not folder.exists():
            continue

        for csv_path in sorted(folder.glob("*.csv")):

            filename = csv_path.stem
            trial = filename[len(subject):]

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
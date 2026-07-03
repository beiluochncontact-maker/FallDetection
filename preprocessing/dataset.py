from preprocessing import reader


def dataset_builder():

    sensor_dict = reader.sensor_reader()
    label_dict = reader.label_reader()
    dataset = {}

    for key, sensor_data in sensor_dict.items():

        if key in label_dict:
            onset = label_dict[key]["onset"]
            impact = label_dict[key]["impact"]

        else:
            onset = 0
            impact = 0

        dataset[key] = {
            "subject": key[0],
            "trial": key[1],
            "csv_path": sensor_data["csv_path"],
            "onset": onset,
            "impact": impact,
            #"fall_type": label_dict[key]["fall_type"],
            "is_fall": key in label_dict
        }

    return dataset


if __name__ == "__main__":

    dataset = dataset_builder()

    # test
    first_key = next(iter(dataset))
    print(first_key)
    print(dataset[first_key])

    print("success")
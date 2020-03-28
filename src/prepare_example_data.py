from why.examples.carinsurance import CarInsurance, InsuranceTransformer
from why.examples import utils
from why.utils import get_root_dir


def main():
    config = utils.read_data_config("car_insurance_cold_calls.json")
    datapath = get_root_dir() / "data"
    train, test = CarInsurance(config=config, datapath=datapath / "raw").prepare_data()
    InsuranceTransformer(
        config, train
    ).split_train_valid().transform_train_valid().save_train_valid(
        datapath / "processed" / "carinsurance"
    )


if __name__ == "__main__":
    main()

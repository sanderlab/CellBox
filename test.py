import pytest
import os
import glob
import numpy as np

from test_utils.dataloader import get_dataloader, yield_data_from_tensorflow_dataloader, yield_data_from_pytorch_dataloader, \
    s2c_row_inds


#def test_model():
#    os.system('python scripts/main.py -config=configs/Example.minimal.json')
#    files = glob.glob('results/Debugging_*/seed_000/3_best.W*')
#    assert False


#################################################### Tests for DataLoaders ####################################################

# Test for correct shape
def test_correct_shape():
    """
    A function to test if the batch yielded by both Tensorflow and Pytorch has the same shape
    """
    experiment_config_path = "/users/ngun7t/Documents/cellbox-jun-6/configs_dev/Example.random_partition.json"
    tensorflow_dataloader_list = get_dataloader(experiment_config_path, tensorflow_code=True)
    pytorch_dataloader_list = get_dataloader(experiment_config_path, tensorflow_code=False)

    # Code to extract the shape of each yield
    for tf_dict, torch_dict in zip(tensorflow_dataloader_list, pytorch_dataloader_list):
        tf_train_pert, tf_train_expr = yield_data_from_tensorflow_dataloader(
            dataloader=tf_dict["iter_train"],
            feed_dict=tf_dict["feed_dict"]
        )
        torch_train_pert, torch_train_expr = yield_data_from_pytorch_dataloader(
            dataloader=torch_dict["iter_train"]
        )

        # Assert that the count of batches obtained is equal
        assert len(tf_train_pert) == len(torch_train_pert), "Length of number of arrays yield for train pert not equal"
        assert len(tf_train_expr) == len(torch_train_expr), "Length of number of arrays yield for train expr not equal"

        # Assert that the shape of each batch is equal
        for tf_arr, torch_arr in zip(tf_train_pert, torch_train_pert):
            assert tf_arr.shape == np.array(torch_arr).shape, f"For pert batches, shape of tf batch = {tf_arr.shape} is not equal to shape of torch batch = {np.array(torch_arr).shape}"

        # Assert that the shape of each batch is equal
        for tf_arr, torch_arr in zip(tf_train_expr, torch_train_expr):
            assert tf_arr.shape == np.array(torch_arr).shape, f"For expr batches, shape of tf batch = {tf_arr.shape} is not equal to shape of torch batch = {np.array(torch_arr).shape}"


# Test for correct input
def test_single_to_combo():
    """
    A function to test if pytorch and tensorflow dataloaders yield the correct rows in the dataset for s2c experiment
    """
    experiment_config_path = "/users/ngun7t/Documents/cellbox-jun-6/configs_dev/Example.single_to_combo.json"
    loo_label_dir = "/users/ngun7t/Documents/cellbox-jun-6/data/loo_label.csv"
    tensorflow_dataloader_list = get_dataloader(experiment_config_path, tensorflow_code=True)
    pytorch_dataloader_list = get_dataloader(experiment_config_path, tensorflow_code=False)

    # Get the row index that contains single drugs
    rows_with_single_drugs, rows_with_multiple_drugs = s2c_row_inds(loo_label_dir)

    # Code to extract the shape of each yield
    for tf_dict, torch_dict in zip(tensorflow_dataloader_list, pytorch_dataloader_list):
        tf_train_pert, tf_train_expr = yield_data_from_tensorflow_dataloader(
            dataloader=tf_dict["iter_train"],
            feed_dict=tf_dict["feed_dict"]
        )
        torch_train_pert, torch_train_expr = yield_data_from_pytorch_dataloader(
            dataloader=torch_dict["iter_train"]
        )
        # Assert that the count of batches obtained is equal
        assert len(tf_train_pert) == len(torch_train_pert), "Length of number of arrays yield for train pert not equal"
        assert len(tf_train_expr) == len(torch_train_expr), "Length of number of arrays yield for train expr not equal"

        # Assert that the shape of each batch is equal, and also it contains the correct row index
        for tf_arr, torch_arr in zip(tf_train_pert, torch_train_pert):
            assert tf_arr.shape == np.array(torch_arr).shape, f"For pert batches, shape of tf batch = {tf_arr.shape} is not equal to shape of torch batch = {np.array(torch_arr).shape}"
            assert np.intersect1d(tf_arr[:, -1], rows_with_multiple_drugs).size == 0, f"batches for tf train set contains data rows that has multiple drugs in s2c mode"
            assert np.intersect1d(torch_arr[:, -1], rows_with_multiple_drugs).size == 0, f"batches for torch train set contains data rows that has multiple drugs in s2c mode"

        # Assert that the shape of each batch is equal
        for tf_arr, torch_arr in zip(tf_train_expr, torch_train_expr):
            assert tf_arr.shape == np.array(torch_arr).shape, f"For expr batches, shape of tf batch = {tf_arr.shape} is not equal to shape of torch batch = {np.array(torch_arr).shape}"
            assert np.intersect1d(tf_arr[:, -1], rows_with_multiple_drugs).size == 0, f"batches for tf train set contains data rows that has multiple drugs in s2c mode"
            assert np.intersect1d(torch_arr[:, -1], rows_with_multiple_drugs).size == 0, f"batches for torch train set contains data rows that has multiple drugs in s2c mode"
               

if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])

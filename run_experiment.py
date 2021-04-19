import json
import os
import sys
import datetime as dt
from uuid import uuid4
import codecs
import munch
import pickle
from compress_pickle import dump
from shutil import copyfile
# from wealth_optimizer.data_slicing import slice_dataset
from wealth_optimizer.common_logger import logger
from wealth_optimizer.data_loader import ExcelDataReader
from wealth_optimizer.executors import ModelExecutor
from wealth_optimizer.models import ModelSelector
from wealth_optimizer.utils import validate_input_config
from wealth_optimizer.performance_stats import plot_asset_allocation_quantiles, plot_asset_allocations
from wealth_optimizer.performance_stats import plot_wealth_trajectories
from wealth_optimizer.performance_stats import plot_trajectory_mean_and_quantiles
from wealth_optimizer.performance_stats import plot_final_wealth_histogram

if __name__ == '__main__':

    config_file = sys.argv[1]
    config_path = os.path.join('configs', config_file)

    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    validate_input_config(config)
    for k, v in config.__dict__.items():
        logger.info('{}: {}'.format(k, v))

    run_date_time = dt.datetime.now()
    experiment_id = str(uuid4().hex)
    run_name = run_date_time.strftime('%Y-%m-%dT%H%M%S') + '_' + config.experiment_name + '_' + experiment_id

    path_to_data_file = os.path.join('datasets', config.dataset_type, config.dataset_name)
    result_dir = os.path.join('results', run_name)
    os.mkdir(result_dir)
    copyfile(config_path, os.path.join(result_dir, 'experiment_config.json'))

    # load data
    reader = ExcelDataReader(path_to_data_file)
    full_data = reader.data

    # create executor and run
    model = ModelSelector().get_model(config.control_model)
    executor = ModelExecutor(full_data,
                             model=model,
                             parameters=config)

    result_object = executor.execute()

    dump(result_object, os.path.abspath(os.path.join(result_dir, 'results')), compression="gzip", set_default_extension=True)

    try:
        Y_LIMITS = tuple(config.chart_limits)
    except:
        Y_LIMITS = (-0.40, 0.40)

    plot_final_wealth_histogram(result_object.test.final_wealth, initial_wealth=config.initial_wealth,
                                target_wealth=config.target_wealth, save_path=result_dir, label='test_set')

    plot_wealth_trajectories(result_object.test.wealth_trajectories,
                             plot_every_nth=1,
                             show=False, save_path=result_dir,
                             label='test_set')

    # plot_wealth_trajectories([result_object.test.backtest.wealth_trajectories],
    #                          plot_every_nth=1,
    #                          show=False, save_path=result_dir,
    #                          label='historical_backtest')
    #
    # plot_asset_allocations([result_object.test.backtest.controls], assets=result_object.config.assets,
    #                        plot_every_nth=1,
    #                        show=False,
    #                        y_limits=Y_LIMITS,
    #                        save_path=result_dir,
    #                        label='historical_backtest')
    #
    plot_asset_allocations(result_object.test.controls, assets=result_object.config.assets,
                           plot_every_nth=1,
                           show=False,
                           y_limits=Y_LIMITS,
                           save_path=result_dir,
                           label='test_set')
    plot_asset_allocation_quantiles(result_object.test.controls,
                                    assets=result_object.config.assets,
                                    quantiles=[0.05, 0.95],
                                    show=False,
                                    y_limits=Y_LIMITS,
                                    save_path=result_dir,
                                    label='test_set')

    plot_trajectory_mean_and_quantiles(result_object.test.wealth_trajectories, quantiles=[0.05, 0.95], show=False,
                                       save_path=result_dir, label='test_set')

    # plt.figure()
    # # epoch_training_losses = [edata['training_loss'] for edata in self.epoch_data]
    # # epoch_validation_losses = [edata['validation_loss'] for edata in self.epoch_data]
    # # epochs = [i for i, _ in enumerate(self.epoch_data)]
    #
    # plt.plot(epochs, epoch_training_losses)
    # plt.title('Training loss')
    # plt.xlabel('Epoch')
    #
    # plt.figure()
    # plt.plot(epochs, epoch_validation_losses)
    # plt.title('Validation loss')
    # plt.xlabel('Epoch')
    #
    # plt.figure()
    # plot_trajectory_mean_and_quantiles(test_paths_wealth_trajectories, quantiles=[0.05, 0.95], show=False)
    #
    # plt.show()

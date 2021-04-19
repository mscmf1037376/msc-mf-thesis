"""Utilities"""

REQUIRED_CONFIG_FIELDS = ['experiment_name',
                          'description',
                          'dataset_type',
                          'dataset_name',
                          'train_length',
                          'validation_length',
                          'test_length',
                          'overlap_validation_training',
                          'min_num_epochs',
                          'max_num_epochs',
                          'num_hidden_units',
                          'num_train_paths',
                          'num_validation_paths',
                          'num_test_paths',
                          'horizon',
                          'target_wealth',
                          'initial_wealth',
                          'expected_block_size_train',
                          'expected_block_size_test',
                          'initial_learning_rate',
                          'loss_function',
                          'control_model']


class ConfigValidationError(Exception):
    pass


def validate_input_config(config_dict):
    for field in REQUIRED_CONFIG_FIELDS:
        if field not in config_dict.keys():
            raise ConfigValidationError('Field: {} missing from configuration'.format(field))

import yaml
import argparse


class ConfigLoader:
    ################################################################################
    # ArgParse and Helper Functions #
    ################################################################################
    def get_config(config_path="config.yaml"):
        with open(config_path, "r") as setting:
            config = yaml.load(setting, Loader=yaml.FullLoader)
        return config

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-config', '--config', required=True,
                            type=str, help='path to the config file')
        parser.add_argument('--test', action='store_true',
                            help='flag: test mode')
        args = vars(parser.parse_args())
        return args

    def print_config(config):
        print("**************** CONFIGURATION ****************")
        for key in sorted(config.keys()):
            val = config[key]
            keystr = "{}".format(key) + (" " * (24 - len(key)))
            print("{} -->   {}".format(keystr, val))
        print("**************** CONFIGURATION ****************")

    def generate_model_name(config):
        model_name = config['model_name']
        if config['pretrain']:
            model_name += '_pretrain_CNN'
            model_name += '_classes' + str(config['pretrain_classes'])
        else:
            if not config['cnn_pretrained']:
                model_name += '_no_CNN_pretrained'
            if config['adaptation']:
                model_name += '_adaptation'
                if config['adaptation_pretrained']:
                    model_name += '_pretrained'
            model_name += '_dataset' + str(config['dataset_index'])
        model_name += '_epoch' + str(config['num_epochs'])
        model_name += '_round' + str(config['training_round'])
        model_name += '_batch' + str(config['batch_size'])
        model_name += '_lr' + str(config['learning_rate'])
        return model_name
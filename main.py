from src.misc import ETL, Analizer, Model, CLIManager, easter_egg
import pandas as pd

def full_process(retrain: bool=True):
    etl = ETL()
    analizer = Analizer(*etl.etl())
    model = Model()
    model.load_data(*analizer.select_features())
    model.train(retrain=retrain)
    model.validate()
    model.registry_models()
    model.publish()


if __name__ == "__main__":
    cli = CLIManager()
    cli.register_command("etl", ETL().etl)
    cli.register_command("full_process", full_process)
    cli.register_command("analize", Analizer().analize)
    cli.register_command("model_data", Model().load_data)
    cli.register_command("train", Model().train)
    cli.register_command("validate", Model().validate)
    cli.register_command("registry_models", Model().registry_models)
    cli.register_command("publish", Model().publish)
    cli.register_command("easter", easter_egg)
    cli.run()

import typer
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params, PopulatedToolingEnvironment
from provenaclient import ProvenaClient, Config
from ProvenaInterfaces.RegistryModels import ItemModelRun
from ProvenaInterfaces.RegistryAPI import ModelRunFetchResponse
from provenaclient.auth import DeviceFlow
from models import BulkStudyLink
from rich import print
import asyncio
from functools import wraps
from typing import cast, List


def coro(f):  # type: ignore
    @wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# Typer CLI typing hint for parameters
ParametersType = List[str]

# Establish env manager
env_manager = EnvironmentManager(environment_file_path="../environments.json")
valid_env_str = env_manager.environment_help_string

app = typer.Typer(pretty_exceptions_show_locals=False)

SAVE_DIRECTORY = "dumps"


def setup_client(env: PopulatedToolingEnvironment) -> ProvenaClient:
    print("setting up client...")
    config = Config(domain=env.domain, realm_name=env.realm_name)
    auth = DeviceFlow(config=config, client_id="client-tools")
    print("client ready...")
    print()
    return ProvenaClient(auth=auth, config=config)


@app.command()
@coro
async def list_datasets(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target for clear and initialisation. If targeting a feature deployment, then this is used for the keycloak instance. One of: {valid_env_str}.",
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    # Process optional environment replacement parameters
    params = process_params(param)
    env = env_manager.get_environment(name=env_name, params=params)

    # provena client
    client = setup_client(env)

    print("Fetching datasets")
    print(await client.datastore.list_all_datasets())


@app.command()
@coro
async def bulk_link_studies(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target for clear and initialisation. If targeting a feature deployment, then this is used for the keycloak instance. One of: {valid_env_str}.",
    ),
    json_path: str = typer.Argument(
        ...,
        help=f"The path to the bulk study link JSON file. See models.py#BulkStudyLink for format."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    
    Links a selection of model runs to a study. 
    
    expects a JSON file - see models.py#BulkStudyLink for spec.

    Parameters
    ----------
    env_name : _type_, optional
        _description_, by default typer.Argument( ..., help=f"The tooling environment to target for clear and initialisation. If targeting a feature deployment, then this is used for the keycloak instance. One of: {valid_env_str}.", )
    json_path : str, optional
        _description_, by default typer.Argument( ..., help=f"The path to the bulk study link JSON file. See models.py#BulkStudyLink for format." )
    param : _type_, optional
        _description_, by default typer.Option( [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")

    Raises
    ------
    e
        _description_
    """
    # Process optional environment replacement parameters
    params = process_params(param)
    env = env_manager.get_environment(name=env_name, params=params)

    # provena client
    client = setup_client(env)

    print("parsing JSON")

    spec = None
    try:
        spec = BulkStudyLink.parse_file(json_path)
    except Exception as e:
        print("Failed parsing...")
        raise e

    assert spec

    # check study is valid
    print("Validating study")
    try:
        await client.registry.study.fetch(id=spec.study)
    except Exception as e:
        print(f"Couldn't fetch study with ID {spec.study}! Skipping...")
        print(f"Exception: {e}...")
        exit(1)

    # for each model run, link to the study
    for model_run_id in spec.model_runs:
        print(f"Fetching model run {model_run_id}")
        existing : ModelRunFetchResponse
        
        # get existing
        try:
            existing = await client.registry.model_run.fetch(id=model_run_id)
        except Exception as e:
            print(f"Couldn't fetch ID {model_run_id}! Skipping...")
            print(f"Exception: {e}...")
            continue

        # then check roles
        if existing.roles is None or not 'metadata-write' in existing.roles:
            print(
                f"Couldn't modify ID {model_run_id} due to insuffient permission! Skipping...")
            continue

        # now update
        assert isinstance(existing.item, ItemModelRun)
        item: ItemModelRun = existing.item
        record = item.record
        record.study_id = spec.study
        print(f"Updating model run {model_run_id}")

        try:
            await client.prov_api.update_model_run(
                model_run_id=model_run_id, reason=f"(bulk) linking to study {spec.study}", record=record)
        except Exception as e:
            print(f"Couldn't update ID {model_run_id}! Skipping...")
            print(f"Exception: {e}...")
            continue

if __name__ == "__main__":
    app()

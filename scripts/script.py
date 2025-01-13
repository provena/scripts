import typer
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params, PopulatedToolingEnvironment
from provenaclient import ProvenaClient, Config
from ProvenaInterfaces.RegistryModels import ItemModelRun
from ProvenaInterfaces.RegistryAPI import ModelRunFetchResponse
from ProvenaInterfaces.DataStoreAPI import CredentialsRequest
from provenaclient.auth import DeviceFlow
from models import BulkStudyLink, PermanentlyDelete
from utils import format_size
from rich import print
import asyncio
from functools import wraps
from typing import cast, List, Dict, Any, Set
import boto3  # type: ignore
import re


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
        existing: ModelRunFetchResponse

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


@app.command()
@coro
async def permanently_delete_files(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target for clear and initialisation. If targeting a feature deployment, then this is used for the keycloak instance. One of: {valid_env_str}.",
    ),
    json_path: str = typer.Argument(
        ...,
        help=f"The path to the permanently delete JSON file. See models.py#PermanentlyDelete for format."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the operation instead of previewing"
    ),
    output_report: str = typer.Option(
        None,
        help="Relative file path to dump an output report to, if desired."
    ),
    allow_new_deletion: bool = typer.Option(
        False,
        help="Allow deletion of files that have never been deleted before. If false, will error if such files are found."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Permanently deletes all versions of files with the given regexes. 

    The regexes are relative to the base storage location of the dataset in Provena.

    Expects a JSON file - see models.py#PermanentlyDelete for spec.

    By default, will not allow deletion of files that have never been deleted before
    (i.e., don't have an existing delete marker). Use --allow-new-deletion to override.
    """

    # Process optional environment replacement parameters
    params = process_params(param)
    env = env_manager.get_environment(name=env_name, params=params)

    # provena client
    client = setup_client(env)

    print("parsing JSON")

    spec = None
    try:
        spec = PermanentlyDelete.parse_file(json_path)
        print("Spec parsed successfully")
    except Exception as e:
        print("Failed parsing...")
        raise e

    assert spec

    # Now read the dataset
    dataset = (await client.datastore.fetch_dataset(spec.dataset_id)).item
    if (not dataset):
        raise Exception("Dataset could not be read successfully.")

    # Now generate read credentials
    creds = await client.datastore.generate_write_access_credentials(
        credentials=CredentialsRequest(console_session_required=False, dataset_id=spec.dataset_id))

    if (not creds.status.success or not creds.credentials):
        raise Exception(
            f"Dataset credentials could not be generated. Message: {creds.status.details or 'unknown'}.")

    # adjusted regex - noting the key returning in listing will be relative to the bucket so don't include bucket name
    adjusted_regex = [dataset.s3.path + r for r in spec.regexes]

    # setup an s3 session with these creds
    cred_object = {
        'aws_access_key_id': creds.credentials.aws_access_key_id,
        'aws_secret_access_key': creds.credentials.aws_secret_access_key,
        'aws_session_token': creds.credentials.aws_session_token,
    }

    # s3 client with specific credentials scoped to this dataset
    s3 = boto3.client('s3', **cred_object)

    # Store matching files and their versions
    matching_files: Dict[str, Any] = {}
    
    # Track which files have delete markers
    files_with_delete_markers: Set[str] = set()

    try:
        # List all objects in the bucket to match against regexes
        paginator = s3.get_paginator('list_object_versions')

        # Paginate through all versions of all objects in the given prefix location
        for page in paginator.paginate(Bucket=dataset.s3.bucket_name, Prefix=dataset.s3.path):
            # Process versions
            if 'Versions' in page:
                for version in page['Versions']:
                    key = version['Key']
                    for regex in adjusted_regex:
                        # Compile the regex pattern
                        pattern = re.compile(regex)
                        if pattern.match(key):
                            if key not in matching_files:
                                matching_files[key] = []
                            matching_files[key].append({
                                'VersionId': version['VersionId'],
                                'LastModified': version['LastModified'],
                                'Size': version['Size']
                            })

            # Process delete markers
            if 'DeleteMarkers' in page:
                for marker in page['DeleteMarkers']:
                    key = marker['Key']
                    for regex in adjusted_regex:
                        # Compile the regex pattern
                        pattern = re.compile(regex)
                        if pattern.match(key):
                            if key not in matching_files:
                                matching_files[key] = []
                            matching_files[key].append({
                                'VersionId': marker['VersionId'],
                                'LastModified': marker['LastModified'],
                                'IsDeleteMarker': True
                            })
                            files_with_delete_markers.add(key)

        # If no files match, exit
        if not matching_files:
            print("No files found matching the provided regex patterns.")
            return

        # Check for files without delete markers
        files_without_delete_markers = set(
            matching_files.keys()) - files_with_delete_markers
        if files_without_delete_markers:
            if not allow_new_deletion:
                print("\nERROR: Found files that have never been deleted before:")
                for file_key in files_without_delete_markers:
                    print(f"  - {file_key}")
                print("\nTo delete these files, rerun with --allow-new-deletion flag.")
                return

        # Print comprehensive list of files to be deleted
        report: List[str] = []

        def build(content: str) -> None:
            report.append(content)

        build("\nFiles matching deletion criteria:")
        build("=================================")

        total_versions = 0
        for file_key, versions in matching_files.items():
            build(f"\nFile: {file_key}")
            if file_key in files_without_delete_markers:
                build("WARNING: This file has never been deleted before")
            build("Versions:")
            for version in sorted(versions, key=lambda x: x['LastModified']):
                total_versions += 1
                build(f"  - Version ID: {version['VersionId']}")
                build(f"    Last Modified: {version['LastModified']}")
                if 'Size' in version:
                    build(f"    Size: {format_size(version['Size'])}")
                if version.get('IsDeleteMarker'):
                    build("    (Delete Marker)")

        build(f"\nTotal files: {len(matching_files)}")
        build(f"Total versions: {total_versions}")
        if files_without_delete_markers:
            build(
                f"WARNING: {len(files_without_delete_markers)} files have never been deleted before")

        report_str = "\n".join(report)
        print(report_str)

        if (output_report):
            try:
                with open(output_report, 'w') as f:
                    f.write(report_str)
            except Exception as e:
                print(f"Failed to write report to {output_report}, error: {e}")

        # If apply flag is not set, exit here
        if not apply:
            print("\nThis was a dry run. Use --apply to perform deletion.")
            return

        # Special confirmation for files that have never been deleted
        if files_without_delete_markers:
            confirmation = input(
                f"\nWARNING: You are about to delete {len(files_without_delete_markers)} files that have never been deleted before. Are you sure? (yes/no): ")
            if confirmation.lower() != 'yes':
                print("Deletion cancelled.")
                return

        # Final confirmation for all deletions
        confirmation = input(
            "\nAre you sure you want to permanently delete these files and all their versions? (yes/no): ")
        if confirmation.lower() != 'yes':
            print("Deletion cancelled.")
            return

        # Perform deletion
        print("\nDeleting files...")
        for file_key, versions in matching_files.items():
            print(f"Deleting {file_key}...")
            for version in versions:
                try:
                    s3.delete_object(
                        Bucket=dataset.s3.bucket_name,
                        Key=file_key,
                        VersionId=version['VersionId']
                    )
                    print(f"  Deleted version {version['VersionId']}")
                except Exception as e:
                    print(
                        f"  Error deleting version {version['VersionId']}: {str(e)}")

        print("\nDeletion complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    app()

import typer
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params
from provenaclient import ProvenaClient, Config
from provenaclient.utils.config import APIOverrides
from ProvenaInterfaces.RegistryModels import ItemModelRun, ItemStudy, ItemSubType, ItemBase
from ProvenaInterfaces.RegistryAPI import ModelRunFetchResponse
from ProvenaInterfaces.DataStoreAPI import CredentialsRequest
from ProvenaInterfaces.AsyncJobAPI import JobStatus
from provenaclient.auth import DeviceFlow
from models import BulkStudyLink, PermanentlyDelete, BulkRelodgeModelRun
from utils import format_size
from rich import print
import asyncio
from functools import wraps
import boto3  # type: ignore
import re
from ProvenaInterfaces.ProvenanceAPI import PostDeleteGraphResponse
from pydantic import BaseModel
from rich.table import Table
from rich.console import Console
import json
from typing import List, Dict, Any, Tuple, Set, Optional
import statistics
from enum import Enum
from pydantic import BaseModel


SAVE_DIRECTORY = "dumps"


def coro(f):  # type: ignore
    """Decorator to turn async function into sync for typer compatibility."""
    @wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# Typer CLI typing hint for parameters
ParametersType = List[str]

# Establish env manager
env_manager = EnvironmentManager(environment_file_path="../environments.json")
valid_env_str = env_manager.environment_help_string

# Console output channel
console = Console()

# Disable locals in exception output for security
app = typer.Typer(pretty_exceptions_show_locals=False)


def setup_client(param: ParametersType, env_name: str) -> ProvenaClient:
    """
    Set up and return a configured Provena client.

    Args:
        env: The populated tooling environment configuration

    Returns:
        ProvenaClient: Configured client instance
    """

    # Process optional environment replacement parameters
    print("Parsing params and environment...")
    params = process_params(param)
    env = env_manager.get_environment(name=env_name, params=params)
    print("Setting up client...")
    # Apply all overrides - will either be original or overridden from
    # params/environment.json
    config = Config(
        domain=env.domain,
        realm_name=env.realm_name,
        api_overrides=APIOverrides(
            auth_api_endpoint_override=env.auth_api_endpoint,
            datastore_api_endpoint_override=env.datastore_api_endpoint,
            registry_api_endpoint_override=env.registry_api_endpoint,
            prov_api_endpoint_override=env.prov_api_endpoint,
            search_api_endpoint_override=env.search_api_endpoint,
            search_service_endpoint_override=env.search_service_endpoint,
            handle_service_api_endpoint_override=env.handle_service_api_endpoint,
            jobs_service_api_endpoint_override=env.jobs_service_api_endpoint,
            keycloak_endpoint_override=env.keycloak_endpoint
        )
    )
    auth = DeviceFlow(config=config, client_id="client-tools")
    client = ProvenaClient(auth=auth, config=config)
    print("Client ready...")
    print()
    return client


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
    client = setup_client(
        env_name=env_name, param=param
    )
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
    client = setup_client(
        env_name=env_name, param=param
    )

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
    client = setup_client(
        env_name=env_name, param=param
    )

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
    compiled_regex = [re.compile(dataset.s3.path + r) for r in spec.regexes]

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
                    for pattern in compiled_regex:
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
                    for pattern in compiled_regex:
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

# Model for bulk deletion JSON file


class BulkModelRunDeletion(BaseModel):
    """
    Schema for bulk model run deletion JSON file.

    Attributes:
        model_run_ids: List of model run IDs to delete
    """
    model_run_ids: List[str]


def analyze_deletion_diff(diff_data: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Analyze the deletion diff to count removed nodes and links.

    Args:
        diff_data: List of diff action dictionaries

    Returns:
        Tuple of (removed_nodes_count, removed_links_count)
    """
    removed_nodes = 0
    removed_links = 0

    print(diff_data)
    for action in diff_data:
        action_type = action.get('action_type', '')
        if action_type == 'REMOVE_NODE':
            removed_nodes += 1
        elif action_type == 'REMOVE_LINK':
            removed_links += 1

    return removed_nodes, removed_links


def display_single_deletion_summary(model_run_id: str, diff_response: PostDeleteGraphResponse) -> None:
    """
    Display summary information for a single model run deletion.

    Args:
        model_run_id: The ID of the model run being deleted
        diff_response: The response from the trial deletion
    """
    removed_nodes, removed_links = analyze_deletion_diff(diff_response.diff)

    table = Table(title=f"Model Run Deletion Summary: {model_run_id}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")

    table.add_row("Nodes to be removed", str(removed_nodes))
    table.add_row("Links to be removed", str(removed_links))
    table.add_row("Total diff actions", str(len(diff_response.diff)))

    console.print(table)


def display_bulk_deletion_summary(deletion_stats: List[Tuple[str, int, int]]) -> None:
    """
    Display summary statistics for bulk model run deletion.

    Args:
        deletion_stats: List of tuples containing (model_run_id, nodes_removed, links_removed)
    """
    if not deletion_stats:
        console.print("[red]No deletion statistics available[/red]")
        return

    # Calculate statistics
    nodes_counts = [stat[1] for stat in deletion_stats]
    links_counts = [stat[2] for stat in deletion_stats]

    avg_nodes = statistics.mean(nodes_counts) if nodes_counts else 0
    avg_links = statistics.mean(links_counts) if links_counts else 0

    # Calculate standard deviation for outlier detection
    if len(nodes_counts) > 1:
        stdev_nodes = statistics.stdev(nodes_counts)
        stdev_links = statistics.stdev(links_counts)
    else:
        stdev_nodes = 0
        stdev_links = 0

    # Create summary table
    summary_table = Table(title="Bulk Deletion Summary")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Total model runs", str(len(deletion_stats)))
    summary_table.add_row("Average nodes per deletion", f"{avg_nodes:.1f}")
    summary_table.add_row("Average links per deletion", f"{avg_links:.1f}")
    summary_table.add_row("Total nodes to be removed", str(sum(nodes_counts)))
    summary_table.add_row("Total links to be removed", str(sum(links_counts)))

    console.print(summary_table)

    # Identify and display outliers (more than 2 standard deviations from mean)
    outliers = []
    for model_run_id, nodes, links in deletion_stats:
        if (abs(nodes - avg_nodes) > 2 * stdev_nodes and stdev_nodes > 0) or \
           (abs(links - avg_links) > 2 * stdev_links and stdev_links > 0):
            outliers.append((model_run_id, nodes, links))

    if outliers:
        console.print("\n[yellow]Outliers detected (>2σ from mean):[/yellow]")
        outliers_table = Table()
        outliers_table.add_column("Model Run ID", style="yellow", no_wrap=True)
        outliers_table.add_column("Nodes", style="red")
        outliers_table.add_column("Links", style="red")
        outliers_table.add_column("Deviation", style="orange1")

        for model_run_id, nodes, links in outliers:
            node_dev = f"±{abs(nodes - avg_nodes):.1f}" if stdev_nodes > 0 else "N/A"
            link_dev = f"±{abs(links - avg_links):.1f}" if stdev_links > 0 else "N/A"
            deviation = f"N: {node_dev}, L: {link_dev}"

            outliers_table.add_row(
                model_run_id,
                str(nodes),
                str(links),
                deviation
            )

        console.print(outliers_table)


@app.command()
@coro
async def delete_model_run(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target. One of: {valid_env_str}.",
    ),
    model_run_id: str = typer.Argument(
        ...,
        help="The ID of the model run to delete."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the deletion instead of running in trial mode. Defaults to False (trial mode)."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Delete a single model run by ID from both registry and provenance store.

    This command will:
    1. First run in trial mode to show what would be deleted
    2. Ask for confirmation if --apply is specified
    3. Delete the model run from both registry and provenance if confirmed

    The deletion removes the model run from both the registry AND the provenance graph,
    including all associated nodes and relationships.
    """
    client = setup_client(
        env_name=env_name, param=param
    )

    console.print(
        f"[cyan]Analyzing deletion of model run: {model_run_id}[/cyan]")

    try:
        # Always run trial mode first to analyze the deletion
        console.print(
            "[yellow]Running trial mode to analyze deletion...[/yellow]")
        trial_response = await client.prov_api.admin.delete_model_run_provenance_and_registry(
            model_run_id=model_run_id,
            trial_mode=True
        )

        # Display summary of what would be deleted
        display_single_deletion_summary(model_run_id, trial_response)

        if not apply:
            console.print(
                "\n[green]Trial mode complete. Use --apply to perform actual deletion.[/green]")
            return

        # Confirm deletion
        console.print(
            f"\n[red]WARNING: This will permanently delete model run {model_run_id}[/red]")
        console.print("[red]This action cannot be undone![/red]")

        confirmation = typer.confirm(
            "Are you sure you want to proceed with the deletion?",
            default=False
        )

        if not confirmation:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            return

        # Perform actual deletion
        console.print(f"[red]Deleting model run {model_run_id}...[/red]")
        deletion_response = await client.prov_api.admin.delete_model_run_provenance_and_registry(
            model_run_id=model_run_id,
            trial_mode=False
        )

        console.print(
            f"[green]Successfully deleted model run {model_run_id}[/green]")

    except Exception as e:
        console.print(
            f"[red]Error deleting model run {model_run_id}: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
@coro
async def delete_model_runs(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target. One of: {valid_env_str}.",
    ),
    json_path: str = typer.Argument(
        ...,
        help="Path to JSON file containing list of model run IDs to delete. See BulkModelRunDeletion schema."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the deletion instead of running in trial mode. Defaults to False (trial mode)."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Delete multiple model runs from a JSON file containing a list of IDs.

    This command will:
    1. Parse the JSON file to get model run IDs
    2. Run trial mode for each model run to analyze deletions
    3. Show summary statistics and outliers
    4. Ask for confirmation if --apply is specified
    5. Delete all model runs if confirmed

    Expected JSON format:
    {
        "model_run_ids": ["id1", "id2", "id3", ...],
    }
    """
    client = setup_client(
        env_name=env_name, param=param
    )

    # Parse JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        deletion_spec = BulkModelRunDeletion(**data)
    except FileNotFoundError:
        console.print(f"[red]JSON file not found: {json_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON format: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error parsing JSON file: {str(e)}[/red]")
        raise typer.Exit(1)

    if not deletion_spec.model_run_ids:
        console.print("[yellow]No model run IDs found in JSON file.[/yellow]")
        return

    console.print(
        f"[cyan]Analyzing deletion of {len(deletion_spec.model_run_ids)} model runs...[/cyan]")

    # Run trial mode for all model runs to collect statistics
    deletion_stats: List[Tuple[str, int, int]] = []
    failed_analyses: List[str] = []

    for i, model_run_id in enumerate(deletion_spec.model_run_ids, 1):
        try:
            console.print(
                f"[yellow]Analyzing {i}/{len(deletion_spec.model_run_ids)}: {model_run_id}[/yellow]")

            trial_response = await client.prov_api.admin.delete_model_run_provenance_and_registry(
                model_run_id=model_run_id,
                trial_mode=True
            )

            removed_nodes, removed_links = analyze_deletion_diff(
                trial_response.diff)
            deletion_stats.append((model_run_id, removed_nodes, removed_links))

        except Exception as e:
            console.print(
                f"[red]Failed to analyze {model_run_id}: {str(e)}[/red]")
            failed_analyses.append(model_run_id)

    # Display summary statistics
    if deletion_stats:
        display_bulk_deletion_summary(deletion_stats)

    # Report any failures
    if failed_analyses:
        console.print(
            f"\n[red]Failed to analyze {len(failed_analyses)} model runs:[/red]")
        for failed_id in failed_analyses:
            console.print(f"  - {failed_id}")

    if not apply:
        console.print(
            "\n[green]Trial mode complete. Use --apply to perform actual deletions.[/green]")
        return

    if not deletion_stats:
        console.print(
            "[red]No model runs could be analyzed. Aborting deletion.[/red]")
        raise typer.Exit(1)

    # Confirm bulk deletion
    total_nodes = sum(stat[1] for stat in deletion_stats)
    total_links = sum(stat[2] for stat in deletion_stats)

    console.print(
        f"\n[red]WARNING: This will permanently delete {len(deletion_stats)} model runs[/red]")
    console.print(f"[red]Total nodes to be removed: {total_nodes}[/red]")
    console.print(f"[red]Total links to be removed: {total_links}[/red]")
    console.print("[red]This action cannot be undone![/red]")

    confirmation = typer.confirm(
        "Are you sure you want to proceed with the bulk deletion?",
        default=False
    )

    if not confirmation:
        console.print("[yellow]Bulk deletion cancelled.[/yellow]")
        return

    # Perform actual deletions
    successful_deletions: List[str] = []
    failed_deletions: List[str] = []

    for i, (model_run_id, _, _) in enumerate(deletion_stats, 1):
        try:
            console.print(
                f"[red]Deleting {i}/{len(deletion_stats)}: {model_run_id}[/red]")

            await client.prov_api.admin.delete_model_run_provenance_and_registry(
                model_run_id=model_run_id,
                trial_mode=False
            )

            successful_deletions.append(model_run_id)

        except Exception as e:
            console.print(
                f"[red]Failed to delete {model_run_id}: {str(e)}[/red]")
            failed_deletions.append(model_run_id)

    # Final summary
    console.print(
        f"\n[green]Successfully deleted {len(successful_deletions)} model runs[/green]")
    if failed_deletions:
        console.print(
            f"[red]Failed to delete {len(failed_deletions)} model runs:[/red]")
        for failed_id in failed_deletions:
            console.print(f"  - {failed_id}")


class BulkActionMode(Enum):
    """Enumeration for bulk action modes."""
    APPLY_ALL = "apply_all"
    INDIVIDUAL = "individual"


class ModelRunAction(BaseModel):
    """Model for a model run action."""
    model_run_id: str
    action_type: str  # "remove" or "replace"
    replacement_study_id: Optional[str] = None


class StudyDeletionAnalysis(BaseModel):
    """Analysis result for a single study deletion."""
    study_id: str
    study_name: str
    model_run_connections: List[str]
    unsupported_connections: List[Tuple[str, str]]


class BulkStudyDeletion(BaseModel):
    """Schema for bulk study deletion JSON file."""
    study_ids: List[str]


async def analyze_study_connections(client: ProvenaClient, study_id: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Analyze connections to a study and categorize them.

    Args:
        client: The Provena client
        study_id: ID of the study to analyze

    Returns:
        Tuple of (model_run_connections, unsupported_connections)
        where unsupported_connections is a list of (id, type) tuples
    """
    # Explore connections
    upstream = (await client.prov_api.explore_upstream(starting_id=study_id, depth=1)).graph
    downstream = (await client.prov_api.explore_downstream(starting_id=study_id, depth=1)).graph

    # Collect all connected entities
    connected_ids = set()
    if hasattr(upstream, 'nodes'):
        for node in upstream.nodes:
            if node.id != study_id:  # Exclude the study itself
                connected_ids.add(node.id)
    if hasattr(downstream, 'nodes'):
        for node in downstream.nodes:
            if node.id != study_id:  # Exclude the study itself
                connected_ids.add(node.id)

    # Analyze each connected entity
    model_run_connections = []
    unsupported_connections = []

    for connected_id in connected_ids:
        try:
            # Fetch the connected entity to determine its type
            connected_response = await client.registry.general_fetch_item(id=connected_id)
            connected_item_possible = connected_response.item

            if not connected_item_possible:
                console.print(
                    f"[yellow]Warning: Could not fetch connected entity {connected_id}[/yellow]")
                continue

            connected_item = ItemBase.model_validate(connected_item_possible)

            if connected_item.item_subtype == ItemSubType.MODEL_RUN:
                model_run_connections.append(connected_id)
            else:
                unsupported_connections.append(
                    (connected_id, connected_item.item_subtype))

        except Exception as e:
            console.print(
                f"[red]Error analyzing connected entity {connected_id}: {str(e)}[/red]")
            unsupported_connections.append((connected_id, "UNKNOWN"))

    return model_run_connections, unsupported_connections


async def analyze_single_study_deletion(client: ProvenaClient, study_id: str) -> StudyDeletionAnalysis:
    """
    Analyze a single study for deletion, including connections.

    Args:
        client: The Provena client
        study_id: ID of the study to analyze

    Returns:
        StudyDeletionAnalysis object containing all analysis results

    Raises:
        Exception: If study cannot be fetched or analyzed
    """
    # Validate study exists
    try:
        study_response = await client.registry.study.fetch(id=study_id)
        if not study_response.item:
            raise Exception(f"Study {study_id} not found!")
        study_name = study_response.item.display_name
    except Exception as e:
        raise Exception(f"Failed to fetch study {study_id}: {str(e)}")

    # Analyze connections
    model_run_connections, unsupported_connections = await analyze_study_connections(client, study_id)

    return StudyDeletionAnalysis(
        study_id=study_id,
        study_name=study_name,
        model_run_connections=model_run_connections,
        unsupported_connections=unsupported_connections
    )


def display_study_deletion_summary(analysis: StudyDeletionAnalysis, is_single: bool = True) -> None:
    """
    Display summary information for a study deletion analysis.

    Args:
        analysis: The StudyDeletionAnalysis object
        is_single: Whether this is for a single study or part of bulk analysis
    """
    title_prefix = "Study" if is_single else "Individual Study"
    table = Table(
        title=f"{title_prefix} Deletion Summary: {analysis.study_id}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Study name", analysis.study_name)
    table.add_row("Connected model runs", str(
        len(analysis.model_run_connections)))
    table.add_row("Unsupported connections", str(
        len(analysis.unsupported_connections)))

    console.print(table)


def display_bulk_study_summary(analyses: List[StudyDeletionAnalysis]) -> None:
    """
    Display summary statistics for bulk study deletion.

    Args:
        analyses: List of StudyDeletionAnalysis objects
    """
    if not analyses:
        console.print("[red]No deletion analyses available[/red]")
        return

    # Calculate totals
    total_studies = len(analyses)
    total_model_runs = sum(len(a.model_run_connections) for a in analyses)

    # Create summary table
    summary_table = Table(title="Bulk Study Deletion Analysis")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Total studies to delete", str(total_studies))
    summary_table.add_row("Connected model runs", str(total_model_runs))

    console.print(summary_table)

    # Show individual study details
    details_table = Table(title="Individual Study Analysis")
    details_table.add_column("Study ID", style="cyan", no_wrap=True)
    details_table.add_column("Name", style="blue", max_width=30)
    details_table.add_column("Model Runs", style="yellow")

    for analysis in analyses:
        details_table.add_row(
            analysis.study_id,
            analysis.study_name[:27] +
            "..." if len(analysis.study_name) > 30 else analysis.study_name,
            str(len(analysis.model_run_connections))
        )

    console.print(details_table)


async def collect_model_run_actions_interactive(client, model_run_ids: List[str], context: str) -> List[ModelRunAction]:
    """
    Interactively collect actions for model runs.

    Args:
        client: The Provena client
        model_run_ids: List of model run IDs to process
        context: Context string for display (e.g., "study study_id")

    Returns:
        List of ModelRunAction objects
    """
    actions = []

    for mr_id in model_run_ids:
        # Get model run details
        try:
            mr_details = (await client.registry.model_run.fetch(id=mr_id)).item
            if not mr_details:
                console.print(
                    f"[red]Failed to fetch details for model run {mr_id}[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(
                f"[red]Error fetching model run {mr_id}: {str(e)}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[yellow]Model Run: {mr_id}[/yellow]")
        console.print(f"Display Name: {mr_details.record.display_name}")
        console.print(f"Description: {mr_details.record.description}")
        console.print("Options:")
        console.print("  1. Remove study reference (set to None)")
        console.print("  2. Replace with different study ID")

        choice = typer.prompt("Enter your choice (1 or 2)", type=int)

        if choice == 1:
            actions.append(ModelRunAction(
                model_run_id=mr_id,
                action_type="remove",
                replacement_study_id=None
            ))
            console.print(
                f"[green]Will remove study reference from {mr_id}[/green]")
        elif choice == 2:
            replacement_study_id = typer.prompt("Enter replacement study ID")

            # Validate replacement study exists
            try:
                replacement_response = await client.registry.study.fetch(id=replacement_study_id)
                if not replacement_response.item:
                    console.print(
                        f"[red]Replacement study {replacement_study_id} not found![/red]")
                    raise typer.Exit(1)
                console.print(
                    f"[green]Replacement study validated: {replacement_response.item.display_name}[/green]")
            except Exception as e:
                console.print(
                    f"[red]Failed to validate replacement study {replacement_study_id}: {str(e)}[/red]")
                raise typer.Exit(1)

            actions.append(ModelRunAction(
                model_run_id=mr_id,
                action_type="replace",
                replacement_study_id=replacement_study_id
            ))
            console.print(
                f"[green]Will replace study reference in {mr_id} with {replacement_study_id}[/green]")
        else:
            console.print("[red]Invalid choice. Aborting.[/red]")
            raise typer.Exit(1)

    return actions


async def collect_bulk_model_run_actions(client, analyses: List[StudyDeletionAnalysis]) -> List[ModelRunAction]:
    """
    Collect model run actions for bulk study deletion with user choice of mode.

    Args:
        client: The Provena client
        analyses: List of study analyses containing model run connections

    Returns:
        List of ModelRunAction objects
    """
    all_actions = []

    # Get total count of model runs
    total_model_runs = sum(len(a.model_run_connections) for a in analyses)
    studies_with_connections = [a for a in analyses if a.model_run_connections]

    if not studies_with_connections:
        return all_actions

    console.print(
        f"\n[yellow]Found {len(studies_with_connections)} studies with {total_model_runs} total model run connections[/yellow]")

    console.print("Choose your approach:")
    console.print(
        "  1. Apply the same action to ALL model runs across all studies")
    console.print("  2. Choose actions for each study individually")
    console.print("  3. Choose actions for each model run individually")

    mode_choice = typer.prompt("Enter your choice (1, 2, or 3)", type=int)

    if mode_choice == 1:
        # Apply same action to all model runs
        console.print("Options for ALL model runs:")
        console.print("  1. Remove study reference (set to None)")
        console.print("  2. Replace with same study ID for all")

        action_choice = typer.prompt("Enter your choice (1 or 2)", type=int)

        if action_choice == 1:
            # Remove all
            for analysis in studies_with_connections:
                for mr_id in analysis.model_run_connections:
                    all_actions.append(ModelRunAction(
                        model_run_id=mr_id,
                        action_type="remove",
                        replacement_study_id=None
                    ))
            console.print(
                f"[green]Will remove study references from all {total_model_runs} model runs[/green]")

        elif action_choice == 2:
            # Replace all with same study
            replacement_study_id = typer.prompt(
                "Enter replacement study ID for ALL model runs")

            # Validate replacement study exists
            try:
                replacement_response = await client.registry.study.fetch(id=replacement_study_id)
                if not replacement_response.item:
                    console.print(
                        f"[red]Replacement study {replacement_study_id} not found![/red]")
                    raise typer.Exit(1)
                console.print(
                    f"[green]Replacement study validated: {replacement_response.item.display_name}[/green]")
            except Exception as e:
                console.print(
                    f"[red]Failed to validate replacement study {replacement_study_id}: {str(e)}[/red]")
                raise typer.Exit(1)

            for analysis in studies_with_connections:
                for mr_id in analysis.model_run_connections:
                    all_actions.append(ModelRunAction(
                        model_run_id=mr_id,
                        action_type="replace",
                        replacement_study_id=replacement_study_id
                    ))
            console.print(
                f"[green]Will replace study references in all {total_model_runs} model runs with {replacement_study_id}[/green]")
        else:
            console.print("[red]Invalid choice. Aborting.[/red]")
            raise typer.Exit(1)

    elif mode_choice == 2:
        # Per-study actions
        for analysis in studies_with_connections:
            console.print(
                f"\n[yellow]Planning actions for study {analysis.study_id} ({len(analysis.model_run_connections)} model runs)[/yellow]")
            console.print(f"Study: {analysis.study_name}")
            console.print(
                f"Connected model runs: {', '.join(analysis.model_run_connections)}")
            console.print("Options:")
            console.print(
                "  1. Remove study reference from ALL connected model runs")
            console.print(
                "  2. Replace study reference with same study ID for ALL connected model runs")

            choice = typer.prompt("Enter your choice (1 or 2)", type=int)

            if choice == 1:
                for mr_id in analysis.model_run_connections:
                    all_actions.append(ModelRunAction(
                        model_run_id=mr_id,
                        action_type="remove",
                        replacement_study_id=None
                    ))
                console.print(
                    f"[green]Will remove study references from all model runs for study {analysis.study_id}[/green]")

            elif choice == 2:
                replacement_study_id = typer.prompt(
                    "Enter replacement study ID for all model runs in this study")

                # Validate replacement study exists
                try:
                    replacement_response = await client.registry.study.fetch(id=replacement_study_id)
                    if not replacement_response.item:
                        console.print(
                            f"[red]Replacement study {replacement_study_id} not found![/red]")
                        raise typer.Exit(1)
                    console.print(
                        f"[green]Replacement study validated: {replacement_response.item.display_name}[/green]")
                except Exception as e:
                    console.print(
                        f"[red]Failed to validate replacement study {replacement_study_id}: {str(e)}[/red]")
                    raise typer.Exit(1)

                for mr_id in analysis.model_run_connections:
                    all_actions.append(ModelRunAction(
                        model_run_id=mr_id,
                        action_type="replace",
                        replacement_study_id=replacement_study_id
                    ))
                console.print(
                    f"[green]Will replace study references with {replacement_study_id} for all model runs for study {analysis.study_id}[/green]")
            else:
                console.print("[red]Invalid choice. Aborting.[/red]")
                raise typer.Exit(1)

    elif mode_choice == 3:
        # Individual model run actions
        for analysis in studies_with_connections:
            console.print(
                f"\n[cyan]Processing model runs for study {analysis.study_id}: {analysis.study_name}[/cyan]")
            study_actions = await collect_model_run_actions_interactive(
                client, analysis.model_run_connections, f"study {analysis.study_id}"
            )
            all_actions.extend(study_actions)
    else:
        console.print("[red]Invalid choice. Aborting.[/red]")
        raise typer.Exit(1)

    return all_actions


async def execute_model_run_actions(client: ProvenaClient, actions: List[ModelRunAction], operation_context: str) -> Tuple[List[str], List[str]]:
    """
    Execute model run updates for study reference changes.

    Args:
        client: The Provena client
        actions: List of ModelRunAction objects to execute
        operation_context: Context string for logging/reason

    Returns:
        Tuple of (successful_updates, failed_updates) lists containing model run IDs
    """
    successful_updates = []
    failed_updates = []

    for action in actions:
        try:
            console.print(
                f"[blue]Processing model run {action.model_run_id}...[/blue]")

            # Fetch the model run
            mr_response = await client.registry.model_run.fetch(id=action.model_run_id)
            if not mr_response.item or not isinstance(mr_response.item, ItemModelRun):
                console.print(
                    f"[red]Failed to fetch model run {action.model_run_id}[/red]")
                failed_updates.append(action.model_run_id)
                continue

            # Check permissions
            if mr_response.roles is None or 'metadata-write' not in mr_response.roles:
                console.print(
                    f"[red]Insufficient permissions to modify {action.model_run_id}[/red]")
                failed_updates.append(action.model_run_id)
                continue

            # Update the model run record
            item = mr_response.item
            record = item.record

            if action.action_type == "remove":
                record.study_id = None
                reason = f"Removing study reference due to {operation_context}"
            else:  # replace
                record.study_id = action.replacement_study_id
                reason = f"Replacing study reference to {action.replacement_study_id} due to {operation_context}"

            # Submit update job
            update_response = await client.prov_api.update_model_run(
                model_run_id=action.model_run_id,
                reason=reason,
                record=record
            )

            if hasattr(update_response, 'session_id'):
                console.print(
                    f"[blue]Waiting for model run update job to complete...[/blue]")
                job_result = await client.job_api.await_successful_job_completion(update_response.session_id)

                if job_result.status == JobStatus.SUCCEEDED:
                    console.print(
                        f"[green]Successfully updated model run {action.model_run_id}[/green]")
                    successful_updates.append(action.model_run_id)
                else:
                    console.print(
                        f"[red]Failed to update model run {action.model_run_id}: {job_result.info}[/red]")
                    failed_updates.append(action.model_run_id)
            else:
                console.print(
                    f"[yellow]Update submitted for model run {action.model_run_id} but no job ID returned[/yellow]")
                # Assume success if no session ID
                successful_updates.append(action.model_run_id)

        except Exception as e:
            console.print(
                f"[red]Error updating model run {action.model_run_id}: {str(e)}[/red]")
            failed_updates.append(action.model_run_id)

    return successful_updates, failed_updates


@app.command()
@coro
async def delete_study(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target. One of: {valid_env_str}.",
    ),
    study_id: str = typer.Argument(
        ...,
        help="The ID of the study to delete."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the deletion instead of running in trial mode. Defaults to False (trial mode)."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Delete a study with enhanced connection handling.

    This command will:
    1. Analyze all upstream/downstream connections to the study
    2. For connected model runs, provide options to remove or replace the study reference
    3. Error if non-model-run connections are found
    4. Delete the study from the registry if confirmed

    The deletion removes the study from the registry. Provenance cleanup is handled automatically.
    """
    client = setup_client(env_name=env_name, param=param)

    console.print(
        f"[cyan]Analyzing deletion of study: {study_id}[/cyan]")

    try:
        # Analyze the study
        console.print("[yellow]Analyzing study deletion...[/yellow]")
        analysis = await analyze_single_study_deletion(client, study_id)

        # Display summary
        display_study_deletion_summary(analysis)

        # Check for unsupported connections
        if analysis.unsupported_connections:
            console.print(
                f"\n[red]ERROR: Found {len(analysis.unsupported_connections)} unsupported connections:[/red]")
            for conn_id, conn_type in analysis.unsupported_connections:
                console.print(f"  - {conn_id} (type: {conn_type})")
            console.print(
                "[red]Cannot proceed with deletion. Only model run connections are supported.[/red]")
            raise typer.Exit(1)

        # Plan actions for model run connections
        model_run_actions = []

        if analysis.model_run_connections:
            console.print(
                f"\n[yellow]Planning actions for {len(analysis.model_run_connections)} connected model runs...[/yellow]")

            if not apply:
                # In trial mode, assume we'll remove the connections
                for mr_id in analysis.model_run_connections:
                    model_run_actions.append(ModelRunAction(
                        model_run_id=mr_id,
                        action_type="remove",
                        replacement_study_id=None
                    ))
                    console.print(
                        f"[blue]Trial mode: Would remove study reference from {mr_id}[/blue]")
            else:
                # Interactive mode for actual execution
                model_run_actions = await collect_model_run_actions_interactive(
                    client, analysis.model_run_connections, f"study {study_id}"
                )

        if not apply:
            console.print(
                "\n[green]Trial mode complete. Use --apply to perform actual deletion.[/green]")
            return

        # Confirm deletion
        console.print(
            f"\n[red]WARNING: This will permanently delete study {study_id}[/red]")
        console.print(
            f"[red]Model runs to be updated: {len(model_run_actions)}[/red]")
        console.print("[red]This action cannot be undone![/red]")

        confirmation = typer.confirm(
            "Are you sure you want to proceed with the deletion?", default=False)

        if not confirmation:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            return

        # Execute model run updates first
        if model_run_actions:
            console.print(
                f"[yellow]Updating {len(model_run_actions)} model runs...[/yellow]")

            successful_updates, failed_updates = await execute_model_run_actions(
                client, model_run_actions, f"study {study_id} deletion"
            )

            if failed_updates:
                console.print(
                    f"[red]Failed to update {len(failed_updates)} model runs:[/red]")
                for failed_id in failed_updates:
                    console.print(f"  - {failed_id}")

        # Perform actual study deletion using admin delete
        console.print(f"[red]Deleting study {study_id}...[/red]")
        delete_response = await client.registry.admin.delete(
            id=study_id,
            item_subtype=ItemSubType.STUDY
        )

        if delete_response.status.success:
            console.print(
                f"[green]Successfully deleted study {study_id}[/green]")
        else:
            console.print(
                f"[red]Failed to delete study {study_id}: {delete_response.status.details or 'unknown error'}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(
            f"[red]Error during study deletion {study_id}: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
@coro
async def delete_studies(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target. One of: {valid_env_str}.",
    ),
    json_path: str = typer.Argument(
        ...,
        help="Path to JSON file containing list of study IDs to delete. See BulkStudyDeletion schema."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the deletion instead of running in trial mode. Defaults to False (trial mode)."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Delete multiple studies from a JSON file containing a list of IDs with enhanced connection handling.

    This command will:
    1. Parse the JSON file to get study IDs
    2. Analyze connections for each study
    3. Handle model run connections with flexible bulk options (remove/replace study references)
    4. Error if any non-model-run connections are found
    5. Show summary statistics
    6. Delete all studies if confirmed

    Expected JSON format:
    {
        "study_ids": ["id1", "id2", "id3", ...]
    }
    """
    client = setup_client(env_name=env_name, param=param)

    # Parse JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        deletion_spec = BulkStudyDeletion(**data)
    except FileNotFoundError:
        console.print(f"[red]JSON file not found: {json_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON format: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error parsing JSON file: {str(e)}[/red]")
        raise typer.Exit(1)

    if not deletion_spec.study_ids:
        console.print("[yellow]No study IDs found in JSON file.[/yellow]")
        return

    console.print(
        f"[cyan]Analyzing deletion of {len(deletion_spec.study_ids)} studies...[/cyan]")

    # Analyze each study
    study_analyses = []
    failed_analyses = []
    all_unsupported_connections = []

    for i, study_id in enumerate(deletion_spec.study_ids, 1):
        try:
            console.print(
                f"[yellow]Analyzing {i}/{len(deletion_spec.study_ids)}: {study_id}[/yellow]")

            analysis = await analyze_single_study_deletion(client, study_id)
            study_analyses.append(analysis)

            if analysis.unsupported_connections:
                all_unsupported_connections.extend(
                    [(study_id, conn_id, conn_type) for conn_id, conn_type in analysis.unsupported_connections])

        except Exception as e:
            console.print(f"[red]Failed to analyze {study_id}: {str(e)}[/red]")
            failed_analyses.append(study_id)

    # Check for unsupported connections across all studies
    if all_unsupported_connections:
        console.print(
            f"\n[red]ERROR: Found unsupported connections across studies:[/red]")
        for study_id, conn_id, conn_type in all_unsupported_connections:
            console.print(
                f"  - Study {study_id} -> {conn_id} (type: {conn_type})")
        console.print(
            "[red]Cannot proceed with deletion. Only model run connections are supported.[/red]")
        raise typer.Exit(1)

    # Display summary statistics
    if study_analyses:
        display_bulk_study_summary(study_analyses)

    # Report any failures
    if failed_analyses:
        console.print(
            f"\n[red]Failed to analyze {len(failed_analyses)} studies:[/red]")
        for failed_id in failed_analyses:
            console.print(f"  - {failed_id}")

    if not apply:
        console.print(
            "\n[green]Trial mode complete. Use --apply to perform actual deletions.[/green]")
        return

    if not study_analyses:
        console.print(
            "[red]No studies could be analyzed. Aborting deletion.[/red]")
        raise typer.Exit(1)

    # Collect all model run actions for bulk processing
    all_model_run_actions = await collect_bulk_model_run_actions(client, study_analyses)

    # Confirm bulk deletion
    console.print(
        f"\n[red]WARNING: This will permanently delete {len(study_analyses)} studies[/red]")
    console.print(
        f"[red]Model runs to be updated: {len(all_model_run_actions)}[/red]")
    console.print("[red]This action cannot be undone![/red]")

    confirmation = typer.confirm(
        "Are you sure you want to proceed with the bulk deletion?", default=False)

    if not confirmation:
        console.print("[yellow]Bulk deletion cancelled.[/yellow]")
        return

    # Execute model run updates first
    if all_model_run_actions:
        console.print(
            f"[yellow]Updating {len(all_model_run_actions)} model runs...[/yellow]")

        successful_updates, failed_updates = await execute_model_run_actions(
            client, all_model_run_actions, "bulk study deletion"
        )

        if failed_updates:
            console.print(
                f"[red]Failed to update {len(failed_updates)} model runs:[/red]")
            for failed_id in failed_updates:
                console.print(f"  - {failed_id}")

        if successful_updates:
            console.print(
                f"[green]Successfully updated {len(successful_updates)} model runs[/green]")

    # Perform actual study deletions using admin delete
    successful_deletions = []
    failed_deletions = []

    for i, analysis in enumerate(study_analyses, 1):
        study_id = analysis.study_id
        try:
            console.print(
                f"[red]Deleting {i}/{len(study_analyses)}: {study_id}[/red]")

            delete_response = await client.registry.admin.delete(
                id=study_id,
                item_subtype=ItemSubType.STUDY
            )

            if delete_response.status.success:
                successful_deletions.append(study_id)
                console.print(
                    f"[green]Successfully deleted study {study_id}[/green]")
            else:
                console.print(
                    f"[red]Failed to delete study {study_id}: {delete_response.status.details or 'unknown error'}[/red]")
                failed_deletions.append(study_id)

        except Exception as e:
            console.print(f"[red]Failed to delete {study_id}: {str(e)}[/red]")
            failed_deletions.append(study_id)

    # Final summary
    console.print(
        f"\n[green]Successfully deleted {len(successful_deletions)} studies[/green]")
    if failed_deletions:
        console.print(
            f"[red]Failed to delete {len(failed_deletions)} studies:[/red]")
        for failed_id in failed_deletions:
            console.print(f"  - {failed_id}")


@app.command()
@coro
async def relodge_model_runs(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target. One of: {valid_env_str}.",
    ),
    json_path: str = typer.Argument(
        ...,
        help="Path to JSON file containing list of model run IDs to relodge. See BulkRelodgeModelRun schema."
    ),
    apply: bool = typer.Option(
        False,
        help="Apply the relodge operation instead of running in trial mode. Defaults to False (trial mode)."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Relodge model runs by fetching them from the registry and re-storing them in the provenance API.

    This command takes a JSON file containing model run IDs and relodges each one by:
    1. Fetching the model run from the registry
    2. Storing it back to the provenance API with validation

    Useful for refreshing model run records or recovering from provenance issues.
    """
    client = setup_client(env_name=env_name, param=param)

    # Parse JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        relodge_spec = BulkRelodgeModelRun(**data)
    except FileNotFoundError:
        console.print(f"[red]JSON file not found: {json_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON format: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error parsing JSON file: {str(e)}[/red]")
        raise typer.Exit(1)

    if not relodge_spec.model_run_ids:
        console.print("[yellow]No model run IDs found in JSON file.[/yellow]")
        return

    if not apply:
        console.print(
            f"[yellow]Trial mode: Would relodge {len(relodge_spec.model_run_ids)} model run(s). Use --apply to execute.[/yellow]")
        return

    console.print(
        f"[blue]Relodging {len(relodge_spec.model_run_ids)} model run(s)...[/blue]")

    success_count = 0
    fail_count = 0

    for model_run_id in relodge_spec.model_run_ids:
        try:
            # Fetch the model run
            model_run: ItemModelRun = (await client.registry.model_run.fetch(id=model_run_id)).item

            # Relodge it
            res = await client.prov_api.admin.store_record(registry_record=model_run, validate_record=True)

            if not res.status.success:
                console.print(
                    f"[red]Failed to relodge model run {model_run_id}: {res.status.details}[/red]")
                fail_count += 1
            else:
                console.print(
                    f"[green]Successfully relodged model run {model_run_id}.[/green]")
                success_count += 1
        except Exception as e:
            console.print(
                f"[red]Error processing model run {model_run_id}: {str(e)}[/red]")
            fail_count += 1

    console.print(
        f"\n[blue]Relodge operation complete. Success: {success_count}, Failed: {fail_count}[/blue]")


if __name__ == "__main__":
    app()

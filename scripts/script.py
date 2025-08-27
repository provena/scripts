import typer
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params
from provenaclient import ProvenaClient, Config
from provenaclient.utils.config import APIOverrides
from ProvenaInterfaces.RegistryModels import ItemModelRun, ItemStudy, ItemSubType, ItemBase
from ProvenaInterfaces.RegistryAPI import ModelRunFetchResponse
from ProvenaInterfaces.DataStoreAPI import CredentialsRequest
from ProvenaInterfaces.AsyncJobAPI import JobStatus
from provenaclient.auth import DeviceFlow
from models import BulkStudyLink, PermanentlyDelete
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
from typing import List, Dict, Any, Tuple, Set
import statistics


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


@app.command()
@coro
async def delete_study(
    env_name: str = typer.Argument(
        ...,
        help=f"The tooling environment to target for clear and initialisation. If targeting a feature deployment, then this is used for the keycloak instance. One of: {valid_env_str}.",
    ),
    study_id: str = typer.Argument(
        ...,
        help="The ID of the study to delete."
    ),
    param: ParametersType = typer.Option(
        [], help=f"List of tooling environment parameter replacements in the format 'id:value' e.g. 'feature_num:1234'. Specify multiple times if required.")
) -> None:
    """
    Deletes a study by ID after confirmation.

    This will permanently remove the study from the registry.

    NOTE: Ensure there are no inbound/outbound links to this study before deletion.
    """
    client = setup_client(
        env_name=env_name, param=param
    )

    print(f"Fetching study {study_id}")

    # Validate study exists and get details
    try:
        study_response = await client.registry.study.fetch(id=study_id)
        study: ItemStudy | None = study_response.item

        if not study:
            print(f"Study {study_id} not found!")
            exit(1)

        print(f"Study found: {study.display_name}")

    except Exception as e:
        print(f"Failed to fetch study {study_id}!")
        print(f"Exception: {e}")
        exit(1)

    # Check if the study has any links
    upstream = await client.prov_api.explore_upstream(starting_id=study_id, depth=1)
    downstream = await client.prov_api.explore_downstream(starting_id=study_id, depth=1)

    if (upstream.record_count > 0 or downstream.record_count > 0):
        print(
            f"Study {study_id} has inbound or outbound links. Please remove these links before deletion.")
        print(
            f"Upstream links: {upstream.record_count}, Downstream links: {downstream.record_count}")
        exit(1)
    else:
        print(
            f"Study {study_id} has no inbound or outbound links. Proceeding with deletion...")

    # Confirmation prompt
    confirmation = input(
        f"\nAre you sure you want to permanently delete study '{study.display_name}' (ID: {study_id})? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Deletion cancelled.")
        return

    # Perform deletion
    print(f"\nDeleting study {study_id}...")
    try:
        response = await client.registry.admin.delete(id=study_id, item_subtype=ItemSubType.STUDY)
        if (response.status.success):
            print(f"Study {study_id} deleted successfully!")
        else:
            print(
                f"Study {study_id} deletion failed! error: {response.status.details or 'unknown'}")

    except Exception as e:
        print(f"Failed to delete study {study_id}!")
        print(f"Exception: {e}")
        exit(1)


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


# Model for bulk study deletion JSON file
class BulkStudyDeletion(BaseModel):
    """
    Schema for bulk study deletion JSON file.

    Attributes:
        study_ids: List of study IDs to delete
    """
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


async def execute_model_run_updates(client: ProvenaClient, model_run_actions: List[Tuple[str, str, str]], operation_context: str) -> Tuple[List[str], List[str]]:
    """
    Execute model run updates for study reference changes.

    Args:
        client: The Provena client
        model_run_actions: List of (action_type, model_run_id, replacement_id) tuples
        operation_context: Context string for logging/reason

    Returns:
        Tuple of (successful_updates, failed_updates) lists containing model run IDs
    """
    successful_updates = []
    failed_updates = []

    for action_type, mr_id, replacement_id in model_run_actions:
        try:
            console.print(f"[blue]Processing model run {mr_id}...[/blue]")

            # Fetch the model run
            mr_response = await client.registry.model_run.fetch(id=mr_id)
            if not mr_response.item or not isinstance(mr_response.item, ItemModelRun):
                console.print(f"[red]Failed to fetch model run {mr_id}[/red]")
                failed_updates.append(mr_id)
                continue

            # Check permissions
            if mr_response.roles is None or 'metadata-write' not in mr_response.roles:
                console.print(
                    f"[red]Insufficient permissions to modify {mr_id}[/red]")
                failed_updates.append(mr_id)
                continue

            # Update the model run record
            item = mr_response.item
            record = item.record

            if action_type == "remove":
                record.study_id = None
                reason = f"Removing study reference due to {operation_context}"
            else:  # replace
                record.study_id = replacement_id
                reason = f"Replacing study reference to {replacement_id} due to {operation_context}"

            # Submit update job
            update_response = await client.prov_api.update_model_run(
                model_run_id=mr_id,
                reason=reason,
                record=record
            )

            if hasattr(update_response, 'session_id'):
                console.print(
                    f"[blue]Waiting for model run update job to complete...[/blue]")
                job_result = await client.job_api.await_successful_job_completion(update_response.session_id)

                if job_result.status == JobStatus.SUCCEEDED:
                    console.print(
                        f"[green]Successfully updated model run {mr_id}[/green]")
                    successful_updates.append(mr_id)
                else:
                    console.print(
                        f"[red]Failed to update model run {mr_id}: {job_result.info}[/red]")
                    failed_updates.append(mr_id)
            else:
                console.print(
                    f"[yellow]Update submitted for model run {mr_id} but no job ID returned[/yellow]")
                # Assume success if no session ID
                successful_updates.append(mr_id)

        except Exception as e:
            console.print(
                f"[red]Error updating model run {mr_id}: {str(e)}[/red]")
            failed_updates.append(mr_id)

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
    4. Delete the study from both registry and provenance if confirmed

    The deletion removes the study from both the registry AND the provenance graph,
    including all associated nodes and relationships.
    """
    client = setup_client(
        env_name=env_name, param=param
    )

    console.print(
        f"[cyan]Analyzing enhanced deletion of study: {study_id}[/cyan]")

    try:
        # Validate study exists
        console.print("[yellow]Validating study exists...[/yellow]")
        try:
            study_response = await client.registry.study.fetch(id=study_id)
            study = study_response.item
            if not study:
                console.print(f"[red]Study {study_id} not found![/red]")
                raise typer.Exit(1)
            console.print(f"[green]Study found: {study.display_name}[/green]")
        except Exception as e:
            console.print(
                f"[red]Failed to fetch study {study_id}: {str(e)}[/red]")
            raise typer.Exit(1)

        # Analyze connections
        console.print("[yellow]Analyzing study connections...[/yellow]")
        model_run_connections, unsupported_connections = await analyze_study_connections(client, study_id)

        console.print(
            f"[cyan]Found {len(model_run_connections)} model run connections[/cyan]")
        for mr_id in model_run_connections:
            console.print(f"[blue]Found connected model run: {mr_id}[/blue]")

        # Error if we have unsupported connections
        if unsupported_connections:
            console.print(
                f"\n[red]ERROR: Found {len(unsupported_connections)} unsupported connections:[/red]")
            for conn_id, conn_type in unsupported_connections:
                console.print(f"  - {conn_id} (type: {conn_type})")
            console.print(
                "[red]Cannot proceed with deletion. Only model run connections are supported.[/red]")
            raise typer.Exit(1)

        # Plan actions for model run connections
        model_run_actions = []

        if model_run_connections:
            console.print(
                f"\n[yellow]Planning actions for {len(model_run_connections)} connected model runs...[/yellow]")

            for mr_id in model_run_connections:
                # get some basic details about the model run
                mr_details: ItemModelRun | None = (await client.registry.model_run.fetch(id=mr_id)).item
                if not mr_details:
                    console.print(
                        f"[red]Failed to fetch details for model run {mr_id}[/red]")
                    raise typer.Exit(1)
                if not apply:
                    # In trial mode, assume we'll remove the connection
                    model_run_actions.append(("remove", mr_id, None))
                    console.print(
                        f"[blue]Trial mode: Would remove study reference from {mr_id}[/blue]")
                else:
                    # Interactive mode for actual execution
                    console.print(f"\n[yellow]Model Run: {mr_id}[/yellow]")
                    console.print(
                        f"Display Name: {mr_details.record.display_name}")
                    console.print(
                        f"Description: {mr_details.record.description}")
                    console.print("Options:")
                    console.print("  1. Remove study reference (set to None)")
                    console.print("  2. Replace with different study ID")

                    choice = typer.prompt(
                        "Enter your choice (1 or 2)", type=int)

                    if choice == 1:
                        model_run_actions.append(("remove", mr_id, None))
                        console.print(
                            f"[green]Will remove study reference from {mr_id}[/green]")
                    elif choice == 2:
                        replacement_study_id = typer.prompt(
                            "Enter replacement study ID")

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

                        model_run_actions.append(
                            ("replace", mr_id, replacement_study_id))
                        console.print(
                            f"[green]Will replace study reference in {mr_id} with {replacement_study_id}[/green]")
                    else:
                        console.print("[red]Invalid choice. Aborting.[/red]")
                        raise typer.Exit(1)

        # Run trial deletion to get provenance impact
        console.print(
            "[yellow]Running trial deletion to analyze provenance impact...[/yellow]")
        trial_response = await client.prov_api.admin.delete_study_provenance_and_registry(
            study_id=study_id,
            trial_mode=True
        )

        # Display summary
        removed_nodes, removed_links = analyze_deletion_diff(
            trial_response.diff)

        # Create summary table
        summary_table = Table(
            title=f"Enhanced Study Deletion Summary: {study_id}")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="magenta")

        summary_table.add_row("Study name", study.display_name)
        summary_table.add_row(
            "Provenance nodes to be removed", str(removed_nodes))
        summary_table.add_row(
            "Provenance links to be removed", str(removed_links))
        summary_table.add_row("Connected model runs",
                              str(len(model_run_connections)))

        remove_count = sum(
            1 for action in model_run_actions if action[0] == "remove")
        replace_count = sum(
            1 for action in model_run_actions if action[0] == "replace")
        summary_table.add_row(
            "Model runs: study ref removal", str(remove_count))
        summary_table.add_row(
            "Model runs: study ref replacement", str(replace_count))

        console.print(summary_table)

        if not apply:
            console.print(
                "\n[green]Trial mode complete. Use --apply to perform actual deletion.[/green]")
            return

        # Confirm deletion
        console.print(
            f"\n[red]WARNING: This will permanently delete study {study_id}[/red]")
        console.print(
            f"[red]Total provenance nodes to be removed: {removed_nodes}[/red]")
        console.print(
            f"[red]Total provenance links to be removed: {removed_links}[/red]")
        console.print(
            f"[red]Model runs to be updated: {len(model_run_actions)}[/red]")
        console.print("[red]This action cannot be undone![/red]")

        confirmation = typer.confirm(
            "Are you sure you want to proceed with the deletion?",
            default=False
        )

        if not confirmation:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            return

        # Execute model run updates first
        if model_run_actions:
            console.print(
                f"[yellow]Updating {len(model_run_actions)} model runs...[/yellow]")

            successful_updates, failed_updates = await execute_model_run_updates(
                client, model_run_actions, f"study {study_id} deletion"
            )

            if failed_updates:
                console.print(
                    f"[red]Failed to update {len(failed_updates)} model runs:[/red]")
                for failed_id in failed_updates:
                    console.print(f"  - {failed_id}")

        # Perform actual study deletion
        console.print(f"[red]Deleting study {study_id}...[/red]")
        deletion_response = await client.prov_api.admin.delete_study_provenance_and_registry(
            study_id=study_id,
            trial_mode=False
        )

        console.print(f"[green]Successfully deleted study {study_id}[/green]")

    except Exception as e:
        console.print(
            f"[red]Error during enhanced study deletion {study_id}: {str(e)}[/red]")
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
    3. Handle model run connections (remove/replace study references)
    4. Error if any non-model-run connections are found
    5. Show summary statistics and outliers
    6. Delete all studies if confirmed

    Expected JSON format:
    {
        "study_ids": ["id1", "id2", "id3", ...]
    }
    """
    client = setup_client(
        env_name=env_name, param=param
    )

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
    study_analysis_results = []
    failed_analyses = []
    all_unsupported_connections = []

    for i, study_id in enumerate(deletion_spec.study_ids, 1):
        try:
            console.print(
                f"[yellow]Analyzing {i}/{len(deletion_spec.study_ids)}: {study_id}[/yellow]")

            # Validate study exists
            try:
                study_response = await client.registry.study.fetch(id=study_id)
                if not study_response.item:
                    console.print(f"[red]Study {study_id} not found![/red]")
                    failed_analyses.append(study_id)
                    continue
            except Exception as e:
                console.print(
                    f"[red]Failed to fetch study {study_id}: {str(e)}[/red]")
                failed_analyses.append(study_id)
                continue

            # Analyze connections
            model_run_connections, unsupported_connections = await analyze_study_connections(client, study_id)

            if unsupported_connections:
                all_unsupported_connections.extend(
                    [(study_id, conn_id, conn_type) for conn_id, conn_type in unsupported_connections])

            # Run trial deletion for provenance impact
            trial_response = await client.prov_api.admin.delete_study_provenance_and_registry(
                study_id=study_id,
                trial_mode=True
            )

            removed_nodes, removed_links = analyze_deletion_diff(
                trial_response.diff)

            study_analysis_results.append({
                'study_id': study_id,
                'study_name': study_response.item.display_name,
                'model_run_connections': len(model_run_connections),
                'removed_nodes': removed_nodes,
                'removed_links': removed_links,
                'model_run_ids': model_run_connections
            })

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
    if study_analysis_results:
        # Create summary table
        summary_table = Table(title="Bulk Study Deletion Analysis")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        total_studies = len(study_analysis_results)
        total_model_runs = sum(s['model_run_connections']
                               for s in study_analysis_results)
        total_nodes = sum(s['removed_nodes'] for s in study_analysis_results)
        total_links = sum(s['removed_links'] for s in study_analysis_results)
        avg_nodes = statistics.mean(
            [s['removed_nodes'] for s in study_analysis_results]) if study_analysis_results else 0
        avg_links = statistics.mean(
            [s['removed_links'] for s in study_analysis_results]) if study_analysis_results else 0

        summary_table.add_row("Total studies to delete", str(total_studies))
        summary_table.add_row("Connected model runs", str(total_model_runs))
        summary_table.add_row("Total provenance nodes", str(total_nodes))
        summary_table.add_row("Total provenance links", str(total_links))
        summary_table.add_row("Avg nodes per study", f"{avg_nodes:.1f}")
        summary_table.add_row("Avg links per study", f"{avg_links:.1f}")

        console.print(summary_table)

        # Show individual study details
        details_table = Table(title="Individual Study Analysis")
        details_table.add_column("Study ID", style="cyan", no_wrap=True)
        details_table.add_column("Name", style="blue", max_width=30)
        details_table.add_column("Model Runs", style="yellow")
        details_table.add_column("Nodes", style="red")
        details_table.add_column("Links", style="red")

        for result in study_analysis_results:
            details_table.add_row(
                result['study_id'],
                result['study_name'][:27] +
                "..." if len(result['study_name']
                             ) > 30 else result['study_name'],
                str(result['model_run_connections']),
                str(result['removed_nodes']),
                str(result['removed_links'])
            )

        console.print(details_table)

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

    if not study_analysis_results:
        console.print(
            "[red]No studies could be analyzed. Aborting deletion.[/red]")
        raise typer.Exit(1)

    # Collect all model run actions for bulk processing
    all_model_run_actions = []
    for result in study_analysis_results:
        if result['model_run_ids']:
            console.print(
                f"\n[yellow]Planning actions for study {result['study_id']} ({len(result['model_run_ids'])} model runs)[/yellow]")

            # For bulk operations, provide a choice for the whole study at once
            if not result['model_run_ids']:
                continue

            console.print(f"Study: {result['study_name']}")
            console.print(
                f"Connected model runs: {', '.join(result['model_run_ids'])}")
            console.print("Options:")
            console.print(
                "  1. Remove study reference from ALL connected model runs")
            console.print(
                "  2. Replace study reference with same study ID for ALL connected model runs")

            choice = typer.prompt("Enter your choice (1 or 2)", type=int)

            if choice == 1:
                for mr_id in result['model_run_ids']:
                    all_model_run_actions.append(("remove", mr_id, None))
                console.print(
                    f"[green]Will remove study references from all model runs for study {result['study_id']}[/green]")
            elif choice == 2:
                replacement_study_id = typer.prompt(
                    "Enter replacement study ID for all model runs")

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

                for mr_id in result['model_run_ids']:
                    all_model_run_actions.append(
                        ("replace", mr_id, replacement_study_id))
                console.print(
                    f"[green]Will replace study references with {replacement_study_id} for all model runs for study {result['study_id']}[/green]")
            else:
                console.print("[red]Invalid choice. Aborting.[/red]")
                raise typer.Exit(1)

    # Confirm bulk deletion
    console.print(
        f"\n[red]WARNING: This will permanently delete {len(study_analysis_results)} studies[/red]")
    console.print(f"[red]Total provenance nodes: {total_nodes}[/red]")
    console.print(f"[red]Total provenance links: {total_links}[/red]")
    console.print(
        f"[red]Model runs to be updated: {len(all_model_run_actions)}[/red]")
    console.print("[red]This action cannot be undone![/red]")

    confirmation = typer.confirm(
        "Are you sure you want to proceed with the bulk deletion?",
        default=False
    )

    if not confirmation:
        console.print("[yellow]Bulk deletion cancelled.[/yellow]")
        return

    # Execute model run updates first
    if all_model_run_actions:
        console.print(
            f"[yellow]Updating {len(all_model_run_actions)} model runs...[/yellow]")

        successful_updates, failed_updates = await execute_model_run_updates(client, all_model_run_actions, "bulk study deletion")

        if failed_updates:
            console.print(
                f"[red]Failed to update {len(failed_updates)} model runs:[/red]")
            for failed_id in failed_updates:
                console.print(f"  - {failed_id}")

        if successful_updates:
            console.print(
                f"[green]Successfully updated {len(successful_updates)} model runs[/green]")

    # Perform actual study deletions
    successful_deletions = []
    failed_deletions = []

    for i, result in enumerate(study_analysis_results, 1):
        study_id = result['study_id']
        try:
            console.print(
                f"[red]Deleting {i}/{len(study_analysis_results)}: {study_id}[/red]")

            await client.prov_api.admin.delete_study_provenance_and_registry(
                study_id=study_id,
                trial_mode=False
            )

            successful_deletions.append(study_id)

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

if __name__ == "__main__":
    app()

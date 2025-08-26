import typer
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params, PopulatedToolingEnvironment
from provenaclient import ProvenaClient, Config
from ProvenaInterfaces.RegistryModels import ItemModelRun, ItemStudy, ItemSubType
from ProvenaInterfaces.RegistryAPI import ModelRunFetchResponse
from ProvenaInterfaces.DataStoreAPI import CredentialsRequest
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
    config = Config(domain=env.domain, realm_name=env.realm_name)
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

if __name__ == "__main__":
    app()

# Provena scripts

This folder contains a set of scripts which can help manage and administrate a Provena deployment.

There are a few key steps to using the tools, described below.

## Installation

Move into the scripts folder

```
cd scripts
```

Setup a python virtual environment (3.10 recommended)

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All of the tools use Typer CLIs - you can execute the corresponding script using

```
python script.py
```

To get help, use `--help`. Some scripts have subcommands. You can also use
`--help` on these sub commands.

## Managing environments

The tooling environment manager is a python package which handles customising the targeted Provena deployment.

A Provena deployment exposes a set of API endpoints which depend on the base domain name.

In the below examples and Schema, note that the "name" field is the environment name parameter which is required in most of the admin tooling CLI functions.

### Schema

The environments.json file must satisfy the `ToolingEnvironmentsFile` Pydantic schema below:

```
class ToolingEnvironment(BaseModel):
    # Core values

    # What is the name of this environment?
    name: str
    # What is the Provena application stage?
    stage: str
    # What is the base domain e.g. your.domain.com
    domain: str
    # What is the auth realm name?
    realm_name: str

    # In the specified overrides - are there any variables to replace? Specify
    # the key (in CLI) and the in text value to find/replace
    replacements: List[ReplacementField] = []

    # Overrides

    # Endpoints are auto generated but can be overrided

    # APIs
    datastore_api_endpoint_override: Optional[str]
    auth_api_endpoint_override: Optional[str]
    registry_api_endpoint_override: Optional[str]
    prov_api_endpoint_override: Optional[str]
    search_api_endpoint_override: Optional[str]
    search_service_endpoint_override: Optional[str]
    handle_service_api_endpoint_override: Optional[str]

    # Keycloak
    keycloak_endpoint_override: Optional[str]

    # Defaults - overridable
    aws_region: str = "ap-southeast-2"

class ToolingEnvironmentsFile(BaseModel):
    envs: List[ToolingEnvironment]
```

### Basic case

In the basic case, you should add an entry for your Provena deployment like so:

```
{
    "envs": [
        {
            "name": "MyProvena",
            "stage": "PROD",
            "domain": "provena.yourdomain.io",
            "realm_name": "TODO"
        }
    ]
}
```

This is the minimum viable environment to target.

- name: The user defined name of the target - keep it short but easy to remember - not used anywhere else
- stage: The application stage - e.g. DEV, STAGE, PROD
- domain: The base domain of the provena deployment- this is postfixed onto the standard set of API endpoints
- realm_name: The keycloak authorisation server realm name

### Advanced case

In the RRAP IS team, we use a process we call the "feature deployment" workflow. This means we spin up sandbox Provena instances for feature development. In these cases, the API endpoints have a special prefix depending on a number specific to our process. Let's call this number the 'feature_number'. The environment file structure is capable of handling in-situ replacements which are fed via parameters from the CLI tooling.

E.g. consider this feature branch workflow environment file

```
{
    "envs":[
        {
            "name": "feature",
            "stage": "DEV",
            "domain": "your.domain.com",
            "realm_name": "example",
            "replacements": [
                {
                    "id": "feature_number",
                    "key": "${feature_number}"
                }
            ],
            "datastore_api_endpoint_override": "https://f${feature_number}-data-api.dev.example.com",
            "auth_api_endpoint_override": "https://f${feature_number}-auth-api.dev.example.com",
            "registry_api_endpoint_override": "https://f${feature_number}-registry-api.dev.example.com",
            "prov_api_endpoint_override": "https://f${feature_number}-prov-api.dev.example.com",
            "search_api_endpoint_override": "https://f${feature_number}-search.dev.example.com"
        }
    ]
}
```

Note that we define an id and key for the feature number replacement. The id is the CLI input key. The Key is the value which will be replaced in the API endpoint overrides.

When we use the CLI tools - all the endpoints feature a `param` option which can be used multiple times, they have a specific syntax

```
python script.py sub_command argument1 argument2 --param id1:value1 --param id2:value2
```

For our feature deployment workflow, we can perform the replacement like so:

```
python environment_bootstrapper.py bootstrap-stage feature --suppress-warnings --param feature_number:${ticket_number}
```

where `ticket_number` is passed from the environment.

## Model Run Deletion Tools

This module provides tools for deleting model runs from both the Provena registry and provenance store. The deletion operations are designed to be safe, with mandatory trial runs and confirmations before any actual deletion occurs.

### `delete_model_run` - Delete a Single Model Run

Deletes a single model run by its ID from both the registry and provenance graph.

**Usage:**

```bash
python model_run_admin.py delete-model-run <env_name> <model_run_id> [OPTIONS]
```

**Arguments:**

- `env_name`: The tooling environment to target (e.g., "MyProvena", "feature")
- `model_run_id`: The handle ID of the model run to delete

**Options:**

- `--apply`: Actually perform the deletion (default: False, runs in trial mode)
- `--param id:value`: Environment parameter replacements for feature deployments

**Example:**

```bash
# Trial run (shows what would be deleted)
python model_run_admin.py delete-model-run MyProvena 10378.1/1234567

# Actually delete the model run
python model_run_admin.py delete-model-run MyProvena 10378.1/1234567 --apply
```

### `delete_model_runs` - Delete Multiple Model Runs

Deletes multiple model runs specified in a JSON file from both the registry and provenance graph.

**Usage:**

```bash
python model_run_admin.py delete-model-runs <env_name> <json_path> [OPTIONS]
```

**Arguments:**

- `env_name`: The tooling environment to target
- `json_path`: Path to JSON file containing model run IDs

**Options:**

- `--apply`: Actually perform the deletions (default: False, runs in trial mode)
- `--param id:value`: Environment parameter replacements for feature deployments

**JSON File Format:**

```json
{
  "model_run_ids": ["10378.1/1234567", "10378.1/2345678", "10378.1/3456789"]
}
```

**Example:**

```bash
# Trial run (analyzes all deletions and shows statistics)
python model_run_admin.py delete-model-runs MyProvena bulk_deletion.json

# Actually delete all model runs
python model_run_admin.py delete-model-runs MyProvena bulk_deletion.json --apply
```

### Output Information

The tools provide detailed information about deletions:

- **Nodes to be removed**: Count of graph nodes that will be deleted
- **Links to be removed**: Count of graph relationships that will be deleted
- **Total diff actions**: Total number of graph operations required
- **Statistical summaries**: For bulk operations, shows averages and identifies outliers
- **Success/failure tracking**: Reports on completed vs failed operations

## Important Notes

- **Irreversible**: Deletions are permanent and cannot be undone
- **Comprehensive**: Removes model runs from BOTH registry and provenance graph
- **Permission Required**: Requires appropriate admin permissions in the target environment
- **Graph Impact**: Deletion affects the entire provenance graph structure
- **Batch Processing**: Bulk operations process items sequentially for better error handling

## Study Deletion Tools

This module provides tools for deleting studies from the Provena registry with intelligent connection handling. The deletion operations are designed to safely handle connected model runs and prevent deletion of studies with unsupported connections.

### `delete_study` - Delete a Single Study

Deletes a single study by its ID from the registry after handling any connected model runs.

**Usage:**

```bash
python model_run_admin.py delete-study <env_name> <study_id> [OPTIONS]
```

**Arguments:**

- `env_name`: The tooling environment to target (e.g., "MyProvena", "feature")
- `study_id`: The handle ID of the study to delete

**Options:**

- `--apply`: Actually perform the deletion (default: False, runs in trial mode)
- `--param id:value`: Environment parameter replacements for feature deployments

**Connection Handling:**

When a study has connected model runs, you'll be prompted to choose how to handle each connection:

1. **Remove study reference**: Set the model run's study_id to None
2. **Replace with different study**: Specify a replacement study ID for the model run

**Example:**

```bash
# Trial run (shows connections and what would be deleted)
python model_run_admin.py delete-study MyProvena 10378.1/1234567

# Interactive deletion with connection handling
python model_run_admin.py delete-study MyProvena 10378.1/1234567 --apply
```

### `delete_studies` - Delete Multiple Studies

Deletes multiple studies specified in a JSON file from the registry with bulk connection handling options.

**Usage:**

```bash
python model_run_admin.py delete-studies <env_name> <json_path> [OPTIONS]
```

**Arguments:**

- `env_name`: The tooling environment to target
- `json_path`: Path to JSON file containing study IDs

**Options:**

- `--apply`: Actually perform the deletions (default: False, runs in trial mode)
- `--param id:value`: Environment parameter replacements for feature deployments

**JSON File Format:**

```json
{
  "study_ids": ["10378.1/1234567", "10378.1/2345678", "10378.1/3456789"]
}
```

**Bulk Connection Handling Options:**

When connected model runs are found, you can choose from three approaches:

1. **Apply same action to ALL model runs**: Choose one action (remove/replace) for all connected model runs across all studies
2. **Choose actions per study**: Select actions for all model runs within each study individually
3. **Choose actions per model run**: Interactive mode for each individual model run connection

**Example:**

```bash
# Trial run (analyzes all studies and shows connection summary)
python model_run_admin.py delete-studies MyProvena bulk_study_deletion.json

# Interactive bulk deletion with connection handling
python model_run_admin.py delete-studies MyProvena bulk_study_deletion.json --apply
```

### Connection Analysis

The tools analyze all upstream and downstream connections to each study:

- **Supported connections**: Model runs (can be handled automatically)
- **Unsupported connections**: Any other entity types (will prevent deletion)

If unsupported connections are found, the deletion will be aborted with details about the problematic connections.

### Workflow

The study deletion process follows this sequence:

1. **Analysis phase**: Examine all connections to the study/studies
2. **Planning**: Collect user decisions for handling model run connections
3. **Model run updates**: Remove or replace study references in connected model runs
4. **Study deletion**: Delete the study from the registry using admin delete

**Note**: The provenance diff engine will automatically remove the Study node
from the provenance graph once it is no longer referenced in any model runs.

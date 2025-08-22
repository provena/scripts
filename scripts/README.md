# Provena scripts

This folder contains a set of scripts which can help manage and administrate a Provena deployment.

There are a few key steps to using the tools, described below.

## Installation

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

## Permanently deleting files

The script available in `script.py` allows you to permanently delete files which match a given regex for a given dataset ID assuming you have write permission into the files in that dataset.

You must produce a file matching the spec available in `models.py#PermanentlyDelete` e.g. this file will delete all files matching the given regex

```json
{
  "dataset_id": "<dataset id here>",
  "regexes": ["NetCDF[^/]*\\.nc$"]
}
```

You can supply multiple regexes if you wish. If the file matches ANY, it will be deleted.

You can then run this script using - this will just dry run and output the files to be deleted which match in the report.txt file.

```
python script.py permanently-delete-files STAGE payloads/your_payload.json --output-report report.txt
```

To run it and delete those matching files, add the --apply flag e.g.

```
python script.py permanently-delete-files STAGE payloads/your_payload.json --output-report report.txt --apply
```

Note that by default, if any files which have not been _soft_ deleted (i.e. have a present delete marker) are found which match the regex, then an error is thrown. To enable deletion of these previously undeleted files, add the `--allow-new-deletion` flag.

# Script docs

## delete_study

**Purpose**: Permanently deletes a study from the Provena registry after validation and confirmation.

**Usage**:

```bash
python script.py delete-study <env_name> <study_id> [OPTIONS]
```

**Arguments**:

- `env_name`: The tooling environment to target (see [Managing environments](#managing-environments))
- `study_id`: The ID of the study to delete

**Options**:

- `--param`: List of tooling environment parameter replacements in the format `id:value` (e.g., `feature_num:1234`). Can be specified multiple times if required.

**Behavior**:

1. Validates the study exists and fetches its details
2. Checks for any inbound or outbound provenance links to/from the study
3. If links exist, prevents deletion and shows link counts
4. If no links exist, displays study information and prompts for confirmation
5. Performs permanent deletion if confirmed

**Example**:

```bash
# Delete a study in production environment
python script.py delete-study MyProvena abc123-def456-789

# Delete a study in feature deployment
python script.py delete-study feature abc123-def456-789 --param feature_number:1234
```

**Important Notes**:

- This operation is **permanent** and cannot be undone
- Studies with existing provenance links (upstream or downstream) cannot be deleted
- Remove all provenance relationships before attempting deletion
- Ensure you have appropriate permissions in the target environment

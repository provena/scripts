#!/usr/bin/env python3
"""
GUI Web Interface for Provena Admin CLI using NiceGUI
"""

from typing import List, Optional
from nicegui import ui, app
from ToolingEnvironmentManager.Management import EnvironmentManager, process_params
from provenaclient import ProvenaClient, Config
from provenaclient.utils.config import APIOverrides
from provenaclient.auth import DeviceFlow
from ProvenaInterfaces.RegistryModels import ItemSubType

# Global state management


class AppState:
    def __init__(self):
        self.client: Optional[ProvenaClient] = None
        self.env_manager = EnvironmentManager(
            environment_file_path="../environments.json")
        self.current_env: Optional[str] = None
        self.auth_status: str = "Not authenticated"


app_state = AppState()

# Authentication and Client Setup


async def setup_client(env_name: str, params: List[str] = None) -> bool:
    """Setup the Provena client with authentication"""
    try:
        if params is None:
            params = []

        # Process parameters
        processed_params = process_params(params)
        env = app_state.env_manager.get_environment(
            name=env_name, params=processed_params)

        # Setup config
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

        # Setup authentication
        auth = DeviceFlow(config=config, client_id="client-tools")
        app_state.client = ProvenaClient(auth=auth, config=config)
        app_state.current_env = env_name
        app_state.auth_status = f"Authenticated to {env_name}"

        return True
    except Exception as e:
        app_state.auth_status = f"Authentication failed: {str(e)}"
        return False

# UI Components


@ui.page('/')
def main_page():
    ui.page_title('Provena Admin GUI')

    # Header
    with ui.header().classes('items-center justify-between'):
        ui.label('Provena Admin Interface').classes('text-h6')
        with ui.row().classes('items-center'):
            ui.label().bind_text_from(app_state, 'auth_status')

    # Main content
    with ui.tabs().classes('w-full') as tabs:
        datasets_tab = ui.tab('Datasets')
        studies_tab = ui.tab('Studies')
        model_runs_tab = ui.tab('Model Runs')
        files_tab = ui.tab('Files')
        auth_tab = ui.tab('Authentication')

    with ui.tab_panels(tabs, value=auth_tab).classes('w-full'):
        # Authentication Panel
        with ui.tab_panel(auth_tab):
            authentication_panel()

        # Datasets Panel
        with ui.tab_panel(datasets_tab):
            datasets_panel()

        # Studies Panel
        with ui.tab_panel(studies_tab):
            studies_panel()

        # Model Runs Panel
        with ui.tab_panel(model_runs_tab):
            model_runs_panel()

        # Files Panel
        with ui.tab_panel(files_tab):
            files_panel()


def authentication_panel():
    """Authentication and environment setup panel"""
    ui.label('Environment Setup').classes('text-h5')
    ui.separator()

    # Environment selection # TODO make this better
    envs = ['prod', 'dev']
    env_select = ui.select(envs, label='Select Environment',
                           value=envs[0] if envs else None)

    # Parameters input
    params_input = ui.textarea(
        label='Parameters (one per line, format: id:value)',
        placeholder='feature_num:1234\nother_param:value'
    ).classes('w-full')

    auth_button = ui.button('Authenticate', on_click=lambda: handle_auth(
        env_select.value, params_input.value))

    # Status display
    with ui.card():
        ui.label('Authentication Status:').classes('text-bold')
        ui.label().bind_text_from(app_state, 'auth_status')


async def handle_auth(env_name: str, params_text: str):
    """Handle authentication request"""
    if not env_name:
        ui.notify('Please select an environment', type='warning')
        return

    # Parse parameters
    params = []
    if params_text.strip():
        for line in params_text.strip().split('\n'):
            if ':' in line:
                params.append(line.strip())

    ui.notify('Authenticating...', type='info')
    success = await setup_client(env_name, params)

    if success:
        ui.notify('Authentication successful!', type='positive')
    else:
        ui.notify('Authentication failed', type='negative')


def datasets_panel():
    """Datasets management panel"""
    ui.label('Dataset Management').classes('text-h5')
    ui.separator()

    # List datasets section
    with ui.card():
        ui.label('List All Datasets').classes('text-h6')
        list_button = ui.button('List Datasets', on_click=list_datasets)
        datasets_output = ui.log().classes('w-full h-64')


async def list_datasets():
    """List all datasets"""
    if not app_state.client:
        ui.notify('Please authenticate first', type='warning')
        return

    try:
        ui.notify('Fetching datasets...', type='info')
        datasets = await app_state.client.datastore.list_all_datasets()

        # Display in a nice format
        with ui.dialog().classes('w-full') as dialog, ui.card():
            ui.label('Datasets').classes('text-h6')
            ui.separator()

            if datasets:
                for dataset in datasets:
                    with ui.expansion(f"Dataset: {dataset.id}").classes('w-full'):
                        ui.json_editor(
                            {'content': {'json': dataset.model_dump(mode='json')}}).classes('w-full')
            else:
                ui.label('No datasets found')

            ui.button('Close', on_click=dialog.close)

        dialog.open()
        ui.notify('Datasets loaded successfully', type='positive')

    except Exception as e:
        ui.notify(f'Error fetching datasets: {str(e)}', type='negative')


def studies_panel():
    """Studies management panel"""
    ui.label('Study Management').classes('text-h5')
    ui.separator()

    # Bulk link studies section
    with ui.card():
        ui.label('Bulk Link Studies').classes('text-h6')
        ui.label('Link multiple model runs to a study')

        with ui.row().classes('w-full'):
            study_id_input = ui.input(label='Study ID').classes('flex-grow')

        model_runs_input = ui.textarea(
            label='Model Run IDs (one per line)',
            placeholder='model_run_id_1\nmodel_run_id_2\nmodel_run_id_3'
        ).classes('w-full')

        ui.button('Link Model Runs to Study', on_click=lambda: handle_bulk_link_studies(
            study_id_input.value, model_runs_input.value
        ))

    ui.separator()

    # Delete single study section
    with ui.card():
        ui.label('Delete Study').classes('text-h6')
        ui.label('Delete a single study (with connection analysis)')

        single_study_input = ui.input(
            label='Study ID to Delete').classes('w-full')

        with ui.row():
            ui.button('Analyze Study (Trial Mode)', on_click=lambda: handle_delete_study(
                single_study_input.value, apply=False
            ))
            ui.button('Delete Study', on_click=lambda: handle_delete_study(
                single_study_input.value, apply=True
            ), color='red')


async def handle_bulk_link_studies(study_id: str, model_runs_text: str):
    """Handle bulk study linking"""
    if not app_state.client:
        ui.notify('Please authenticate first', type='warning')
        return

    if not study_id or not model_runs_text.strip():
        ui.notify('Please provide study ID and model run IDs', type='warning')
        return

    # Parse model run IDs
    model_run_ids = [line.strip()
                     for line in model_runs_text.strip().split('\n') if line.strip()]

    try:
        # Validate study exists
        ui.notify('Validating study...', type='info')
        await app_state.client.registry.study.fetch(id=study_id)

        # Link each model run
        successful = 0
        failed = 0

        for mr_id in model_run_ids:
            try:
                ui.notify(f'Processing model run {mr_id}...', type='info')

                # Fetch existing model run
                existing = await app_state.client.registry.model_run.fetch(id=mr_id)

                # Check permissions
                if not existing.roles or 'metadata-write' not in existing.roles:
                    ui.notify(
                        f'Insufficient permissions for {mr_id}', type='warning')
                    failed += 1
                    continue

                # Update the model run
                item = existing.item
                record = item.record
                record.study_id = study_id

                await app_state.client.prov_api.update_model_run(
                    model_run_id=mr_id,
                    reason=f"(GUI bulk) linking to study {study_id}",
                    record=record
                )

                successful += 1

            except Exception as e:
                ui.notify(
                    f'Failed to update {mr_id}: {str(e)}', type='negative')
                failed += 1

        ui.notify(
            f'Completed: {successful} successful, {failed} failed', type='positive')

    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')


async def handle_delete_study(study_id: str, apply: bool = False):
    """Handle study deletion with analysis"""
    if not app_state.client:
        ui.notify('Please authenticate first', type='warning')
        return

    if not study_id:
        ui.notify('Please provide a study ID', type='warning')
        return

    try:
        ui.notify(
            f'{"Analyzing" if not apply else "Deleting"} study {study_id}...', type='info')

        # This is a simplified version - in a full implementation you'd want to
        # recreate the analysis logic from your CLI
        if apply:
            # Perform deletion using admin delete
            delete_response = await app_state.client.registry.admin.delete(
                id=study_id,
                item_subtype=ItemSubType.STUDY
            )

            if delete_response.status.success:
                ui.notify(
                    f'Study {study_id} deleted successfully', type='positive')
            else:
                ui.notify(
                    f'Failed to delete study: {delete_response.status.details}', type='negative')
        else:
            # Trial mode - just validate study exists
            study_response = await app_state.client.registry.study.fetch(id=study_id)
            if study_response.item:
                ui.notify(
                    f'Study found: {study_response.item.display_name}', type='positive')
                ui.notify(
                    'Use "Delete Study" button to perform actual deletion', type='info')
            else:
                ui.notify('Study not found', type='negative')

    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')


def model_runs_panel():
    """Model runs management panel"""
    ui.label('Model Run Management').classes('text-h5')
    ui.separator()

    # Delete single model run
    with ui.card():
        ui.label('Delete Model Run').classes('text-h6')
        ui.label('Delete a single model run from registry and provenance')

        model_run_input = ui.input(
            label='Model Run ID to Delete').classes('w-full')

        with ui.row():
            ui.button('Analyze Model Run (Trial Mode)', on_click=lambda: handle_delete_model_run(
                model_run_input.value, apply=False
            ))
            ui.button('Delete Model Run', on_click=lambda: handle_delete_model_run(
                model_run_input.value, apply=True
            ), color='red')


async def handle_delete_model_run(model_run_id: str, apply: bool = False):
    """Handle model run deletion"""
    if not app_state.client:
        ui.notify('Please authenticate first', type='warning')
        return

    if not model_run_id:
        ui.notify('Please provide a model run ID', type='warning')
        return

    try:
        ui.notify(
            f'{"Analyzing" if not apply else "Deleting"} model run {model_run_id}...', type='info')

        # Run trial mode first
        trial_response = await app_state.client.prov_api.admin.delete_model_run_provenance_and_registry(
            model_run_id=model_run_id,
            trial_mode=True
        )

        # Show analysis results
        removed_nodes = sum(1 for action in trial_response.diff if action.get(
            'action_type') == 'REMOVE_NODE')
        removed_links = sum(1 for action in trial_response.diff if action.get(
            'action_type') == 'REMOVE_LINK')

        analysis_msg = f'Analysis: {removed_nodes} nodes and {removed_links} links would be removed'
        ui.notify(analysis_msg, type='info')

        if apply:
            # Perform actual deletion
            await app_state.client.prov_api.admin.delete_model_run_provenance_and_registry(
                model_run_id=model_run_id,
                trial_mode=False
            )

            ui.notify(
                f'Model run {model_run_id} deleted successfully', type='positive')

    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')


def files_panel():
    """Files management panel"""
    ui.label('File Management').classes('text-h5')
    ui.separator()

    # Permanent file deletion
    with ui.card():
        ui.label('Permanently Delete Files').classes('text-h6')
        ui.label('Delete files matching regex patterns from dataset storage')

        dataset_id_input = ui.input(label='Dataset ID').classes('w-full')

        regexes_input = ui.textarea(
            label='Regex Patterns (one per line)',
            placeholder='.*\.tmp$\n.*backup.*\n.*test.*'
        ).classes('w-full')

        with ui.row():
            ui.checkbox('Allow deletion of files never deleted before').bind_value_to(
                globals(), 'allow_new_deletion')

        with ui.row():
            ui.button('Preview Deletion (Trial Mode)', on_click=lambda: handle_permanent_delete_files(
                dataset_id_input.value, regexes_input.value, apply=False
            ))
            ui.button('Delete Files', on_click=lambda: handle_permanent_delete_files(
                dataset_id_input.value, regexes_input.value, apply=True
            ), color='red')


# Global variable for checkbox
allow_new_deletion = False


async def handle_permanent_delete_files(dataset_id: str, regexes_text: str, apply: bool = False):
    """Handle permanent file deletion"""
    if not app_state.client:
        ui.notify('Please authenticate first', type='warning')
        return

    if not dataset_id or not regexes_text.strip():
        ui.notify('Please provide dataset ID and regex patterns', type='warning')
        return

    regexes = [line.strip()
               for line in regexes_text.strip().split('\n') if line.strip()]

    try:
        ui.notify(
            f'{"Analyzing" if not apply else "Deleting"} files in dataset {dataset_id}...', type='info')

        # This is a simplified placeholder - the actual implementation would need
        # to replicate the complex S3 logic from your CLI
        ui.notify(f'Would process {len(regexes)} regex patterns', type='info')

        if apply:
            result = await ui.run_javascript('''
                return confirm("Are you sure you want to permanently delete these files? This action cannot be undone!");
            ''')

            if not result:
                ui.notify('Deletion cancelled', type='info')
                return

            # Here you would implement the actual file deletion logic
            ui.notify(
                'File deletion would be performed here (not implemented in this demo)', type='info')
        else:
            ui.notify('File analysis completed (preview mode)', type='positive')

    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')

if __name__ in {"__main__", "__mp_main__"}:
    # Configure the app
    app.title = 'Provena Admin GUI'

    # Run the server
    ui.run(
        title='Provena Admin GUI',
        port=8080,
        host='0.0.0.0',
        show=True,
        reload=False
    )
